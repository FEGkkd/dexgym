"""Train the original standalone Hand21 task with real WM8 and imagined BPTT32."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch
from isaaclab.app import launch_simulation
from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint
from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.optimization import build_optimizers
from dexgym.algorithms.supermimic.core.replay_buffer import SchemaReplay
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.algorithms.supermimic.core.trainer import SuperMimicTrainer, TrainerConfig
from dexgym.algorithms.supermimic.core.trust_evaluator import TrustConfig
from dexgym.tasks.direct.right_hand_tracking.env import RightHandTrackingDirectEnv
from dexgym.tasks.direct.right_hand_tracking.teledexter_env_cfg import WujiHand21SuperMimicCfg
from dexgym.tasks.direct.right_hand_tracking.supermimic_robot import Hand21DynamicsAdapter
from dexgym.tasks.direct.right_hand_tracking.supermimic_task import Hand21Task
from dexgym.tasks.direct.right_hand_tracking.supermimic_model import HandWorldModel
from dexgym.tasks.direct.right_hand_tracking.supermimic_collector import HandNewtonCollector
from dexgym.tasks.direct.right_hand_tracking.supermimic_trust import HandModelTrust


def components(env, experiment):
    robot = Hand21DynamicsAdapter(device=env.device)
    task = Hand21Task(robot, env.bank, env.cfg)
    network_fields = experiment['network'].copy()
    for key in ('world_hidden', 'goal_hidden', 'actor_hidden', 'critic_hidden', 'termination_hidden'):
        network_fields[key] = tuple(network_fields[key])
    network = NetworkSpec(**network_fields)
    modules = NetworkFactory.build(robot.spec, task.spec, network, robot_adapter=robot)
    dynamics_signature = modules.world_model.dynamics_signature
    modules.world_model = HandWorldModel(robot.spec, network, task)
    modules.world_model.dynamics_signature = dynamics_signature
    for module in (modules.world_model, modules.actor, modules.critic, modules.termination):
        module.to(env.device)
    return robot, task, modules


def physics_contract(cfg, robot):
    from dexgym.assets.wuji_hand21.robot_cfg import OFFICIAL_STIFFNESS, OFFICIAL_DAMPING, OFFICIAL_EFFORT_LIMIT
    return {'num_envs': cfg.scene.num_envs, 'seed': cfg.seed, 'control_dt': 1/30,
            'physics_dt': cfg.sim.dt, 'integrator': 'implicitfast', 'gravity': tuple(cfg.sim.gravity),
            'stiffness': OFFICIAL_STIFFNESS, 'damping': OFFICIAL_DAMPING,
            'effort_limits': OFFICIAL_EFFORT_LIMIT, 'action_scale': .1, 'deadzone': .1,
            'robot': robot.checkpoint_contract(), 'full_self_collision': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/supermimic/hand_s0_train.json'))
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, help='Bound this segment; total schedule stays in config')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    experiment = json.loads(args.config.read_text())
    if experiment['version'] != 'hand21_s0_v1':
        raise ValueError('Unknown experiment version')
    cfg = WujiHand21SuperMimicCfg()
    cfg.library_path, cfg.scene.num_envs, cfg.seed = str(args.library), experiment['num_envs'], experiment['seed']
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    torch.manual_seed(cfg.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'experiment.json').write_text(json.dumps(experiment, indent=2))
    started = time.monotonic()
    with launch_simulation(cfg):
        env = RightHandTrackingDirectEnv(cfg)
        try:
            robot, task, modules = components(env, experiment)
            collector = HandNewtonCollector(env, robot, task, action_noise_std=experiment['action_noise_std'])
            replay = SchemaReplay(experiment['replay_capacity_steps'], env.num_envs, robot.spec, task.spec, env.device)
            lr = experiment['lr']
            optimizers = build_optimizers(modules, world_lr=lr, actor_lr=lr, critic_lr=lr,
                                         termination_mode=task.termination_mode, actor_betas=(.9, .95))
            trust = experiment['trust'].copy()
            trust['max_rmse'] = tuple(trust['max_rmse'])
            train_count = int(env.num_envs * experiment['train_env_fraction'])
            trainer = SuperMimicTrainer(
                collector=collector, replay=replay, robot=robot, task=task, modules=modules,
                optimizers=optimizers, trust_evaluator=HandModelTrust(TrustConfig(**trust), experiment['tip_mean_error_limits_m']),
                config=TrainerConfig(**experiment['trainer']), train_env_range=(0, train_count),
                validation_env_range=(train_count, env.num_envs))
            checkpoint = CoreCheckpoint(
                git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                reference_hashes={'manifest': hashlib.sha256(args.library.read_bytes()).hexdigest()},
                physics_config=physics_contract(cfg, robot))
            resume = checkpoint.load(args.resume, trainer) if args.resume else None
            if resume:
                env.curriculum_steps[:] = trainer.counts['real_transitions'] // env.num_envs
                # Sample fresh goals AFTER restoring curriculum progress; the
                # collector built before checkpoint.load used the initial stage.
                collector = HandNewtonCollector(env, robot, task, action_noise_std=experiment['action_noise_std'])
                trainer.collector = collector
                resume['restored_curriculum_steps'] = int(env.curriculum_steps[0])
            iterations = args.iterations or experiment['iterations'] - trainer.iteration
            if iterations < 1 or trainer.iteration + iterations > experiment['iterations']:
                raise ValueError('Training segment exceeds configured budget')
            with (args.output / 'iterations.jsonl').open('a') as log:
                for _ in range(iterations):
                    fraction = trainer.iteration / experiment['iterations']
                    scale = experiment['cosine_floor'] + (1 - experiment['cosine_floor']) * .5 * (1 + math.cos(math.pi * fraction))
                    for optimizer in optimizers.values():
                        for group in optimizer.param_groups:
                            group['lr'] = lr * scale
                    result = trainer.run_iteration()
                    result.update({'tracking': collector.last_metrics, 'event_model': modules.world_model.last_event_metrics,
                                   'elapsed_seconds': time.monotonic() - started, 'lr': lr * scale})
                    log.write(json.dumps(result) + '\n')
                    log.flush()
                    if trainer.iteration % 10 == 0:
                        print(json.dumps({'iteration': trainer.iteration, 'tracking': collector.last_metrics,
                                          'horizon': trainer.allowed_horizon, 'counts': trainer.counts}), flush=True)
                    if trainer.iteration % experiment['checkpoint_interval'] == 0:
                        checkpoint.save(args.output / f'model_{trainer.iteration}.pt', trainer, resume_mode='fresh_physics')
            saved = checkpoint.save(args.output / f'model_{trainer.iteration}.pt', trainer, resume_mode='fresh_physics')
            report = {'iteration': trainer.iteration, 'counts': trainer.counts, 'trust': trainer.trust_report,
                      'allowed_horizon': trainer.allowed_horizon, 'replay_bytes': replay.bytes_allocated,
                      'elapsed_seconds': time.monotonic() - started, 'checkpoint': str(saved), 'resume': resume,
                      'policy_accuracy_passed': None, 'split': 'train', 'robot': asdict(robot.spec), 'task': asdict(task.spec)}
            (args.output / 'segment_report.json').write_text(json.dumps(report, indent=2))
        finally:
            env.close()


if __name__ == '__main__':
    main()
