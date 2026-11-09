"""Real-Newton hand-only collector, pre-reset, FK and WM8 acceptance check."""

import argparse
import json
from pathlib import Path

import torch
from isaaclab.app import launch_simulation
from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.optimization import build_optimizers, safe_gradient_step
from dexgym.algorithms.supermimic.core.replay_buffer import SchemaReplay
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.tasks.direct.right_hand_tracking.env import RightHandTrackingDirectEnv
from dexgym.tasks.direct.right_hand_tracking.teledexter_env_cfg import WujiHand21SuperMimicCfg
from dexgym.tasks.direct.right_hand_tracking.supermimic_robot import Hand21DynamicsAdapter
from dexgym.tasks.direct.right_hand_tracking.supermimic_task import Hand21Task
from dexgym.tasks.direct.right_hand_tracking.supermimic_model import HandWorldModel
from dexgym.tasks.direct.right_hand_tracking.supermimic_collector import HandNewtonCollector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--replay-capacity', type=int, help='Allocate full replay without extending this short run')
    args = parser.parse_args()
    cfg = WujiHand21SuperMimicCfg()
    cfg.library_path, cfg.scene.num_envs, cfg.seed = args.library, args.num_envs, 42
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    torch.manual_seed(42)
    report = {'physics': 'standalone Hand21 Newton 480Hz/30Hz', 'trained_policy': False}
    with launch_simulation(cfg):
        env = RightHandTrackingDirectEnv(cfg)
        try:
            robot = Hand21DynamicsAdapter(device=env.device)
            task = Hand21Task(robot, env.bank, cfg)
            config = NetworkSpec(world_hidden=(128, 128), goal_hidden=(64,),
                                 actor_hidden=(128, 64), critic_hidden=(64,), latent_dim=16)
            modules = NetworkFactory.build(robot.spec, task.spec, config, robot_adapter=robot)
            modules.world_model = HandWorldModel(robot.spec, config, task)
            for module in (modules.world_model, modules.actor, modules.critic, modules.termination):
                module.to(env.device)
            collector = HandNewtonCollector(env, robot, task, action_noise_std=.1)
            capacity = args.replay_capacity or args.steps
            if capacity < args.steps:
                raise ValueError('Smoke replay capacity must retain the complete short run')
            replay = SchemaReplay(capacity, args.num_envs, robot.spec, task.spec, env.device)
            optimizers = build_optimizers(modules, world_lr=2e-4, actor_lr=2e-4, critic_lr=2e-4,
                                         termination_mode=task.termination_mode)
            max_fk_error = 0.
            terminal_count = reach_count = 0
            # Force a one-step failure on one lane. Other lanes retain normal rules.
            env.fail_frames[0] = torch.ceil(env.fail_budget[0]).long() + 1
            env.dwell[0] = 0
            pre_reset_difference = None
            for i in range(args.steps):
                delta = torch.linalg.vector_norm(robot.fk(robot.physical_state(env)[:, :20]) - env._current_tips(), dim=-1)
                max_fk_error = max(max_fk_error, float(delta.max()))
                batch = collector.step(modules.actor)
                replay.append(batch)
                modules.world_model.update_norms(batch.state, batch.action, batch.dynamics_context, batch.next_state, robot)
                terminal_count += int(batch.done.sum())
                reach_count += int(env.last_hand_reached.sum())
                if i == 0:
                    assert batch.terminated[0], 'Forced failure must terminate now'
                    pre_reset_difference = float((batch.next_state[0] - robot.physical_state(env)[0]).abs().max())
                    assert pre_reset_difference > 1e-5, 'Collector must preserve pre-reset arrival'
            window = replay.sample_windows(32, 8)
            loss, errors = modules.world_model.autoregressive_loss(
                window, robot, termination_head=modules.termination, task=task)
            parameters = list(modules.world_model.parameters()) + list(modules.termination.parameters())
            result = safe_gradient_step(loss, parameters, optimizers['world_model'])
            assert result['updated'], result
            assert max_fk_error < 1e-4, max_fk_error
            assert terminal_count > 0
            report.update({'fk_max_error_m': max_fk_error, 'transitions': args.steps * args.num_envs,
                           'terminals': terminal_count, 'reached': reach_count,
                           'pre_reset_state_difference': pre_reset_difference,
                           'world8_loss': float(loss.detach()), 'world_update': result,
                           'reach_head': modules.world_model.last_event_metrics, 'passed': True})
            report.update({'replay_capacity': capacity, 'replay_bytes': replay.bytes_allocated,
                           'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                           'torch_peak_reserved_bytes': torch.cuda.max_memory_reserved()})
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
