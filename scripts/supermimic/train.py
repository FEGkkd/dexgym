"""Unified M4.2 Newton / learned world-model SuperMimic trainer."""

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.app import launch_simulation

from dexgym.algorithms.supermimic.checkpoint import CheckpointManager
from dexgym.algorithms.supermimic.networks import DeltaWorldModel, GoalActor, PrivilegedCritic
from dexgym.algorithms.supermimic.replay_buffer import MotionReplay
from dexgym.algorithms.supermimic.trainer import SuperMimicTrainer, TrainerConfig
from dexgym.algorithms.supermimic.trust_evaluator import ModelTrustEvaluator
from dexgym.tasks.marvin_supermimic.collector import AlignedNewtonCollector
from dexgym.tasks.marvin_supermimic.env_cfg import MarvinWaveDirectCfg
from dexgym.tasks.marvin_supermimic.task_adapter import MarvinTrackingTask
from dexgym.tasks.marvin_supermimic.robot_adapter import DEFAULT_MJCF


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("pure_model_based", "teacher_bootstrap_model_based",
                                          "supervised_init_model_based"), default="pure_model_based")
    parser.add_argument("--action-schema", choices=("legacy_wave_v1", "robot_fixed_v2", "robot_fixed_v3"), default="robot_fixed_v3")
    parser.add_argument("--reward-profile", default="paper_multilink")
    parser.add_argument("--termination-mode", choices=("analytic", "learned"), default="analytic")
    parser.add_argument("--normalization-mode", choices=("legacy", "running"), default="running")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--physics-steps-per-iteration", type=int, default=1)
    parser.add_argument("--replay-steps", type=int, default=400)
    parser.add_argument("--world-hidden", type=int, default=256)
    parser.add_argument("--world-batch", type=int, default=64)
    parser.add_argument("--imag-batch", type=int, default=32)
    parser.add_argument("--world-updates-per-iteration", type=int, default=1)
    parser.add_argument("--actor-updates-per-iteration", type=int, default=1)
    parser.add_argument("--world-warmup-updates", type=int, default=30)
    parser.add_argument("--reference-pd-bootstrap-steps", type=int, default=0)
    parser.add_argument("--supervised-init-updates", type=int, default=0)
    parser.add_argument("--critic-epochs", type=int, default=2)
    parser.add_argument("--critic-minibatches", type=int, default=4)
    parser.add_argument("--trust-interval", type=int, default=25)
    parser.add_argument("--validation-batch", type=int, default=32)
    parser.add_argument("--max-imagination-horizon", type=int, default=32)
    parser.add_argument("--action-noise", type=float, default=0.05)
    parser.add_argument("--reset-perturb-rad", type=float, default=0.01)
    parser.add_argument("--random-reference-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=140116)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.supervised_init_updates and args.mode != "supervised_init_model_based":
        parser.error("Supervised actor initialization is only legal in supervised_init_model_based")
    if args.mode == "supervised_init_model_based" and not args.supervised_init_updates:
        parser.error("supervised_init_model_based requires --supervised-init-updates")
    if args.iterations < 1 or args.num_envs < 2 or args.checkpoint_interval < 1:
        parser.error("Need positive iterations/checkpoint interval and at least two environments")
    return args


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    actor = GoalActor(goal_latent_tanh=True, normalization_mode=args.normalization_mode).to(device)
    model = DeltaWorldModel(hidden=args.world_hidden).to(device)
    critic = PrivilegedCritic(normalization_mode=args.normalization_mode).to(device)
    optimizers = {
        "actor": torch.optim.Adam(actor.parameters(), lr=2e-5, betas=(0.9, 0.95)),
        "world_model": torch.optim.Adam(model.parameters(), lr=2e-4),
        "critic": torch.optim.Adam(critic.parameters(), lr=2e-4),
    }
    cfg = MarvinWaveDirectCfg()
    cfg.reference_path = str(args.reference)
    cfg.action_schema = args.action_schema
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.reward_profile = args.reward_profile
    cfg.random_reference_start = args.random_reference_start
    cfg.reset_joint_perturb_rad = args.reset_perturb_rad
    cfg.initialize_hands_from_pd_replay = args.action_schema == "legacy_wave_v1"
    cfg.audit_contacts_each_physics_step = True
    cfg.episode_length_s = 28.0
    config = TrainerConfig(
        mode=args.mode, physics_steps_per_iteration=args.physics_steps_per_iteration,
        world_updates_per_iteration=args.world_updates_per_iteration, world_batch=args.world_batch,
        imagination_batch=args.imag_batch, actor_updates_per_iteration=args.actor_updates_per_iteration,
        max_imagination_horizon=args.max_imagination_horizon,
        world_warmup_updates=args.world_warmup_updates,
        reference_pd_bootstrap_steps=args.reference_pd_bootstrap_steps,
        critic_epochs=args.critic_epochs, critic_minibatches=args.critic_minibatches,
        trust_interval=args.trust_interval, validation_batch=args.validation_batch)
    metadata = {
        "algorithm_version": "m42_model_based_v1", "robot_adapter_version": "marvin_arms_v2",
        "task_adapter_version": "marvin_tracking_v2", "action_schema": args.action_schema,
        "state_schema": "arm_q14_qd14_command14_v1",
        "observation_schema": "arm_proprio_history_10x56_v1",
        "goal_schema": "six_link_future_10x90_v1",
        "reference_hashes": {str(args.reference): sha256(args.reference)},
        "robot_asset_hashes": {"mjcf": sha256(DEFAULT_MJCF)},
        "termination_mode": args.termination_mode,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    manager = CheckpointManager(metadata)
    record = None
    if args.resume:
        record = manager.load(args.resume, models={"actor": actor, "world_model": model, "critic": critic},
                              optimizers=optimizers, map_location=device)
        if record["resume_requires_recollection"]:
            print("Replay was not saved; collecting new Newton transitions before model trust.", flush=True)
    training_log = args.output / "training.jsonl"
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with launch_simulation(cfg):
        env = gym.make("DexGym-Marvin-SuperMimic-Wave-Direct-v0", cfg=cfg)
        try:
            native = env.unwrapped
            robot = native.dynamics_adapter
            task = MarvinTrackingTask(native.motion, robot, reward_profile=args.reward_profile,
                                      termination_mode=args.termination_mode)
            replay = MotionReplay(args.replay_steps, args.num_envs, device)
            if args.supervised_init_updates:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from train_wave import pd_actor_examples
                history, goal, target = pd_actor_examples(native.motion, robot.action_map, device)
                for _ in range(args.supervised_init_updates):
                    index = torch.randint(len(history), (128,), device=device)
                    optimizers["actor"].zero_grad(set_to_none=True)
                    loss = (actor(history[index], goal[index]) - target[index]).square().mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    optimizers["actor"].step()
            collector = AlignedNewtonCollector(env, actor, replay, action_noise_std=args.action_noise)
            train_envs = max(1, int(args.num_envs * 0.8))
            trainer = SuperMimicTrainer(
                collector=collector, replay=replay, robot=robot, task=task,
                world_model=model, actor=actor, critic=critic,
                world_optimizer=optimizers["world_model"], actor_optimizer=optimizers["actor"],
                critic_optimizer=optimizers["critic"],
                trust_evaluator=ModelTrustEvaluator(robot.fk, native.motion), config=config,
                train_env_range=(0, train_envs), validation_env_range=(train_envs, args.num_envs))
            if record is not None:
                trainer.iteration = int(record["training"]["iteration"])
                trainer.counts.update(record["training"]["counts"])
                trainer.trust_report = {}
                trainer.allowed_horizon = 0
            for _ in range(args.iterations):
                result = trainer.run_iteration()
                result["wall_time_s"] = time.perf_counter() - started
                with training_log.open("a") as stream:
                    stream.write(json.dumps(_jsonable(result)) + "\n")
                if trainer.iteration % args.checkpoint_interval == 0 or trainer.iteration == args.iterations:
                    manager.save(args.output / f"iteration_{trainer.iteration:06d}.pt",
                                 models={"actor": actor, "world_model": model, "critic": critic},
                                 optimizers=optimizers,
                                 training={"iteration": trainer.iteration,
                                           "real_transitions": trainer.counts["real_transitions"],
                                           "counts": trainer.counts.copy(), "trust": trainer.trust_report},
                                 config={**vars(args), **dataclasses.asdict(config)}, seed=args.seed)
                if trainer.iteration % 10 == 0:
                    print(json.dumps({"iteration": trainer.iteration, "counts": trainer.counts,
                                      "horizon": trainer.allowed_horizon,
                                      "last_world": result["world"][-1],
                                      "last_actor": result["actor"][-1],
                                      "wall_time_s": result["wall_time_s"]}), flush=True)
        finally:
            env.close()
    summary = {"counts": trainer.counts, "peak_torch_vram_bytes": torch.cuda.max_memory_allocated(),
               "wall_time_s": time.perf_counter() - started, "metadata": metadata,
               "config": _jsonable({**vars(args), **dataclasses.asdict(config)})}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
