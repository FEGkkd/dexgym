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
from dexgym.algorithms.supermimic.scheduling import step_schedulers_after_updates
from dexgym.algorithms.supermimic.trainer import SuperMimicTrainer, TrainerConfig
from dexgym.algorithms.supermimic.trust_evaluator import ModelTrustEvaluator
from dexgym.algorithms.supermimic.termination import TaskTerminationHead
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
    parser.add_argument("--legacy-world-head", action="store_true",
                        help="Match M4.2 v1 dynamics initialization for controlled ablations")
    parser.add_argument("--world-batch", type=int, default=64)
    parser.add_argument("--imag-batch", type=int, default=32)
    parser.add_argument("--world-updates-per-iteration", type=int, default=1)
    parser.add_argument("--actor-updates-per-iteration", type=int, default=1)
    parser.add_argument("--world-warmup-updates", type=int, default=30)
    parser.add_argument("--reference-pd-bootstrap-steps", type=int, default=0)
    parser.add_argument("--boundary-probe-steps", type=int, default=0)
    parser.add_argument("--termination-pos-weight", type=float, default=1.0)
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
    parser.add_argument("--actor-lr", type=float, default=2e-5)
    parser.add_argument("--world-lr", type=float, default=2e-4)
    parser.add_argument("--critic-lr", type=float, default=2e-4)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default=None)
    parser.add_argument("--schedule-total-iterations", type=int)
    parser.add_argument("--min-lr-fraction", type=float, default=0.1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-replay", action=argparse.BooleanOptionalAction, default=None,
                        help="Save full physical Replay; defaults on for learned termination")
    args = parser.parse_args()
    if args.supervised_init_updates and args.mode != "supervised_init_model_based":
        parser.error("Supervised actor initialization is only legal in supervised_init_model_based")
    if args.mode == "supervised_init_model_based" and not args.supervised_init_updates:
        parser.error("supervised_init_model_based requires --supervised-init-updates")
    if args.iterations < 1 or args.num_envs < 2 or args.checkpoint_interval < 1:
        parser.error("Need positive iterations/checkpoint interval and at least two environments")
    if min(args.actor_lr, args.world_lr, args.critic_lr) <= 0 or not 0 <= args.min_lr_fraction <= 1:
        parser.error("Learning rates must be positive and minimum LR fraction within [0, 1]")
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
    preview = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    previous_config = preview["config"] if preview is not None else {}
    args.lr_schedule = args.lr_schedule or previous_config.get("lr_schedule", "constant")
    if args.schedule_total_iterations is None:
        args.schedule_total_iterations = previous_config.get("schedule_total_iterations")
    if args.lr_schedule == "cosine" and args.schedule_total_iterations is None:
        args.schedule_total_iterations = args.iterations
    for key, legacy_default in (("actor_lr", 2e-5), ("world_lr", 2e-4),
                                ("critic_lr", 2e-4), ("min_lr_fraction", 0.1),
                                ("lr_schedule", "constant"), ("schedule_total_iterations", None)):
        if preview is not None and getattr(args, key) != previous_config.get(key, legacy_default):
            raise ValueError(f"Resume learning-rate contract differs: {key}")
    if args.lr_schedule == "cosine" and args.schedule_total_iterations < 1:
        raise ValueError("Cosine schedule needs a positive total iteration count")
    separate_task_head = preview is None or "termination" in preview["models"]
    legacy_model_head = args.legacy_world_head or (preview is not None and any(
        name.startswith("termination_head.") for name in preview["models"]["world_model"]))
    actor = GoalActor(goal_latent_tanh=True, normalization_mode=args.normalization_mode).to(device)
    model = DeltaWorldModel(hidden=args.world_hidden,
                            legacy_termination_head=legacy_model_head).to(device)
    termination = (TaskTerminationHead(42, 90).to(device)
                   if args.termination_mode == "learned" and separate_task_head else None)
    critic = PrivilegedCritic(normalization_mode=args.normalization_mode).to(device)
    models = {"actor": actor, "world_model": model, "critic": critic}
    if termination is not None:
        models["termination"] = termination
    optimizers = {
        "actor": torch.optim.Adam(actor.parameters(), lr=args.actor_lr, betas=(0.9, 0.95)),
        "world_model": torch.optim.Adam(
            list(model.parameters()) + (list(termination.parameters()) if termination is not None else []),
            lr=args.world_lr),
        "critic": torch.optim.Adam(critic.parameters(), lr=args.critic_lr),
    }
    schedulers = ({name: torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.schedule_total_iterations,
        eta_min=optimizer.param_groups[0]["lr"] * args.min_lr_fraction)
        for name, optimizer in optimizers.items()} if args.lr_schedule == "cosine" else {})
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
        boundary_probe_steps=args.boundary_probe_steps,
        termination_pos_weight=args.termination_pos_weight,
        critic_epochs=args.critic_epochs, critic_minibatches=args.critic_minibatches,
        trust_interval=args.trust_interval, validation_batch=args.validation_batch)
    metadata = {
        "algorithm_version": (preview["metadata"]["algorithm_version"] if preview else
                              "m42_model_based_v1" if args.legacy_world_head else
                              "m42_model_based_v2_split_task_head"),
        "robot_adapter_version": "marvin_arms_v2",
        "task_adapter_version": "marvin_tracking_v2", "action_schema": args.action_schema,
        "state_schema": "arm_q14_qd14_command14_v1",
        "observation_schema": "arm_proprio_history_10x56_v1",
        "goal_schema": "six_link_future_10x90_v1",
        "reference_hashes": {str(args.reference): sha256(args.reference)},
        "robot_asset_hashes": {"mjcf": sha256(DEFAULT_MJCF)},
        "termination_mode": args.termination_mode,
        "reward_profile": args.reward_profile,
        "normalization_mode": args.normalization_mode,
        "lr_schedule": args.lr_schedule,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    manager = CheckpointManager(metadata)
    record = None
    replay = MotionReplay(args.replay_steps, args.num_envs, device)
    if args.resume:
        record = manager.load(args.resume, models=models,
                              optimizers=optimizers, schedulers=schedulers,
                              map_location="cpu", replay=replay)
        if record["resume_requires_recollection"]:
            print("Replay was not saved; collecting new Newton transitions before model trust.", flush=True)
            if args.termination_mode == "learned":
                print("Recollecting hard failure probes so the learned head retains positive labels.", flush=True)
    training_log = args.output / "training.jsonl"
    started = time.perf_counter()
    phase_seconds = {"collection": 0.0, "world": 0.0, "trust": 0.0, "actor_critic": 0.0}
    torch.cuda.reset_peak_memory_stats()
    with launch_simulation(cfg):
        env = gym.make("DexGym-Marvin-SuperMimic-Wave-Direct-v0", cfg=cfg)
        try:
            native = env.unwrapped
            robot = native.dynamics_adapter
            task = MarvinTrackingTask(native.motion, robot, reward_profile=args.reward_profile,
                                      termination_mode=args.termination_mode,
                                      termination_module=termination)
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
                if record["resume_requires_recollection"]:
                    trainer.resume_recollection_steps_required = 32
                    if args.termination_mode == "learned":
                        trainer.resume_boundary_probe_steps_remaining = 32
            starting_counts = trainer.counts.copy()
            for local_iteration in range(1, args.iterations + 1):
                before_updates = trainer.counts.copy()
                result = trainer.run_iteration()
                step_schedulers_after_updates(schedulers, before_updates, trainer.counts,
                                              max_steps=args.schedule_total_iterations or 0)
                result["learning_rates"] = {name: optimizer.param_groups[0]["lr"]
                                            for name, optimizer in optimizers.items()}
                for phase, seconds in result["phase_seconds"].items():
                    phase_seconds[phase] += seconds
                result["wall_time_s"] = time.perf_counter() - started
                with training_log.open("a") as stream:
                    stream.write(json.dumps(_jsonable(result)) + "\n")
                if trainer.iteration % args.checkpoint_interval == 0 or local_iteration == args.iterations:
                    manager.save(args.output / f"iteration_{trainer.iteration:06d}.pt",
                                 models=models,
                                 optimizers=optimizers,
                                 schedulers=schedulers,
                                 training={"iteration": trainer.iteration,
                                           "real_transitions": trainer.counts["real_transitions"],
                                           "counts": trainer.counts.copy(), "trust": trainer.trust_report},
                                 config={**vars(args), **dataclasses.asdict(config)}, seed=args.seed,
                                 replay=(replay if (args.save_replay if args.save_replay is not None
                                                    else args.termination_mode == "learned") else None))
                if trainer.iteration % 10 == 0:
                    print(json.dumps({"iteration": trainer.iteration, "counts": trainer.counts,
                                      "horizon": trainer.allowed_horizon,
                                      "last_world": result["world"][-1],
                                      "last_actor": result["actor"][-1],
                                      "wall_time_s": result["wall_time_s"]}), flush=True)
        finally:
            env.close()
    elapsed = time.perf_counter() - started
    try:
        gpu_memory_mb = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        gpu_memory_mb = None
    summary = {"counts": trainer.counts, "peak_torch_vram_bytes": torch.cuda.max_memory_allocated(),
               "torch_reserved_vram_bytes": torch.cuda.memory_reserved(),
               "device_memory_used_total_mb": gpu_memory_mb,
               "replay_bytes_allocated": replay.bytes_allocated,
               "phase_wall_time_s": phase_seconds,
               "physics_transitions_per_collection_s": (
                   args.iterations * args.physics_steps_per_iteration * args.num_envs /
                   max(phase_seconds["collection"], 1e-9)),
               "world_updates_per_world_phase_s": (
                   args.iterations * args.world_updates_per_iteration /
                   max(phase_seconds["world"], 1e-9)),
               "actor_updates_per_actor_phase_s": (
                   (trainer.counts["actor_updates"] - starting_counts["actor_updates"]) /
                   max(phase_seconds["actor_critic"], 1e-9)),
               "physics_transitions_per_s": (trainer.counts["real_transitions"] - starting_counts["real_transitions"]) / elapsed,
               "world_updates_per_s": (trainer.counts["world_updates"] - starting_counts["world_updates"]) / elapsed,
               "actor_updates_per_s": (trainer.counts["actor_updates"] - starting_counts["actor_updates"]) / elapsed,
               "critic_updates_per_s": (trainer.counts["critic_updates"] - starting_counts["critic_updates"]) / elapsed,
               "wall_time_s": time.perf_counter() - started, "metadata": metadata,
               "config": _jsonable({**vars(args), **dataclasses.asdict(config)})}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
