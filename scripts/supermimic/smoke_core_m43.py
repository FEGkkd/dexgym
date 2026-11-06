"""Bounded M4.3 Newton integration smoke for two distinct Task MDP plugins."""

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import torch
from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint
from dexgym.algorithms.supermimic.core.config import CoreExperimentConfig
from dexgym.algorithms.supermimic.core.evaluator import PhysicalEvaluator
from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.observation_noise import ObservationNoiseModel
from dexgym.algorithms.supermimic.core.replay_buffer import SchemaReplay
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.algorithms.supermimic.core.trainer import SuperMimicTrainer, TrainerConfig
from dexgym.algorithms.supermimic.core.trust_evaluator import ModelTrustEvaluator, TrustConfig
from dexgym.algorithms.supermimic.registry import PluginRegistry
from dexgym.assets.supermimic import WAVE_REFERENCE
from dexgym.motion.library import MotionLibrary
from dexgym.tasks.marvin_supermimic.core_collector import CoreNewtonCollector
from dexgym.tasks.marvin_supermimic.env_cfg import MarvinWaveDirectCfg
from dexgym.tasks.marvin_supermimic.physics_contract import marvin_physics_config
from dexgym.tasks.marvin_supermimic.reach_env import PROCEDURAL_RESET, MarvinReachDirectCfg
from dexgym.tasks.marvin_supermimic.register_core import register_marvin
from isaaclab.app import launch_simulation

WAVE = str(WAVE_REFERENCE)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", choices=("tracking", "reach"))
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--termination-mode", choices=("analytic", "learned"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configured = CoreExperimentConfig.from_json(args.config) if args.config else None
    if configured is not None:
        if configured.backend != "newton" or configured.robot_id != "marvin_arm_only":
            parser.error("This physical launcher requires Newton and the Marvin adapter")
        selected = {"marvin_motion_tracking": "tracking",
                    "marvin_dual_tcp_reach": "reach"}.get(configured.task_id)
        if selected is None or (args.task is not None and args.task != selected):
            parser.error("Task config and selected Marvin physical task differ")
        args.task = selected
        args.reference_path = configured.reference_path
        args.action_schema = configured.action_schema
        args.network = configured.network
        args.trainer = configured.trainer
        args.trust = configured.trust
        args.observation_noise_spec = configured.observation_noise
        args.termination_mode = args.termination_mode or configured.termination_mode
    else:
        args.reference_path = WAVE if args.task == "tracking" else PROCEDURAL_RESET
        args.action_schema = "robot_fixed_v3"
        args.network = NetworkSpec(world_hidden=(64, 64), goal_hidden=(64,),
                                   actor_hidden=(128, 64), critic_hidden=(64,), latent_dim=16)
        args.trainer = TrainerConfig(world_batch=32, imagination_batch=8,
                                     world_updates_per_iteration=2,
                                     world_warmup_updates=8, trust_interval=4,
                                     validation_batch=16, critic_epochs=2, critic_minibatches=4)
        args.trust = TrustConfig(min_distinct_starts=4,
                                 max_rmse=(0.9, 1.0, 1.1),
                                 require_baseline_improvement=True, metric="normalized")
        args.observation_noise_spec = None
        args.termination_mode = args.termination_mode or "analytic"
    args.num_envs = args.num_envs or (configured.num_envs if configured else 16)
    args.iterations = args.iterations or (configured.iterations if configured else 40)
    args.eval_steps = args.eval_steps or (configured.eval_steps if configured else 12)
    args.seed = args.seed if args.seed is not None else (configured.seed if configured else 43)
    if args.task is None or not args.reference_path:
        parser.error("Select a task and a physical reset/reference path")
    if args.task == "reach" and args.termination_mode != "analytic":
        parser.error("Reach task currently supports analytic termination only")
    if args.num_envs < 4 or not 1 <= args.iterations <= 80 or not 1 <= args.eval_steps <= 50:
        parser.error("Smoke is bounded to 4+ environments, 1-80 iterations and 1-50 evaluation steps")
    return args


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    cfg = MarvinReachDirectCfg() if args.task == "reach" else MarvinWaveDirectCfg()
    if args.task == "reach":
        cfg.reset_state_path = args.reference_path
    else:
        cfg.reference_path = args.reference_path
    cfg.action_schema = args.action_schema
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.audit_contacts_each_physics_step = True
    if args.task == "tracking":
        cfg.initialize_hands_from_pd_replay = False
        cfg.random_reference_start = False
        cfg.reset_joint_perturb_rad = 0
    started = time.perf_counter()
    with launch_simulation(cfg):
        gym_id = ("DexGym-Marvin-SuperMimic-Reach-Direct-v0" if args.task == "reach" else
                  "DexGym-Marvin-SuperMimic-Wave-Direct-v0")
        env = gym.make(gym_id, cfg=cfg)
        try:
            native = env.unwrapped
            registry = PluginRegistry()
            register_marvin(registry)
            if args.task == "reach":
                robot = native.core_robot
                task = registry.task("marvin_dual_tcp_reach", robot=robot)
                hashes = {args.reference_path: hashlib.sha256(
                    Path(args.reference_path).read_bytes()).hexdigest()}
            else:
                library = MotionLibrary({native.motion.motion_id: args.reference_path}, native.device)
                robot = registry.robot("marvin_arm_only", reference_for_joint_order=native.motion,
                                       device=native.device, action_schema=cfg.action_schema)
                task = registry.task("marvin_motion_tracking", reference=library, robot=robot,
                                     termination_mode=args.termination_mode)
                hashes = {args.reference_path: library.hashes[native.motion.motion_id]}
            network = args.network
            modules = NetworkFactory.build(robot.spec, task.spec, network)
            observation_noise = (ObservationNoiseModel(args.observation_noise_spec,
                                                       seed=args.seed + 432)
                                 if args.observation_noise_spec is not None else None)
            for module in (modules.world_model, modules.actor, modules.critic, modules.termination):
                module.to(native.device)
            from dexgym.algorithms.supermimic.core.optimization import build_optimizers
            optimizers = build_optimizers(modules, world_lr=3e-4, actor_lr=2e-4,
                                          critic_lr=3e-4,
                                          termination_mode=task.termination_mode)
            train_envs = max(1, int(0.75 * args.num_envs))
            trainer = SuperMimicTrainer(
                collector=CoreNewtonCollector(env, robot, task, action_noise_std=0.03,
                                               observation_noise=observation_noise),
                replay=SchemaReplay(128, args.num_envs, robot.spec, task.spec, native.device),
                robot=robot, task=task, modules=modules, optimizers=optimizers,
                trust_evaluator=ModelTrustEvaluator(args.trust),
                config=args.trainer,
                train_env_range=(0, train_envs),
                validation_env_range=(train_envs, args.num_envs),
                observation_noise=observation_noise)
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            checkpoint = CoreCheckpoint(
                git_commit=commit, reference_hashes=hashes,
                physics_config=marvin_physics_config(cfg, robot, task_id=task.spec.task_id))
            resume_report = checkpoint.load(args.resume, trainer) if args.resume else None
            start_counts = trainer.counts.copy()
            logs, timings = [], {name: 0.0 for name in
                                 ("collection", "world", "trust", "actor_critic")}
            update_timings = {"actor_bptt_seconds": 0.0,
                              "critic_optimization_seconds": 0.0}
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            for _ in range(args.iterations):
                result = trainer.run_iteration()
                logs.append(result)
                for name, duration in result["phase_seconds"].items():
                    timings[name] += duration
                for update in result["actor"]:
                    for name in update_timings:
                        update_timings[name] += update.get(name, 0.0)
                if trainer.iteration % 10 == 0:
                    print(json.dumps({"iteration": trainer.iteration,
                                      "counts": trainer.counts,
                                      "trusted_horizon": trainer.allowed_horizon}), flush=True)
            eval_report = PhysicalEvaluator(
                lambda: CoreNewtonCollector(env, robot, task, action_noise_std=0,
                                             observation_noise=observation_noise,
                                             observation_mode="eval"),
                task).evaluate(modules.actor, steps=args.eval_steps)
            args.output.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint.save(args.output / f"{args.task}_m43_smoke.pt",
                                              trainer, include_replay=False)
            try:
                used, total = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                     "--format=csv,noheader,nounits"], text=True).splitlines()[0].split(",")
                device_memory = {"used_mb": int(used), "total_mb": int(total)}
            except (OSError, ValueError, subprocess.CalledProcessError):
                device_memory = None
            report = {"task": asdict(task.spec), "robot": asdict(robot.spec),
                      "network": asdict(network), "trainer": asdict(trainer.config),
                      "counts": trainer.counts, "trust": trainer.trust_report,
                      "new_counts": {name: trainer.counts[name] - start_counts[name]
                                     for name in trainer.counts},
                      "resume": resume_report,
                      "trusted_horizon": trainer.allowed_horizon,
                      "evaluation": eval_report, "phase_wall_seconds": timings,
                      "update_wall_seconds": update_timings,
                      "mean_world_update_seconds": timings["world"] / max(
                          trainer.counts["world_updates"] - start_counts["world_updates"], 1),
                      "mean_actor_bptt_seconds": update_timings["actor_bptt_seconds"] / max(
                          trainer.counts["actor_updates"] - start_counts["actor_updates"], 1),
                      "mean_critic_gradient_step_seconds":
                          update_timings["critic_optimization_seconds"] / max(
                              trainer.counts["critic_updates"] - start_counts["critic_updates"], 1),
                      "physical_transitions_per_collection_s":
                          args.iterations * args.num_envs / max(timings["collection"], 1e-9),
                      "peak_torch_vram_bytes": (torch.cuda.max_memory_allocated()
                                               if torch.cuda.is_available() else None),
                      "device_memory_at_end": device_memory,
                      "checkpoint": str(checkpoint_path), "git_commit": commit,
                      "wall_time_s": time.perf_counter() - started}
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
            (args.output / "iterations.jsonl").write_text("".join(
                json.dumps(value) + "\n" for value in logs))
            print(json.dumps(report), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
