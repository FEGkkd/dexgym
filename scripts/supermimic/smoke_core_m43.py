"""Bounded M4.3 Newton integration smoke for two distinct Task MDP plugins."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import gymnasium as gym
import torch
from isaaclab.app import launch_simulation

import dexgym.tasks.marvin_supermimic  # register both task-owned Newton environments
from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint
from dexgym.algorithms.supermimic.core.config import CoreExperimentConfig
from dexgym.algorithms.supermimic.core.evaluator import PhysicalEvaluator
from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.replay_buffer import SchemaReplay
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.algorithms.supermimic.core.trainer import SuperMimicTrainer, TrainerConfig
from dexgym.algorithms.supermimic.core.trust_evaluator import ModelTrustEvaluator, TrustConfig
from dexgym.algorithms.supermimic.registry import PluginRegistry
from dexgym.motion.library import MotionLibrary
from dexgym.tasks.marvin_supermimic.core_collector import CoreNewtonCollector
from dexgym.tasks.marvin_supermimic.core_plugins import (
    MarvinCoreRobot, MarvinDualTCPReach, MarvinMotionTracking)
from dexgym.tasks.marvin_supermimic.env_cfg import MarvinWaveDirectCfg
from dexgym.tasks.marvin_supermimic.reach_env import MarvinReachDirectCfg, PROCEDURAL_RESET
from dexgym.tasks.marvin_supermimic.register_core import register_marvin


WAVE = ("/home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/"
        "cmu_141_16_wave_bimanual.npz")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", choices=("tracking", "reach"))
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--seed", type=int)
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
    args.num_envs = args.num_envs or (configured.num_envs if configured else 16)
    args.iterations = args.iterations or (configured.iterations if configured else 40)
    args.eval_steps = args.eval_steps or (configured.eval_steps if configured else 12)
    args.seed = args.seed if args.seed is not None else (configured.seed if configured else 43)
    if args.task is None or not args.reference_path:
        parser.error("Select a task and a physical reset/reference path")
    if args.num_envs < 4 or not 1 <= args.iterations <= 80 or not 1 <= args.eval_steps <= 50:
        parser.error("Smoke is bounded to 4+ environments, 1-80 iterations and 1-50 evaluation steps")
    return args


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    cfg = MarvinReachDirectCfg() if args.task == "reach" else MarvinWaveDirectCfg()
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
                task = registry.task("marvin_motion_tracking", reference=library, robot=robot)
                hashes = {args.reference_path: library.hashes[native.motion.motion_id]}
            network = args.network
            modules = NetworkFactory.build(robot.spec, task.spec, network)
            for module in (modules.world_model, modules.actor, modules.critic, modules.termination):
                module.to(native.device)
            optimizers = {
                "world_model": torch.optim.Adam(modules.world_model.parameters(), lr=3e-4),
                "actor": torch.optim.Adam(modules.actor.parameters(), lr=2e-4),
                "critic": torch.optim.Adam(modules.critic.parameters(), lr=3e-4),
            }
            train_envs = max(1, int(0.75 * args.num_envs))
            trainer = SuperMimicTrainer(
                collector=CoreNewtonCollector(env, robot, task, action_noise_std=0.03),
                replay=SchemaReplay(128, args.num_envs, robot.spec, task.spec, native.device),
                robot=robot, task=task, modules=modules, optimizers=optimizers,
                trust_evaluator=ModelTrustEvaluator(args.trust),
                config=args.trainer,
                train_env_range=(0, train_envs),
                validation_env_range=(train_envs, args.num_envs))
            logs, timings = [], {name: 0.0 for name in
                                 ("collection", "world", "trust", "actor_critic")}
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            for _ in range(args.iterations):
                result = trainer.run_iteration()
                logs.append(result)
                for name, duration in result["phase_seconds"].items():
                    timings[name] += duration
                if trainer.iteration % 10 == 0:
                    print(json.dumps({"iteration": trainer.iteration,
                                      "counts": trainer.counts,
                                      "trusted_horizon": trainer.allowed_horizon}), flush=True)
            eval_report = PhysicalEvaluator(
                lambda: CoreNewtonCollector(env, robot, task, action_noise_std=0),
                task).evaluate(modules.actor, steps=args.eval_steps)
            args.output.mkdir(parents=True, exist_ok=True)
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            checkpoint = CoreCheckpoint(
                git_commit=commit, reference_hashes=hashes,
                physics_config={"num_envs": args.num_envs, "physics_dt": cfg.sim.dt,
                                "policy_dt": 0.02})
            checkpoint_path = checkpoint.save(args.output / f"{args.task}_m43_smoke.pt",
                                              trainer, include_replay=False)
            report = {"task": asdict(task.spec), "robot": asdict(robot.spec),
                      "network": asdict(network), "trainer": asdict(trainer.config),
                      "counts": trainer.counts, "trust": trainer.trust_report,
                      "trusted_horizon": trainer.allowed_horizon,
                      "evaluation": eval_report, "phase_wall_seconds": timings,
                      "physical_transitions_per_collection_s":
                          args.iterations * args.num_envs / max(timings["collection"], 1e-9),
                      "peak_torch_vram_bytes": (torch.cuda.max_memory_allocated()
                                               if torch.cuda.is_available() else None),
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
