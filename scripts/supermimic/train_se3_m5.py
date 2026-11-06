"""Bounded strict-pure SuperMimic pilot for IK-free Marvin SE(3) reach and hold."""

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

import dexgym.tasks.marvin_supermimic  # noqa: F401
from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint
from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.optimization import build_optimizers
from dexgym.algorithms.supermimic.core.replay_buffer import SchemaReplay
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.algorithms.supermimic.core.trainer import SuperMimicTrainer, TrainerConfig
from dexgym.algorithms.supermimic.core.trust_evaluator import ModelTrustEvaluator, TrustConfig
from dexgym.tasks.marvin_supermimic.physics_contract import marvin_se3_physics_config
from dexgym.tasks.marvin_supermimic.se3_collector import SE3CorrelatedExplorationCollector
from dexgym.tasks.marvin_supermimic.se3_env import MarvinSE3ReachHoldCfg


DEFAULT_CONFIG = Path("configs/supermimic/m5_se3_near_pilot.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--level", choices=("near", "medium", "wider_feasible"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    experiment = json.loads(config_bytes)
    if experiment.get("version") != "m5_se3_near_pilot_v1":
        parser.error("Unknown SE(3) experiment config version")
    args.num_envs = args.num_envs if args.num_envs is not None else experiment["num_envs"]
    args.iterations = args.iterations if args.iterations is not None else experiment["iterations"]
    args.seed = args.seed if args.seed is not None else experiment["seed"]
    args.level = args.level or experiment["curriculum_level"]
    network_fields = experiment["network"].copy()
    for name in ("world_hidden", "goal_hidden", "actor_hidden", "critic_hidden",
                 "termination_hidden"):
        if name in network_fields:
            network_fields[name] = tuple(network_fields[name])
    network = NetworkSpec(**network_fields)
    trust_fields = experiment["trust"].copy()
    trust_fields["max_rmse"] = tuple(trust_fields["max_rmse"])
    trust = TrustConfig(**trust_fields)
    config = TrainerConfig(**experiment["trainer"])
    exploration = experiment["exploration"]
    if not 0 <= exploration["rho"] < 1 or exploration["sigma"] < 0:
        parser.error("Invalid temporal exploration config")
    if not 16 <= args.num_envs <= 256 or not 0 <= args.iterations <= 200:
        parser.error("Pilot is bounded to 16-256 environments and at most 200 iterations")
    cfg = MarvinSE3ReachHoldCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.goal_train_seed = args.seed + 101
    cfg.goal_eval_seed = args.seed + 100101
    cfg.curriculum_level = args.level
    cfg.audit_contacts_each_physics_step = True
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    with launch_simulation(cfg):
        env = gym.make("DexGym-Marvin-SuperMimic-SE3-Reach-Hold-Direct-v0", cfg=cfg)
        try:
            native = env.unwrapped
            robot, task = native.core_robot, native.se3_task
            modules = NetworkFactory.build(robot.spec, task.spec, network, robot_adapter=robot)
            for module in (modules.world_model, modules.actor, modules.critic, modules.termination):
                module.to(native.device)
            collector = SE3CorrelatedExplorationCollector(
                env, robot, task, seed=args.seed + 7001,
                rho=exploration["rho"], sigma=exploration["sigma"])
            replay = SchemaReplay(512, args.num_envs, robot.spec, task.spec, native.device)
            optimizers = build_optimizers(modules, world_lr=3e-4, actor_lr=2e-4,
                                          critic_lr=3e-4,
                                          termination_mode=task.termination_mode)
            if config.mode != "pure_model_based" or config.reference_pd_bootstrap_steps:
                raise ValueError("This entry is strict-pure and cannot use teacher data")
            train_count = max(1, int(0.75 * args.num_envs))
            trainer = SuperMimicTrainer(
                collector=collector, replay=replay, robot=robot, task=task,
                modules=modules, optimizers=optimizers,
                trust_evaluator=ModelTrustEvaluator(trust), config=config,
                train_env_range=(0, train_count),
                validation_env_range=(train_count, args.num_envs))
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            checkpoint = CoreCheckpoint(
                git_commit=git_commit, reference_hashes={},
                physics_config=marvin_se3_physics_config(cfg, robot))
            resume = checkpoint.load(args.resume, trainer) if args.resume else None
            args.output.mkdir(parents=True, exist_ok=True)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            log_path = args.output / "iterations.jsonl"
            with log_path.open("w") as log:
                for _ in range(args.iterations):
                    result = trainer.run_iteration()
                    log.write(json.dumps(result) + "\n")
                    log.flush()
                    if trainer.iteration % 10 == 0:
                        print(json.dumps({"iteration": trainer.iteration,
                                          "counts": trainer.counts,
                                          "trusted_horizon": trainer.allowed_horizon,
                                          "trust": trainer.trust_report}), flush=True)
            checkpoint_path = checkpoint.save(args.output / "model.pt", trainer,
                                              resume_mode="replay")
            report = {"task": asdict(task.spec), "robot": asdict(robot.spec),
                      "network": asdict(network), "trainer": asdict(config),
                      "trust_config": asdict(trust), "counts": trainer.counts,
                      "config_path": str(args.config),
                      "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                      "exploration": exploration,
                      "trust": trainer.trust_report,
                      "allowed_horizon": trainer.allowed_horizon,
                      "replay_steps": replay.steps, "resume": resume,
                      "strict_pure": True, "teacher_data": False,
                      "goal_curriculum_level": args.level,
                      "checkpoint": str(checkpoint_path), "git_commit": git_commit,
                      "peak_torch_vram_bytes": (torch.cuda.max_memory_allocated()
                                               if torch.cuda.is_available() else None),
                      "wall_time_s": time.perf_counter() - started}
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
