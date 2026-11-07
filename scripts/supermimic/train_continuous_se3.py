"""Strict-pure WM8/BPTT32 training for continuous arm-only dual-TCP following."""

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import dexgym.tasks.marvin_supermimic  # noqa: F401
import gymnasium as gym
import torch
from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint
from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.optimization import build_optimizers
from dexgym.algorithms.supermimic.core.replay_buffer import SchemaReplay
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.algorithms.supermimic.core.trainer import SuperMimicTrainer, TrainerConfig
from dexgym.algorithms.supermimic.core.trust_evaluator import TrustConfig
from dexgym.tasks.marvin_supermimic.physics_contract import (
    marvin_continuous_se3_physics_config, marvin_teleop_workspace_physics_config)
from dexgym.tasks.marvin_supermimic.se3_collector import SE3CorrelatedExplorationCollector
from dexgym.tasks.marvin_supermimic.se3_trajectory_env import MarvinContinuousSE3FollowCfg
from dexgym.tasks.marvin_supermimic.teleop_trajectory_env import MarvinTeleopWorkspaceCfg
from dexgym.tasks.marvin_supermimic.workspace_reach_env import MarvinWorkspaceSE3ReachHoldCfg
from dexgym.tasks.marvin_supermimic.se3_trust import SE3DualDistributionTrustEvaluator, SE3GeometryTrustConfig
from isaaclab.app import launch_simulation

DEFAULT_CONFIG = Path("configs/supermimic/continuous_se3_follow_4096_full32_v1.json")


def _model_integration_version(experiment):
    version = experiment.get("world_model_integration", "robot_adapter_delta_v1")
    if version not in ("robot_adapter_delta_v1", "command_centered_residual_v2"):
        raise ValueError(f"Unknown World Model integration version: {version}")
    return version


def _task_profile(version):
    if version == "marvin_continuous_se3_follow_v1":
        return (MarvinContinuousSE3FollowCfg,
                "DexGym-Marvin-SuperMimic-Continuous-SE3-Follow-Direct-v0", None)
    if version in ("marvin_teleop_workspace_follow_v1",
                   "marvin_teleop_workspace_follow_v2",
                   "marvin_teleop_workspace_follow_v3"):
        return (MarvinTeleopWorkspaceCfg,
                "DexGym-Marvin-SuperMimic-Teleop-Workspace-SE3-Direct-v0",
                "marvin_teleop_workspace_bank_" + version.rsplit("_", 1)[-1])
    if version == "marvin_workspace_reach_hold_v1":
        return (MarvinWorkspaceSE3ReachHoldCfg,
                "DexGym-Marvin-SuperMimic-Workspace-Reach-Hold-Direct-v0",
                "marvin_teleop_workspace_bank_v3")
    if version in ("marvin_workspace_reach_hold_v2", "marvin_workspace_reach_hold_v3"):
        return (MarvinWorkspaceSE3ReachHoldCfg,
                "DexGym-Marvin-SuperMimic-Workspace-Reach-Hold-Direct-v0",
                "marvin_teleop_workspace_bank_v4")
    if version == "marvin_workspace_reach_hold_v4":
        return (MarvinWorkspaceSE3ReachHoldCfg,
                "DexGym-Marvin-SuperMimic-Workspace-Reach-Hold-Direct-v0",
                "marvin_teleop_workspace_bank_v5")
    raise ValueError(f"Unknown continuous SE(3) experiment version: {version}")


def _configured_components(experiment):
    network_fields = experiment["network"].copy()
    for key in ("world_hidden", "goal_hidden", "actor_hidden", "critic_hidden"):
        network_fields[key] = tuple(network_fields[key])
    trust_fields = experiment["trust"].copy()
    trust_fields["max_rmse"] = tuple(trust_fields["max_rmse"])
    geometry_fields = {key: tuple(values) for key, values in experiment["geometry_trust"].items()}
    return (NetworkSpec(**network_fields), TrainerConfig(**experiment["trainer"]),
            TrustConfig(**trust_fields), SE3GeometryTrustConfig(**geometry_fields))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-from", type=Path,
                        help="Explicit compatible model transfer; fresh Replay, optimizer and trust")
    parser.add_argument("--init-dynamics-from", type=Path,
                        help="Transfer only compatible World Model; keep random Actor and fresh Critic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-bank", type=Path,
                        help="FK-generated target bank for the causal workspace task")
    parser.add_argument("--curriculum-level", type=int, choices=(0, 1, 2, 3),
                        help="Explicitly expand the generated workspace target difficulty")
    args = parser.parse_args()
    if sum(path is not None for path in (args.resume, args.init_from,
                                        args.init_dynamics_from)) > 1:
        parser.error("Choose one of strict resume, full model transfer or dynamics-only transfer")
    config_bytes = args.config.read_bytes()
    experiment = json.loads(config_bytes)
    try:
        cfg_class, environment_id, bank_version = _task_profile(experiment.get("version"))
    except ValueError as error:
        parser.error(str(error))
    workspace = bank_version is not None
    if not workspace and (args.reference_bank is not None or args.curriculum_level is not None):
        parser.error("Workspace reference and curriculum overrides require the workspace config")
    num_envs = args.num_envs if args.num_envs is not None else experiment["num_envs"]
    iterations = args.iterations if args.iterations is not None else experiment["iterations_per_segment"]
    seed = args.seed if args.seed is not None else experiment["seed"]
    if not 16 <= num_envs <= 4096 or not 1 <= iterations <= experiment["iterations_per_segment"]:
        parser.error("Training segment must use 16-4096 environments and a bounded iteration count")
    network, schedule, trust, geometry = _configured_components(experiment)
    try:
        world_integration = _model_integration_version(experiment)
    except ValueError as error:
        parser.error(str(error))
    if world_integration == "command_centered_residual_v2" and not workspace:
        parser.error("Command-centered dynamics is only validated for the Marvin workspace task")
    if (schedule.mode != "pure_model_based" or schedule.reference_pd_bootstrap_steps
            or schedule.world_window != 8 or schedule.max_imagination_horizon != 32
            or experiment["minimum_actor_horizon"] != 32):
        parser.error("This task trains strict-pure WM8 with only trusted BPTT32 Actor updates")
    replay_capacity = int(experiment["replay_capacity_steps"])
    if replay_capacity < max(schedule.world_window, schedule.max_imagination_horizon):
        parser.error("Replay cannot hold a full World Model and imagination horizon")
    cfg = cfg_class()
    if experiment["version"] == "marvin_workspace_reach_hold_v1":
        cfg.reach_success_semantics = "pre_deadline_arrival_v1"
    if experiment["version"] in ("marvin_workspace_reach_hold_v3",
                                 "marvin_workspace_reach_hold_v4"):
        cfg.reach_temporal_context = "normalized_time_remaining_v3"
        cfg.observation_space = 10 * 69 + 18
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    if workspace:
        if args.reference_bank is None:
            parser.error("The workspace task requires --reference-bank")
        with np.load(args.reference_bank, allow_pickle=False) as bank:
            if str(bank["version"]) != bank_version:
                parser.error("Workspace training config and target bank version differ")
        cfg.reference_bank_path = str(args.reference_bank)
        cfg.reach_curriculum_version = bank_version
        cfg.episode_length_s = float(experiment.get("episode_length_s", 4.0))
        cfg.curriculum_level = (args.curriculum_level if args.curriculum_level is not None
                                else int(experiment["curriculum_level"]))
        cfg.train_env_fraction = 0.75
    else:
        cfg.include_reverse = bool(experiment["include_reverse"])
    cfg.random_reference_start = bool(experiment["random_reference_start"])
    cfg.max_reference_start_frame = int(experiment["max_reference_start_frame"])
    cfg.reset_joint_perturb_rad = float(experiment["reset_joint_perturb_rad"])
    source_reference = cfg.reference_bank_path if workspace else cfg.reference_path
    if hashlib.sha256(Path(source_reference).read_bytes()).hexdigest() != experiment["reference_sha256"]:
        parser.error("Reference bank or validated procedural reference hash differs")
    exploration = experiment["exploration"]
    train_count = int(0.75 * num_envs)
    exploratory_count = int(train_count * exploration["train_env_fraction"])
    val_noisy_count = int((num_envs - train_count) * exploration["validation_noisy_fraction"])
    if (exploratory_count < 1 or exploratory_count >= train_count
            or val_noisy_count < 1 or val_noisy_count >= num_envs - train_count):
        parser.error("Dual-distribution training/validation lanes must be nonempty")
    torch.manual_seed(seed)
    began = time.perf_counter()
    with launch_simulation(cfg):
        env = gym.make(environment_id, cfg=cfg)
        try:
            native = env.unwrapped
            robot, task = native.core_robot, native.follow_task
            modules = NetworkFactory.build(
                robot.spec, task.spec, network, robot_adapter=robot,
                world_integration_version=world_integration)
            for module in (modules.world_model, modules.actor, modules.critic, modules.termination):
                module.to(native.device)
            collector = SE3CorrelatedExplorationCollector(
                env, robot, task, seed=seed + 7001,
                rho=exploration["rho"], sigma=exploration["sigma"],
                exploration_env_count=exploratory_count,
                train_env_count=train_count,
                validation_exploration_env_count=val_noisy_count)
            replay = SchemaReplay(replay_capacity, num_envs, robot.spec, task.spec, native.device)
            optimizers = build_optimizers(modules, world_lr=3e-4, actor_lr=2e-4,
                                          critic_lr=3e-4, termination_mode=task.termination_mode)
            trainer = SuperMimicTrainer(
                collector=collector, replay=replay, robot=robot, task=task,
                modules=modules, optimizers=optimizers,
                trust_evaluator=SE3DualDistributionTrustEvaluator(
                    trust, geometry,
                    noisy_validation_range=(train_count, train_count + val_noisy_count),
                    deterministic_validation_range=(train_count + val_noisy_count, num_envs),
                    minimum_actor_horizon=32),
                config=schedule,
                train_env_range=(0, train_count),
                validation_env_range=(train_count, num_envs))
            checkpoint = CoreCheckpoint(
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                reference_hashes=native.reference.hashes,
                physics_config=(marvin_teleop_workspace_physics_config(cfg, robot)
                                if workspace else marvin_continuous_se3_physics_config(cfg, robot)))
            resume_info = (checkpoint.load(args.resume, trainer) if args.resume else
                           checkpoint.initialize_modules_for_new_reference(args.init_from, trainer)
                           if args.init_from else
                           checkpoint.initialize_dynamics_for_new_reference(
                               args.init_dynamics_from, trainer)
                           if args.init_dynamics_from else None)
            if trainer.iteration + iterations > experiment["max_total_iterations"]:
                raise ValueError("Training continuation exceeds the versioned iteration budget")
            args.output.mkdir(parents=True, exist_ok=True)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            with (args.output / "iterations.jsonl").open("w") as log:
                for _ in range(iterations):
                    result = trainer.run_iteration()
                    log.write(json.dumps(result) + "\n")
                    log.flush()
                    if trainer.iteration % 10 == 0:
                        print(json.dumps({"iteration": trainer.iteration,
                                          "counts": trainer.counts,
                                          "trusted_horizon": trainer.allowed_horizon}), flush=True)
            checkpoint_path = checkpoint.save(args.output / "model.pt", trainer,
                                              resume_mode="replay")
            report = {"task": asdict(task.spec), "robot": asdict(robot.spec),
                      "network": asdict(network), "world_model_integration": world_integration,
                      "trainer": asdict(schedule),
                      "trust_config": asdict(trust), "geometry_trust_config": asdict(geometry),
                      "reference_hashes": native.reference.hashes,
                      "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                      "counts": trainer.counts, "iteration": trainer.iteration,
                      "trust": trainer.trust_report, "trusted_horizon": trainer.allowed_horizon,
                      "replay_bytes_allocated": replay.bytes_allocated,
                      "strict_pure": True, "teacher_data": False,
                      "target_stream_contract": (
                          "static_far_goal_50hz_fixed_base_v1"
                          if experiment["version"] in ("marvin_workspace_reach_hold_v1",
                                                       "marvin_workspace_reach_hold_v2",
                                                       "marvin_workspace_reach_hold_v3",
                                                       "marvin_workspace_reach_hold_v4") else
                          "causal_current_50hz_fixed_base_v1" if workspace else
                          "future_10_frames_v1"),
                      "curriculum_level": cfg.curriculum_level if workspace else None,
                      "resume": resume_info, "checkpoint": str(checkpoint_path),
                      "wall_time_s": time.perf_counter() - began,
                      "peak_torch_vram_bytes": (torch.cuda.max_memory_allocated()
                                               if torch.cuda.is_available() else None)}
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({key: value for key, value in report.items()
                              if key not in ("task", "robot", "network", "trainer", "trust")}), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
