"""Independent Newton process evaluation of one episode per environment from a v2 Core checkpoint."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.app import launch_simulation

import dexgym.tasks.marvin_supermimic  # noqa: F401 - Gym registration
from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint
from dexgym.algorithms.supermimic.core.evaluator import PhysicalEvaluator
from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.algorithms.supermimic.termination import TerminationThresholds
from dexgym.motion.library import MotionLibrary
from dexgym.tasks.marvin_supermimic.core_collector import CoreNewtonCollector
from dexgym.tasks.marvin_supermimic.core_plugins import (
    MarvinCoreRobot, MarvinDualTCPReach, MarvinMotionTracking)
from dexgym.tasks.marvin_supermimic.env_cfg import MarvinWaveDirectCfg
from dexgym.tasks.marvin_supermimic.physics_contract import marvin_physics_config
from dexgym.tasks.marvin_supermimic.reach_env import MarvinReachDirectCfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Tracking reference or Reach safe reset NPZ used by the checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=431)
    args = parser.parse_args()
    record = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if record.get("schema") != CoreCheckpoint.schema_version:
        parser.error("Independent Core evaluation requires a semantic v2 checkpoint")
    manifest = record["contract_manifest"]
    task_id = manifest["task_spec"]["task_id"]
    if task_id not in ("marvin_motion_tracking", "marvin_dual_tcp_reach"):
        parser.error("Checkpoint task is not a registered Marvin Newton task")
    tracking = task_id == "marvin_motion_tracking"
    cfg = MarvinWaveDirectCfg() if tracking else MarvinReachDirectCfg()
    if tracking:
        cfg.reference_path = str(args.reference)
    else:
        cfg.reset_state_path = str(args.reference)
    physics = record["physics_config"]
    cfg.action_schema = manifest["robot_spec"]["action_schema"]
    cfg.scene.num_envs = int(physics["num_envs"])
    cfg.seed = args.seed
    cfg.audit_contacts_each_physics_step = physics["audit_contacts_each_physics_step"]
    cfg.episode_length_s = physics["episode_length_s"]
    if tracking:
        for key in ("random_reference_start", "reset_joint_perturb_rad",
                    "active_reset_fraction", "early_reset_fraction",
                    "observation_position_noise_rad", "observation_velocity_noise_rad_s",
                    "initialize_hands_from_pd_replay"):
            stored = "reset_random_reference_start" if key == "random_reference_start" else key
            setattr(cfg, key, physics[stored])
    torch.manual_seed(args.seed)
    with launch_simulation(cfg):
        gym_id = ("DexGym-Marvin-SuperMimic-Wave-Direct-v0" if tracking else
                  "DexGym-Marvin-SuperMimic-Reach-Direct-v0")
        env = gym.make(gym_id, cfg=cfg)
        try:
            native = env.unwrapped
            params = manifest["task_contract"]
            thresholds = TerminationThresholds(**params["termination_thresholds"])
            if tracking:
                library = MotionLibrary({native.motion.motion_id: str(args.reference)}, native.device)
                robot = MarvinCoreRobot(native.motion, native.device, action_schema=cfg.action_schema)
                task = MarvinMotionTracking(library, robot,
                                            reward_profile=params["reward_profile"],
                                            termination_mode=record["termination_mode"],
                                            thresholds=thresholds)
                steps = native.motion.length - 1
            else:
                robot = native.core_robot
                task = MarvinDualTCPReach(robot, episode_length=params["episode_length"],
                                          thresholds=thresholds)
                steps = task.episode_length
            network = NetworkSpec(**manifest["networks"]["actor"]["network_spec"])
            modules = NetworkFactory.build(robot.spec, task.spec, network)
            modules.actor.to(native.device)
            hashes = {str(args.reference): hashlib.sha256(args.reference.read_bytes()).hexdigest()}
            checkpoint = CoreCheckpoint(
                git_commit=record["git_commit"], reference_hashes=hashes,
                physics_config=marvin_physics_config(cfg, robot, task_id=task.spec.task_id))
            load_report = checkpoint.load_actor_for_evaluation(
                args.checkpoint, robot=robot, task=task, modules=modules)
            collector_factory = lambda: CoreNewtonCollector(env, robot, task, action_noise_std=0.0)
            args.output.mkdir(parents=True, exist_ok=True)
            report = PhysicalEvaluator(
                collector_factory, task, evaluation_kind="independent_physics_evaluation"
            ).evaluate(modules.actor, steps=steps, one_episode_per_env=True,
                       trace_path=args.output / "physical_trace.pt")
            output = {"checkpoint": str(args.checkpoint), "reference": str(args.reference),
                      "seed": args.seed, "load": load_report,
                      "robot_spec": asdict(robot.spec), "task_spec": asdict(task.spec),
                      "evaluation": report}
            (args.output / "report.json").write_text(json.dumps(output, indent=2))
            print(json.dumps({"episode_count": report["episode_count"],
                              "completed_episode_count": report["completed_episode_count"],
                              "episode_success_rate": report["episode_success_rate"],
                              "hard_terminations": report["hard_terminations"],
                              "truncations": report["truncations"],
                              "report": str(args.output / "report.json")}), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
