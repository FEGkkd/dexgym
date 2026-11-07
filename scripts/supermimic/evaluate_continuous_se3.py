"""Independent full-episode Newton evaluation for continuous dual-TCP tracking."""

import argparse
import json
from pathlib import Path

import numpy as np
import dexgym.tasks.marvin_supermimic  # noqa: F401
import gymnasium as gym
import torch
from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint
from dexgym.algorithms.supermimic.core.evaluator import PhysicalEvaluator
from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.tasks.marvin_supermimic.core_collector import CoreNewtonCollector
from dexgym.tasks.marvin_supermimic.physics_contract import (
    marvin_continuous_se3_physics_config, marvin_teleop_workspace_physics_config)
from dexgym.tasks.marvin_supermimic.se3_trajectory_env import MarvinContinuousSE3FollowCfg
from dexgym.tasks.marvin_supermimic.teleop_trajectory_env import MarvinTeleopWorkspaceCfg
from dexgym.tasks.marvin_supermimic.workspace_reach_env import MarvinWorkspaceSE3ReachHoldCfg
from dexgym.motion.teleop_workspace import reference_episode_control_steps
from isaaclab.app import launch_simulation


class ZeroPolicy(torch.nn.Module):
    def forward(self, history, goal):
        return history.new_zeros((len(history), 14))


class ReferenceServoDiagnostic(torch.nn.Module):
    """Explicit reference-joint physical feasibility check, never a training policy."""

    def __init__(self, native):
        super().__init__()
        self.native = native

    def forward(self, history, goal):
        native = self.native
        next_frame = torch.minimum(native.reference_frame + 1,
                                   native.reference.lengths[native.reference_id] - 1)
        q_ref = native.reference.reset_q_at(native.reference_id, next_frame)
        requested = ((q_ref - native.command) / native.action_map.delta_q_max).clamp(-0.95, 0.95)
        return torch.atanh(requested)


def _evaluation_task_profile(name):
    if name == "pinned":
        return (MarvinContinuousSE3FollowCfg,
                "DexGym-Marvin-SuperMimic-Continuous-SE3-Follow-Direct-v0",
                "marvin_bimanual_continuous_se3_follow")
    if name == "workspace":
        return (MarvinTeleopWorkspaceCfg,
                "DexGym-Marvin-SuperMimic-Teleop-Workspace-SE3-Direct-v0",
                "marvin_bimanual_teleop_workspace_se3_follow")
    if name == "workspace_reach":
        return (MarvinWorkspaceSE3ReachHoldCfg,
                "DexGym-Marvin-SuperMimic-Workspace-Reach-Hold-Direct-v0",
                "marvin_bimanual_workspace_se3_reach_hold")
    raise ValueError(f"Unknown SE(3) evaluation task: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--baseline", choices=("zero", "reference_pd"))
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--motion-id", type=int)
    parser.add_argument("--reference-bank", type=Path)
    parser.add_argument("--task", choices=("pinned", "workspace", "workspace_reach"),
                        default="pinned")
    parser.add_argument("--evaluation-split", choices=("train", "validation", "test"),
                        default="test")
    parser.add_argument("--evaluation-max-difficulty", type=int, choices=(0, 1, 2, 3),
                        default=2, help="Maximum held-out goal difficulty, inclusive")
    parser.add_argument("--allow-new-reference", action="store_true",
                        help="Explicitly evaluate the causal workspace Actor on a different target bank")
    parser.add_argument("--reset-perturb-rad", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 64 or not 0 <= args.reset_perturb_rad <= 0.03:
        parser.error("Independent evaluation uses 1-64 environments and bounded reset perturbation")
    saved = (torch.load(args.checkpoint, map_location="cpu", weights_only=False)
             if args.checkpoint else None)
    cfg_class, environment_id, expected_task = _evaluation_task_profile(args.task)
    workspace = args.task in ("workspace", "workspace_reach")
    if workspace and args.baseline == "reference_pd":
        parser.error("The causal workspace task has no joint-witness policy or IK baseline")
    if args.allow_new_reference and not workspace:
        parser.error("Reference source override is only supported for the causal workspace task")
    if saved and (saved.get("schema") != CoreCheckpoint.schema_version
                  or saved["task_spec"]["task_id"] != expected_task):
        parser.error("Expected a compatible continuous SE(3) Core checkpoint")
    if workspace and args.reference_bank is None:
        parser.error("Workspace evaluation requires --reference-bank")
    cfg = cfg_class()
    if args.task == "workspace_reach" and saved:
        task_version = saved["contract_manifest"]["task_contract"]["task_adapter_version"]
        if task_version == "marvin_workspace_se3_reach_hold_v1":
            cfg.reach_success_semantics = "pre_deadline_arrival_v1"
        elif task_version == "marvin_workspace_se3_reach_hold_v3":
            cfg.reach_temporal_context = "normalized_time_remaining_v3"
            cfg.observation_space = 10 * 69 + 18
        elif task_version != "marvin_workspace_se3_reach_hold_v2":
            parser.error(f"Unsupported workspace reach success contract: {task_version}")
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.evaluation_mode = True
    cfg.evaluation_motion_id = args.motion_id
    cfg.random_reference_start = False
    cfg.reset_joint_perturb_rad = args.reset_perturb_rad
    if workspace:
        cfg.reference_bank_path = str(args.reference_bank)
        cfg.evaluation_split = args.evaluation_split
        cfg.evaluation_max_difficulty = args.evaluation_max_difficulty
        with np.load(args.reference_bank, allow_pickle=False) as bank:
            cfg.reach_curriculum_version = str(bank["version"])
            if (args.task == "workspace_reach" and str(bank["version"]) not in
                    ("marvin_teleop_workspace_bank_v3", "marvin_teleop_workspace_bank_v4",
                     "marvin_teleop_workspace_bank_v5")):
                parser.error("Workspace reach-and-hold requires a static far-goal bank")
            cfg.episode_length_s = (bank["tcp_pos"].shape[1] - 1) / 50
    torch.manual_seed(args.seed)
    with launch_simulation(cfg):
        env = gym.make(environment_id, cfg=cfg)
        try:
            native = env.unwrapped
            robot, task = native.core_robot, native.follow_task
            if saved:
                network = NetworkSpec(**saved["contract_manifest"]["networks"]["actor"]["network_spec"])
                integration = saved["contract_manifest"]["networks"]["world_model"]["integration_version"]
                modules = NetworkFactory.build(
                    robot.spec, task.spec, network, robot_adapter=robot,
                    world_integration_version=integration)
                for module in (modules.world_model, modules.actor, modules.critic, modules.termination):
                    module.to(native.device)
                checkpoint = CoreCheckpoint(
                    git_commit=saved["git_commit"], reference_hashes=native.reference.hashes,
                    physics_config=(marvin_teleop_workspace_physics_config(cfg, robot)
                                    if workspace else marvin_continuous_se3_physics_config(cfg, robot)))
                load = checkpoint.load_actor_for_evaluation(
                    args.checkpoint, robot=robot, task=task, modules=modules,
                    allow_reference_override=args.allow_new_reference)
                policy = modules.actor
                before = {name: tensor.clone() for name, tensor in policy.state_dict().items()}
            else:
                load = None
                policy = ZeroPolicy() if args.baseline == "zero" else ReferenceServoDiagnostic(native)
            collector = CoreNewtonCollector(env, robot, task, action_noise_std=0)
            actual_reset_perturb = float((
                native.command - native.reference.reset_q_at(
                    native.reference_id, native.reference_frame)).abs().max())
            evaluator = PhysicalEvaluator(lambda: collector, task,
                                          evaluation_kind="independent_physics_evaluation")
            args.output.mkdir(parents=True, exist_ok=True)
            steps = reference_episode_control_steps(native.reference.lengths,
                                                    cfg.episode_length_s)
            report = evaluator.evaluate(policy, steps=steps, one_episode_per_env=True,
                                        trace_path=args.output / "physics_trace.pt")
            if saved and any(not torch.equal(value, policy.state_dict()[name])
                             for name, value in before.items()):
                raise RuntimeError("Independent evaluation modified Actor or normalizers")
            report.update({"checkpoint": str(args.checkpoint) if args.checkpoint else None,
                           "baseline": args.baseline, "checkpoint_load": load,
                           "seed": args.seed, "num_envs": args.num_envs,
                           "evaluation_motion_id": args.motion_id,
                           "evaluation_split": args.evaluation_split if workspace else None,
                           "evaluation_max_difficulty": (
                               args.evaluation_max_difficulty if workspace else None),
                           "reach_success_semantics": (
                               cfg.reach_success_semantics if args.task == "workspace_reach" else None),
                           "allow_new_reference": args.allow_new_reference,
                           "reference_names": native.reference.names,
                           "reference_hashes": native.reference.hashes,
                           "reset_perturb_rad": args.reset_perturb_rad,
                           "actual_reset_perturb_max_abs_rad": actual_reset_perturb,
                           "exploration_noise": 0.0})
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({key: value for key, value in report.items()
                              if key not in ("episodes", "reference_names")}), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
