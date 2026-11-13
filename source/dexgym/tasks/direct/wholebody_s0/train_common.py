"""Common Isaac Lab launch and provenance for the independent learner tasks."""

import argparse
import json
from pathlib import Path
import random
import time

import mujoco
import numpy as np
import torch

from ..marvin54_s0.physics import open_physics
from .bag import digest
from .domain import DELAY_PROBABILITY, IMPULSE_PROBABILITY, PHYSICAL_FIELDS, SCALE_RANGES
from .prepare import write_json
from .fingertip import HAND_SCALE
from .intent import GOAL_VERSION, INTENT_DIM
from .residual import (DR_DIM, INTENT_EXECUTION_COEF, INTENT_RESIDUAL_COEF, INTENT_SMOOTHNESS_COEF,
                       N, OBS_DIM, POSE_GOAL_VERSION, POSE_HAND_WEIGHTS, POSE_TRACKING_SCALES,
                       REWARD_PROFILES, ReferenceBank)
from .robot import NEWTON_USD
from .residual_env import ResidualEnv


def train_parser(method, *, default_control_steps=1024):
    p = argparse.ArgumentParser(description=f"Train wholebody_s0 with {method} and Isaac Lab Newton")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--control-steps", type=int, default=default_control_steps)
    p.add_argument("--checkpoints", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--robust", action="store_true")
    p.add_argument("--transfer-model", type=Path, help="Blend the teleoperation XML's servo and gravity profiles in robust Newton worlds")
    p.add_argument("--reward-profile", choices=REWARD_PROFILES, default="v1")
    p.add_argument("--init", type=Path)
    return p


def atomic_checkpoint(value, path):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def run(args, task_class, learn):
    if min(args.num_envs, args.control_steps, args.checkpoints) <= 0:
        raise ValueError("Environment, budget and checkpoint counts must be positive")
    if args.transfer_model and not args.robust:
        raise ValueError("--transfer-model requires --robust")
    resume = getattr(args, "resume", None)
    if args.reward_profile == "fingertip" and (not args.transfer_model or not (args.init or resume)):
        raise ValueError("The fingertip profile needs a transfer model and a previously trained arm policy")
    if args.reward_profile in ("human_intent", "retarget_pose") and not args.transfer_model:
        raise ValueError(f"{args.reward_profile} needs the teleoperation model for task-space targets")
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    source = args.manifest.resolve(strict=True)
    manifest = json.loads(source.read_text())
    model_info = manifest["provenance"]["robot_model"]
    model_path = Path(model_info["path"]).resolve(strict=True)
    if digest(model_path) != model_info["sha256"] or "native_replay" not in manifest:
        raise ValueError("Use the verified native replay Motion Library and its matching packaged robot")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    transfer_model, transfer_info = None, None
    if args.transfer_model:
        transfer_path = args.transfer_model.resolve(strict=True)
        transfer_model = mujoco.MjModel.from_xml_path(str(transfer_path))
        transfer_info = {"path": str(transfer_path), "sha256": digest(transfer_path)}
    device = "cuda:0"
    bank = ReferenceBank(source, "train", model, device,
                         intent_model=transfer_model if args.reward_profile == "human_intent" else None)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    with open_physics("newton", model, bank.names, args.num_envs, robot_usd=NEWTON_USD) as backend:
        env = ResidualEnv(backend, bank, task_class, robust=args.robust, seed=args.seed,
                          transfer_model=transfer_model, reward_profile=args.reward_profile)
        contract = env.contract()
        task = env.task
        provenance = {
            "task_id": task.task_id, "method": task.method, "manifest": str(source),
            "manifest_sha256": bank.manifest_sha256, "model": str(model_path), "model_sha256": model_info["sha256"],
            "train_clips": len(bank.entries), "train_frames": bank.frames,
            "train_episodes": sorted({entry["source_episode"] for entry in bank.entries}),
            "joint_names": bank.names, "policy_dim": task.policy_dim, "context_dim": task.context_dim,
            "dynamics_dim": task.dynamics_dim, "transfer_model": transfer_info,
            "action_scale_rad": task.scale.tolist(), "command_delta_limit_rad": task.slew.tolist(),
            "reward_profile": task.reward_profile, "reward_weights": task.reward_weights,
            "seed": args.seed, "num_envs": args.num_envs,
            "control_steps": args.control_steps, "robust": args.robust,
            "init": str(args.init.resolve()) if args.init else None, "newton": contract,
            "setup_wall_seconds": time.perf_counter() - started,
        }
        if task.fingertip_priority:
            provenance["hand_servo"] = {
                "fingertip_site_model": transfer_info, "hand_scale_rad": HAND_SCALE,
                "architecture": "per_hand_mlp_v1", "frozen_arm": True,
            }
        if task.human_intent:
            provenance["intent"] = {
                "geometry_model": transfer_info, "goal_version": GOAL_VERSION,
                "goal_feature_dim": INTENT_DIM, "architecture": "arm_and_per_hand_goal_adapters",
                "execution_coefficient": INTENT_EXECUTION_COEF, "smoothness_coefficient": INTENT_SMOOTHNESS_COEF,
                "residual_coefficient": INTENT_RESIDUAL_COEF,
            }
        if task.retarget_pose:
            provenance["pose_tracking"] = {
                "geometry_model": transfer_info, "goal_version": POSE_GOAL_VERSION,
                "reference": "forward_kinematics_of_q_ref", "wrist_sites": ["tcp_L", "tcp_R"],
                "wrist_relative": "right_tcp_in_left_tcp", "error_scales": POSE_TRACKING_SCALES,
                "hand_weights": POSE_HAND_WEIGHTS,
            }
        if args.robust:
            provenance["domain_profile"] = {
                "multipliers": {field: list(bounds) for field, bounds in zip(PHYSICAL_FIELDS, SCALE_RANGES, strict=True)},
                "delay_probability": DELAY_PROBABILITY, "delay_s": .02,
                "mean_velocity_impulses_per_control_interval": IMPULSE_PROBABILITY,
            }
        args.init_privileged = False
        args.init_v1_arm = False
        args.init_intent_actor = False
        args.init_pose_actor = False
        args.init_dynamics = False
        if args.init:
            prior = json.loads((args.init.parent / "provenance.json").read_text())
            args.init_v1_arm = task.fingertip_priority and prior.get("reward_profile", "v1") == "v1"
            args.init_intent_actor = task.human_intent and prior.get("reward_profile") == "fingertip"
            args.init_pose_actor = task.retarget_pose and prior.get("reward_profile") == "fingertip"
            cross_profile = args.init_intent_actor or args.init_pose_actor
            keys = ["task_id", "model_sha256", "joint_names", "command_delta_limit_rad"]
            if cross_profile:
                keys += ["action_scale_rad", "transfer_model"]
                if set(prior["train_episodes"]) - set(provenance["train_episodes"]):
                    raise ValueError("The starting actor used recordings outside the new training split")
                if prior["policy_dim"] != OBS_DIM or prior["hand_servo"]["architecture"] != "per_hand_mlp_v1":
                    raise ValueError("This warm start needs the existing fingertip actor")
            else:
                keys.append("manifest_sha256")
            if not (args.init_v1_arm or cross_profile):
                keys += ["action_scale_rad", "policy_dim", "reward_weights"]
                if task.fingertip_priority:
                    keys.append("hand_servo")
                if task.human_intent:
                    keys.append("intent")
                if task.retarget_pose:
                    keys.append("pose_tracking")
            for key in keys:
                if prior[key] != provenance[key]:
                    raise ValueError(f"Initialization contract mismatch: {key}")
            args.init_dynamics = (prior.get("dynamics_dim", 2 * N + DR_DIM) == task.dynamics_dim
                                  and prior.get("transfer_model") == transfer_info)
            args.init_privileged = (not (args.init_v1_arm or cross_profile)
                                    and prior.get("context_dim", OBS_DIM + DR_DIM + 1) == task.context_dim
                                    and prior.get("transfer_model") == transfer_info)
            if args.init_v1_arm:
                provenance["initialization_scope"] = "frozen_v1_arm"
            elif cross_profile:
                provenance["initialization_scope"] = ("fingertip_actor_and_compatible_dynamics" if task.method == "supermimic"
                                                     else "fingertip_actor_weights")
            else:
                provenance["initialization_scope"] = "all_weights" if args.init_privileged else "policy_weights"
            provenance["init_sha256"] = digest(args.init)
        if resume:
            prior = json.loads((resume.parent / "provenance.json").read_text())
            keys = ("task_id", "method", "manifest_sha256", "model_sha256", "joint_names", "policy_dim", "context_dim",
                    "action_scale_rad", "command_delta_limit_rad", "reward_weights", "hand_servo", "transfer_model",
                    "domain_profile", "seed", "num_envs", "robust", "intent", "pose_tracking")
            for key in keys:
                if prior.get(key) != provenance.get(key):
                    raise ValueError(f"Resume contract mismatch: {key}")
            if prior.get("reward_profile", "v1") != task.reward_profile:
                raise ValueError("Resume contract mismatch: reward_profile")
            provenance["resume"] = str(resume.resolve())
            provenance["resume_sha256"] = digest(resume)
            provenance["initialization_scope"] = "learner_state"
        write_json(out / "provenance.json", provenance)
        print("WHOLEBODY_NEWTON", json.dumps(contract), flush=True)
        result = learn(args, env)
        result.update(total_wall_seconds=time.perf_counter() - started,
                      peak_torch_bytes=torch.cuda.max_memory_allocated())
        if result.get("cumulative_sim_transitions", result["sim_transitions"]) != args.num_envs * args.control_steps:
            raise RuntimeError(f"Physical control budget was not met: {result}")
        write_json(out / "training_proof.json", result)
        print("WHOLEBODY_TRAINING_PROOF", json.dumps(result), flush=True)
