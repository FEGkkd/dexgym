"""Score static far-goal arrivals from an independent Newton physics trace.

The regular tracking success contract measures full-clip error. A distant
step target must instead be scored by dual TCP arrival and sustained hold.
This analysis reads only saved physical states and TCP goals; no goal q exists.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dexgym.algorithms.supermimic.differentiable_fk import DifferentiableMarvinFK
from dexgym.algorithms.supermimic.rewards import so3_geodesic
from dexgym.motion.teleop_workspace import DEFAULT_MJCF


def summarize_reach_errors(position_m, rotation_rad, net_speed_rad_s, *,
                           hold_steps=25, position_limit_m=0.005,
                           rotation_limit_rad=np.deg2rad(2),
                           speed_limit_rad_s=0.15):
    position_m = np.asarray(position_m)
    rotation_rad = np.asarray(rotation_rad)
    speed = np.asarray(net_speed_rad_s)
    if (position_m.ndim != 2 or position_m.shape[1] != 2
            or rotation_rad.shape != position_m.shape
            or speed.shape != (len(position_m),) or len(position_m) == 0
            or not np.isfinite(position_m).all()
            or not np.isfinite(rotation_rad).all() or not np.isfinite(speed).all()):
        raise ValueError("Reach errors must be finite aligned physical Episode arrays")
    position_ok = (position_m <= position_limit_m).all(1)
    complete_ok = position_ok & (rotation_rad <= rotation_limit_rad).all(1)
    hold_ok = complete_ok & (speed <= speed_limit_rad_s)
    first_position = np.flatnonzero(position_ok)
    first_complete = np.flatnonzero(complete_ok)
    first_hold = None
    streak = longest = 0
    for index, good in enumerate(hold_ok):
        streak = streak + 1 if good else 0
        longest = max(longest, streak)
        if first_hold is None and streak >= hold_steps:
            first_hold = index + 1
    return {"first_position_arrival_step": int(first_position[0] + 1) if len(first_position) else None,
            "first_complete_arrival_step": int(first_complete[0] + 1) if len(first_complete) else None,
            "first_hold_step": first_hold,
            "longest_hold_steps": longest,
            "success": first_hold is not None,
            "initial_position_m": position_m[0].tolist(),
            "final_position_m": position_m[-1].tolist(),
            "best_position_m": position_m.min(0).tolist(),
            "initial_rotation_deg": np.rad2deg(rotation_rad[0]).tolist(),
            "final_rotation_deg": np.rad2deg(rotation_rad[-1]).tolist(),
            "best_rotation_deg": np.rad2deg(rotation_rad.min(0)).tolist(),
            "max_net_joint_speed_rad_s": float(speed.max())}


def analyze(trace_path: Path, bank_path: Path):
    trace = torch.load(trace_path, map_location="cpu", weights_only=False)
    if trace.get("evaluation_kind") != "independent_physics_evaluation":
        raise ValueError("Reach probe requires independent Newton physics")
    transitions, masks = trace["transitions"], trace["evaluated_mask"]
    if len(transitions) != len(masks):
        raise ValueError("Trace and Episode masks differ")
    with np.load(bank_path, allow_pickle=False) as bank:
        if str(bank["version"]) != "marvin_teleop_workspace_bank_v3":
            raise ValueError("Expected a hidden-witness far-goal probe bank")
        goal_position = bank["tcp_pos"][:, 0].copy()
        goal_quaternion = bank["tcp_quat_xyzw"][:, 0].copy()
        difficulty = bank["difficulty"].copy()
    fk = DifferentiableMarvinFK(DEFAULT_MJCF, "cpu")
    episodes = []
    for env, physical_episode in enumerate(trace["episodes"]):
        valid = [step for step in range(len(transitions)) if bool(masks[step][env])]
        if not valid:
            continue
        motion_id = int(transitions[valid[0]].task_state.motion_id[env])
        if any(int(transitions[step].task_state.motion_id[env]) != motion_id for step in valid):
            raise ValueError("An Episode changed its goal bank ID")
        q = torch.stack([transitions[step].next_state[env, :14] for step in valid])
        q_previous = torch.stack([transitions[step].state[env, :14] for step in valid])
        with torch.no_grad():
            position, rotation = fk(q)
            target_rotation = torch.tensor(
                Rotation.from_quat(goal_quaternion[motion_id]).as_matrix(), dtype=q.dtype)
            pos_error = torch.linalg.vector_norm(
                position[:, :, 2] - torch.tensor(goal_position[motion_id], dtype=q.dtype), dim=-1)
            rot_error = so3_geodesic(rotation[:, :, 2], target_rotation[None])
        net_speed = ((q - q_previous) / 0.02).abs().amax(-1)
        result = summarize_reach_errors(pos_error.numpy(), rot_error.numpy(), net_speed.numpy())
        unsafe = any(bool(transitions[step].terminated[env]) for step in valid)
        result.update({"env_id": env, "motion_id": motion_id,
                       "difficulty": int(difficulty[motion_id]),
                       "physical_steps": len(valid), "hard_failure": unsafe,
                       "complete": bool(physical_episode["complete"]),
                       "success": bool(result["success"] and not unsafe
                                       and physical_episode["complete"])})
        episodes.append(result)
    return {"evaluation_kind": "independent_newton_far_goal_reach_hold_v1",
            "trace": str(trace_path), "reference_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
            "episodes": episodes, "episode_count": len(episodes),
            "success_rate": sum(ep["success"] for ep in episodes) / len(episodes),
            "hard_failure_count": sum(ep["hard_failure"] for ep in episodes),
            "mean_final_position_m": np.mean([ep["final_position_m"] for ep in episodes], axis=0).tolist(),
            "mean_final_rotation_deg": np.mean([ep["final_rotation_deg"] for ep in episodes], axis=0).tolist(),
            "mean_initial_position_m": np.mean([ep["initial_position_m"] for ep in episodes], axis=0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--reference-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.trace, args.reference_bank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
