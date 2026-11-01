"""Fresh-process PhysX Reset replay of one unvalidated scripted Mimic candidate.

The candidate is injected only into this one-environment diagnostic bank. It
is never promoted to a validated training bank by this script. Bottle, cap,
hands, fingers and screw are restored at an episode boundary, then advanced
under gravity with zero actions and normal contact physics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidate-result", type=Path, required=True)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--external-forces-every-iteration", action="store_true")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 50:
    parser.error("At least 50 policy steps are required for a one-second replay")
app = AppLauncher(args).app

import torch
import warp as wp
from isaaclab_physx.physics import PhysxManager

from dexteroushands_lab.scenes.grasp_init import grasp_sample_from_dict
from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env import BimanualBottleDirectEnv
from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env_cfg import GraspHoldDirectEnvCfg


def _angle(q0: torch.Tensor, q1: torch.Tensor) -> float:
    dot = torch.dot(q0, q1).abs().clamp(max=1)
    return float(2 * torch.acos(dot))


def main():
    probe = json.loads(args.candidate_result.read_text())
    if "diagnostic_candidate" not in probe:
        raise ValueError("Probe has no captured live PhysX candidate")
    sample = grasp_sample_from_dict(probe["diagnostic_candidate"])
    if (sample.validated or not sample.metadata.get("mimic_candidate_unverified")
            or sample.source != "scripted_physx_contact_candidate_requires_fresh_reset_replay"):
        raise ValueError("Only unvalidated scripted Mimic candidates may be replayed")
    if (sample.metadata.get("enable_external_forces_every_iteration", False)
            != args.external_forces_every_iteration):
        raise ValueError("Replay TGS force-iteration setting differs from candidate generation")

    cfg = GraspHoldDirectEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.thread_backend = "mimic"
    cfg.right_cap_contact_gate = "manifold_sidewall"
    cfg.sim.physics.enable_external_forces_every_iteration = args.external_forces_every_iteration
    cfg.stable_reset_probability = 0.0
    env = BimanualBottleDirectEnv(cfg)
    try:
        for side, spec in zip(("left", "right"), env.specs):
            if sample.metadata.get(f"{side}_usd_sha256") != spec.usd_sha256:
                raise ValueError(f"Candidate {side} official USD hash mismatch")
        if sample.metadata.get("bottle_parameters") != vars(env.bottle_cfg):
            raise ValueError("Candidate bottle configuration differs from live scene")
        env.reset()

        # Replace only this diagnostic process's pregrasp rows. The actual
        # episode-boundary _reset_idx path restores every selected state.
        first = env._stable_count
        for name in ("left", "right", "bottle", "cap"):
            state = getattr(sample, name)
            for field in ("root_pose", "root_velocity"):
                env._bank[f"{name}_{field}"][first:] = torch.tensor(
                    getattr(state, field), device=env.device)
        for name in ("left", "right"):
            state = getattr(sample, name)
            for field in ("joint_position", "joint_velocity", "joint_target", "wrist_target_pose"):
                env._bank[f"{name}_{field}"][first:] = torch.tensor(
                    getattr(state, field), device=env.device)
        env._bank["thread_angle"][first:] = sample.thread_initial_angle_rad
        env._bank["thread_spin_velocity"][first:] = sample.metadata["mimic_spin_velocity_rad_s"]
        drive_state = sample.metadata["mimic_drive_state"]
        env._bank["thread_stick_reference"][first:] = drive_state["stick_reference_rad"]
        env._bank["thread_sliding"][first:] = drive_state["sliding"]
        env._bank["thread_breakaway_seen"][first:] = drive_state["breakaway_seen"]
        env._reset_idx(torch.tensor([0], device=env.device))
        origin = env.scene.env_origins[0]
        expected = torch.tensor(sample.bottle.root_pose, device=env.device)
        expected[:3] += origin
        state = env.thread.view.read()
        reset_error = {
            "bottle_pose_max_abs_m_or_quat": float((state.root_pose_w[0] - expected).abs().max()),
            "cap_pose_max_abs_m_or_quat": float((state.cap_pose_w[0] - torch.tensor(
                sample.cap.root_pose, device=env.device)).abs().max()),
            "theta_abs_rad": float((state.theta_rad[0] - sample.thread_initial_angle_rad).abs()),
            "spin_abs_rad_s": float((state.spin_velocity_rad_s[0]
                                   - sample.metadata["mimic_spin_velocity_rad_s"]).abs()),
            "stick_reference_abs_rad": float((env.thread.drive.stick_reference[0]
                                              - drive_state["stick_reference_rad"]).abs()),
            "sliding_mismatch": float(bool(env.thread.drive.sliding[0]) != drive_state["sliding"]),
            "breakaway_seen_mismatch": float(bool(env.thread.drive.breakaway_seen[0])
                                             != drive_state["breakaway_seen"]),
        }
        for side, hand in (("left", env.left), ("right", env.right)):
            canonical = env.specs[0 if side == "left" else 1].joint_names
            actual = torch.stack([hand.data.joint_pos.torch[0, hand.joint_names.index(name)]
                                  for name in canonical])
            target = torch.tensor(getattr(sample, side).joint_position, device=env.device)
            reset_error[f"{side}_joint_max_abs_rad"] = float((actual - target).abs().max())

        # DirectRLEnv auto-resets on success. Keep this diagnostic episode
        # running for a full second while still computing the real done gate.
        original_get_dones = env._get_dones

        def keep_episode_for_measurement():
            terminated, truncated = original_get_dones()
            return torch.zeros_like(terminated), torch.zeros_like(truncated)

        env._get_dones = keep_episode_for_measurement
        initial_bottle = env.thread.view.read().root_pose_w[0].clone()
        rows = []
        max_left_force = max_right_force = 0.0
        min_separation = math.inf
        view = PhysxManager.get_physics_sim_view()
        contact_views = []
        for obj, side, hand_name in (("Bottle", "left", "LeftHand"),
                                     ("Cap", "right", "RightHand")):
            contact_views.append(view.create_rigid_contact_view(
                f"/World/envs/env_0/{obj}",
                filter_patterns=[f"/World/envs/env_0/{hand_name}/{body}"
                                 for body in env.contacts.body_names[side]],
                max_contact_data_count=256))
        for step in range(args.steps):
            _, reward, _, _, extras = env.step(torch.zeros(1, 52, device=env.device))
            screw = env.thread.view.read()
            bottle = screw.root_pose_w[0]
            drift = float(torch.linalg.vector_norm(bottle[:3] - initial_bottle[:3]))
            tilt = _angle(bottle[3:], initial_bottle[3:])
            left = env.contacts.force_magnitudes("Bottle", "left")[0]
            right = env.contacts.force_magnitudes("Cap", "right")[0]
            max_left_force = max(max_left_force, float(left.max()))
            max_right_force = max(max_right_force, float(right.max()))
            if step == 0:
                for contact_view in contact_views:
                    data = contact_view.get_contact_data(dt=cfg.sim.dt)
                    distances = wp.to_torch(data[3]).reshape(-1)
                    counts = wp.to_torch(data[4]).reshape(-1).to(torch.int64)
                    starts = wp.to_torch(data[5]).reshape(-1).to(torch.int64)
                    if int(counts.sum()) > len(distances):
                        raise RuntimeError("Initial contact view overflowed")
                    for count, start in zip(counts.tolist(), starts.tolist()):
                        if count:
                            min_separation = min(min_separation,
                                                 float(distances[start:start + count].min()))
            episode = extras["phase1_episode"]
            rows.append({
                "step": step + 1, "time_s": (step + 1) * env.step_dt,
                "bottle_pose_w_xyzw": bottle.detach().cpu().tolist(),
                "bottle_velocity_w": screw.root_velocity_w[0].detach().cpu().tolist(),
                "bottle_drift_from_replay_m": drift,
                "bottle_rotation_from_replay_rad": tilt,
                "cap_spin_rad_s": float(screw.spin_velocity_rad_s[0]),
                "thread_error_m": float(episode["constraint_error_m"][0]),
                "left_contact": bool(episode["left_contact"][0]),
                "right_sidewall_tripod": bool(episode["right_contact"][0]),
                "bottle_stable": bool(episode["bottle_stable"][0]),
                "strict_success": bool(episode["success"][0]),
                "failure": bool(episode["failure"][0]),
                "ground_contact": bool(episode["ground_contact"][0]),
                "reward": float(reward[0]),
            })
        left_ratio = sum(row["left_contact"] for row in rows) / len(rows)
        right_ratio = sum(row["right_sidewall_tripod"] for row in rows) / len(rows)
        result = {
            "status": "FRESH_PROCESS_PHYSX_RESET_REPLAY_DIAGNOSTIC",
            "source_result": str(args.candidate_result),
            "source_sample_index": probe["sample_index"],
            "enable_external_forces_every_iteration": args.external_forces_every_iteration,
            "candidate_validated": False,
            "reset_error": reset_error,
            "duration_s": args.steps * env.step_dt,
            "left_contact_ratio": left_ratio,
            "right_sidewall_tripod_ratio": right_ratio,
            "maximum_bottle_drift_m": max(row["bottle_drift_from_replay_m"] for row in rows),
            "maximum_bottle_rotation_rad": max(row["bottle_rotation_from_replay_rad"] for row in rows),
            "maximum_left_net_body_force_n": max_left_force,
            "maximum_right_net_body_force_n": max_right_force,
            "initial_min_contact_separation_m": (min_separation if math.isfinite(min_separation)
                                                 else None),
            "maximum_thread_error_m": max(abs(row["thread_error_m"]) for row in rows),
            "ground_contact_any": any(row["ground_contact"] for row in rows),
            "failure_any": any(row["failure"] for row in rows),
            "strict_success_reached": any(row["strict_success"] for row in rows),
            "rows": rows,
        }
        result["diagnostic_pass"] = (
            max(reset_error.values()) < 1e-3 and result["strict_success_reached"]
            and left_ratio > 0.8 and right_ratio > 0.8
            and result["maximum_bottle_drift_m"] < 0.02
            and result["maximum_bottle_rotation_rad"] < math.radians(10)
            and max(max_left_force, max_right_force) < 100
            and min_separation >= -0.002
            and result["maximum_thread_error_m"] < 0.001
            and not result["ground_contact_any"] and not result["failure_any"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("MIMIC_GRASP_REPLAY", json.dumps({
            key: result[key] for key in (
                "diagnostic_pass", "strict_success_reached", "left_contact_ratio",
                "right_sidewall_tripod_ratio", "maximum_bottle_drift_m",
                "maximum_bottle_rotation_rad", "initial_min_contact_separation_m",
                "maximum_left_net_body_force_n", "maximum_right_net_body_force_n")
        }), flush=True)
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    main()
