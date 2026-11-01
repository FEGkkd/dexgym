"""Fresh-process PhysX Reset replay of one unvalidated scripted Mimic candidate.

The candidate is injected only into this one-environment diagnostic bank. It
is never promoted to a validated training bank by this script. Bottle, cap,
hands, fingers and screw are restored at an episode boundary, then advanced
under gravity with zero actions and normal contact physics.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from dataclasses import asdict
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidate-result", type=Path, required=True)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--external-forces-every-iteration", action="store_true")
parser.add_argument("--trace-thread-substeps", action="store_true",
                    help="Record screw DOF position, velocity and drive mode each physics substep")
parser.add_argument("--spin-reset-override", type=float,
                    help="Diagnostic only: replace captured screw speed at Reset, e.g. 0")
parser.add_argument("--resistance-scale", type=float, default=1.0,
                    help="Diagnostic only: scale all stick/slip torque and drive gains together")
parser.add_argument("--ablate-right-cap-contact", action="store_true",
                    help="Diagnostic only: move the right wrist away at Reset")
parser.add_argument("--ablate-right-digit",
                    choices=("thumb", "index_finger", "middle_finger"),
                    help="Diagnostic only: restore one right digit to its screened pregrasp pose")
parser.add_argument("--mimic-natural-frequency", type=float,
                    help="Diagnostic only: author a compliant screw Mimic joint (rad/s)")
parser.add_argument("--mimic-damping-ratio", type=float, default=1.2,
                    help="Damping ratio for diagnostic compliant Mimic joint")
parser.add_argument("--solver-type", choices=("tgs", "pgs"), default="tgs",
                    help="Diagnostic solver comparison; captured candidate used TGS")
parser.add_argument("--cross-solver-ablation", action="store_true",
                    help="Diagnostic only: permit a solver mismatch with the captured state")
parser.add_argument("--capture-final-candidate", action="store_true",
                    help="Save the live end state for an independent, same-solver Reset replay")
parser.add_argument("--right-middle-pip-target-offset", type=float, default=0.0,
                    help="Diagnostic only: close the right middle PIP through its normal joint drive")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 50:
    parser.error("At least 50 policy steps are required for a one-second replay")
if args.spin_reset_override is not None and not math.isfinite(args.spin_reset_override):
    parser.error("--spin-reset-override must be finite")
if not math.isfinite(args.resistance_scale) or not 0 < args.resistance_scale <= 20:
    parser.error("--resistance-scale must be in (0, 20]")
if args.ablate_right_cap_contact and args.ablate_right_digit:
    parser.error("Choose a right-wrist or one right-digit ablation")
if args.mimic_natural_frequency is not None and (
        not math.isfinite(args.mimic_natural_frequency)
        or args.mimic_natural_frequency <= 0):
    parser.error("--mimic-natural-frequency must be positive and finite")
if not math.isfinite(args.mimic_damping_ratio) or args.mimic_damping_ratio <= 0:
    parser.error("--mimic-damping-ratio must be positive and finite")
if args.solver_type == "pgs" and args.external_forces_every_iteration:
    parser.error("PGS does not support --external-forces-every-iteration")
if not math.isfinite(args.right_middle_pip_target_offset) or not 0 <= args.right_middle_pip_target_offset <= 0.2:
    parser.error("--right-middle-pip-target-offset must be in [0, 0.2] rad")
app = AppLauncher(args).app

import torch
import warp as wp
from isaaclab_physx.physics import PhysxManager
from pxr import PhysxSchema, UsdPhysics

from dexteroushands_lab.assets.bottle.batched_thread import _rotate
from dexteroushands_lab.scenes.grasp_init import grasp_sample_from_dict
from dexteroushands_lab.scenes.mimic_grasp_capture import capture_mimic_grasp_sample
from dexteroushands_lab.tasks.direct.bimanual_bottle import bimanual_bottle_env as env_module
from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env import BimanualBottleDirectEnv
from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env_cfg import GraspHoldDirectEnvCfg


def _angle(q0: torch.Tensor, q1: torch.Tensor) -> float:
    dot = torch.dot(q0, q1).abs().clamp(max=1)
    return float(2 * torch.acos(dot))


def main():
    print("MIMIC_REPLAY_PHASE start", flush=True)
    probe = json.loads(args.candidate_result.read_text())
    if "diagnostic_candidate" not in probe:
        raise ValueError("Probe has no captured live PhysX candidate")
    sample = grasp_sample_from_dict(probe["diagnostic_candidate"])
    accepted_sources = {
        "scripted_physx_contact_candidate_requires_fresh_reset_replay",
        "pgs_physx_settled_candidate_requires_fresh_reset_replay",
    }
    if (sample.validated or not sample.metadata.get("mimic_candidate_unverified")
            or sample.source not in accepted_sources):
        raise ValueError("Only unvalidated live PhysX Mimic candidates may be replayed")
    physics_match = (
        sample.metadata.get("solver_type", "tgs") == args.solver_type
        and sample.metadata.get("enable_external_forces_every_iteration", False)
        == args.external_forces_every_iteration)
    if not physics_match and not args.cross_solver_ablation:
        raise ValueError("Replay solver settings differ from candidate generation")

    cfg = GraspHoldDirectEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.sim.physics.solver_type = 1 if args.solver_type == "tgs" else 0
    cfg.thread_backend = "mimic"
    cfg.right_cap_contact_gate = "manifold_sidewall"
    cfg.no_right_cap_contact_counterfactual = args.ablate_right_cap_contact
    cfg.sim.physics.enable_external_forces_every_iteration = args.external_forces_every_iteration
    cfg.stable_reset_probability = 0.0
    if args.mimic_natural_frequency is not None:
        ordinary_author = env_module.author_mimic_thread

        def author_compliant_diagnostic(stage, env_path, config, *, cap_local_pose_xyzw):
            slide, spin = ordinary_author(
                stage, env_path, config, cap_local_pose_xyzw=cap_local_pose_xyzw)
            mimic = PhysxSchema.PhysxMimicJointAPI.Apply(
                slide.GetPrim(), UsdPhysics.Tokens.rotX)
            mimic.GetNaturalFrequencyAttr().Set(args.mimic_natural_frequency)
            mimic.GetDampingRatioAttr().Set(args.mimic_damping_ratio)
            return slide, spin

        env_module.author_mimic_thread = author_compliant_diagnostic
    env = BimanualBottleDirectEnv(cfg)
    print("MIMIC_REPLAY_PHASE scene_created", flush=True)
    try:
        for side, spec in zip(("left", "right"), env.specs):
            if sample.metadata.get(f"{side}_usd_sha256") != spec.usd_sha256:
                raise ValueError(f"Candidate {side} official USD hash mismatch")
        if sample.metadata.get("bottle_parameters") != vars(env.bottle_cfg):
            raise ValueError("Candidate bottle configuration differs from live scene")
        if args.resistance_scale != 1.0:
            original = env.thread.config
            scaled = replace(original, **{
                name: getattr(original, name) * args.resistance_scale
                for name in ("breakaway_torque", "loose_breakaway_torque",
                             "breakaway_stiffness", "breakaway_damping",
                             "running_resistance_torque", "loose_running_resistance_torque")
            })
            env.thread.config = scaled
            env.thread.drive.config = scaled
        env.reset()
        print("MIMIC_REPLAY_PHASE default_reset", flush=True)

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
        reset_spin = (sample.metadata["mimic_spin_velocity_rad_s"]
                      if args.spin_reset_override is None else args.spin_reset_override)
        env._bank["thread_spin_velocity"][first:] = reset_spin
        drive_state = sample.metadata["mimic_drive_state"]
        env._bank["thread_stick_reference"][first:] = drive_state["stick_reference_rad"]
        env._bank["thread_sliding"][first:] = drive_state["sliding"]
        env._bank["thread_breakaway_seen"][first:] = drive_state["breakaway_seen"]
        if args.right_middle_pip_target_offset:
            joint = sample.metadata["right_joint_order"].index("r_middle_finger_pip")
            env._bank["right_joint_target"][first:, joint] += args.right_middle_pip_target_offset
        if args.ablate_right_digit:
            pregrasp_rows = json.loads((Path(__file__).resolve().parents[2]
                                        / "docs/assets/phase0_pregrasp_reset_bank.json").read_text())
            pregrasp = grasp_sample_from_dict(pregrasp_rows[probe.get("sample_index", probe.get("source_sample_index"))])
            canonical = sample.metadata["right_joint_order"]
            if pregrasp.metadata["right_joint_order"] != canonical:
                raise ValueError("Digit ablation joint order differs from captured candidate")
            digit_ids = [i for i, name in enumerate(canonical)
                         if name.startswith(f"r_{args.ablate_right_digit}_")]
            if not digit_ids:
                raise ValueError("Digit ablation found no official joints")
            for field in ("joint_position", "joint_target"):
                env._bank[f"right_{field}"][first:, digit_ids] = torch.tensor(
                    [getattr(pregrasp.right, field)[i] for i in digit_ids],
                    device=env.device)
            env._bank["right_joint_velocity"][first:, digit_ids] = 0
        env._reset_idx(torch.tensor([0], device=env.device))
        print("MIMIC_REPLAY_PHASE candidate_reset", flush=True)
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
                                   - reset_spin).abs()),
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

        thread_substeps = []
        if args.trace_thread_substeps:
            original_thread_step = env.thread.step

            def traced_thread_step():
                dof_pos = wp.to_torch(env.thread.art.get_dof_positions()).reshape(1, 2)
                dof_vel = wp.to_torch(env.thread.art.get_dof_velocities()).reshape(1, 2)
                root_pose = wp.to_torch(env.thread.art.get_root_transforms()).reshape(1, 7)
                root_velocity = wp.to_torch(env.thread.art.get_root_velocities()).reshape(1, 6)
                link_velocity = wp.to_torch(env.thread.art.get_link_velocities()).reshape(1, 3, 6)
                axis_w = _rotate(root_pose[:, 3:], torch.tensor(
                    [[0., 0., 1.]], device=env.device))
                cap_angular = link_velocity[:, env.thread.view.cap_id, 3:]
                relative_spin = ((cap_angular - root_velocity[:, 3:]) * axis_w).sum(-1)
                spin_id = env.thread.view.spin_id
                slide_id = env.thread.view.slide_id
                thread_substeps.append({
                    "substep": len(thread_substeps),
                    "theta_rad": float(dof_pos[0, spin_id]),
                    "spin_velocity_rad_s": float(dof_vel[0, spin_id]),
                    "axial_m": float(dof_pos[0, slide_id]),
                    "axial_velocity_m_s": float(dof_vel[0, slide_id]),
                    "physical_relative_spin_rad_s": float(relative_spin[0]),
                    "stick_reference_rad": float(env.thread.drive.stick_reference[0]),
                    "sliding_before_update": bool(env.thread.drive.sliding[0]),
                })
                return original_thread_step()

            env.thread.step = traced_thread_step

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
        minimum_separation = math.inf
        left_names = env.contacts.body_names["left"]
        thumb_ids = [i for i, name in enumerate(left_names) if name.startswith("l_thumb_")]
        other_ids = [i for i, name in enumerate(left_names)
                     if any(name.startswith(f"l_{digit}_") for digit in (
                         "index_finger", "middle_finger", "ring_finger", "pinky"))]
        view = PhysxManager.get_physics_sim_view()
        contact_views = []
        for obj, side, hand_name in (("Bottle", "left", "LeftHand"),
                                     ("Cap", "right", "RightHand")):
            contact_views.append(view.create_rigid_contact_view(
                f"/World/envs/env_0/{obj}",
                filter_patterns=[f"/World/envs/env_0/{hand_name}/{body}"
                                 for body in env.contacts.body_names[side]],
                max_contact_data_count=256))
        hand_hand_views = [view.create_rigid_contact_view(
            f"/World/envs/env_0/LeftHand/{body}",
            filter_patterns=[f"/World/envs/env_0/RightHand/{partner}"
                             for partner in env.contacts.body_names["right"]],
            max_contact_data_count=64)
            for body in env.contacts.body_names["left"]]
        for step in range(args.steps):
            if step == 0:
                print("MIMIC_REPLAY_PHASE first_step", flush=True)
            _, reward, _, _, extras = env.step(torch.zeros(1, 52, device=env.device))
            screw = env.thread.view.read()
            bottle = screw.root_pose_w[0]
            drift = float(torch.linalg.vector_norm(bottle[:3] - initial_bottle[:3]))
            tilt = _angle(bottle[3:], initial_bottle[3:])
            left = env.contacts.force_magnitudes("Bottle", "left")[0]
            right = env.contacts.force_magnitudes("Cap", "right")[0]
            left_vectors = env.contacts.latest_forces[("Bottle", "left")][0]
            thumb_vector = left_vectors[thumb_ids].sum(dim=0)
            other_vector = left_vectors[other_ids].sum(dim=0)
            thumb_norm = torch.linalg.vector_norm(thumb_vector)
            other_norm = torch.linalg.vector_norm(other_vector)
            left_opposition = bool(
                (thumb_norm > env.contacts.force_floor_n)
                & (other_norm > env.contacts.force_floor_n)
                & (torch.dot(thumb_vector, other_vector)
                   < -0.5 * thumb_norm * other_norm))
            right_digit_force = {
                digit: float(right[[i for i, name in enumerate(env.contacts.body_names["right"])
                                    if name.startswith(f"r_{digit}_")]].max())
                for digit in ("thumb", "index_finger", "middle_finger")
            }
            max_left_force = max(max_left_force, float(left.max()))
            max_right_force = max(max_right_force, float(right.max()))
            hand_hand_force = max(float(wp.to_torch(contact_view.get_contact_force_matrix(
                dt=cfg.sim.dt)).norm(dim=-1).max())
                for contact_view in hand_hand_views)
            for contact_view in (*contact_views, *hand_hand_views):
                data = contact_view.get_contact_data(dt=cfg.sim.dt)
                distances = wp.to_torch(data[3]).reshape(-1)
                counts = wp.to_torch(data[4]).reshape(-1).to(torch.int64)
                starts = wp.to_torch(data[5]).reshape(-1).to(torch.int64)
                if int(counts.sum()) > len(distances):
                    raise RuntimeError("Contact view overflowed")
                for count, start in zip(counts.tolist(), starts.tolist()):
                    if count:
                        observed = float(distances[start:start + count].min())
                        minimum_separation = min(minimum_separation, observed)
                        if step == 0:
                            min_separation = min(min_separation, observed)
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
                "left_thumb_force_n": float(left[thumb_ids].max()),
                "left_other_force_n": float(left[other_ids].max()),
                "left_force_opposition": left_opposition,
                "left_cap_cross_contact": bool(
                    (env.contacts.force_magnitudes("Cap", "left")[0]
                     >= env.contacts.force_floor_n).any()),
                "right_bottle_cross_contact": bool(
                    (env.contacts.force_magnitudes("Bottle", "right")[0]
                     >= env.contacts.force_floor_n).any()),
                "hand_hand_contact": hand_hand_force >= env.contacts.force_floor_n,
                "hand_hand_force_n": hand_hand_force,
                "right_sidewall_tripod": bool(episode["right_contact"][0]),
                "right_cap_digit_force_n": right_digit_force,
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
            "source_sample_index": probe.get("sample_index", probe.get("source_sample_index")),
            "enable_external_forces_every_iteration": args.external_forces_every_iteration,
            "spin_reset_override_rad_s": args.spin_reset_override,
            "resistance_scale": args.resistance_scale,
            "right_cap_contact_ablated": args.ablate_right_cap_contact,
            "right_digit_ablated": args.ablate_right_digit,
            "right_middle_pip_target_offset_rad": args.right_middle_pip_target_offset,
            "mimic_natural_frequency_rad_s": args.mimic_natural_frequency,
            "mimic_damping_ratio": (args.mimic_damping_ratio
                                    if args.mimic_natural_frequency is not None else None),
            "solver_type": args.solver_type,
            "source_physics_settings_match": physics_match,
            "candidate_validated": False,
            "reset_error": reset_error,
            "duration_s": args.steps * env.step_dt,
            "left_contact_ratio": left_ratio,
            "left_force_opposition_ratio": (sum(row["left_force_opposition"] for row in rows)
                                            / len(rows)),
            "cross_contact_ratios": {
                name: sum(row[field] for row in rows) / len(rows)
                for name, field in (("left_cap", "left_cap_cross_contact"),
                                    ("right_bottle", "right_bottle_cross_contact"),
                                    ("hand_hand", "hand_hand_contact"))},
            "right_sidewall_tripod_ratio": right_ratio,
            "maximum_bottle_drift_m": max(row["bottle_drift_from_replay_m"] for row in rows),
            "maximum_bottle_rotation_rad": max(row["bottle_rotation_from_replay_rad"] for row in rows),
            "maximum_left_net_body_force_n": max_left_force,
            "maximum_right_net_body_force_n": max_right_force,
            "initial_min_contact_separation_m": (min_separation if math.isfinite(min_separation)
                                                 else None),
            "minimum_contact_separation_m": (minimum_separation
                                              if math.isfinite(minimum_separation) else None),
            "maximum_thread_error_m": max(abs(row["thread_error_m"]) for row in rows),
            "ground_contact_any": any(row["ground_contact"] for row in rows),
            "failure_any": any(row["failure"] for row in rows),
            "strict_success_reached": any(row["strict_success"] for row in rows),
            "rows": rows,
        }
        if args.trace_thread_substeps:
            result["thread_substeps"] = thread_substeps
            reported = [row["spin_velocity_rad_s"] for row in thread_substeps]
            physical = [row["physical_relative_spin_rad_s"] for row in thread_substeps]
            actual = [(thread_substeps[i + 1]["theta_rad"]
                       - thread_substeps[i]["theta_rad"]) / cfg.sim.dt
                      for i in range(len(thread_substeps) - 1)]
            residual = [reported[i + 1] - actual[i]
                        for i in range(len(actual))]
            result["thread_velocity_consistency"] = {
                "reported_mean_rad_s": sum(reported) / len(reported),
                "actual_angle_difference_mean_rad_s": sum(actual) / len(actual),
                "reported_vs_angle_difference_rms_rad_s": math.sqrt(
                    sum(value * value for value in residual) / len(residual)),
                "reported_vs_angle_difference_max_abs_rad_s": max(
                    abs(value) for value in residual),
                "reported_vs_link_velocity_max_abs_rad_s": max(
                    abs(a - b) for a, b in zip(reported, physical)),
                "angle_change_rad": (thread_substeps[-1]["theta_rad"]
                                     - thread_substeps[0]["theta_rad"]),
            }
        result["diagnostic_pass"] = (
            args.spin_reset_override is None and args.resistance_scale == 1.0
            and not args.ablate_right_cap_contact and args.ablate_right_digit is None
            and args.mimic_natural_frequency is None and physics_match
            and args.right_middle_pip_target_offset == 0.0
            and max(reset_error.values()) < 1e-3 and result["strict_success_reached"]
            and left_ratio > 0.8 and right_ratio > 0.8
            and result["maximum_bottle_drift_m"] < 0.02
            and result["maximum_bottle_rotation_rad"] < math.radians(10)
            and max(max_left_force, max_right_force) < 100
            and min_separation is not None and min_separation >= -0.002
            and minimum_separation >= -0.002
            and result["maximum_thread_error_m"] < 0.001
            and not result["ground_contact_any"] and not result["failure_any"])
        if args.capture_final_candidate:
            result["diagnostic_candidate"] = asdict(capture_mimic_grasp_sample(
                env, sample,
                source="pgs_physx_settled_candidate_requires_fresh_reset_replay"
                if args.solver_type == "pgs" else sample.source,
                metadata_extra={
                    "solver_type": args.solver_type,
                    "enable_external_forces_every_iteration": args.external_forces_every_iteration,
                    "source_replay": str(args.output),
                    "cross_solver_seed": not physics_match,
                }))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("MIMIC_GRASP_REPLAY", json.dumps({
            key: result[key] for key in (
                "diagnostic_pass", "strict_success_reached", "left_contact_ratio",
                "right_sidewall_tripod_ratio", "maximum_bottle_drift_m",
                "maximum_bottle_rotation_rad", "initial_min_contact_separation_m",
                "maximum_left_net_body_force_n", "maximum_right_net_body_force_n")
        }), flush=True)
    except BaseException as exc:
        print(f"MIMIC_REPLAY_EXCEPTION {type(exc).__name__}: {exc!r}", flush=True)
        raise
    finally:
        print("MIMIC_REPLAY_PHASE cleanup", flush=True)
        env.close()
        app.close()
        if args.mimic_natural_frequency is not None:
            env_module.author_mimic_thread = ordinary_author


if __name__ == "__main__":
    main()
