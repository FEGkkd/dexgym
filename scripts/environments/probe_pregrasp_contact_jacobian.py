"""Diagnose finger-contact reachability from a screened pregrasp Reset.

Official MJCF collision-mesh finite differences choose one closing joint per
required digit. A scripted, contact-gated finger command is then tested in the
real Isaac/PhysX DirectRLEnv with gravity on and no support. This is a probe,
never a learned grasp or a bank validation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sample-index", type=int, default=3)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--maintain-contact", action="store_true",
                    help="Resume bounded joint closing when a digit loses contact")
parser.add_argument("--contact-target-n", type=float, default=0.5)
parser.add_argument("--contact-points", action="store_true",
                    help="Inspect cap-sidewall manifold points; expensive and may overflow PhysX point buffer")
parser.add_argument("--right-wrist-down-steps", type=int, default=0,
                    help="Diagnostic bounded right wrist descent during initial free fall")
parser.add_argument("--right-index-mcp-flex", action="store_true",
                    help="Also close the right index MCP flex after PIP reaches its limit")
parser.add_argument("--right-middle-mcp-abd", action="store_true",
                    help="Also close the right middle MCP abduction toward the cap sidewall")
parser.add_argument("--right-middle-pip-target-max", type=float,
                    help="Diagnostic upper target bound for right middle PIP; avoids blindly curling past sidewall")
parser.add_argument("--right-index-pip-target-max", type=float,
                    help="Diagnostic upper target bound for right index PIP to test bottom-rim contact")
parser.add_argument("--right-middle-abd-target-min", type=float,
                    help="Diagnostic lower target bound for right middle MCP abduction")
parser.add_argument("--right-index-abd-target-min", type=float,
                    help="Diagnostic lower target bound for right index MCP abduction to avoid index-middle collision")
parser.add_argument("--right-index-dip-extra-rad", type=float, default=0.0,
                    help="Additional bounded right-index DIP target from the bank pose")
parser.add_argument("--right-wrist-track-cap", action="store_true",
                    help="Command a bounded right wrist translation following measured free-cap translation")
parser.add_argument("--max-bottle-displacement-m", type=float, default=0.05,
                    help="Stop the diagnostic if the free bottle moves beyond this limit")
parser.add_argument("--candidate-json", type=Path,
                    help="After scripted closing, command an offline MJCF candidate through bounded wrist and finger drives")
parser.add_argument("--candidate-transition-steps", type=int, default=10)
parser.add_argument("--candidate-hold-steps", type=int, default=50)
parser.add_argument("--candidate-offset-xyz", type=float, nargs=3,
                    default=(0.0, 0.0, 0.0),
                    help="Small world-frame approach offset from offline candidate for PhysX calibration")
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 0 <= args.sample_index < 32 or args.steps < 1:
    parser.error("sample-index must be in [0,31] and steps must be positive")
if args.contact_target_n <= 0:
    parser.error("contact-target-n must be positive")
if args.right_wrist_down_steps < 0:
    parser.error("right-wrist-down-steps must be nonnegative")
if args.max_bottle_displacement_m <= 0:
    parser.error("max-bottle-displacement-m must be positive")
if not 0 <= args.right_index_dip_extra_rad <= 0.5:
    parser.error("right-index-dip-extra-rad must be in [0, 0.5]")
if args.candidate_transition_steps < 1 or args.candidate_hold_steps < 1:
    parser.error("candidate transition and hold steps must be positive")
if any(abs(value) > 0.005 for value in args.candidate_offset_xyz):
    parser.error("candidate offset components must stay within 5 mm")
app = AppLauncher(args).app

import torch
import warp as wp
from isaaclab_physx.physics import PhysxManager

from dexteroushands_lab.assets.bottle import BottleConfig
from dexteroushands_lab.assets.bottle.batched_thread import _rotate
from dexteroushands_lab.assets.wuji_hand2_beta import load_hand2_spec
from dexteroushands_lab.scenes.grasp_init import grasp_sample_from_dict
from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env import BimanualBottleDirectEnv
from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env_cfg import GraspHoldDirectEnvCfg

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.environments.generate_pregrasp_reset_candidates import (  # noqa: E402
    MjcfGeometryScreen, combined_model,
)


def closing_directions(sample, screen, specs):
    baseline = screen.check(sample)
    commands = {}
    evidence = {}
    for side, short, digits in (
        ("left", "l", ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")),
        ("right", "r", ("thumb", "index_finger", "middle_finger")),
    ):
        spec = specs[0] if side == "left" else specs[1]
        for digit in digits:
            original = baseline["per_digit"][short][digit]
            field = "sidewall_gap_m" if side == "right" else "gap_m"
            start_gap = original[field]
            trials = []
            for index, joint in enumerate(spec.joints):
                if not joint.name.startswith(f"{short}_{digit}_"):
                    continue
                for delta in (-0.05, 0.05):
                    q = list(getattr(sample, side).joint_position)
                    if not joint.lower <= q[index] + delta <= joint.upper:
                        continue
                    q[index] += delta
                    hand = replace(getattr(sample, side), joint_position=tuple(q),
                                   joint_target=tuple(q))
                    shifted = replace(sample, **{side: hand})
                    gap = screen.check(shifted)["per_digit"][short][digit].get(field)
                    if gap is not None:
                        trials.append((gap - start_gap, joint.name, delta, gap))
            best = min(trials, default=None)
            key = f"{side}_{digit}"
            if best is not None and best[0] < -1e-6:
                commands[key] = (side, digit, best[1], 1.0 if best[2] > 0 else -1.0)
            evidence[key] = {
                "initial_gap_m": start_gap,
                "best_delta_gap_m": best[0] if best else None,
                "joint": best[1] if best else None,
                "finite_difference_rad": best[2] if best else None,
                "predicted_gap_m": best[3] if best else None,
            }
    return commands, evidence


def main():
    root = Path(__file__).resolve().parents[2]
    official = Path(__import__("os").environ["WUJI_DESCRIPTION_ROOT"])
    sample = grasp_sample_from_dict(json.loads(
        (root / "docs/assets/phase0_pregrasp_reset_bank.json").read_text())[args.sample_index])
    config = BottleConfig()
    specs = (load_hand2_spec(official, side="left"),
             load_hand2_spec(official, side="right"))
    screen = MjcfGeometryScreen(combined_model(official, config), config)
    commands, evidence = closing_directions(sample, screen, specs)
    candidate = None
    if args.candidate_json is not None:
        proposal = json.loads(args.candidate_json.read_text())
        if (proposal.get("status") != "MJCF_ONLY_REQUIRES_NATIVE_USD_PHYSX_REPLAY"
                or proposal.get("source_sample_index") != args.sample_index):
            raise ValueError("Candidate does not match this bank row and geometry screen")
        candidate = proposal["candidate"]
    cfg = GraspHoldDirectEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.thread_backend = "mimic"
    cfg.stable_reset_probability = 0.0
    cfg.evaluation_mode = args.sample_index >= 24
    env = BimanualBottleDirectEnv(cfg)
    try:
        sim_view = PhysxManager.get_physics_sim_view()
        right_root = "/World/envs/env_0/RightHand"
        right_index_self_views = []
        for body in env.contacts.body_names["right"]:
            if "index_finger" not in body:
                continue
            filters = [f"{right_root}/{partner}" for partner in
                       env.contacts.body_names["right"]
                       if "index_finger" not in partner]
            view = sim_view.create_rigid_contact_view(
                f"{right_root}/{body}", filter_patterns=filters,
                max_contact_data_count=128)
            right_index_self_views.append((body, filters, view))
        env.reset()
        first = 24 if cfg.evaluation_mode else 0
        expected = env._stable_count + args.sample_index - first
        for _ in range(256):
            if int(env._reset_sample_index[0]) == expected:
                break
            env._reset_idx(torch.tensor([0], device=env.device))
        else:
            raise RuntimeError("Could not select requested bank row in 256 episode-boundary Resets")
        active = set(commands)
        ever_contacted = set()
        previous_forces = {key: 0.0 for key in commands}
        cap_contact_view = None
        if args.contact_points:
            # The batched production force view has a deliberately small
            # point budget. This one-env diagnostic needs full manifolds when
            # all three fingers touch at once; it is not used in PPO.
            filters = [f"/World/envs/env_0/RightHand/{body}"
                       for body in env.contacts.body_names["right"]]
            cap_contact_view = sim_view.create_rigid_contact_view(
                "/World/envs/env_0/Cap", filter_patterns=filters,
                max_contact_data_count=256)
        initial_bottle = env.bottle.data.root_pose_w.torch[0, :3].clone()
        initial_cap = env.cap.data.root_pose_w.torch[0, :3].clone()
        initial_right_wrist = env.right.data.root_pose_w.torch[0, :3].clone()
        rows = []
        total_steps = args.steps + (
            args.candidate_transition_steps + args.candidate_hold_steps
            if candidate is not None else 0)
        for step in range(total_steps):
            if candidate is not None and step == args.steps:
                proposed_pos = torch.tensor(
                    [value + offset for value, offset in zip(
                        candidate["wrist_pose_w_xyzw"][:3], args.candidate_offset_xyz)],
                    device=env.device)
                proposed_quat = torch.tensor(
                    candidate["wrist_pose_w_xyzw"][3:], device=env.device)
                current_pose = env.right.data.root_pose_w.torch[0]
                if (float(torch.linalg.vector_norm(proposed_pos - current_pose[:3])) > 0.02
                        or float(torch.linalg.vector_norm(proposed_quat)) < 0.999
                        or float(torch.linalg.vector_norm(proposed_quat)) > 1.001
                        or float(2 * torch.acos(torch.abs(torch.dot(
                            proposed_quat, current_pose[3:])).clamp(max=1))) > 0.2):
                    raise ValueError("Candidate wrist transition exceeds diagnostic safety bounds")
                target = torch.tensor(
                    [candidate["right_joint_position"][name]
                     for name in env.right.joint_names], device=env.device)
                limits = env._joint_limits[1]
                if bool(((target < limits[:, 0]) | (target > limits[:, 1])).any()):
                    raise ValueError("Candidate exceeds official finger joint limits")
                env.wrists[1].target_pos[0] = proposed_pos
                env.wrists[1].target_quat[0] = proposed_quat
                env._finger_targets[1][0] = target
            if args.maintain_contact and step < args.steps:
                active = {key for key, force in previous_forces.items()
                          if force < args.contact_target_n}
            action = torch.zeros(1, 52, device=env.device)
            if step >= args.steps:
                pass  # Candidate targets were set once; zero action holds them.
            elif args.right_wrist_track_cap:
                desired = initial_right_wrist + (
                    env.cap.data.root_pose_w.torch[0, :3] - initial_cap)
                error = desired - env.wrists[1].target_pos[0]
                action[0, 26:29] = (error / cfg.wrist_translation_scale).clamp(-1, 1)
            elif step < args.right_wrist_down_steps:
                action[0, 28] = -1.0
            for key in active if step < args.steps else ():
                side, _, name, direction = commands[key]
                hand = env.left if side == "left" else env.right
                if (args.right_middle_pip_target_max is not None
                        and name == "r_middle_finger_pip" and direction > 0
                        and float(env._finger_targets[1][0, hand.joint_names.index(name)])
                        >= args.right_middle_pip_target_max):
                    continue
                if (args.right_index_pip_target_max is not None
                        and name == "r_index_finger_pip" and direction > 0
                        and float(env._finger_targets[1][0, hand.joint_names.index(name)])
                        >= args.right_index_pip_target_max):
                    continue
                action[0, (6 if side == "left" else 32) + hand.joint_names.index(name)] = direction
            if step < args.steps and args.right_index_mcp_flex and "right_index_finger" in active:
                action[0, 32 + env.right.joint_names.index("r_index_finger_mcp_flex")] = 1.0
            if step < args.steps and args.right_middle_mcp_abd and (
                    "right_middle_finger" in active
                    or args.right_middle_abd_target_min is not None):
                abd_index = env.right.joint_names.index("r_middle_finger_mcp_abd")
                if (args.right_middle_abd_target_min is None
                        or float(env._finger_targets[1][0, abd_index])
                        > args.right_middle_abd_target_min):
                    action[0, 32 + abd_index] = -1.0
            if step < args.steps and args.right_index_abd_target_min is not None:
                abd_index = env.right.joint_names.index("r_index_finger_mcp_abd")
                if (float(env._finger_targets[1][0, abd_index])
                        > args.right_index_abd_target_min):
                    action[0, 32 + abd_index] = -1.0
            if step < args.steps and args.right_index_dip_extra_rad > 0:
                dip_index = env.right.joint_names.index("r_index_finger_dip")
                canonical_index = sample.metadata["right_joint_order"].index(
                    "r_index_finger_dip")
                upper_target = (sample.right.joint_target[canonical_index]
                                + args.right_index_dip_extra_rad)
                if float(env._finger_targets[1][0, dip_index]) < upper_target:
                    action[0, 32 + dip_index] = 1.0
            _, reward, terminated, truncated, extras = env.step(action)
            # The force matrix identifies the finger but cannot distinguish a
            # cap sidewall grasp from a top/edge collision. Inspect actual
            # PhysX manifold points in this one-environment diagnostic.
            cap_contact_points = []
            sidewall_forces = {digit: 0.0 for digit in ("thumb", "index_finger", "middle_finger")}
            if cap_contact_view is not None:
                cap_normal_forces, cap_points, _, cap_separations, cap_counts, cap_starts = map(
                    wp.to_torch, cap_contact_view.get_contact_data(dt=cfg.sim.dt))
                cap_counts, cap_starts = cap_counts.reshape(-1), cap_starts.reshape(-1)
                cap_pose = env.cap.data.root_pose_w.torch[0]
                cap_inverse = torch.cat((-cap_pose[3:6], cap_pose[6:7]))
                for body_index, body_name in enumerate(env.contacts.body_names["right"]):
                    begin = int(cap_starts[body_index])
                    end = begin + int(cap_counts[body_index])
                    for point_index in range(begin, end):
                        force_n = float(torch.linalg.vector_norm(cap_normal_forces[point_index]))
                        if force_n < env.contacts.force_floor_n:
                            continue
                        point_local = _rotate(
                            cap_inverse[None], (cap_points[point_index] - cap_pose[:3])[None])[0]
                        radial_m = float(torch.linalg.vector_norm(point_local[:2]))
                        axial_m = float(point_local[2])
                        on_sidewall = (abs(axial_m) <= config.cap_height * 0.5 - 0.001
                                       and abs(radial_m - config.cap_outer_radius) <= 0.004)
                        cap_contact_points.append({
                            "body": body_name, "force_n": force_n,
                            "radial_m": radial_m, "axial_m": axial_m,
                            "separation_m": float(cap_separations.reshape(-1)[point_index]),
                            "sidewall": on_sidewall,
                        })
                        if on_sidewall:
                            for digit in sidewall_forces:
                                if digit in body_name:
                                    sidewall_forces[digit] += force_n
            forces = {}
            cross_forces = {}
            for key, (side, digit, _, _) in commands.items():
                obj = "Bottle" if side == "left" else "Cap"
                matrix = env.contacts.force_magnitudes(obj, side)[0]
                names = env.contacts.body_names[side]
                force = float(matrix[[i for i, body in enumerate(names) if digit in body]].max())
                forces[key] = force
                wrong_obj = "Cap" if side == "left" else "Bottle"
                wrong_matrix = env.contacts.force_magnitudes(wrong_obj, side)[0]
                cross_forces[key] = float(wrong_matrix[
                    [i for i, body in enumerate(names) if digit in body]].max())
                if force > env.contacts.force_floor_n:
                    ever_contacted.add(key)
                if step < args.steps and not args.maintain_contact and force > env.contacts.force_floor_n:
                    active.discard(key)
            previous_forces = forces
            displacement = float(torch.linalg.vector_norm(
                env.bottle.data.root_pose_w.torch[0, :3] - initial_bottle))
            cap = env.cap.data.root_pose_w.torch[0]
            inverse_cap = torch.cat((-cap[3:6], cap[6:7]))
            tip_world = env.right.data.body_link_pose_w.torch[0, env._tip_ids[1][:3], :3]
            tip_local = _rotate(inverse_cap.expand(3, 4), tip_world - cap[:3])
            radial = torch.linalg.vector_norm(tip_local[:, :2], dim=-1)
            right_joint_names = (
                "r_index_finger_mcp_flex", "r_index_finger_pip",
                "r_middle_finger_mcp_abd", "r_middle_finger_pip")
            joint_ids = [env.right.joint_names.index(name) for name in right_joint_names]
            phase = ("closing" if step < args.steps else
                     "candidate_transition" if step < args.steps + args.candidate_transition_steps else
                     "candidate_hold")
            rows.append({"step": step + 1, "time_s": (step + 1) * env.step_dt,
                         "phase": phase,
                         "reward": float(reward[0]), "terminated": bool(terminated[0]),
                         "truncated": bool(truncated[0]),
                         "bottle_displacement_m": displacement,
                         "bottle_translation_m": (
                             env.bottle.data.root_pose_w.torch[0, :3] - initial_bottle
                         ).detach().cpu().tolist(),
                         "cap_translation_m": (cap[:3] - initial_cap).detach().cpu().tolist(),
                         "cap_pose_w_xyzw": cap.detach().cpu().tolist(),
                         "bottle_pose_w_xyzw": env.bottle.data.root_pose_w.torch[0].detach().cpu().tolist(),
                         "right_wrist_position_w_m": (
                             env.right.data.root_pose_w.torch[0, :3].detach().cpu().tolist()),
                         "right_wrist_pose_w_xyzw": (
                             env.right.data.root_pose_w.torch[0].detach().cpu().tolist()),
                         "contact_force_n": forces,
                         "cross_object_force_n": cross_forces,
                         "right_cap_contact_points": cap_contact_points,
                         "right_cap_sidewall_force_n": sidewall_forces,
                         "right_tip_radial_gap_m": (radial - config.cap_outer_radius).detach().cpu().tolist(),
                         "right_tip_axial_m": tip_local[:, 2].detach().cpu().tolist(),
                         "right_tip_cap_local_xyz_m": tip_local.detach().cpu().tolist(),
                         "right_selected_joint_rad": {
                             name: [float(env.right.data.joint_pos.torch[0, joint_id]),
                                    float(env._finger_targets[1][0, joint_id])]
                             for name, joint_id in zip(right_joint_names, joint_ids)},
                         "opening_rad": float(extras["phase1_episode"]["opening_rad"][0]),
                         "thread_error_m": float(extras["phase1_episode"]["constraint_error_m"][0])})
            if bool(terminated[0] or truncated[0]) or displacement > args.max_bottle_displacement_m:
                break
        result = {"sample_index": args.sample_index, "sample_seed": sample.seed,
                  "thread_backend": "mimic", "geometry_finite_difference": evidence,
                  "maintain_contact": args.maintain_contact,
                  "right_wrist_down_steps": args.right_wrist_down_steps,
                  "right_index_mcp_flex": args.right_index_mcp_flex,
                  "right_middle_mcp_abd": args.right_middle_mcp_abd,
                  "right_middle_pip_target_max": args.right_middle_pip_target_max,
                  "right_index_pip_target_max": args.right_index_pip_target_max,
                  "right_middle_abd_target_min": args.right_middle_abd_target_min,
                  "right_index_abd_target_min": args.right_index_abd_target_min,
                  "right_index_dip_extra_rad": args.right_index_dip_extra_rad,
                  "right_wrist_track_cap": args.right_wrist_track_cap,
                  "candidate_json": str(args.candidate_json) if args.candidate_json else None,
                  "candidate_offset_xyz_m": list(args.candidate_offset_xyz),
                  "contact_points_enabled": args.contact_points,
                  "final_right_joint_position": dict(zip(
                      env.right.joint_names,
                      env.right.data.joint_pos.torch[0].detach().cpu().tolist())),
                  "final_right_joint_target": dict(zip(
                      env.right.joint_names,
                      env._finger_targets[1][0].detach().cpu().tolist())),
                  "final_right_applied_torque_nm": dict(zip(
                      env.right.joint_names,
                      env.right.data.applied_torque.torch[0].detach().cpu().tolist())),
                  "commands": commands, "rows": rows,
                  "contacted_digits": sorted(ever_contacted),
                  "unsafe_motion": rows[-1]["bottle_displacement_m"] > args.max_bottle_displacement_m,
                  "terminated": rows[-1]["terminated"] if rows else None}
        self_contacts = []
        for body, filters, view in right_index_self_views:
            normal_forces, _, _, distances, counts, starts = map(
                wp.to_torch, view.get_contact_data(dt=cfg.sim.dt))
            counts, starts = counts.reshape(-1), starts.reshape(-1)
            for partner_index, partner in enumerate(filters):
                begin = int(starts[partner_index])
                end = begin + int(counts[partner_index])
                for point_index in range(begin, end):
                    force = float(torch.linalg.vector_norm(normal_forces[point_index]))
                    if force > 0.01:
                        self_contacts.append({
                            "index_body": body, "partner": partner,
                            "normal_force_n": force,
                            "separation_m": float(distances.reshape(-1)[point_index]),
                        })
        result["right_index_self_contacts_final"] = self_contacts
        print("PREGRASP_CONTACT_JACOBIAN", json.dumps({
            "sample_index": args.sample_index, "contacted_digits": result["contacted_digits"],
            "steps": len(rows), "final_displacement_m": rows[-1]["bottle_displacement_m"],
            "unsafe_motion": result["unsafe_motion"],
            "terminated": result["terminated"],
            "right_index_self_contact_count": len(self_contacts)}), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    main()
