"""Measure Beta 2 PGS Mimic Reset compatibility and the first 0.3 s contact window.

The scripted inward action below is a feasibility probe, never a task action
schedule or an RL policy. The training environment receives only actor output.
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--native-bank", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--admitted-bank", type=Path)
parser.add_argument("--count", type=int, default=16)
parser.add_argument("--steps", type=int, default=15)
parser.add_argument("--probe-fingers", choices=("stable", "none", "fit", "fit_approach"), default="fit")
parser.add_argument("--probe-order", choices=("active_first", "zero_first"),
                    default="active_first")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import warp as wp
from dexteroushands_lab.assets.bottle.mimic_contract import mimic_reset_geometry
from dexteroushands_lab.assets.bottle.batched_thread import _rotate
from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.env import PregraspCapTurn30DirectEnv
from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.env_cfg import PregraspCapTurn30DirectEnvCfg


def retain_last_live(previous, observed, alive_before_step, done):
    """Ignore post-step states from environments already auto-reset by DirectRLEnv."""
    if (previous.shape != observed.shape or previous.ndim < 1
            or alive_before_step.shape != previous.shape[:1]
            or done.shape != previous.shape[:1]):
        raise ValueError("Probe diagnostic batch shapes differ")
    mask = alive_before_step & ~done
    while mask.ndim < observed.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, observed, previous)


def run_probe(env, *, approach: bool, steps: int, probe_fingers: str = "stable"):
    n = env.num_envs
    env.reset()
    alive = torch.ones(n, device=env.device, dtype=torch.bool)
    first_left = torch.full((n,), -1, device=env.device, dtype=torch.long)
    first_right = first_left.clone()
    first_bilateral = first_left.clone()
    first_right_digits = {digit: first_left.clone() for digit in
                          ("thumb", "index_finger", "middle_finger")}
    first_left_digits = {digit: first_left.clone() for digit in
                         ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")}
    max_left_digit_force = {digit: torch.zeros(n, device=env.device)
                            for digit in first_left_digits}
    first_failure = first_left.clone()
    min_z = torch.full((n,), float("inf"), device=env.device)
    max_constraint = torch.zeros(n, device=env.device)
    max_pitch = torch.zeros(n, device=env.device)
    max_joint_error = torch.zeros(n, device=env.device)
    max_velocity_bias = torch.zeros(n, device=env.device)
    max_right_score = torch.zeros(n, device=env.device)
    max_right_proximity = torch.zeros(n, device=env.device)
    max_cross_force = torch.zeros(n, device=env.device)
    max_correct_force = torch.zeros(n, device=env.device)
    initial_right_wrist_to_cap = torch.linalg.vector_norm(
        env.right.data.root_pose_w.torch[:, :3] - env.cap.data.root_pose_w.torch[:, :3], dim=-1).clone()
    def right_site_geometry():
        cap = env.cap.data.root_pose_w.torch
        tip = env.right.data.body_link_pose_w.torch[:, env._tip_ids[1][:3], :3]
        inverse = torch.cat((-cap[:, 3:6], cap[:, 6:7]), dim=-1)
        local = _rotate(inverse[:, None, :].expand(-1, 3, -1).reshape(-1, 4),
                        (tip - cap[:, None, :3]).reshape(-1, 3)).reshape(n, 3, 3)
        return (torch.linalg.vector_norm(local[:, :, :2], dim=-1)
                - env.bottle_cfg.cap_outer_radius), local[:, :, 2]
    initial_site_gap, initial_site_axial = right_site_geometry()
    last_right_site_gap = initial_site_gap.clone()
    last_right_site_axial = initial_site_axial.clone()
    last_right_wrist_to_cap = initial_right_wrist_to_cap.clone()
    last_right_wrist_clip = torch.zeros(n, device=env.device, dtype=torch.long)
    last_right_wrist_target_displacement = torch.zeros(n, device=env.device)
    last_live_observed_step = torch.zeros(n, device=env.device, dtype=torch.long)
    def left_site_geometry():
        bottle = env.bottle.data.root_pose_w.torch
        tip = env.left.data.body_link_pose_w.torch[:, env._tip_ids[0], :3]
        inverse = torch.cat((-bottle[:, 3:6], bottle[:, 6:7]), dim=-1)
        local = _rotate(inverse[:, None, :].expand(-1, tip.shape[1], -1).reshape(-1, 4),
                        (tip - bottle[:, None, :3]).reshape(-1, 3)).reshape(n, -1, 3)
        return (torch.linalg.vector_norm(local[:, :, :2], dim=-1)
                - env.bottle_cfg.bottle_radius), local[:, :, 2]
    initial_left_site_gap, initial_left_site_axial = left_site_geometry()
    last_left_site_gap = initial_left_site_gap.clone()
    last_left_site_axial = initial_left_site_axial.clone()
    minimum_left_site_gap = initial_left_site_gap.clone()
    first_step_separation = {"Bottle/left": [None] * n, "Cap/right": [None] * n}
    previous_theta = env.thread.theta_unwrapped.clone()
    left_stable = torch.tensor(env._validation_stable["left"]["joint_position"],
                               device=env.device)[env._joint_reorder[0]]
    right_stable = torch.tensor(env._validation_stable["right"]["joint_position"],
                                device=env.device)[env._joint_reorder[1]]
    for step in range(1, steps + 1):
        action = torch.zeros(n, 52, device=env.device)
        if approach and probe_fingers in ("stable", "none", "fit_approach"):
            for side, obj, start, target in ((0, env.bottle, 0, left_stable),
                                             (1, env.cap, 26, right_stable)):
                wrist = (env.left if side == 0 else env.right).data.root_pose_w.torch
                object_pose = obj.data.root_pose_w.torch
                bottle_q = env.bottle.data.root_pose_w.torch[:, 3:]
                axis = _rotate(bottle_q, torch.tensor([0., 0., 1.],
                    device=env.device).expand(n, 3))
                radial = wrist[:, :3] - object_pose[:, :3]
                radial -= (radial * axis).sum(-1, keepdim=True) * axis
                radial /= torch.linalg.vector_norm(radial, dim=-1, keepdim=True).clamp_min(1e-6)
                action[:, start:start+3] = -radial
                if probe_fingers == "stable":
                    action[:, start+6:start+26] = torch.sign(
                        target[None, :] - env._finger_targets[side])
        if approach and probe_fingers in ("fit", "fit_approach"):
            left_target = getattr(env, "_validation_probe_left_q", left_stable[None, :])
            action[:, 6:26] = torch.sign(left_target - env._finger_targets[0])
            action[:, 32:52] = torch.sign(
                env._validation_probe_right_q - env._finger_targets[1])
        _, _, terminated, truncated, info = env.step(action)
        if step == 1:
            for key, label in ((('Bottle', 'left'), "Bottle/left"),
                               (('Cap', 'right'), "Cap/right")):
                view = env.contacts.views[key]
                buffers = tuple(map(wp.to_torch, view.get_contact_data(dt=env.physics_dt)))
                distance = buffers[3].reshape(-1)
                count = buffers[4].reshape(n, -1)
                start = buffers[5].reshape(n, -1)
                body_count = len(env.contacts.body_names[key[1]])
                for env_id in range(n):
                    values = []
                    for body in range(body_count):
                        column = env_id * body_count + body
                        first = int(start[env_id, column])
                        last = first + int(count[env_id, column])
                        values.extend(distance[first:last].detach().cpu().tolist())
                    first_step_separation[label][env_id] = min(values) if values else None
        episode = info["phase1_episode"]
        left = episode["left_contact"] & alive
        right = episode["right_contact"] & alive
        bilateral = left & right
        failure = (terminated | truncated) & alive
        live_after_step = alive & ~failure
        first_left = torch.where((first_left < 0) & left, step, first_left)
        first_right = torch.where((first_right < 0) & right, step, first_right)
        first_bilateral = torch.where((first_bilateral < 0) & bilateral,
                                      step, first_bilateral)
        right_forces = env.contacts.force_magnitudes("Cap", "right")
        left_forces = env.contacts.force_magnitudes("Bottle", "left")
        cross_force = torch.maximum(
            env.contacts.force_magnitudes("Cap", "left").amax(-1),
            env.contacts.force_magnitudes("Bottle", "right").amax(-1))
        max_cross_force = torch.where(live_after_step, torch.maximum(max_cross_force, cross_force),
                                      max_cross_force)
        correct_force = torch.maximum(
            env.contacts.force_magnitudes("Bottle", "left").amax(-1),
            right_forces.amax(-1))
        max_correct_force = torch.where(live_after_step, torch.maximum(max_correct_force,
                                               correct_force), max_correct_force)
        for digit, previous in first_right_digits.items():
            columns = [i for i, name in enumerate(env.contacts.body_names["right"])
                       if digit in name]
            touching = (right_forces[:, columns].amax(-1) >= env.contacts.force_floor_n) & live_after_step
            first_right_digits[digit] = torch.where((previous < 0) & touching, step, previous)
        for digit, previous in first_left_digits.items():
            columns = [i for i, name in enumerate(env.contacts.body_names["left"])
                       if digit in name]
            peak = left_forces[:, columns].amax(-1)
            touching = (peak >= env.contacts.force_floor_n) & live_after_step
            first_left_digits[digit] = torch.where((previous < 0) & touching, step, previous)
            max_left_digit_force[digit] = torch.where(
                live_after_step, torch.maximum(max_left_digit_force[digit], peak),
                max_left_digit_force[digit])
        current_left_gap, current_left_axial = left_site_geometry()
        minimum_left_site_gap = torch.where(
            live_after_step[:, None], torch.minimum(minimum_left_site_gap, current_left_gap),
            minimum_left_site_gap)
        current_right_gap, current_right_axial = right_site_geometry()
        current_right_wrist_to_cap = torch.linalg.vector_norm(
            env.right.data.root_pose_w.torch[:, :3]
            - env.cap.data.root_pose_w.torch[:, :3], dim=-1)
        current_right_wrist_target_displacement = torch.linalg.vector_norm(
            env.wrists[1].target_pos - env.wrists[1].initial_pos, dim=-1)
        last_right_site_gap = retain_last_live(
            last_right_site_gap, current_right_gap, alive, failure)
        last_right_site_axial = retain_last_live(
            last_right_site_axial, current_right_axial, alive, failure)
        last_left_site_gap = retain_last_live(
            last_left_site_gap, current_left_gap, alive, failure)
        last_left_site_axial = retain_last_live(
            last_left_site_axial, current_left_axial, alive, failure)
        last_right_wrist_to_cap = retain_last_live(
            last_right_wrist_to_cap, current_right_wrist_to_cap, alive, failure)
        last_right_wrist_target_displacement = retain_last_live(
            last_right_wrist_target_displacement,
            current_right_wrist_target_displacement, alive, failure)
        last_right_wrist_clip = retain_last_live(
            last_right_wrist_clip, env.wrists[1].workspace_clip_count, alive, failure)
        last_live_observed_step = torch.where(
            live_after_step, torch.full_like(last_live_observed_step, step),
            last_live_observed_step)
        if env._metrics is not None:
            max_right_score = torch.where(live_after_step, torch.maximum(
                max_right_score, env._metrics["right_contact_score"]), max_right_score)
            max_right_proximity = torch.where(live_after_step, torch.maximum(
                max_right_proximity, env._metrics["right_proximity"]), max_right_proximity)
        first_failure = torch.where((first_failure < 0) & failure, step, first_failure)
        min_z = torch.where(alive, torch.minimum(min_z, episode["bottle_z_m"]), min_z)
        max_constraint = torch.where(alive, torch.maximum(
            max_constraint, episode["constraint_error_m"].abs()), max_constraint)
        pitch = (episode["axial_travel_m"] -
                 env.bottle_cfg.thread_pitch * episode["opening_rad"] / (2 * math.pi)).abs()
        max_pitch = torch.where(alive, torch.maximum(max_pitch, pitch), max_pitch)
        joint_error = torch.maximum(
            (env.left.data.joint_pos.torch - env._finger_targets[0]).abs().amax(-1),
            (env.right.data.joint_pos.torch - env._finger_targets[1]).abs().amax(-1))
        max_joint_error = torch.where(live_after_step, torch.maximum(max_joint_error, joint_error),
                                      max_joint_error)
        pose_velocity = (env.thread.theta_unwrapped - previous_theta) / env.step_dt
        velocity_bias = (pose_velocity - env.thread.state.spin_velocity_rad_s).abs()
        max_velocity_bias = torch.where(live_after_step, torch.maximum(max_velocity_bias,
                                             velocity_bias), max_velocity_bias)
        previous_theta = env.thread.theta_unwrapped.clone()
        alive &= ~failure
    result = {"first_left_contact_step": first_left, "first_right_contact_step": first_right,
              "first_bilateral_contact_step": first_bilateral,
              "first_failure_step": first_failure, "minimum_bottle_z_m": min_z,
              "maximum_constraint_error_m": max_constraint,
              "maximum_pitch_error_m": max_pitch,
              "maximum_joint_tracking_error_rad": max_joint_error,
              "maximum_spin_velocity_bias_rad_s": max_velocity_bias}
    result["maximum_right_contact_score"] = max_right_score
    result["maximum_right_proximity"] = max_right_proximity
    result["maximum_cross_object_contact_force_n"] = max_cross_force
    result["maximum_correct_object_contact_force_n"] = max_correct_force
    result["initial_right_wrist_to_cap_m"] = initial_right_wrist_to_cap
    result["final_right_wrist_to_cap_m"] = last_right_wrist_to_cap
    result["right_wrist_workspace_clip_count"] = last_right_wrist_clip
    result["right_wrist_target_displacement_m"] = last_right_wrist_target_displacement
    result["last_live_observed_step"] = last_live_observed_step
    result["initial_right_site_radial_gap_m"] = initial_site_gap
    result["initial_right_site_axial_m"] = initial_site_axial
    result["final_right_site_radial_gap_m"] = last_right_site_gap
    result["final_right_site_axial_m"] = last_right_site_axial
    result["initial_left_site_radial_gap_m"] = initial_left_site_gap
    result["initial_left_site_axial_m"] = initial_left_site_axial
    result["minimum_left_site_radial_gap_m"] = minimum_left_site_gap
    result["final_left_site_radial_gap_m"] = last_left_site_gap
    result["final_left_site_axial_m"] = last_left_site_axial
    for digit, value in first_left_digits.items():
        result[f"first_left_{digit}_body_contact_step"] = value
        result[f"maximum_left_{digit}_contact_force_n"] = max_left_digit_force[digit]
    for digit, value in first_right_digits.items():
        result[f"first_right_{digit}_body_contact_step"] = value
    report = {key: value.detach().cpu().tolist() for key, value in result.items()}
    report["first_step_signed_separation_m"] = first_step_separation
    return report


def main():
    source = json.loads(args.native_bank.read_text())
    if not 1 <= args.count <= len(source) or args.steps < 1:
        raise ValueError("Invalid candidate count or probe duration")
    cfg = PregraspCapTurn30DirectEnvCfg()
    cfg.pregrasp_bank_path = str(args.native_bank.resolve())
    cfg.pregrasp_reset_levels = ()
    cfg.pregrasp_candidate_validation_mode = True
    cfg.scene.num_envs = args.count
    cfg.sim.device = args.device
    cfg.seed = 1001
    env = PregraspCapTurn30DirectEnv(cfg)
    env._validation_stable = json.loads((Path(__file__).resolve().parents[2] /
        "docs/assets/phase1_mimic_pgs_grasp_init_bank.json").read_text())[0]
    if args.probe_fingers in ("fit", "fit_approach"):
        if any("right_probe_joint_target" not in row["metadata"]
               for row in source[:args.count]):
            raise ValueError("Fitted probe requires per-sample right target metadata")
        env._validation_probe_right_q = torch.tensor(
            [row["metadata"]["right_probe_joint_target"]
             for row in source[:args.count]], device=env.device,
            dtype=torch.float32)[:, env._joint_reorder[1]]
        left_targets = [row["metadata"].get("left_probe_joint_target")
                        for row in source[:args.count]]
        if any(value is not None for value in left_targets):
            if any(value is None or len(value) != 20 for value in left_targets):
                raise ValueError("Fitted left probe targets must cover every candidate")
            env._validation_probe_left_q = torch.tensor(
                left_targets, device=env.device,
                dtype=torch.float32)[:, env._joint_reorder[0]]
            if not bool(torch.isfinite(env._validation_probe_left_q).all()):
                raise ValueError("Fitted left probe targets must be finite")
    try:
        print("PHASE1B_PGS_ENV_READY", flush=True)
        if args.probe_order == "active_first":
            active = run_probe(env, approach=True, steps=args.steps,
                               probe_fingers=args.probe_fingers)
            print("PHASE1B_PGS_APPROACH_READY", flush=True)
            zero = run_probe(env, approach=False, steps=args.steps)
            print("PHASE1B_PGS_ZERO_READY", flush=True)
        else:
            zero = run_probe(env, approach=False, steps=args.steps)
            print("PHASE1B_PGS_ZERO_READY", flush=True)
            active = run_probe(env, approach=True, steps=args.steps,
                               probe_fingers=args.probe_fingers)
            print("PHASE1B_PGS_APPROACH_READY", flush=True)
        results = []
        admitted = []
        for index, sample in enumerate(source[:args.count]):
            geometry = mimic_reset_geometry(sample["bottle"]["root_pose"],
                sample["cap"]["root_pose"], sample["thread_initial_angle_rad"],
                env.bottle_cfg)
            def slice_report(report):
                return {name: ({key: values[index] for key, values in value.items()}
                               if name == "first_step_signed_separation_m" else value[index])
                        for name, value in report.items()}
            near = slice_report(zero)
            reach = slice_report(active)
            compatible = (abs(geometry["pitch_error_m"]) < 5e-5
                          and abs(geometry["twist_error_rad"]) < .01
                          and reach["maximum_constraint_error_m"] < .001
                          and reach["maximum_pitch_error_m"] < .00005
                          and reach["first_failure_step"] != 1
                          and reach["maximum_cross_object_contact_force_n"] < .1
                          and reach["maximum_correct_object_contact_force_n"] < 50
                          and all(value is None or value > -.001
                                  for value in reach["first_step_signed_separation_m"].values()))
            # The active probe demonstrates a contact opportunity; the
            # strategy still has to discover its own actions and sustain it.
            reachable = (reach["first_bilateral_contact_step"] > 0
                         and near["first_bilateral_contact_step"] < 0
                         and (reach["first_failure_step"] < 0
                              or reach["first_bilateral_contact_step"]
                              <= reach["first_failure_step"]))
            record = {"seed": sample["seed"],
                      "level": sample["metadata"]["phase1b_reset_level"],
                      "mimic_reset_geometry": geometry,
                      "zero_action": near, "active_contact_probe": reach,
                      "pgs_compatible": compatible,
                      "active_contact_reachable": reachable}
            results.append(record)
            if compatible and reachable:
                row = json.loads(json.dumps(sample))
                row["metadata"]["phase1b_pgs_reach_evidence"] = record
                row["metadata"]["geometry_qualified"] = True
                row["metadata"].setdefault("geometry_evidence", {})["native_physx"] = {
                    "qualified": True,
                    "backend": "official_native_USD_PhysX_one_step_contact",
                    "solver_type": "pgs",
                    "initial_signed_separation_m": reach["first_step_signed_separation_m"],
                    "maximum_cross_object_contact_force_n":
                        reach["maximum_cross_object_contact_force_n"],
                    "maximum_correct_object_contact_force_n":
                        reach["maximum_correct_object_contact_force_n"],
                    "first_bilateral_contact_step":
                        reach["first_bilateral_contact_step"],
                    "probe_kind": "ordinary_finger_target_only_feasibility",
                }
                row["metadata"].pop("right_probe_joint_target", None)
                row["metadata"].pop("left_probe_joint_target", None)
                admitted.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"solver": "pgs_mimic", "policy_hz": 50,
            "physx_hz": 500, "steps": args.steps, "probe_order": args.probe_order,
            "endpoint_semantics": "last_live_post_step_before_auto_reset",
            "attempted": args.count,
            "admitted": len(admitted), "results": results}, indent=2) + "\n")
        if args.admitted_bank:
            args.admitted_bank.parent.mkdir(parents=True, exist_ok=True)
            args.admitted_bank.write_text(json.dumps(admitted, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"attempted": args.count, "admitted": len(admitted),
                          "first_16": [(r["seed"], r["active_contact_reachable"])
                                       for r in results[:16]]}), flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        env.close()


try:
    main()
finally:
    app.close()
