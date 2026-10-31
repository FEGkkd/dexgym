"""Gravity-on, no-support PhysX hold checks for bimanual candidate initial states."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--duration", type=float, default=1.0)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--bank", type=Path, default=None)
parser.add_argument("--max_peak_force", type=float, default=100.0,
                    help="Engineering anomaly guard in newtons; not hardware calibrated")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.duration <= 0 or args.max_peak_force <= 0:
    parser.error("Duration and force guard must be positive")
app = AppLauncher(args).app

import torch  # noqa: E402
from dexteroushands_lab.scenes.grasp_init import GraspInitBank  # noqa: E402
from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402
from dexteroushands_lab.scenes.antipodal_contact_analysis import analyze_bottle_contacts  # noqa: E402


def orientation_change(q0, q1):
    dot = float(torch.dot(q0, q1).abs().clamp(max=1))
    return 2.0 * math.acos(dot)


def run_hold(fixture, sample, *, duration):
    fixture.reset_manager.reset([0], [sample])
    start = fixture.bottle.data.root_pose_w.torch[0].detach().clone()
    initial_geometry = {}
    for hand, spec, obj, key in ((fixture.left, fixture.specs[0], fixture.bottle, "left"),
                                  (fixture.right, fixture.specs[1], fixture.cap, "right")):
        tips = {}
        for body in (f"{'l' if spec.side == 'left' else 'r'}_{digit}_tip"
                     for digit in ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")):
            point = hand.data.body_link_pose_w.torch[0, hand.body_names.index(body), :3]
            tips[body] = {"position_w_m": point.detach().cpu().tolist(),
                          "center_distance_m": float(torch.linalg.vector_norm(
                              point - obj.data.root_pose_w.torch[0, :3]))}
        initial_geometry[key] = {"object_center_w_m":
                                 obj.data.root_pose_w.torch[0, :3].detach().cpu().tolist(),
                                 "tips": tips}
    trajectory = []
    initial_contact_summary = None
    initial_contact_points = None
    first_step_bottle_velocity_w = None
    first_step_left_velocity_w = None
    count = math.ceil(duration / fixture.dt)
    opposing_steps = 0
    for step in range(count):
        fixture.step()
        bottle_pose = fixture.bottle.data.root_pose_w.torch[0].detach().cpu().tolist()
        points_now = fixture.monitor.raw_contact_points(0, fixture.dt)
        opposing_steps += int(analyze_bottle_contacts(points_now, bottle_pose)[
            "surrounds_bottle_axis"])
        if step == 0:
            initial_contact_summary = copy.deepcopy(fixture.monitor.summary(0))
            initial_contact_points = points_now
            first_step_bottle_velocity_w = fixture.bottle.data.root_vel_w.torch[
                0].detach().cpu().tolist()
            first_step_left_velocity_w = fixture.left.data.root_vel_w.torch[
                0].detach().cpu().tolist()
        pose = fixture.bottle.data.root_pose_w.torch[0].detach().clone()
        if step % max(1, count // 20) == 0 or step == count - 1:
            trajectory.append({"t_s": (step + 1) * fixture.dt,
                               "position_w_m": pose[:3].cpu().tolist(),
                               "quaternion_xyzw": pose[3:7].cpu().tolist()})
    end = fixture.bottle.data.root_pose_w.torch[0].detach()
    summary = fixture.monitor.summary(0)
    displacement = float(torch.linalg.vector_norm(end[:3] - start[:3]))
    tilt = orientation_change(start[3:7], end[3:7])
    separations = [item["separation_m"] for item in initial_contact_points]
    return {"duration_s": count * fixture.dt, "dt_s": fixture.dt,
            "bottle_displacement_m": displacement, "bottle_rotation_rad": tilt,
            "correct_contact_ratios": {role: summary.get(role, {}).get("contact_ratio", 0.0)
                                       for role in ("left_bottle", "right_cap")},
            "contact_summary": summary, "initial_contact_summary": initial_contact_summary,
            "initial_raw_contact_points": sorted(initial_contact_points,
                                                 key=lambda item: item["separation_m"])[:30],
            "first_step_bottle_velocity_w": first_step_bottle_velocity_w,
            "first_step_left_velocity_w": first_step_left_velocity_w,
            "initial_min_separation_m": min(separations) if separations else None,
            "opposing_contact_coverage_ratio": opposing_steps / count,
            "bottle_trajectory": trajectory,
            "initial_geometry": initial_geometry, "finite": True}


def pass_hold(result, *, require_right=True):
    contacts = result["contact_summary"]
    separation = result["initial_min_separation_m"]
    return (result["duration_s"] >= 1.0 and result["bottle_displacement_m"] < 0.02
            and result["bottle_rotation_rad"] < math.radians(10)
            and separation is not None and separation >= -0.002
            and result["opposing_contact_coverage_ratio"] > 0.8
            and result["correct_contact_ratios"]["left_bottle"] > 0.8
            and (not require_right or result["correct_contact_ratios"]["right_cap"] > 0.8)
            and contacts.get("bottle_ground", {}).get("contact_ratio", 0.0) == 0.0
            and all(contacts.get(role, {}).get("peak_force_n", 0.0) < args.max_peak_force
                    for role in (("left_bottle", "right_cap") if require_right else ("left_bottle",))))


def main():
    fixture = BimanualPhase0Fixture(num_envs=1, device=args.device,
                                    include_hand_hand=False)
    results = {"fixture": "gravity_on_no_support", "dt_s": fixture.dt,
               "bottle_profile": fixture.config.source_profile,
               "left_asset_sha256": fixture.specs[0].usd_sha256,
               "right_asset_sha256": fixture.specs[1].usd_sha256,
               "asset_commit": fixture.specs[0].source_commit,
               "max_peak_force_guard_n": args.max_peak_force,
               "seeds": {}, "validated_samples": 0}
    bank = GraspInitBank(args.bank) if args.bank else None
    for seed in args.seeds:
        sample = fixture.initializer.generate(mode="stable_pregrasp", seed=seed)
        hold = run_hold(fixture, sample, duration=args.duration)
        # The cap's D6 connection cannot be allowed to mask a failed left hold.
        left_only = replace(sample, right=replace(sample.right,
            root_pose=(sample.right.root_pose[0] + 0.35, *sample.right.root_pose[1:]),
            wrist_target_pose=(sample.right.wrist_target_pose[0] + 0.35,
                               *sample.right.wrist_target_pose[1:])))
        ablation = run_hold(fixture, left_only, duration=args.duration)
        passed = (pass_hold(hold) and pass_hold(ablation, require_right=False)
                  and ablation["bottle_displacement_m"] < 0.02
                  and ablation["correct_contact_ratios"]["left_bottle"] > 0.8)
        replay = run_hold(fixture, sample, duration=args.duration) if passed else None
        passed = passed and replay is not None and pass_hold(replay)
        results["seeds"][str(seed)] = {"sample": asdict(sample), "hold": hold, "left_only": ablation,
                                       "replay": replay, "passed": passed}
        print("GRASP_SEED_RESULT " + json.dumps({"seed": seed,
              "hold_displacement_m": hold["bottle_displacement_m"],
              "left_contact_ratio": hold["correct_contact_ratios"]["left_bottle"],
              "right_contact_ratio": hold["correct_contact_ratios"]["right_cap"],
              "left_only_displacement_m": ablation["bottle_displacement_m"],
              "passed": passed}), flush=True)
        if passed:
            results["validated_samples"] += 1
            if bank:
                contacts = hold["contact_summary"]
                bank.add(sample.with_validation({"passed": True,
                    "fixture": "gravity_on_no_support", "duration_s": hold["duration_s"],
                    "hold_displacement_m": hold["bottle_displacement_m"],
                    "hold_rotation_rad": hold["bottle_rotation_rad"],
                    "left_only_displacement_m": ablation["bottle_displacement_m"],
                    "left_contact_ratio": hold["correct_contact_ratios"]["left_bottle"],
                    "right_contact_ratio": hold["correct_contact_ratios"]["right_cap"],
                    "left_only_contact_ratio": ablation["correct_contact_ratios"]["left_bottle"],
                    "ground_contact_ratio": contacts.get("bottle_ground", {}).get("contact_ratio", 0.0),
                    "max_peak_force_guard_n": args.max_peak_force,
                    "initial_min_separation_m": hold["initial_min_separation_m"],
                    "max_initial_penetration_guard_m": 0.002,
                    "opposing_contact_coverage_ratio": hold[
                        "opposing_contact_coverage_ratio"],
                    "left_thumb_contact_time_s": max(
                        (data["samples"] * fixture.dt for name, data in
                         contacts["left_bottle"]["bodies"].items() if "|l_thumb_" in name),
                        default=0.0),
                    "left_other_finger_count": len({
                        name.split("|")[-1].split("_")[1]
                        for name, data in contacts["left_bottle"]["bodies"].items()
                        if "|l_thumb_" not in name and data["samples"] * fixture.dt >= 0.05}),
                    "support_released": True, "reset_replay_passed": True,
                    "max_contact_peak_n": max(contacts.get(role, {}).get("peak_force_n", 0.0)
                                              for role in ("left_bottle", "right_cap"))}))
    # A separate no-contact episode checks gravity and absence of hidden support.
    sample = fixture.initializer.generate(mode="near_contact", seed=999)
    sample = replace(sample,
        left=replace(sample.left,
            root_pose=(sample.left.root_pose[0] - 0.4, *sample.left.root_pose[1:]),
            wrist_target_pose=(sample.left.wrist_target_pose[0] - 0.4,
                               *sample.left.wrist_target_pose[1:])),
        right=replace(sample.right,
            root_pose=(sample.right.root_pose[0] + 0.4, *sample.right.root_pose[1:]),
            wrist_target_pose=(sample.right.wrist_target_pose[0] + 0.4,
                               *sample.right.wrist_target_pose[1:])))
    fall = run_hold(fixture, sample, duration=0.10)
    results["no_contact_free_fall"] = {
        "displacement_m": fall["bottle_displacement_m"],
        "expected_g_t2_over_2_m": 0.5 * 9.81 * 0.1**2,
        "contacts": fall["correct_contact_ratios"],
    }
    if (not 0.035 < fall["bottle_displacement_m"] < 0.065
            or any(fall["correct_contact_ratios"].values())):
        raise AssertionError("No-contact free-fall control did not match gravity-only motion")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print("SUSPENDED_GRASP_RESULT " + json.dumps({"validated_samples": results["validated_samples"],
          "requested_samples": len(args.seeds),
          "free_fall_displacement_m": fall["bottle_displacement_m"]}), flush=True)


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
