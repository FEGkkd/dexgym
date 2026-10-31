"""Search bottle-relative pregrasp candidates; promote only gravity-on holds.

This is a geometric/physics search utility, not a learned grasp policy. An
empty bank is a valid result when no candidate passes the hold and ablation.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import itertools
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--short_duration", type=float, default=0.2)
parser.add_argument("--hold_duration", type=float, default=1.0)
parser.add_argument("--bank", type=Path, default=Path("docs/assets/grasp_init_bank.json"))
parser.add_argument("--report", type=Path, default=Path("docs/assets/grasp_search_results.json"))
parser.add_argument("--left_only_probe", action="store_true",
                    help="Search unsupported left-only bottle holds before bimanual acceptance")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.short_duration <= 0 or args.hold_duration <= 0:
    parser.error("Search durations must be positive")
app = AppLauncher(args).app

import torch  # noqa: E402
from dexteroushands_lab.scenes.grasp_init import GraspInitBank  # noqa: E402
from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402


def relocate(hand, dx, dy):
    def shift(p):
        return (p[0] + dx, p[1] + dy, *p[2:])
    return replace(hand, root_pose=shift(hand.root_pose),
                   wrist_target_pose=shift(hand.wrist_target_pose))


def finger_pose(hand, spec, mcp_fraction, pip_fraction=0.75):
    values = list(hand.joint_position)
    for index, joint in enumerate(spec.joints):
        if joint.name.endswith("_mcp_flex"):
            values[index] = max(joint.lower, min(joint.upper, joint.upper * mcp_fraction))
        elif joint.name.endswith(("_pip", "_dip")):
            values[index] = max(joint.lower, min(joint.upper, joint.upper * pip_fraction))
    return replace(hand, joint_position=tuple(values), joint_target=tuple(values))


def evaluate(fixture, sample, duration):
    fixture.reset_manager.reset([0], [sample])
    start = fixture.bottle.data.root_pose_w.torch[0].detach().clone()
    steps = math.ceil(duration / fixture.dt)
    initial_summary = None
    for step in range(steps):
        fixture.step()
        if step == 0:
            initial_summary = copy.deepcopy(fixture.monitor.summary(0))
    end = fixture.bottle.data.root_pose_w.torch[0].detach()
    contacts = fixture.monitor.summary(0)
    separations = [detail["minimum_separation_m"]
                   for data in initial_summary.values()
                   for detail in data["bodies"].values()
                   if detail["minimum_separation_m"] is not None]
    dot = float(torch.dot(end[3:7], start[3:7]).abs().clamp(max=1))
    return {"duration_s": steps * fixture.dt,
            "displacement_m": float(torch.linalg.vector_norm(end[:3] - start[:3])),
            "rotation_rad": 2.0 * math.acos(dot),
            "left_ratio": contacts.get("left_bottle", {}).get("contact_ratio", 0),
            "right_ratio": contacts.get("right_cap", {}).get("contact_ratio", 0),
            "ground_ratio": contacts.get("bottle_ground", {}).get("contact_ratio", 0),
            "initial_min_separation_m": min(separations) if separations else None,
            "left_peak_n": contacts.get("left_bottle", {}).get("peak_force_n", 0),
            "right_peak_n": contacts.get("right_cap", {}).get("peak_force_n", 0)}


def main():
    fixture = BimanualPhase0Fixture(num_envs=1, device=args.device,
                                    include_hand_hand=False)
    candidates = []
    provenance = fixture.initializer.generate(mode="stable_pregrasp", seed=0).metadata
    if args.left_only_probe:
        offsets = itertools.product((-0.05, -0.035, -0.025, -0.015, 0.0),
                                    (-0.015, 0.0, 0.015), (0.0, 0.25, 0.5))
        for index, (dx, dy, mcp_fraction) in enumerate(offsets):
            sample = fixture.initializer.generate(mode="stable_pregrasp", seed=0)
            sample = replace(sample,
                left=finger_pose(relocate(sample.left, dx, dy), fixture.specs[0],
                                 mcp_fraction),
                right=relocate(sample.right, 0.35, 0.0),
                metadata={**sample.metadata, "left_only_probe_dx_m": dx,
                          "left_only_probe_dy_m": dy,
                          "mcp_fraction": mcp_fraction},
                source="left_only_unsupported_geometry_probe")
            short = evaluate(fixture, sample, args.short_duration)
            record = {"candidate": index, "dx_m": dx, "dy_m": dy,
                      "seed": 0, "mcp_fraction": mcp_fraction, "short": short}
            if (short["displacement_m"] < 0.03 and short["left_ratio"] > 0.7
                    and short["left_peak_n"] < 100
                    and (short["initial_min_separation_m"] is None
                         or short["initial_min_separation_m"] > -0.005)):
                record["hold"] = evaluate(fixture, sample, args.hold_duration)
            candidates.append(record)
            print("LEFT_ONLY_SEARCH " + json.dumps(record, sort_keys=True), flush=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"metadata": provenance,
            "search": "left_only_unsupported", "pip_dip_fraction": 0.75,
            "right_hand_ablation_dx_m": 0.35, "candidates": candidates},
            indent=2, sort_keys=True) + "\n")
        print("LEFT_ONLY_SEARCH_RESULT " + json.dumps({"candidates": len(candidates),
              "long_trials": sum("hold" in x for x in candidates),
              "report": str(args.report)}), flush=True)
        return
    # Small deterministic local search around the measured fingertip geometry.
    offsets = itertools.product((-0.025, 0.0, 0.025), (0.0, 0.25, 0.5))
    for index, (dx, mcp_fraction) in enumerate(offsets):
        sample = fixture.initializer.generate(mode="stable_pregrasp", seed=index)
        sample = replace(sample,
                         left=finger_pose(relocate(sample.left, dx, 0.0),
                                          fixture.specs[0], mcp_fraction),
                         right=finger_pose(sample.right, fixture.specs[1],
                                           mcp_fraction),
                         metadata={**sample.metadata,
                                   "search_left_dx_m": dx,
                                   "search_mcp_fraction": mcp_fraction,
                                   "search_pip_dip_fraction": 0.75},
                         source="deterministic_physics_search_candidate")
        short = evaluate(fixture, sample, args.short_duration)
        record = {"seed": index, "left_dx_m": dx, "mcp_fraction": mcp_fraction,
                  "short": short, "passed": False}
        candidates.append(record)
        print("GRASP_SEARCH_SHORT " + json.dumps(record, sort_keys=True), flush=True)
        if (short["displacement_m"] < 0.03 and short["left_ratio"] > 0.5
                and short["right_ratio"] > 0.5):
            hold = evaluate(fixture, sample, args.hold_duration)
            ablated = replace(sample, right=relocate(sample.right, 0.35, 0.0))
            left_only = evaluate(fixture, ablated, args.hold_duration)
            passed = (args.hold_duration >= 1.0 and hold["displacement_m"] < 0.02
                      and left_only["displacement_m"] < 0.02
                      and hold["rotation_rad"] < math.radians(10)
                      and hold["left_ratio"] > 0.8 and hold["right_ratio"] > 0.8
                      and left_only["left_ratio"] > 0.8
                      and hold["ground_ratio"] == 0
                      and max(hold["left_peak_n"], hold["right_peak_n"]) < 100)
            record.update({"hold": hold, "left_only": left_only, "passed": passed})
            if passed:
                GraspInitBank(args.bank).add(sample.with_validation({
                    "passed": True, "fixture": "gravity_on_no_support",
                    "duration_s": hold["duration_s"],
                    "hold_displacement_m": hold["displacement_m"],
                    "hold_rotation_rad": hold["rotation_rad"],
                    "left_only_displacement_m": left_only["displacement_m"],
                    "left_contact_ratio": hold["left_ratio"],
                    "right_contact_ratio": hold["right_ratio"],
                    "left_only_contact_ratio": left_only["left_ratio"],
                    "ground_contact_ratio": hold["ground_ratio"],
                    "max_contact_peak_n": max(hold["left_peak_n"], hold["right_peak_n"]),
                    "max_peak_force_guard_n": 100.0}))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"metadata": provenance,
        "search": "bimanual_unsupported", "pip_dip_fraction": 0.75,
        "candidates": candidates},
        indent=2, sort_keys=True) + "\n")
    print("GRASP_SEARCH_RESULT " + json.dumps({"candidates": len(candidates),
          "validated": sum(x["passed"] for x in candidates),
          "bank": str(args.bank), "report": str(args.report)}), flush=True)


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
