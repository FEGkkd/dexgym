"""Diagnostic Reset of separately held left-power and right-tripod grasps.

The complete right-tripod bottle/cap/thread snapshot stays authoritative.
The validated left hand is rigidly reframed around that bottle at Reset.
Both objects remain dynamic; no support or per-step pose write is used.
This is diagnostic evidence, not bank admission or cross-family training data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--left_bank", type=Path, required=True)
parser.add_argument("--right_bank", type=Path, required=True)
parser.add_argument("--left_index", type=int, default=0)
parser.add_argument("--right_index", type=int, default=0)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from dexteroushands_lab.assets.bottle import BottleConfig  # noqa: E402
from dexteroushands_lab.scenes.bimanual_grasp_synthesis import BimanualGraspSynthesisFixture  # noqa: E402
from dexteroushands_lab.scenes.grasp_init import GraspInitBank, _reframe_sample  # noqa: E402
from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402


def left_power_contact(hold):
    digits = hold.get("left_finger_contact_time_s", {})
    duration = hold.get("duration_s", 1.0)
    return (digits.get("thumb", 0.0) > .8*duration and
            sum(digits.get(name, 0.0) > .8*duration
                for name in ("index", "middle", "ring", "pinky")) >= 2)


def main():
    left = GraspInitBank(args.left_bank).load()[args.left_index]
    right = GraspInitBank(args.right_bank).load()[args.right_index]
    if (not left.validated or left.validation.get("scope") != "left_only" or
            not right.validated or right.validation.get("right_grasp_topology") !=
            "thumb_index_middle_tripod" or
            left.metadata["bottle_parameters"] != right.metadata["bottle_parameters"] or
            any(left.metadata[key] != right.metadata[key] for key in
                ("left_usd_sha256", "right_usd_sha256", "asset_source_commit"))):
        raise ValueError("Source grasps are not validated or asset compatible")
    reframed = _reframe_sample(left, right.bottle.root_pose)
    candidate = replace(right, left=reframed.left, validated=False, validation=None,
                        source="diagnostic_composite_left_power_right_tripod",
                        metadata={**right.metadata,
                                  "composite_left_source_sha256": sha256(
                                      args.left_bank.read_bytes()).hexdigest(),
                                  "composite_right_source_sha256": sha256(
                                      args.right_bank.read_bytes()).hexdigest(),
                                  "composite_requires_fresh_physx_validation": True})
    fixture = BimanualPhase0Fixture(
        num_envs=1, device=args.device,
        config=BottleConfig(**right.metadata["bottle_parameters"]),
        control_mode="pose_command_wrist",
        left_initial_pose=candidate.left.root_pose,
        right_initial_pose=candidate.right.root_pose,
        include_hand_hand=True)
    experiment = BimanualGraspSynthesisFixture(
        fixture, require_right_tripod=True)
    hold = experiment.replay(candidate)
    second = experiment.replay(candidate) if hold["passed"] else None
    cross = (max(hold["cross_contact_ratios"].values()) == 0 and
             (second is None or max(second["cross_contact_ratios"].values()) == 0))
    passed = bool(hold["passed"] and left_power_contact(hold) and cross and
                  second and second["passed"] and left_power_contact(second))
    output = {"passed": passed,
              "left_bank": str(args.left_bank),
              "right_bank": str(args.right_bank),
              "left_source_validation": left.validation,
              "right_source_validation": right.validation,
              "candidate": asdict(candidate),
              "unsupported_hold": hold,
              "same_process_reset_replay": second,
              "independent_fresh_replay_required": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print("COMPOSITE_GRASP_DIAG " + json.dumps({
        "passed": passed, "left_power_contact": left_power_contact(hold),
        "right_tripod": hold["right_tripod_held"],
        "left_fingers": hold["left_finger_contact_time_s"],
        "right_fingers": hold["right_finger_contact_time_s"],
        "bottle_displacement_m": hold["bottle_displacement_m"],
        "minimum_separation_m": hold["minimum_separation_m"],
        "cross_contact_ratios": hold["cross_contact_ratios"],
        "same_process_replay_passed": bool(second and second["passed"]),
    }), flush=True)


try:
    main()
finally:
    app.close()
