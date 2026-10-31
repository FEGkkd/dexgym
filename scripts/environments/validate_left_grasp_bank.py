"""Replay one persisted left-only grasp from ResetManager in a fresh PhysX scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bank", type=Path, required=True)
parser.add_argument("--sample_index", type=int, required=True)
parser.add_argument("--output", type=Path)
parser.add_argument("--duration_s", type=float, default=1.0,
                    help="Unsupported hold duration after Reset; default Phase 0 acceptance is 1 s")
parser.add_argument("--left_wrist_dx_m", type=float, default=0.0,
                    help="Diagnostic inward/outward X command after Reset; not bank replay evidence")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from dexteroushands_lab.assets.bottle import BottleConfig  # noqa: E402
from dexteroushands_lab.scenes.grasp_init import GraspInitBank  # noqa: E402
from dexteroushands_lab.scenes.grasp_synthesis import GraspSynthesisFixture, SynthesisLimits  # noqa: E402
from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402


def main():
    samples = GraspInitBank(args.bank).load()
    if not 0 <= args.sample_index < len(samples):
        raise IndexError("Sample index is outside the validated bank")
    sample = samples[args.sample_index]
    if sample.validation["scope"] != "left_only":
        raise ValueError("This fixture accepts a left-only validated bank entry")
    config = BottleConfig(**sample.metadata["bottle_parameters"])
    fixture = BimanualPhase0Fixture(
        num_envs=1, device=args.device, config=config,
        control_mode=sample.validation["wrist_control_mode"],
        left_initial_pose=sample.left.root_pose,
        right_initial_pose=sample.right.root_pose,
        include_hand_hand=False)
    replay = GraspSynthesisFixture(
        fixture, SynthesisLimits(release_s=args.duration_s)).replay_left(
            sample, wrist_target_offset_xyz=(args.left_wrist_dx_m, 0.0, 0.0))
    summary = {"sample_index": args.sample_index, "seed": sample.seed,
               "hold_duration_s": args.duration_s,
               "left_wrist_dx_m": args.left_wrist_dx_m,
               "persisted_validation_passed": sample.validation["passed"],
               "fresh_process_replay_passed": replay["passed"] and
                                              args.left_wrist_dx_m == 0.0,
               "diagnostic_hold_passed": replay["passed"],
               "bottle_displacement_m": replay["bottle_displacement_m"],
               "bottle_rotation_rad": replay["bottle_rotation_rad"],
               "contact_ratio": replay["left_contact_ratio"],
               "maximum_contact_force_n": replay["left_contact_peak_n"],
               "minimum_separation_m": replay["minimum_separation_m"]}
    print("LEFT_BANK_REPLAY " + json.dumps(summary), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"summary": summary, "replay": replay},
                                          indent=2) + "\n")
    if not replay["passed"]:
        raise RuntimeError("Persisted grasp failed fresh-process Reset replay")


try:
    main()
finally:
    app.close()
