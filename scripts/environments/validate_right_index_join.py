"""Try adding the missing right index or middle digit to a verified hold.

The existing bank provides a physical starting state. All index movement is
through official finger joint drives; the bottle and cap remain free.
Results are diagnostic until unsupported hold and fresh Reset replay pass.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bimanual_bank", type=Path, required=True)
parser.add_argument("--sample_index", type=int, default=0)
parser.add_argument("--right_proposal", type=Path, required=True)
parser.add_argument("--right_proposal_index", type=int, required=True)
parser.add_argument("--digit", choices=("index", "middle"), default="index")
parser.add_argument("--close_s", type=float, default=.6)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from dexteroushands_lab.assets.bottle import BottleConfig  # noqa: E402
from dexteroushands_lab.scenes.bimanual_grasp_synthesis import BimanualGraspSynthesisFixture  # noqa: E402
from dexteroushands_lab.scenes.grasp_init import GraspInitBank  # noqa: E402
from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402


def main():
    bank = GraspInitBank(args.bimanual_bank).load()
    sample = bank[args.sample_index]
    proposals = json.loads(args.right_proposal.read_text())["best"]
    matches = [item for item in proposals if item["index"] == args.right_proposal_index]
    if len(matches) != 1:
        raise ValueError("Right proposal index absent or ambiguous")
    fixture = BimanualPhase0Fixture(
        num_envs=1, device=args.device,
        config=BottleConfig(**sample.metadata["bottle_parameters"]),
        control_mode="pose_command_wrist",
        left_initial_pose=sample.left.root_pose,
        right_initial_pose=sample.right.root_pose,
        include_hand_hand=True,
    )
    experiment = BimanualGraspSynthesisFixture(
        fixture, require_right_tripod=True)
    result = experiment.join_digit(sample, matches[0],
                                   digit=args.digit, close_s=args.close_s)
    output = {**result,
              "candidate": asdict(result["candidate"])
              if result["candidate"] is not None else None,
              "bimanual_bank": str(args.bimanual_bank),
              "source_sample_index": args.sample_index,
              "right_proposal": str(args.right_proposal),
              "asset_commit": fixture.specs[0].source_commit,
              "right_usd_sha256": fixture.specs[1].usd_sha256,
              "dt_s": fixture.dt,
              "hand_hand_monitored": True,
              "support_used_during_right_formation": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    hold = result["unsupported_hold"] or {}
    print("RIGHT_DIGIT_JOIN_RESULT " + json.dumps({
        "digit": args.digit,
        "source_sample_index": args.sample_index,
        "aborted": result["aborted"],
        "formation_minimum_separation_m": result["formation_minimum_separation_m"],
        "formation_peak_contact_force_n": result["formation_peak_contact_force_n"],
        "hold_passed": hold.get("passed"),
        "right_finger_contact_time_s": hold.get("right_finger_contact_time_s"),
        "bottle_displacement_m": hold.get("bottle_displacement_m"),
        "reset_replay_passed": bool(result["reset_replay"] and result["reset_replay"]["passed"]),
        "same_process_passed": result["passed"],
        "fresh_process_replay_required_for_bank": True,
    }), flush=True)


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    raise
else:
    app.close()
