"""Add left index contact to a verified right-tripod, free-bottle hold.

The left index moves only through its official PhysX joint drive. This
candidate requires a separate 1-second hold and fresh-process Reset replay.
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
parser.add_argument("--joint", choices=("mcp_abd", "mcp_flex", "pip"), required=True)
parser.add_argument("--delta_rad", type=float, required=True)
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
    sample = GraspInitBank(args.bimanual_bank).load()[args.sample_index]
    fixture = BimanualPhase0Fixture(
        num_envs=1, device=args.device,
        config=BottleConfig(**sample.metadata["bottle_parameters"]),
        control_mode="pose_command_wrist",
        left_initial_pose=sample.left.root_pose,
        right_initial_pose=sample.right.root_pose,
        include_hand_hand=True)
    experiment = BimanualGraspSynthesisFixture(
        fixture, require_right_tripod=True)
    result = experiment.join_left_index(
        sample, joint_name=f"l_index_finger_{args.joint}",
        delta_rad=args.delta_rad,
        close_s=args.close_s)
    output = {**result,
              "candidate": asdict(result["candidate"])
              if result["candidate"] is not None else None,
              "bimanual_bank": str(args.bimanual_bank),
              "source_sample_index": args.sample_index,
              "asset_commit": fixture.specs[0].source_commit,
              "left_usd_sha256": fixture.specs[0].usd_sha256,
              "right_usd_sha256": fixture.specs[1].usd_sha256,
              "dt_s": fixture.dt,
              "hand_hand_monitored": True,
              "support_used_during_right_formation": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    hold = result["unsupported_hold"] or {}
    print("LEFT_INDEX_JOIN_RESULT " + json.dumps({
        "joint": args.joint, "delta_rad": args.delta_rad,
        "aborted": result["aborted"],
        "formation_minimum_separation_m": result["formation_minimum_separation_m"],
        "formation_peak_contact_force_n": result["formation_peak_contact_force_n"],
        "hold_passed": hold.get("passed"),
        "left_index_middle_held": hold.get("left_index_middle_held"),
        "left_finger_contact_time_s": hold.get("left_finger_contact_time_s"),
        "right_finger_contact_time_s": hold.get("right_finger_contact_time_s"),
        "reset_replay_passed": bool(result["reset_replay"] and
                                    result["reset_replay"]["passed"]),
        "same_process_passed": result["passed"],
    }), flush=True)


try:
    main()
finally:
    app.close()
