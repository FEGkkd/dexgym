"""Validate a later state of an already PhysX-proven bimanual grasp family."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--parent_result", type=Path, required=True)
parser.add_argument("--parent_fresh_replay", type=Path, required=True)
parser.add_argument("--offset_s", type=float, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from dexteroushands_lab.assets.bottle import BottleConfig  # noqa: E402
from dexteroushands_lab.scenes.bimanual_grasp_synthesis import BimanualGraspSynthesisFixture  # noqa: E402
from dexteroushands_lab.scenes.grasp_init import grasp_sample_from_dict  # noqa: E402
from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402


def main():
    raw = args.parent_result.read_bytes()
    parent = json.loads(raw)
    fresh = json.loads(args.parent_fresh_replay.read_text())
    if (parent.get("passed") is not True
            or fresh["summary"].get("fresh_process_replay_passed") is not True
            or fresh["summary"].get("source_result_sha256") != sha256(raw).hexdigest()):
        raise ValueError("A verified parent result and matching fresh replay are required")
    sample = grasp_sample_from_dict(parent["candidate"])
    fixture = BimanualPhase0Fixture(
        num_envs=1, device=args.device,
        config=BottleConfig(**sample.metadata["bottle_parameters"]),
        control_mode="pose_command_wrist",
        left_initial_pose=sample.left.root_pose,
        right_initial_pose=sample.right.root_pose,
        include_hand_hand=True,
    )
    result = BimanualGraspSynthesisFixture(fixture).derive_temporal_snapshot(
        sample, parent_result_sha256=sha256(raw).hexdigest(), offset_s=args.offset_s)
    output = {**result, "candidate": asdict(result["candidate"])
              if result["candidate"] is not None else None,
              "left_sample_index": parent["left_sample_index"],
              "asset_commit": fixture.specs[0].source_commit,
              "left_usd_sha256": fixture.specs[0].usd_sha256,
              "right_usd_sha256": fixture.specs[1].usd_sha256,
              "hand_hand_monitored": True,
              "support_used_during_right_formation": False,
              "parent_result_sha256": sha256(raw).hexdigest(),
              "same_grasp_family": True,
              "temporal_offset_s": args.offset_s,
              "dt_s": fixture.dt}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print("BIMANUAL_TEMPORAL_SNAPSHOT " + json.dumps({
        "offset_s": args.offset_s, "passed": result["passed"],
        "hold_displacement_m": result["unsupported_hold"]["bottle_displacement_m"]
        if result["unsupported_hold"] else None,
        "radial_opposition_ratio": result["unsupported_hold"]["right_radial_opposition_ratio"]
        if result["unsupported_hold"] else None,
        "reset_replay_passed": result["reset_replay"] is not None and
                               result["reset_replay"]["passed"],
    }), flush=True)


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    raise
else:
    app.close()
