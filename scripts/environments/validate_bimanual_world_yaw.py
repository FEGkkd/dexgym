"""PhysX-test a world-yaw variant of one proven bimanual grasp family."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--parent_result", type=Path, required=True)
parser.add_argument("--parent_fresh_replay", type=Path, required=True)
parser.add_argument("--yaw_deg", type=float, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from dexteroushands_lab.assets.bottle import BottleConfig  # noqa: E402
from dexteroushands_lab.scenes.bimanual_grasp_synthesis import BimanualGraspSynthesisFixture  # noqa: E402
from dexteroushands_lab.scenes.grasp_init import grasp_sample_from_dict, rotate_sample_world_z  # noqa: E402
from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402


def main():
    raw = args.parent_result.read_bytes()
    parent = json.loads(raw)
    fresh = json.loads(args.parent_fresh_replay.read_text())
    digest = sha256(raw).hexdigest()
    if (parent.get("passed") is not True
            or fresh["summary"].get("fresh_process_replay_passed") is not True
            or fresh["summary"].get("source_result_sha256") != digest):
        raise ValueError("A verified parent result and matching fresh replay are required")
    sample = rotate_sample_world_z(
        grasp_sample_from_dict(parent["candidate"]), math.radians(args.yaw_deg),
        parent_result_sha256=digest)
    fixture = BimanualPhase0Fixture(
        num_envs=1, device=args.device,
        config=BottleConfig(**sample.metadata["bottle_parameters"]),
        control_mode="pose_command_wrist",
        left_initial_pose=sample.left.root_pose,
        right_initial_pose=sample.right.root_pose,
        include_hand_hand=True,
    )
    experiment = BimanualGraspSynthesisFixture(fixture)
    hold = experiment.replay(sample)
    replay = experiment.replay(sample) if hold["passed"] else None
    output = {
        "passed": bool(hold["passed"] and replay and replay["passed"]),
        "aborted": None,
        "formation_minimum_separation_m": hold["minimum_separation_m"],
        "formation_peak_contact_force_n": hold["peak_contact_force_n"],
        "unsupported_hold": hold, "reset_replay": replay,
        "candidate": asdict(sample),
        "left_sample_index": parent["left_sample_index"],
        "asset_commit": fixture.specs[0].source_commit,
        "left_usd_sha256": fixture.specs[0].usd_sha256,
        "right_usd_sha256": fixture.specs[1].usd_sha256,
        "hand_hand_monitored": True,
        "support_used_during_right_formation": False,
        "same_relative_grasp_family": True,
        "parent_result_sha256": digest,
        "world_yaw_deg": args.yaw_deg,
        "dt_s": fixture.dt,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print("BIMANUAL_WORLD_YAW " + json.dumps({
        "yaw_deg": args.yaw_deg, "passed": output["passed"],
        "bottle_displacement_m": hold["bottle_displacement_m"],
        "radial_opposition_ratio": hold["right_radial_opposition_ratio"],
        "reset_replay_passed": replay is not None and replay["passed"],
    }), flush=True)


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    raise
else:
    app.close()
