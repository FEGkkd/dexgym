"""PhysX right-cap formation on a validated pose-command left-only grasp.

The output is diagnostic until the unsupported hold and independent Reset
replay both pass. It never promotes an offline reach proposal by itself.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--left_bank", type=Path, required=True)
parser.add_argument("--left_sample_index", type=int, default=0)
parser.add_argument("--right_preshape", type=Path, required=True)
parser.add_argument("--right_proposal_index", type=int, required=True)
parser.add_argument("--approach_m", type=float, default=0.0)
parser.add_argument("--approach_down_m", type=float, default=0.0,
                    help="Bounded right wrist downward approach during formation")
parser.add_argument("--middle_mcp_delta_rad", type=float, default=0.0,
                    help="Directed right middle MCP preload within 0.1 rad")
parser.add_argument("--right_wrist_dy_m", type=float, default=0.0,
                    help="Directed right wrist Y offset within 2 mm")
parser.add_argument("--close_s", type=float, default=1.2)
parser.add_argument("--settle_s", type=float, default=.3)
parser.add_argument("--follow_cap", action="store_true",
                    help="Command bounded right wrist motion to follow measured cap translation during formation")
parser.add_argument("--hand_hand", action="store_true",
                    help="Include per-link inter-hand contact views")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from dexteroushands_lab.assets.bottle import BottleConfig  # noqa: E402
from dexteroushands_lab.scenes.bimanual_grasp_synthesis import BimanualGraspSynthesisFixture  # noqa: E402
from dexteroushands_lab.scenes.grasp_init import GraspInitBank  # noqa: E402
from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402


def main():
    samples = GraspInitBank(args.left_bank).load()
    sample = samples[args.left_sample_index]
    proposals = json.loads(args.right_preshape.read_text())["best"]
    matches = [p for p in proposals if p["index"] == args.right_proposal_index]
    if len(matches) != 1:
        raise ValueError("Right proposal index absent or ambiguous")
    proposal = dict(matches[0])
    if abs(args.right_wrist_dy_m) > .002:
        raise ValueError("Right wrist Y offset exceeds the 2 mm directed search bound")
    proposal["wrist_position_w_m"] = list(proposal["wrist_position_w_m"])
    proposal["wrist_position_w_m"][1] += args.right_wrist_dy_m
    proposal["right_wrist_dy_m"] = args.right_wrist_dy_m
    print("BIMANUAL_STAGE proposal_loaded", flush=True)
    q = proposal["wrist_quaternion_wxyz"]
    right_pose = (*proposal["wrist_position_w_m"], *q[1:4], q[0])
    fixture = BimanualPhase0Fixture(
        num_envs=1, device=args.device,
        config=BottleConfig(**sample.metadata["bottle_parameters"]),
        control_mode="pose_command_wrist",
        left_initial_pose=sample.left.root_pose,
        right_initial_pose=right_pose,
        include_hand_hand=args.hand_hand,
    )
    print("BIMANUAL_STAGE fixture_ready", flush=True)
    experiment = BimanualGraspSynthesisFixture(fixture)
    print("BIMANUAL_STAGE formation_start", flush=True)
    result = experiment.run(sample, proposal, close_s=args.close_s,
                            settle_s=args.settle_s,
                            approach_m=args.approach_m,
                            approach_down_m=args.approach_down_m,
                            follow_cap_during_formation=args.follow_cap,
                            middle_mcp_delta_rad=args.middle_mcp_delta_rad)
    print("BIMANUAL_STAGE formation_done", flush=True)
    output = {**result,
              "candidate": asdict(result["candidate"])
              if result["candidate"] is not None else None,
              "left_bank": str(args.left_bank),
              "left_sample_index": args.left_sample_index,
              "right_preshape": str(args.right_preshape),
              "control_mode": "pose_command_wrist",
              "asset_commit": fixture.specs[0].source_commit,
              "left_usd_sha256": fixture.specs[0].usd_sha256,
              "right_usd_sha256": fixture.specs[1].usd_sha256,
              "dt_s": fixture.dt,
              "hand_hand_monitored": args.hand_hand,
              "support_used_during_right_formation": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    hold = result["unsupported_hold"] or {}
    print("BIMANUAL_SYNTHESIS_RESULT " + json.dumps({
        "left_seed": sample.seed, "right_proposal": proposal["index"],
        "aborted": result["aborted"],
        "formation_minimum_separation_m": result["formation_minimum_separation_m"],
        "formation_peak_contact_force_n": result["formation_peak_contact_force_n"],
        "unsupported_hold_passed": hold.get("passed"),
        "left_contact_ratio": hold.get("left_contact_ratio"),
        "right_contact_ratio": hold.get("right_contact_ratio"),
        "hold_bottle_displacement_m": hold.get("bottle_displacement_m"),
        "reset_replay_passed": result["reset_replay"] is not None and
                               result["reset_replay"]["passed"],
        "bank_eligible": result["passed"],
    }), flush=True)


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    raise
else:
    app.close()
