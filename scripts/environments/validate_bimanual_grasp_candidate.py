"""Fresh-process, no-support PhysX Reset replay of a bimanual grasp candidate."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--result", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from dexteroushands_lab.assets.bottle import BottleConfig  # noqa: E402
from dexteroushands_lab.scenes.bimanual_grasp_synthesis import BimanualGraspSynthesisFixture  # noqa: E402
from dexteroushands_lab.scenes.grasp_init import grasp_sample_from_dict  # noqa: E402
from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402


def main():
    raw = args.result.read_bytes()
    result = json.loads(raw)
    if result.get("passed") is not True or result.get("candidate") is None:
        raise ValueError("Only a candidate with an unsupported hold and Reset replay may be checked")
    sample = grasp_sample_from_dict(result["candidate"])
    fixture = BimanualPhase0Fixture(
        num_envs=1, device=args.device,
        config=BottleConfig(**sample.metadata["bottle_parameters"]),
        control_mode="pose_command_wrist",
        left_initial_pose=sample.left.root_pose,
        right_initial_pose=sample.right.root_pose,
        include_hand_hand=True,
    )
    require_tripod = sample.metadata.get("right_cap_synthesis", {}).get(
        "require_right_tripod", False)
    replay = BimanualGraspSynthesisFixture(
        fixture, require_right_tripod=require_tripod).replay(sample)
    summary = {
        "source_result": str(args.result), "left_seed": sample.seed,
        "source_result_sha256": sha256(raw).hexdigest(),
        "right_proposal_index": sample.metadata.get("right_cap_proposal_index"),
        "fresh_process_replay_passed": replay["passed"],
        "bottle_displacement_m": replay["bottle_displacement_m"],
        "bottle_rotation_rad": replay["bottle_rotation_rad"],
        "left_contact_ratio": replay["left_contact_ratio"],
        "right_contact_ratio": replay["right_contact_ratio"],
        "right_radial_opposition_ratio": replay["right_radial_opposition_ratio"],
        "right_tripod_held": replay["right_tripod_held"],
        "minimum_separation_m": replay["minimum_separation_m"],
        "max_thread_lateral_error_m": replay["max_thread_lateral_error_m"],
        "cross_contact_ratios": replay["cross_contact_ratios"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "replay": replay}, indent=2) + "\n")
    print("BIMANUAL_FRESH_REPLAY " + json.dumps(summary), flush=True)
    if not replay["passed"]:
        raise RuntimeError("Bimanual candidate failed fresh-process Reset replay")


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    raise
else:
    app.close()
