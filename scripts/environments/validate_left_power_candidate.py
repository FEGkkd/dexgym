"""Independently replay a captured left cylindrical-grasp candidate in PhysX.

This accepts an unvalidated synthesis report for diagnosis. A passing bottle
hold alone does not mark the five-digit topology or bank admission as passed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument("--synthesis_report", type=Path)
source.add_argument("--candidate", type=Path,
                    help="Packaged or exported unvalidated GraspInitSample JSON")
source.add_argument("--bank", type=Path,
                    help="Replay a validated bank sample without altering its state")
parser.add_argument("--sample_index", type=int, default=0)
parser.add_argument("--duration_s", type=float, default=1.0)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.duration_s < 1.0:
    parser.error("The independent hold must last at least 1 second")
app = AppLauncher(args).app

from dexteroushands_lab.assets.bottle import BottleConfig  # noqa: E402
from dexteroushands_lab.scenes.grasp_init import (  # noqa: E402
    GraspInitBank, grasp_sample_from_dict)
from dexteroushands_lab.scenes.grasp_synthesis import (  # noqa: E402
    GraspSynthesisFixture, SynthesisLimits)
from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402


def main():
    if args.bank:
        record = None
        sample = GraspInitBank(args.bank).load()[args.sample_index]
    elif args.candidate:
        record = None
        sample = grasp_sample_from_dict(json.loads(
            args.candidate.read_text(encoding="utf-8")))
    else:
        report = json.loads(args.synthesis_report.read_text(encoding="utf-8"))
        record = report["results"][args.sample_index]
        if record.get("candidate") is None:
            raise ValueError("Synthesis result has no physically captured release state")
        sample = grasp_sample_from_dict(record["candidate"])
    if sample.validated and not args.bank:
        raise ValueError("This diagnostic expects an unvalidated candidate")
    if sample.metadata.get("wrist_control_mode") != "fixed_wrist":
        raise ValueError("This fixture checks the fixed-wrist left grasp only")
    fixture = BimanualPhase0Fixture(
        num_envs=1, device=args.device,
        config=BottleConfig(**sample.metadata["bottle_parameters"]),
        control_mode="fixed_wrist",
        left_initial_pose=sample.left.root_pose,
        right_initial_pose=sample.right.root_pose,
        include_hand_hand=False)
    image_topology = (sample.validation is not None and
                      sample.validation.get("contact_topology") ==
                      "image_cylindrical_v2")
    replay = GraspSynthesisFixture(
        fixture, SynthesisLimits(release_s=args.duration_s),
        require_power_grasp=not image_topology).replay_left(sample)
    image_passed = replay.get("image_grasp", {}).get("passed") is True
    result = {
        "status": ("independent_PhysX_Reset_replay_of_validated_bank_sample"
                   if args.bank else
                   "independent_PhysX_Reset_replay_of_unvalidated_candidate"),
        "source_report": str(args.synthesis_report) if args.synthesis_report else None,
        "source_candidate": str(args.candidate) if args.candidate else None,
        "source_bank": str(args.bank) if args.bank else None,
        "source_proposal_index": (record.get("reach_proposal_index")
                                  if record else sample.seed),
        "source_release_passed": (bool(record.get("release_passed"))
                                  if record else None),
        "sample_metadata": sample.metadata,
        "sample_seed": sample.seed,
        "sample_source": sample.source,
        "captured_state": asdict(sample),
        "replay": replay,
        "validated": bool(sample.validated and replay.get("passed") and
                          (not image_topology or image_passed)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("LEFT_POWER_REPLAY_RESULT " + json.dumps({
        "proposal": result["source_proposal_index"],
        "replay_passed": replay["passed"],
        "bottle_displacement_m": replay.get("bottle_displacement_m"),
        "power_grasp_ratio": replay.get("power_grasp", {}).get(
            "valid_topology_ratio"),
        "image_grasp_ratio": replay.get("image_grasp", {}).get(
            "valid_topology_ratio"),
        "validated": result["validated"],
    }), flush=True)


try:
    main()
finally:
    app.close()
