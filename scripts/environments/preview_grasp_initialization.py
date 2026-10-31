"""Preview a pinned bimanual initial state; candidates are labelled unverified."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("stable_pregrasp", "weak_contact", "near_contact"),
                    default="stable_pregrasp")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--bank", type=Path)
parser.add_argument("--synthesis_report", type=Path,
                    help="Preview an unvalidated support-release candidate from a PhysX report")
parser.add_argument("--candidate", type=Path,
                    help="Preview one unvalidated GraspInitSample JSON")
parser.add_argument("--sample_index", type=int, default=0)
parser.add_argument("--steps", type=int, default=500,
                    help="Maximum physics steps in headless or GUI preview")
parser.add_argument("--contact_debug", action="store_true",
                    help="Display Isaac Lab contact markers when the visualizer is installed")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from dexteroushands_lab.scenes.grasp_init import (  # noqa: E402
    GraspInitBank, grasp_sample_from_dict)
from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402


def main():
    fixture = BimanualPhase0Fixture(num_envs=1, device=args.device,
                                    contact_debug=args.contact_debug)
    if sum(bool(x) for x in (args.bank, args.synthesis_report, args.candidate)) > 1:
        raise ValueError("Choose one bank, synthesis report, or candidate JSON")
    if args.bank:
        samples = GraspInitBank(args.bank).load()
        if not samples:
            raise RuntimeError(f"No validated initial states in {args.bank}")
        sample = samples[args.sample_index]
    elif args.synthesis_report:
        records = json.loads(args.synthesis_report.read_text())["results"]
        candidate = records[args.sample_index]["candidate"]
        if candidate is None:
            raise ValueError("Selected synthesis attempt was rejected before support release")
        sample = grasp_sample_from_dict(candidate)
    elif args.candidate:
        sample = grasp_sample_from_dict(json.loads(args.candidate.read_text()))
    else:
        sample = fixture.initializer.generate(mode=args.mode, seed=args.seed)
    fixture.reset_manager.reset([0], [sample])
    print("GRASP_PREVIEW_INITIAL " + json.dumps({"mode": sample.mode,
          "seed": sample.seed, "validated": sample.validated,
          "source": sample.source, "asset_variant": sample.metadata["asset_variant"]}),
          flush=True)
    for step in range(args.steps):
        if not app.is_running():
            break
        fixture.step()
        if step % 100 == 0 or step == args.steps - 1:
            pose = fixture.bottle.data.root_pose_w.torch[0].detach().cpu().tolist()
            contacts = fixture.monitor.summary(0)
            print("GRASP_PREVIEW_FRAME " + json.dumps({"step": step + 1,
                  "bottle_pose_xyzw": pose,
                  "contact_categories": list(contacts),
                  "contact_ratios": {name: data["contact_ratio"]
                                     for name, data in contacts.items()}}), flush=True)


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
