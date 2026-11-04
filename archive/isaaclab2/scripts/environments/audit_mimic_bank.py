"""Audit Reset-bank cap poses against the PhysX mimic screw constraint.

This is an offline geometry test. It never edits the bank or claims that a
geometrically consistent pose is a stable hand grasp.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dexteroushands_lab.assets.bottle import BottleConfig
from dexteroushands_lab.assets.bottle.mimic_contract import mimic_reset_geometry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("--pitch-tolerance-m", type=float, default=0.00005)
    parser.add_argument("--lateral-tolerance-m", type=float, default=0.0001)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero when any sample fails the geometry gate")
    args = parser.parse_args()
    rows = json.loads(args.bank.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("expected a nonempty JSON array of GraspInitSample rows")
    details = []
    for index, row in enumerate(rows):
        cfg = BottleConfig(**row["metadata"]["bottle_parameters"])
        geometry = mimic_reset_geometry(row["bottle"]["root_pose"],
                                        row["cap"]["root_pose"],
                                        row["thread_initial_angle_rad"], cfg)
        qualified = (abs(geometry["pitch_error_m"]) <= args.pitch_tolerance_m
                     and geometry["lateral_m"] <= args.lateral_tolerance_m)
        details.append({"index": index, "seed": row.get("seed"),
                        "source": row.get("source"),
                        "legacy_validated": row.get("validated"),
                        "mimic_geometry_qualified": qualified,
                        **geometry})
    report = {"bank": str(args.bank), "sample_count": len(details),
              "mimic_geometry_qualified_count": sum(x["mimic_geometry_qualified"] for x in details),
              "pitch_tolerance_m": args.pitch_tolerance_m,
              "lateral_tolerance_m": args.lateral_tolerance_m,
              "samples": details}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}))
    for sample in details:
        print(json.dumps(sample))
    if args.strict and report["mimic_geometry_qualified_count"] != len(details):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
