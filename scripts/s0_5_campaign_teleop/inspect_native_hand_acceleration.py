#!/usr/bin/env python3
"""Measure hand acceleration from consecutive actual signed 500 Hz samples."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from audit_root_five_geometry_layers import stats
from dexgym.tasks.direct.wholebody_s05.v1.rmb.schema import file_sha, write_json


def main(args):
    summary = json.loads((args.physics / "fixed-arm-native-summary.json").read_bytes())
    path = args.physics / "fixed-arm-native.jsonl.gz"
    if file_sha(path) != summary["sha256"]["ledger"] or summary["physics_hz"] != 500:
        raise ValueError("Native physics receipt does not bind the 500 Hz trajectory")
    last = {}
    sample = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for row in map(json.loads, stream):
            for mode, steps in row["experiments"].items():
                for step in steps:
                    if "qd_measured" not in step:
                        last.pop(mode, None)
                        continue
                    speed = np.asarray(step["qd_measured"][14:])
                    time = step["feedback_available_ns"]
                    previous = last.get(mode)
                    if previous and previous[0] == row["window"] and time - previous[1] == 2_000_000:
                        sample[mode].append(abs(speed - previous[2]) / .002)
                    last[mode] = row["window"], time, speed
    write_json(args.out, {"scope": "DIAGNOSTIC_ONLY", "unit": "rad/s^2",
                          "sample_basis": "Absolute finite difference of consecutive actual 2 ms qd samples inside one window",
                          "threshold_evaluated": False, "hand_joint_order": summary["signed_joints"][14:],
                          "result": {mode: {"by_joint": stats(values), "maximum_axis_per_step": stats(
                              [float(v.max()) for v in values])} for mode, values in sample.items()},
                          "sha256": {"ledger": file_sha(path), "source": file_sha(Path(__file__))}})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physics", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
