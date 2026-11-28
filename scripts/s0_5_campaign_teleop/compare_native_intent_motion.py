#!/usr/bin/env python3
"""Compare dense C and sparse B on jointly observed frozen gesture frames."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from audit_root_five_geometry_layers import stats
from dexgym.tasks.direct.wholebody_s05.v1.rmb.schema import file_sha, write_json
from summarize_teleop_parity_hand_slices import shape


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def complete(row, mode):
    steps = row["experiments"][mode]
    return len(steps) == 10 and all("q_measured" in step and not step["event"] for step in steps)


def main(args):
    physical = json.loads((args.physics / "fixed-arm-native-summary.json").read_bytes())
    functional = json.loads((args.functional / "summary.json").read_bytes())
    a = args.physics / "fixed-arm-native.jsonl.gz"
    b = args.functional / "rows.jsonl.gz"
    if file_sha(a) != physical["sha256"]["ledger"] or file_sha(b) != functional["sha256"]["ledger"]:
        raise ValueError("Frozen evidence was modified")
    planned = defaultdict(int)
    paired = defaultdict(int)
    results = defaultdict(lambda: defaultdict(list))
    gestures = {p["gesture"]: p["side"] for p in json.loads(args.windows.read_bytes())["proxy_evidence"]
                if p["status"] == "PLANNED_PROXY_PROBE"}
    if file_sha(args.windows) != physical["sha256"]["windows"]:
        raise ValueError("Frozen gesture window changed")
    geometry = {row["tick_ns"]: row for row in read(b)}
    for row in read(a):
        target = geometry[row["tick_ns"]]
        if target["source_index"] != row["input_source_index"]:
            raise ValueError("Different source identities during pair comparison")
        names = target["source_proxy_windows"]
        for name in names:
            planned[name] += 1
        if not names or not all(complete(row, m) for m in ("B", "C")):
            continue
        for name in names:
            side_choice = gestures[name]
            sides = ("left", "right") if side_choice == "both" else (side_choice,)
            goals = target["HUMAN_FUNCTIONAL_INTENT"]
            if not goals or any(s not in goals for s in sides):
                continue
            paired[name] += 1
            for side in sides:
                index = ("left", "right").index(side)
                human = goals[side]
                measured = {m: np.asarray(row["fk_at_last_observed_substep"][m]["landmark_local_m"][index])
                            for m in ("B", "C")}
                for m, local in measured.items():
                    function = shape(local.tolist())
                    values = np.asarray(function["thumb_gaps_mm"])
                    human_points = np.asarray(human["rotated_raw_landmarks_m"])
                    human_shape = shape(human_points.tolist())
                    expected = np.asarray(human_shape["thumb_gaps_mm"])
                    prefix = f"{m}.{side}"
                    results[name][f"{prefix}.thumb_index_gap_mm"].append(float(values[0]))
                    results[name][f"{prefix}.thumb_index_gap_error_mm"].append(float(abs(values[0] - expected[0])))
                    results[name][f"{prefix}.finger_curl_error"].append(
                        float(np.mean(abs(np.asarray(function["curl"]) - np.asarray(human_shape["curl"])))))
                results[name][f"B_vs_C.{side}.local_tip_delta_mm"].append(
                    (1000 * np.linalg.norm(measured["B"][:, -1] - measured["C"][:, -1], axis=1)).tolist())
    write_json(args.out, {"scope": "DIAGNOSTIC_ONLY", "sample_basis": "Jointly completed 20 ms intervals, final actual FK",
                         "geometry_basis": "Same signed model, unscaled original MANUS functional intent",
                         "planned_selected_proxy_ticks": dict(planned), "paired_completed_proxy_ticks": dict(paired),
                         "results": {name: {key: stats(values) for key, values in series.items()}
                                     for name, series in results.items()},
                         "sha256": {"physical_ledger": file_sha(a), "functional_ledger": file_sha(b),
                                     "windows": file_sha(args.windows), "source": file_sha(Path(__file__))}})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for opt in ("physics", "functional", "windows", "out"):
        p.add_argument("--" + opt, type=Path, required=True)
    main(p.parse_args())
