#!/usr/bin/env python3
"""Preselect bounded physical windows using only recorded hand geometry."""

import argparse
from pathlib import Path

from dexgym.tasks.direct.wholebody_s05.v1.rmb.schema import file_sha, write_json
from summarize_teleop_parity_hand_slices import (
    ROOT, from_input, jsonl, margin, passes, rapid_runs, stable_runs,
)


PROXIES = ("natural_open", "fist", "thumb_index_pinch", "thumb_middle_pinch", "spread",
           "one_finger_flexion", "asymmetric_hands", "rapid_repeated_open_close")


def plan(source, output):
    rows = [row for row in jsonl(source) if row["in_start_window"]]
    if len(rows) != 4804:
        raise ValueError("Fixed parity audit requires the signed 4,804-row timeline")
    hands = [from_input(row) for row in rows]
    candidates = []
    source_windows = [{"first": 0, "stop": 105, "names": ["initial_handoff_control"]}]
    for gesture in PROXIES:
        sides = ("both",) if gesture == "asymmetric_hands" else ("left", "right")
        options = []
        for side in sides:
            if gesture == "rapid_repeated_open_close":
                options.extend((0.0, start, stop, side) for start, stop in rapid_runs(rows, hands, side))
            else:
                scores = [margin(gesture, pair, side) for pair in hands]
                options.extend((max(scores[start:stop]), start, stop, side) for start, stop in
                               stable_runs(rows, [passes(value) for value in scores], side))
        if not options:
            candidates.append({"gesture": gesture, "status": "INSUFFICIENT_INPUT_EVIDENCE"})
            continue
        score, first, stop, side = min(options, key=lambda o: (-o[0], o[2] - o[1], o[1], o[3]))
        middle = first + (stop - first - 1) // 2
        if gesture == "rapid_repeated_open_close":
            probe_start, probe_end = first, stop
        else:
            probe_start, probe_end = max(first, middle - 4), min(stop, middle + 5)
        source_windows.append({"first": max(0, probe_start - 76), "stop": probe_end, "names": [gesture]})
        candidates.append({"gesture": gesture, "status": "PLANNED_PROXY_PROBE", "side": side, "proxy_margin": score,
                           "input_support_first_row": first, "input_support_stop_row_exclusive": stop,
                           "within_proxy_first_row": probe_start, "within_proxy_stop_row_exclusive": probe_end})
    merged = []
    for item in sorted(source_windows, key=lambda x: x["first"]):
        if merged and item["first"] <= merged[-1]["stop"]:
            merged[-1]["stop"] = max(merged[-1]["stop"], item["stop"])
            merged[-1]["names"].extend(item["names"])
        else:
            merged.append(item)
    physical = [{"start_tick_ns": rows[w["first"]]["tick_ns"],
                 "stop_exclusive_tick_ns": rows[w["stop"]]["tick_ns"] if w["stop"] < len(rows) else
                 rows[-1]["tick_ns"] + 20_000_000, "start_row": w["first"], "stop_row_exclusive": w["stop"],
                 "names": w["names"]} for w in merged]
    write_json(output, {"scope": "DIAGNOSTIC_ONLY", "root": ROOT, "frozen_planned_rows": len(rows),
                        "selection_input": "Signed HMB geometry only, independent of SDK or physics results",
                        "input_sha256": file_sha(source), "physical_windows": physical,
                        "start_each_window": "Signed Home, use recorded preceding movement on unchanged time grid",
                        "proxy_evidence": candidates,
                        "rows_in_physical_windows": sum(w["stop"] - w["first"] for w in merged)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan(args.input, args.out)
