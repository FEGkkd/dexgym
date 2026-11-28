#!/usr/bin/env python3
"""Compare dense and HMB callback histories at identical message IDs and reset anchors."""

import argparse
import json
from pathlib import Path

import numpy as np

from parity_wuji_probe import NativePair, SIDES, digest, read_jsonl, row_error, summarize


def main(args):
    identity = json.loads((args.wuji / "wuji-summary.json").read_bytes())
    tick_path = args.wuji / "wuji-bc-ticks.jsonl"
    if (digest(args.events) != identity["source_sha256"]["event_jsonl"]
            or digest(tick_path) != identity["output_sha256"][tick_path.name]):
        raise ValueError("Cold-anchor input not bound to the native Wuji receipt")
    ticks = [row for row in read_jsonl(tick_path) if row["nominal_pair_eligible"] and row["in_start_window"]]
    expected = [{row["source_index"][side]: row["b_raw_rad"][side] for row in ticks} for side in range(2)]
    events = read_jsonl(args.events)
    anchor = ticks[0]["source_index"]
    time0 = min(row["receive_ns"] for row in events if row["source_index"] in anchor)
    native = NativePair(args.wuji / "build/libparity_wuji_probe.so", identity["profile_by_side"])
    started = [False, False]
    dense = [{}, {}]
    calls = [0, 0]
    try:
        native.reset()
        for row in events:
            side = SIDES.index(row["side"])
            token = row["source_index"]
            started[side] |= token == anchor[side]
            if not started[side]:
                continue
            result = native.step(row, row["receive_ns"] - time0)
            if result["raw_rad"] is not None:
                calls[side] += 1
            if token in expected[side]:
                dense[side][token] = result["raw_rad"]
    finally:
        native.close()
    initial = [row_error(dense[side][anchor[side]], expected[side][anchor[side]]) for side in range(2)]
    if not np.array_equal(initial, [0., 0.]):
        raise ValueError("Dense and sparse analyses did not start from identical SDK outputs")
    differences = [[row_error(dense[side][row["source_index"][side]], row["b_raw_rad"][side])
                    for row in ticks] for side in range(2)]
    report = {"scope": "DIAGNOSTIC_ONLY", "paired_rows": len(ticks), "same_source_start_token": anchor,
              "same_start_max_abs_rad": initial, "dense_calls_after_start": calls,
              "per_side_max_absolute_raw_joint_error_rad": [summarize(values) for values in differences],
              "sha256": {"raw_events": digest(args.events), "sdk_ticks": digest(tick_path),
                          "script": digest(Path(__file__)),
                          "original_native_bridge": digest(args.wuji / "build/libparity_wuji_probe.so")}}
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, allow_nan=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("events", "wuji", "out"):
        parser.add_argument("--" + field, type=Path, required=True)
    main(parser.parse_args())
