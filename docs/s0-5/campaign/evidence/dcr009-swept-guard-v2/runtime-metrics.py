import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
attempt = run / "run/attempt-01"
summary = json.loads((run / "analysis/history-summary.json").read_bytes())
stop = json.loads((attempt / "stop-audit.json").read_bytes())
def rows(path):
    return [json.loads(line) for line in (attempt / path).read_bytes().splitlines()]

def quantiles(values):
    if not values:
        return {"n": 0}
    return {"n": len(values), **dict(zip(("p50", "p95", "p99", "max"), np.percentile(values, [50, 95, 99, 100]).tolist()))}

observed = {row["sequence"]: row for row in rows("sweep-v2-metrics.jsonl")}
native = rows("sweep-trace.jsonl")
begin = summary["recorded_timebase"]["origin_mono_ns"]
end = summary["halt"]["mono_ns"]
matched = {row["sequence"] for row in native}
assert len(observed) == len(matched) == len(native) and set(observed) == matched
out = {}
for label, request in (("entire_guard_session", native), ("historical_window_observed", [row for row in native if begin <= row["receipt_ns"] <= end])):
    assert all(observed[row["sequence"]]["reason"] == row["reason"] for row in request)
    out[label] = {"reason_counts": dict(Counter(row["reason"] for row in request)),
                  "proof_duration_ms": quantiles([(row["proof_complete_ns"] - row["receipt_ns"]) / 1e6 for row in request]),
                  "models": {}, "denials": []}
    for model in ("live", "signed"):
        metrics = [observed[row["sequence"]]["models"][model] for row in request if observed[row["sequence"]]["models"][model] is not None]
        gaps = [m["min_sample_gap_m"] * 1000 for m in metrics if m["min_sample_gap_m"] is not None]
        out[label]["models"][model] = {"evaluated": len(metrics), "cache_hits": sum(m["cache_hit"] for m in metrics),
            "cache_kinds": dict(Counter(m["cache_kind"] for m in metrics if m["cache_hit"])),
            "center_probe_counts": quantiles([m["queries"] for m in metrics]),
            "min_reported_sample_gap_mm": min(gaps) if gaps else None,
            "termination_counts": dict(Counter(m["termination"] for m in metrics))}
    if label == "historical_window_observed":
        for row in request:
            if row["reason"] == "PERMIT":
                continue
            detail = observed[row["sequence"]]
            out[label]["denials"].append({**detail, "receipt_offset_s": (row["receipt_ns"] - begin) / 1e9,
                "proof_duration_ms": (row["proof_complete_ns"] - row["receipt_ns"]) / 1e6})
out["timing_boundary"] = {"replay_input_halt_offset_s": (end - begin) / 1e9,
                         "native_stop_barrier_offset_s": (stop["native_barrier"]["mono_ns"] - begin) / 1e9,
                         "accepted_stop_offset_s": (stop["accepted_stop_receipt_mono_ns"] - begin) / 1e9}
out["geometry_scope"] = "Reported minimum among sampled and capped proof distance probes, not a whole-path minimum or an executed-contact measurement."
Path(sys.argv[2]).write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
