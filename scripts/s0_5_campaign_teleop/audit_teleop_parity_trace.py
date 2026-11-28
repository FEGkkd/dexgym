#!/usr/bin/env python3
"""Create a compact, receipt-verified ledger without copying the raw MANUS bank."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from audit_root_five_geometry_layers import stats
from dexgym.tasks.direct.wholebody_s05.v1.rmb.schema import file_sha, write_json


def compact(trial):
    steps = trial.get("physical", [])
    status = {"status": trial["status"], "cause": trial.get("cause"),
              "physical_steps": len(steps), "contact": any(s["contact_pairs"] for s in steps),
              "events": [s["event"] for s in steps], "signed_gap_mm": min(
                  (1000 * s["minimum_gap_m"] for s in steps), default=None)}
    if "snapshot" not in trial:
        return status
    previous = trial["snapshot"][0]
    status.update(candidate_changed=previous != trial["nominal"],
                  command_changed="issued" in trial and previous != trial["issued"],
                  common=trial["common_proof"], guarded=trial["guard_v2"], hold=trial["hold_proof"],
                  guard_termination=trial["guard_v2_detail"].get("termination"),
                  guard_pair=trial["guard_v2_detail"].get("unresolved_pair_indices", []),
                  common_termination=trial["common_proof_detail"].get("termination"),
                  common_pair=trial["common_proof_detail"].get("witness_pair_index"),
                  source_index=trial["command_source_index"])
    return status


def audit(args):
    source = args.physics / "fixed-arm-parity.jsonl.gz"
    receipt = args.physics / "fixed-arm-parity-summary.json"
    signed = json.loads(receipt.read_bytes())
    hmb = json.loads(args.events.read_bytes())
    if (file_sha(source) != signed["sha256"]["ledger"] or file_sha(args.sdk) != signed["sha256"]["sdk_ticks"]
            or signed["scope"] != "DIAGNOSTIC_ONLY" or hmb["root"] != signed["root"]):
        raise ValueError("Evidence identity mismatch")
    sdk = {row["tick_ns"]: row for row in map(json.loads, args.sdk.read_text(encoding="utf-8").splitlines())}
    counts = defaultdict(Counter)
    core_c = Counter()
    latency = defaultdict(list)
    contacts = defaultdict(Counter)
    selected_witness = {}
    args.out.mkdir()
    compact_path = args.out / "row-ledger.jsonl.gz"
    with gzip.open(source, "rt", encoding="utf-8") as src, compact_path.open("xb") as raw, (
        gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0)) as out:
        for index, row in enumerate(map(json.loads, src)):
            linked = sdk[row["tick_ns"]]
            if (row["partition"], row["partition_tick"], row["source_index"]) != (
                    linked["partition"], linked["partition_tick"], linked["source_index"]):
                raise ValueError("Paired physical input does not match the SDK schedule")
            result = {}
            for mode, trials in row["experiments"].items():
                status = [compact(trial) for trial in trials]
                result[mode] = status
                counts[mode]["complete"] += sum(x["physical_steps"] for x in status) == 10
                for sub, (trial, detail) in enumerate(zip(trials, status, strict=True)):
                    if mode == "C" and detail.get("candidate_changed"):
                        key = f"{detail.get('common')}:{detail.get('guarded')}:{detail.get('guard_termination')}"
                        core_c[key] += 1
                        if key not in selected_witness:
                            selected_witness[key] = {"row": index, "tick_ns": row["tick_ns"], "subcycle": sub,
                                                     "exact_trial": trial}
                    for step in trial.get("physical", []):
                        for pair in step["contact_pairs"]:
                            if pair["gap_m"] <= 0:
                                contacts[mode][tuple(sorted((pair["geom1"], pair["geom2"])))] += 1
                    if not detail["physical_steps"] or "snapshot" not in trial:
                        continue
                    if mode == "A":
                        ns = trial["source_receive_ns"]
                    else:
                        ns = [None if a is None else a - hmb["hmb_arrival_delay_ns"] for a in linked["arrival_ns"]]
                    for side, timestamp in enumerate(ns):
                        if trial["handoff_mask"][side] and timestamp is not None:
                            age = trial["physical_selection_ns"] - timestamp
                            if age < 0:
                                raise ValueError("Noncausal physical selection")
                            latency[f"{mode}_{side}"].append(age / 1e6)
                counts[mode]["physical_substeps"] += sum(item["physical_steps"] for item in status)
            out.write((json.dumps({"row": index, "tick_ns": row["tick_ns"], "partition": row["partition"],
                                  "partition_tick": row["partition_tick"], "source_index": row["source_index"],
                                  "status": result}, allow_nan=False, separators=(",", ":")) + "\n").encode())
    if index + 1 != signed["window"]["plan_ticks"]:
        raise ValueError("Compact trace did not retain all historical plan rows")
    for mode, count in counts.items():
        expected = signed["experiments"][mode]
        if (count["physical_substeps"] != expected["physics"].get("substeps", 0)
                or count["complete"] != expected["row_status"].get("COMPLETE", 0)):
            raise ValueError(f"Trace does not reconcile physical denominators for {mode}")
    write_json(args.out / "trace-audit.json", {
        "scope": "DIAGNOSTIC_ONLY", "row_count": index + 1, "physical_counts": {k: dict(v) for k, v in counts.items()},
        "c_changed_candidate_snapshot_comparisons": dict(core_c),
        "source_to_scheduled_issue_age_ms_by_mode_side": {k: stats(v) for k, v in latency.items()},
        "actual_contacts_by_pair": {mode: [{"geoms": list(pair), "nonpositive_contact_records": n} for pair, n in c.items()]
                                    for mode, c in contacts.items()},
        "representative_exact_c_witnesses": selected_witness,
        "sha256": {"physics_summary": file_sha(receipt), "full_physics": file_sha(source),
                    "compact_rows": file_sha(compact_path), "sdk_ticks": file_sha(args.sdk), "event_audit": file_sha(args.events),
                    "self": file_sha(Path(__file__))}})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("physics", "sdk", "events", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    audit(p.parse_args())
