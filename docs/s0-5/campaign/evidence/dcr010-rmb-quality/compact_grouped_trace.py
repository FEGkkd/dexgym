"""Export status and proof provenance from a receipt-verified grouped run."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory


METRIC_KEYS = (
    "termination", "queries", "max_depth", "min_sample_gap_m", "endpoint_gap_m",
    "witness_pair_index", "unresolved_pair_indices", "original_hold_proof",
    "escalation_max_depth",
)


def metrics(fields):
    result = {key: fields[key] for key in METRIC_KEYS if key in fields}
    if "original_hold_metrics" in fields:
        result["original_hold_metrics"] = metrics(fields["original_hold_metrics"])
    return result


def compact_step(step):
    groups = [
        {
            "group": group["group"],
            "nominal_status": group["nominal_status"],
            "choice": group["choice"],
            "committed": group.get("committed"),
            "outcome": group.get("label", "UNCOMMITTED_FAULT"),
            "trials": [
                {"proof": trial["proof"], "termination": trial["metrics"].get("termination")}
                for trial in group["trials"]
            ],
        }
        for group in step["groups"]
    ]
    return {
        "proof": step["proof"],
        "metrics": metrics(step["metrics"]),
        "hold": step["hold"],
        "hold_proof": step["hold_proof"],
        "hold_metrics": metrics(step["hold_metrics"] or {}),
        "fault": step["fault"],
        "groups": groups,
    }


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def export(run, out):
    original = run / "diagnostic-events.jsonl"
    receipt_bytes = (run / "receipt.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    if receipt["schema"] != "s05.rmb.offline_diagnostic.v2" or not receipt["group_admission"]:
        raise ValueError("Expected a grouped offline diagnostic receipt")
    part_index = {part["partition"]: index for index, part in enumerate(receipt["partitions"])}
    group_names = tuple(receipt["group_joint_names"])
    event_hash = hashlib.sha256()
    statuses, proofs, holds, hold_escalations = Counter(), Counter(), Counter(), Counter()
    group_outcomes = {name: Counter() for name in group_names}
    previous_position = (-1, -1)
    previous_time = -1
    terminal = None
    row_count = 0
    if out.exists():
        raise FileExistsError(out)
    with TemporaryDirectory(dir=out.parent) as scratch:
        temp = Path(scratch)
        ledger = temp / "step-status.jsonl.gz"
        with original.open("rb") as source, ledger.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                for row_count, line in enumerate(source, start=1):
                    event_hash.update(line)
                    row = json.loads(line)
                    position = (part_index[row["partition"]], row["tick"])
                    if position <= previous_position or row["time_ns"] <= previous_time:
                        raise ValueError(f"Unexpected event order at source row {row_count}")
                    previous_position, previous_time = position, row["time_ns"]
                    result = {key: row[key] for key in ("partition", "tick", "time_ns", "control_cutoff_ns", "status")}
                    statuses[result["status"]] += 1
                    result.update(source_row=row_count - 1, safety=[])
                    for arm_subcycle, step in enumerate(row["safety"]):
                        if terminal is None and step["fault"]:
                            terminal = {"source_row": row_count - 1, "arm_subcycle": arm_subcycle, "event": row}
                        entry = compact_step(step)
                        result["safety"].append({"arm_subcycle": arm_subcycle, **entry})
                        proofs[step["proof"]] += 1
                        if step["hold_proof"] is not None:
                            holds[step["hold_proof"]] += 1
                        if "original_hold_proof" in entry["hold_metrics"]:
                            hold_escalations[step["hold_proof"]] += 1
                        for group in entry["groups"]:
                            group_outcomes[group["group"]][group["outcome"]] += 1
                    zipped.write((json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n").encode())
        source_sha = event_hash.hexdigest()
        if (source_sha != receipt["diagnostic_events_sha256"]
                or row_count != receipt["historical"]["processed_ticks"]
                or dict(statuses) != receipt["statuses"] or dict(proofs) != receipt["proofs"]):
            raise ValueError("Trace identity or denominators disagree with the run receipt")
        summary = {
            "root": receipt["root"],
            "source_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "source_diagnostic_events_sha256": source_sha,
            "compressed_step_status_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
            "source_rows": row_count,
            "statuses": statuses,
            "final_combination_proofs": proofs,
            "hold_proofs": holds,
            "unchanged_hold_escalations_by_result": hold_escalations,
            "unchanged_hold_proof_max_depth": receipt.get("unchanged_hold_proof_max_depth"),
            "group_outcomes": group_outcomes,
            "physical_fault": receipt["physical_fault"],
        }
        write_json(temp / "step-status-summary.json", summary)
        write_json(temp / "terminal-proof.json", terminal)
        out.mkdir()
        for result in temp.iterdir():
            result.rename(out / result.name)
    print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("out", type=Path)
    options = parser.parse_args()
    export(options.run, options.out)
