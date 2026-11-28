#!/usr/bin/env python3
"""Bind diagnostic publication receipts to both-model proofs and observed ROS output."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from run_start_handoff_cohort import digest


GROUPS = ("arm", "left", "right")
TOPICS = {
    "/tianji/joint_command": "arm",
    "/left_hand/command": "left",
    "/right_hand/command": "right",
}
LINE = re.compile(
    r"S05_DCR009_GUARD group=(arm|left|right) session=(\S+) generation=(\d+) command=(\d+) "
    r"proof=(\d+) published=([01]) reason=(\d+) feedback_ns=(\d+)"
)


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError(f"Truncated stream: {path}")
            yield json.loads(line)


def inspect_attempt(path, initial):
    guard_path = path / "sweep-trace.jsonl"
    trace_path = path / "evidence" / "trace.jsonl"
    results = {"root": initial["root"], "phase": initial["phase"], "start_status": initial.get("start_status"),
               "historical_packets": initial.get("historical_packets"), "formal_coverage_credit": False}
    if not guard_path.is_file() or not trace_path.is_file():
        results.update(status="INSUFFICIENT_TRACE", observed_missing=[item.name for item in (guard_path, trace_path) if not item.is_file()])
        return results
    proofs, reasons = {}, Counter()
    for row in rows(guard_path):
        reasons[row["reason"]] += 1
        if row["reason"] == "PERMIT":
            if row["sequence"] in proofs or row["issued_measured_proposed_rad"] is None:
                raise ValueError(f"Ambiguous sweep admission: {path}")
            proofs[row["sequence"]] = GROUPS[row["group"]]
    reports, controller_denials, attested_ids = {key: 0 for key in GROUPS}, Counter(), set()
    reported = {key: set() for key in GROUPS}
    unproven = []
    with (path / "sim.log").open(encoding="utf-8", errors="replace") as log:
        for line in log:
            record = LINE.search(line)
            if not record:
                continue
            group, _, _, command, proof, publish, reason, _ = record.groups()
            command, proof = int(command), int(proof)
            if publish == "0":
                controller_denials[group, reason] += 1
                continue
            reports[group] += 1
            reported[group].add(command)
            if proofs.get(proof) != group or proof in attested_ids or reason != "0":
                unproven.append({"group": group, "command": command, "proof": proof})
            attested_ids.add(proof)

    actual = {name: Counter() for name in GROUPS}
    guard_home = None
    for row in rows(trace_path):
        if row["kind"] == "internal_home_guard":
            guard_home = row["statuses"]
        if row["kind"] == "output" and row["topic"] in TOPICS:
            value = row["value"]
            sequence = value.get("sequence")
            if type(sequence) is int:
                actual[TOPICS[row["topic"]]][sequence] += 1
    unexplained = {group: sorted(actual[group].keys() - reported[group]) for group in GROUPS}
    results.update(
        status="PREPUBLICATION_PROOFS_MATCH" if not unproven and not any(unexplained.values()) and guard_home == ["PERMIT", "PERMIT"]
        else "UNBOUND_OR_MISSING_EVIDENCE",
        sweep_request_reasons=dict(reasons),
        controller_pub_proven=reports,
        controller_denials={f"{group}:{reason}": count for (group, reason), count in controller_denials.items()},
        total_observed_ros_command_messages={group: sum(counter.values()) for group, counter in actual.items()},
        unbound_publish_proofs=unproven,
        observed_without_publish_receipt=unexplained,
        internal_home_proof=guard_home,
        sweep_trace_sha256=digest(guard_path),
        native_trace_sha256=digest(trace_path),
        plant_substep_application_qualified=False,
        dynamic_overshoot_qualified=False,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.run / "run_receipt.json"
    receipt = read(receipt_path)
    summaries = [inspect_attempt(args.run / f"attempt-{row['attempt']:02d}", row) for row in receipt["attempts"]]
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump({"scope": "DIAGNOSTIC_ONLY", "formal_coverage_credit": False, "input_receipt_sha256": digest(receipt_path),
                   "run_status": receipt["status"], "results": summaries}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if all(row.get("status") == "PREPUBLICATION_PROOFS_MATCH" for row in summaries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
