#!/usr/bin/env python3
"""Summarize offline coverage, fault splits and errors from verified artifacts."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from dexgym.tasks.direct.wholebody_s05.v1.rmb.offline_nominal import OfflineNominalStatus
from dexgym.tasks.direct.wholebody_s05.v1.rmb.schema import file_sha, write_json


def percentiles(values):
    return np.percentile(values, [50, 95, 99]).tolist() if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    path = args.run / "receipt.json"
    receipt = json.loads(path.read_bytes())
    if receipt["schema"] != "s05.rmb.offline_diagnostic.v1" or receipt["formal_coverage_credit"]:
        raise ValueError("Expected an isolated offline diagnostic receipt")
    events_path = args.run / "diagnostic-events.jsonl"
    start_path = args.run / "start-events.jsonl"
    if file_sha(events_path) != receipt["diagnostic_events_sha256"] or file_sha(start_path) != receipt["start_events_sha256"]:
        raise ValueError("Diagnostic runtime evidence hash mismatch")
    rows = {}
    with events_path.open(encoding="utf-8") as stream:
        for event in map(json.loads, stream):
            key = event["partition"], event["tick"]
            if key in rows:
                raise ValueError("Duplicate original scheduled row")
            rows[key] = event
    counts_by_group = [Counter() for _ in range(4)]
    proof_causes = Counter()
    classes = Counter()
    tip_errors, active_tip_errors = [], []
    first_hold = first_active = first_stop = None
    geometry_unknown = retained_history = bad_velocity_boundaries = 0
    smallest_observed_gap = None
    largest_sampled_penetration = 0.
    safety_conservation = Counter()
    divisions = []
    for part in receipt["partitions"]:
        def data(name):
            path = args.run / part["directory"] / f"{name}.npy"
            if file_sha(path) != part["arrays"][name]["sha256"]:
                raise ValueError(f"Array hash mismatch: {part['partition']}/{name}")
            return np.load(path, allow_pickle=False)

        statuses = data("nominal_status")
        applied = data("applied_substeps")
        issued = data("q_issued_mask")
        legal = data("q_nominal_mask")
        target_tip, actual_tip = data("target_world_tip_xyz"), data("actual_world_tip_xyz")
        target_mask, actual_mask = data("target_world_tip_xyz_mask"), data("actual_world_tip_xyz_mask")
        gaps, depths, intervals = (data(name) for name in (
            "physical_clearance_m", "physical_penetration_m", "physical_interval_mask"))
        history = data("history_eligible")
        segment, velocity = data("hmb_segment"), data("qd_nominal_mask")
        bad_velocity_boundaries += int(np.count_nonzero(velocity[1:] & (segment[1:] != segment[:-1])))
        partitions = Counter()
        for i in range(len(applied)):
            event = rows.get((part["partition"], i))
            if event is None:
                continue
            elapsed = (event["time_ns"] - receipt["window"]["hmb_cutoff_start_ns"]) / 1e9
            status = event["status"]
            marker = {"elapsed_s": elapsed, "partition": part["partition"], "tick": i, "status": status}
            first_hold = marker if first_hold is None and "HOLD" in status else first_hold
            first_active = marker if first_active is None and status == "ACTIVE" else first_active
            first_stop = marker if first_stop is None and int(applied[i]) != 10 else first_stop
            classes[status] += 1
            partitions[status] += 1
            retained_history += int(history[i])
            geometry_unknown += int(not target_mask[i].all())
            if applied[i] == 10 and target_mask[i].all() and actual_mask[i].all():
                error = float(np.linalg.norm(target_tip[i] - actual_tip[i], axis=-1).max() * 1000.)
                tip_errors.append(error)
                if legal[i]:
                    active_tip_errors.append(error)
            for axis, records in enumerate(statuses[i].T):
                counts_by_group[axis].update(OfflineNominalStatus(int(code)).name for code in records if code)
            for subcycle, evidence in enumerate(event["safety"]):
                proof, metrics = evidence["proof"], evidence["metrics"]
                proof_causes[(proof, metrics.get("termination", proof))] += 1
                if issued[i, subcycle]:
                    approved = evidence["hold_proof"] if evidence["hold"] else proof
                    if approved != "PERMIT" or proof == "NATIVE_PAIR_HARD_FAULT":
                        raise ValueError(f"Issued command lacks protection evidence: {marker} subcycle {subcycle}")
                    safety_conservation["proved_issued_subcycles"] += 1
                    safety_conservation["proved_issued_hold_subcycles"] += int(evidence["hold"])
                if intervals[i, subcycle] and not issued[i, subcycle]:
                    raise ValueError(f"Physics without an issued command: {marker} subcycle {subcycle}")
            if issued[i, len(event["safety"]):].any():
                raise ValueError(f"Unrecorded proof for issued commands: {marker}")
            if not int(intervals[i].sum()) <= int(applied[i]) <= 2 * int(intervals[i].sum()):
                raise ValueError(f"Physical substep count disagrees with active intervals: {marker}")
            if intervals[i].any():
                observed = float(gaps[i, intervals[i]].min())
                smallest_observed_gap = observed if smallest_observed_gap is None else min(smallest_observed_gap, observed)
                largest_sampled_penetration = max(largest_sampled_penetration, float(depths[i].max()))
        divisions.append({"partition": part["partition"], "statuses": dict(partitions)})
    if retained_history != receipt["historical"]["diagnostic_five_frame_ticks"] or dict(classes) != receipt["statuses"]:
        raise ValueError("Computed diagnostic denominators disagree")
    write_json(args.out, {"scope": "DIAGNOSTIC_ONLY", "formal_coverage_credit": False,
              "run_receipt_sha256": file_sha(path), "geometry_unknown_ticks": geometry_unknown,
              "status_counts": dict(classes), "native_group_order": ["left arm", "right arm", "left hand", "right hand"],
              "native_status_per_4ms_subcycle": list(map(dict, counts_by_group)),
              "timing": {"first_active": first_active, "first_hold": first_hold, "first_short_physics": first_stop},
              "signed_geometry": {"minimum_sampled_gap_m": smallest_observed_gap,
                                  "maximum_sampled_penetration_m": largest_sampled_penetration},
              "postinterval_max_ten_tip_mm": {"all_observed": {"n": len(tip_errors), "p50_p95_p99": percentiles(tip_errors)},
                                             "active_only": {"n": len(active_tip_errors), "p50_p95_p99": percentiles(active_tip_errors)}},
              "velocity_masks_crossing_partition_local_segment": bad_velocity_boundaries,
              "safety_conservation": dict(safety_conservation),
              "proof_status": [{"decision": reason, "termination": termination, "count": count}
                               for (reason, termination), count in sorted(proof_causes.items())],
              "partitions": divisions})


if __name__ == "__main__":
    main()
