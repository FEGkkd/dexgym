#!/usr/bin/env python3
"""Export status and geometric witnesses from two immutable diagnostic lanes."""

import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import json
from pathlib import Path

from dexgym.tasks.direct.wholebody_s05.v1.rmb.schema import file_sha, write_json


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as ledger:
        yield from map(json.loads, ledger)


@contextmanager
def compress(path):
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as ledger:
            yield lambda row: ledger.write((json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n").encode())


def compact_step(step):
    base = {k: step[k] for k in ("cutoff_ns", "status", "cause", "common_proof", "current_extra_v2", "hold_proof",
                                 "event", "minimum_gap_m", "depth_m", "contact_pairs", "feedback_available_ns")
            if k in step}
    if "common_detail" in step:
        for field in ("common_detail", "extra_detail", "hold_detail"):
            detail = step.get(field)
            if detail:
                base[field] = {k: detail[k] for k in ("termination", "witness_pair_index",
                                                      "unresolved_pair_indices", "endpoint_gap_m") if k in detail}
    if "sources" in step:
        base["sources"] = [None if s is None else [s["source_index"], s["receive_ns"]] for s in step["sources"]]
    if "qd_measured" in step:
        base["max_abs_hand_qd_rad_s"] = max(map(abs, step["qd_measured"][14:]))
        base["moved"] = step["pre_issue_signed_snapshot"][0] != step["q_issued"]
    return base


def compact_physics(path, output, report):
    count = Counter()
    with compress(output) as write:
        for row in read(path):
            count["rows"] += 1
            record = {name: row[name] for name in ("partition", "partition_tick", "tick_ns", "input_source_index",
                                                    "control_cutoff_ns", "window")}
            record["experiments"] = {}
            record["row_status"] = {}
            record["strict_sampled_feedback_ns"] = {}
            for mode, steps in row["experiments"].items():
                chosen = row["rmb_feedback_at_exact_signed_tick"][mode]
                record["strict_sampled_feedback_ns"][mode] = chosen["feedback_available_ns"] if chosen else None
                feedback = sum("q_measured" in step for step in steps)
                status = "COMPLETE" if feedback == 10 else "PARTIAL" if feedback else "NOT_EXECUTED"
                record["experiments"][mode] = [compact_step(s) for s in steps]
                record["row_status"][mode] = status
                count[(mode, status)] += 1
            write(record)
    if count["rows"] != 4804:
        raise ValueError("Compact result changed the frozen denominator")
    for mode, expected in report["experiment_status"].items():
        if any(count[(mode, key)] != n for key, n in expected.items()):
            raise ValueError(f"Compact result changed {mode} physical status")


def compact_functional(path, output):
    counter = Counter()
    with compress(output) as write:
        for row in read(path):
            counter["root_rows"] += 1
            record = {k: row[k] for k in ("partition", "partition_tick", "tick_ns", "start_window_row",
                                          "source_index", "legacy_raw_near_pinch_cohort", "source_proxy_windows",
                                          "raw_manus_mask", "frozen_target_mask", "candidate_mask")}
            human = row.get("HUMAN_FUNCTIONAL_INTENT") or {}
            old = row.get("FROZEN_ROBOT_MORPHOLOGY") or {}
            new = row.get("ROBOT_FUNCTIONAL_CANDIDATE") or {}
            witness = row.get("KINEMATIC_WITNESS") or {}
            record["functional_intent_defined"] = sorted(human)
            record["witness_status"] = witness.get("status")
            counter["right_near_pinch"] += bool(record["legacy_raw_near_pinch_cohort"])
            if record["legacy_raw_near_pinch_cohort"] or record["source_proxy_windows"]:
                record["raw_functional_aperture_m"] = {side: value["raw_tip_pair_aperture_m"]
                                                       for side, value in human.items()}
                record["frozen_scores"] = old.get("scores_by_side")
                record["candidate_scores"] = new.get("scores_by_side")
                record["witness_scores"] = witness.get("scores_by_side")
                record["attempts"] = witness.get("attempts")
                record["witness_meets_near_pinch_tolerance"] = witness.get("near_pinch_score_tolerance_met")
            write(record)
    if counter != {"root_rows": 4844, "right_near_pinch": 187}:
        raise ValueError(f"Compact ledger changed signed root or near-pinch count: {counter}")


def main(args):
    physics = json.loads((args.physics / "fixed-arm-native-summary.json").read_bytes())
    functional = json.loads((args.functional / "summary.json").read_bytes())
    actual = args.physics / "fixed-arm-native.jsonl.gz"
    geometry = args.functional / "rows.jsonl.gz"
    if file_sha(actual) != physics["sha256"]["ledger"] or file_sha(geometry) != functional["sha256"]["ledger"]:
        raise ValueError("Runtime evidence differs from its receipt")
    args.out.mkdir()
    status = args.out / "fixed-arm-status.jsonl.gz"
    witness = args.out / "functional-witnesses.jsonl.gz"
    compact_physics(actual, status, physics)
    compact_functional(geometry, witness)
    write_json(args.out / "compact.json", {"scope": "DIAGNOSTIC_ONLY", "sha256": {
        "physical_ledger": file_sha(actual), "functional_ledger": file_sha(geometry),
        "fixed_arm_status": file_sha(status), "functional_witnesses": file_sha(witness)}})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("physics", "functional", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    main(p.parse_args())
