#!/usr/bin/env python3
"""Compare one original and one segmented DCR-009 root-five START attempt."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_dcr009_guarded_diagnostic import LINE, read, rows
from replay_dcr009_guarded_diagnostic import LIVE_SHA, SIGNED_SHA
from summarize_dcr009_center_offsets import (
    METRICS, MIN_COMMON_BINS, PREPOSITION, ROOT, SCOPE, STEP_NS, digest, error_summary, errors, parse_attempt,
)


SIDES = ("left", "right")
GOAL_POSITION_TOLERANCE_M = .0001
GOAL_SO3_TOLERANCE_RAD = .001


def finite_number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def named_pair(value):
    return dict(zip(SIDES, map(finite_number, value))) if isinstance(value, list) and len(value) == 2 else dict.fromkeys(SIDES)


def endpoint(trace):
    origins = [row["mono_ns"] for row in trace if row["kind"] == "preposition_clock"]
    if not origins:
        origins = [row["mono_ns"] for row in trace if row["stage"] == PREPOSITION and row["kind"] == "input"]
    if not origins:
        return None, dict.fromkeys(SIDES)
    handoff = next((row for row in trace if row["kind"] == "start_handoff"), None)
    boundary = next((row["mono_ns"] for row in trace if row["stage"] == PREPOSITION and
                     (row["kind"] == "start_handoff" or (row["kind"] == "pilot_lifecycle"
                      and row.get("status") == "FAULT_STOP"))), trace[-1]["mono_ns"])
    measured = [row for row in trace if row["kind"] == "start_measurement" and row["stage"] == PREPOSITION
                and origins[0] <= row["mono_ns"] <= boundary and errors(row["value"].get("wrist_error_m_deg"))]
    if not measured:
        return None, dict.fromkeys(SIDES)
    current = measured[-1]
    value = current["value"]
    result = {
        "elapsed_s": (current["mono_ns"] - origins[0]) / 1e9,
        "observed_mono_ns": current["mono_ns"],
        "terminal_gap_ms": (boundary - current["mono_ns"]) / 1e6,
        "feedback_header_ns": value.get("feedback_header_ns"),
        "matches_start_handoff_receipt": handoff is not None and handoff.get("measurement") == value,
        "ready_sample": current.get("ready") is True,
        "measured_speed_rad_s": finite_number(value.get("measured_speed_rad_s")),
        "hand_error_to_issued_rad": named_pair(value.get("hand_error_to_issued_rad")),
        "hand_error_to_unsegmented_wuji_rad": named_pair(value.get("hand_error_to_unsegmented_wuji_rad")),
        "hand_segment_guard_proofs": named_pair(value.get("hand_segment_guard_proofs")),
    }
    for name, field in METRICS.items():
        measured_errors = errors(value.get(field))
        result[name] = ({side: {"position_mm": p, "so3_deg": angle} for side, (p, angle) in zip(SIDES, measured_errors)}
                        if measured_errors else None)
    last_hands = dict.fromkeys(SIDES)
    for row in measured:
        values = named_pair(row["value"].get("hand_error_to_unsegmented_wuji_rad"))
        proofs = named_pair(row["value"].get("hand_segment_guard_proofs"))
        for side in SIDES:
            if values[side] is not None and proofs[side] is not None and proofs[side] > 0:
                last_hands[side] = {"error_rad": values[side], "guard_proof": proofs[side],
                                    "elapsed_s": (row["mono_ns"] - origins[0]) / 1e9,
                                    "terminal_gap_ms": (boundary - row["mono_ns"]) / 1e6}
    return result, last_hands


def goal_difference(original, segmented, field):
    if field not in original or field not in segmented:
        return None
    before, after = (np.asarray(anchor[field], dtype=float) for anchor in (original, segmented))
    for value in (before, after):
        if (value.shape != (2, 4, 4) or not np.isfinite(value).all() or
                not np.allclose(value[:, 3, :], [0, 0, 0, 1], atol=1e-12, rtol=0)):
            raise ValueError(f"Expected two finite SE(3) target poses for {field}")
        axes = value[:, :3, :3]
        if not np.allclose(axes.swapaxes(1, 2) @ axes, np.eye(3), atol=1e-6, rtol=0) or not np.allclose(np.linalg.det(axes), 1, atol=1e-6, rtol=0):
            raise ValueError(f"Improper START target rotation for {field}")
    position = np.linalg.norm(after[:, :3, 3] - before[:, :3, 3], axis=1)
    angle = Rotation.from_matrix(after[:, :3, :3] @ before[:, :3, :3].swapaxes(1, 2)).magnitude()
    return {"exact_numeric_match": bool(np.array_equal(before, after)),
            "within_bound_target_tolerance": bool(np.all(position <= GOAL_POSITION_TOLERANCE_M) and np.all(angle <= GOAL_SO3_TOLERANCE_RAD)),
            "per_wrist": {side: {"position_mm": float(p * 1000), "so3_deg": math.degrees(a)}
                          for side, p, a in zip(SIDES, position, angle)}}


def endpoint_hand_joins(sim_path, trace, endpoint_record):
    wanted = endpoint_record["hand_segment_guard_proofs"] if endpoint_record else dict.fromkeys(SIDES)
    found = dict.fromkeys(SIDES)
    with sim_path.open(encoding="utf-8", errors="replace") as log:
        for line in log:
            match = LINE.search(line)
            if not match:
                continue
            side, _, _, sequence, guard, published, reason, _ = match.groups()
            if side in SIDES and wanted[side] == int(guard) and published == "1" and reason == "0":
                found[side] = int(sequence)
    return {side: (None if wanted[side] is None else found[side] is not None and any(
        row["kind"] == "output" and row.get("topic") == f"/{side}_hand/command" and
        row["value"].get("sequence") == found[side] and row["mono_ns"] <= endpoint_record["observed_mono_ns"]
        for row in trace)) for side in SIDES}


def inspect(path, profile):
    attempt, bins, end, input_records = parse_attempt(path, 1, 0)
    result = read(path / "result.json")
    permitted_profiles = {None, "original"} if profile == "original" else {"segmented"}
    if result.get("start_only_profile") not in permitted_profiles or attempt["recorded_history_inputs"]:
        raise ValueError(f"Unexpected profile or historical input in START-only comparison: {path}")
    trace = list(rows(path / "evidence/trace.jsonl"))
    model_rows = tuple(row["sha256"] for row in trace if row["kind"] == "model")
    if model_rows != (LIVE_SHA,):
        raise ValueError(f"Loaded MuJoCo model differs or is unverified in {path}")
    outcome = [row for row in trace if row["kind"] == "start_only_completed"]
    if outcome and (len(outcome) != 1 or outcome[0].get("original_history_started") is not False or outcome[0].get("profile") != profile):
        raise ValueError(f"Invalid START-only completion identity in {path}")
    if profile == "segmented" and bool(outcome) != (attempt["start_status"] == "START_REACHED"):
        raise ValueError(f"START-only scope receipt disagrees with measured START status in {path}")
    guard_path = path / "sweep-ready.json"
    guard = read(guard_path)
    if digest(guard_path) != result.get("guard_ready_sha256") or (guard["live_model_sha256"], guard["signed_model_sha256"]) != (LIVE_SHA, SIGNED_SHA):
        raise ValueError(f"Swept guard preflight identity differs in {path}")
    scrutiny = {key: value for key, value in guard.items() if key not in {"pid", "socket"}}
    last, last_hands = endpoint(trace)
    hand_joins = endpoint_hand_joins(path / "sim.log", trace, last) if profile == "segmented" else dict.fromkeys(SIDES)
    if last:
        last["hand_endpoint_proof_joins_observed_command"] = hand_joins
    attempt.update(profile=profile, measured_start_reached=attempt["start_status"] == "START_REACHED", endpoint=last,
                   last_preterminal_hand_error_to_unsegmented_wuji_rad=last_hands,
                   start_only_scope_completion=bool(outcome), live_model_sha256=model_rows[0],
                   identical_two_model_guard_contract_source_sha256=guard["guard_script_sha256"])
    if profile == "segmented" and attempt["measured_start_reached"]:
        valid_hands = last and all(last["hand_error_to_unsegmented_wuji_rad"][side] is not None and
                                  last["hand_error_to_unsegmented_wuji_rad"][side] <= .10 and
                                  hand_joins[side] is True for side in SIDES)
        if not valid_hands or not last["matches_start_handoff_receipt"] or not last["ready_sample"]:
            raise ValueError("Segmented START reached requires fresh original Wuji target and final wrist/hand dwell evidence")
    return attempt, bins, end, input_records, next((row for row in trace if row["kind"] == "start_anchor"), {}), scrutiny


def compare(original_path, segmented_path):
    original, original_bins, original_end, original_inputs, original_anchor, original_guard = inspect(original_path, "original")
    segmented, segmented_bins, segmented_end, segmented_inputs, segmented_anchor, segmented_guard = inspect(segmented_path, "segmented")
    cutoff = original_anchor.get("source_cutoff_ns")
    if (not original_inputs or original_inputs != segmented_inputs or cutoff != segmented_anchor.get("source_cutoff_ns")
            or original_anchor.get("selected_anchor_status") != segmented_anchor.get("selected_anchor_status")
            or any(anchor.get("center_offset_world_m", [0, 0, 0]) != [0, 0, 0] for anchor in (original_anchor, segmented_anchor))):
        raise ValueError("Both attempts must bind the same original START input records")
    guard_same = original_guard == segmented_guard
    goal_compare = {key: goal_difference(original_anchor, segmented_anchor, key)
                    for key in ("runtime_native_goal", "start_world_registered_wrist")}
    iso_match = all(goal and goal["within_bound_target_tolerance"] for goal in goal_compare.values())
    latest = min(original_end, segmented_end)
    common = [index for index in range(1, latest + 1) if index in original_bins and index in segmented_bins]
    deltas = {}
    for name in METRICS:
        matched = [index for index in common if original_bins[index][name] and segmented_bins[index][name]]
        for attempt, series in ((original, original_bins), (segmented, segmented_bins)):
            attempt["common_support_metrics"][name] = error_summary([series[index][name] for index in matched])
        if matched:
            before, after = (attempt["common_support_metrics"][name] for attempt in (original, segmented))
            deltas[name] = {"segmented_minus_original_worst_wrist_p95_mm": after["worst_wrist_position_p95_mm"] - before["worst_wrist_position_p95_mm"],
                            "segmented_minus_original_worst_wrist_p95_deg": after["worst_wrist_so3_p95_deg"] - before["worst_wrist_so3_p95_deg"]}
    matches_publish = all(not attempt["publication_mismatches"] and attempt["internal_home_proofs"] == [["PERMIT", "PERMIT"]]
                          and attempt["stop_ack_with_no_late_observed_command"] and attempt["owned_groups_stopped"]
                          and attempt["domain_empty_after"] for attempt in (original, segmented))
    return {"schema": "s05.teleop.dcr009_root_five_segmented_start.v1", "scope": SCOPE, "root": ROOT, "center_offset_mm": 0,
            "formal_coverage_credit": False, "stage_credit": False, "physical_qualification": False, "residual_enabled": False,
            "original_goal_equality": {"same_original_input_records": True, "source_cutoff_ns": cutoff,
                                       "within_tolerance": iso_match, **goal_compare,
                                       "position_tolerance_mm": GOAL_POSITION_TOLERANCE_M * 1000,
                                       "so3_tolerance_deg": math.degrees(GOAL_SO3_TOLERANCE_RAD)},
            "same_two_model_guard_configuration": guard_same,
            "publication_home_and_cleanup_evidence_complete": matches_publish,
            "equal_time_support": {"bin_ms": STEP_NS / 1e6, "common_terminal_cutoff_ms": latest * STEP_NS / 1e6,
                                   "common_bin_end_ms": [index * STEP_NS / 1e6 for index in common], "eligible_bins": len(common),
                                   "rule": "Last distinct observed sample within each shared right-closed bin before either terminal event; no cross-bin fill"},
            "comparison_status": "DIAGNOSTIC_PAIRED" if len(common) >= MIN_COMMON_BINS and guard_same and iso_match and matches_publish else "INCONCLUSIVE",
            "paired_p95_deltas": deltas, "original": original, "segmented": segmented}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("original", "segmented", "out"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.original, args.segmented)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(result["comparison_status"])


if __name__ == "__main__":
    main()
