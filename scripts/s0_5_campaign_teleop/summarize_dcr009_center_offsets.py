#!/usr/bin/env python3
"""Compare the five guarded DCR-009 root-05 center-offset diagnostics offline."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean

from audit_dcr009_guarded_diagnostic import GROUPS, LINE, TOPICS, read, rows
from replay_dcr009_guarded_diagnostic import CENTER_OFFSET_SEQUENCE_MM as OFFSETS, ROOT_FIVE as ROOT, ROOT_FIVE_PLAN_SHA, SCOPE
from run_start_handoff_cohort import digest


PREPOSITION = "preposition_start"
HISTORY = "historical_human_inputs"
SAFETY_TOPIC = "/tianji/safety"
STEP_NS = 20_000_000
MIN_COMMON_BINS = 5
JOINT_SLICES = {"arm": slice(0, 14), "left": slice(14, 34), "right": slice(34, 54)}
JOINT_NAMES = {"arm": [f"{side}_j{index}" for side in ("left", "right") for index in range(1, 8)]}
JOINT_NAMES.update({side: [f"{side[0]}_{finger}J{index}" for finger in ("TH", "FF", "MF", "RF", "LF") for index in range(4)]
                    for side in ("left", "right")})
METRICS = {
    "engineering_tcp": "wrist_error_m_deg",
    "registered_wrist_marker": "registered_wrist_error_m_deg",
    "unshifted_engineering_tcp": "wrist_original_error_m_deg",
}
GUARD_VALID = {"PERMIT", "LIMIT_OR_COUPLING", "COLLISION", "SWEEP_UNPROVEN", "TIME_BUDGET"}


def errors(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        if any(type(x) not in (float, int) or not math.isfinite(x) or x < 0 for x in pair) or pair[1] > 180.0001:
            return None
    return tuple((p * 1000, angle) for p, angle in value)


def p95(numbers):
    values = sorted(numbers)
    index = (len(values) - 1) * .95
    left, right = math.floor(index), math.ceil(index)
    return values[left] + (index - left) * (values[right] - values[left])


def error_summary(samples):
    if not samples:
        return None
    positions = [[value[side][0] for value in samples] for side in range(2)]
    rotations = [[value[side][1] for value in samples] for side in range(2)]
    worst_position = [max(pair) for pair in zip(*positions)]
    worst_rotation = [max(pair) for pair in zip(*rotations)]
    return {
        "samples": len(samples),
        "left_position_mm": {"mean": mean(positions[0]), "p95": p95(positions[0]), "min": min(positions[0])},
        "right_position_mm": {"mean": mean(positions[1]), "p95": p95(positions[1]), "min": min(positions[1])},
        "left_so3_deg": {"mean": mean(rotations[0]), "p95": p95(rotations[0]), "min": min(rotations[0])},
        "right_so3_deg": {"mean": mean(rotations[1]), "p95": p95(rotations[1]), "min": min(rotations[1])},
        "worst_wrist_position_p95_mm": p95(worst_position),
        "worst_wrist_so3_p95_deg": p95(worst_rotation),
        "bilateral_geometric_start_scale_hits": sum(p <= 10 and a <= 5 for p, a in zip(worst_position, worst_rotation)),
        "bilateral_geometric_near_misses": sum(p <= 20 and a <= 10 and not (p <= 10 and a <= 5)
                                               for p, a in zip(worst_position, worst_rotation)),
    }


def feedback_rows(trace, origin, terminal):
    bins = {}
    all_valid = []
    seen = set()
    for row in trace:
        if row["stage"] != PREPOSITION or row["kind"] != "start_measurement" or not origin <= row["mono_ns"] <= terminal:
            continue
        value = row["value"]
        header = value.get("feedback_header_ns")
        metrics = {name: errors(value.get(field)) for name, field in METRICS.items()}
        if type(header) is not int or header <= 0 or header in seen or not value.get("active_task") or not metrics["engineering_tcp"]:
            continue
        seen.add(header)
        all_valid.append(metrics)
        elapsed = row["mono_ns"] - origin
        if elapsed > 0:
            bins[(elapsed + STEP_NS - 1) // STEP_NS] = metrics
    return bins, all_valid


def sweep_audit(path):
    proofs = {}
    reasons = {group: Counter() for group in GROUPS}
    unattributed = Counter()
    for row in rows(path):
        index, reason = row["group"], row["reason"]
        if type(index) is not int or not 0 <= index < len(GROUPS):
            unattributed[reason] += 1
            continue
        group = GROUPS[index]
        reasons[group][reason] += 1
        if reason == "PERMIT":
            states = row["issued_measured_proposed_rad"]
            if row["sequence"] in proofs or not isinstance(states, list) or len(states) != 3 or len(states[2]) != 54:
                raise ValueError(f"Ambiguous or incomplete permit in {path}")
            proofs[row["sequence"]] = (group, states[2][JOINT_SLICES[group]])
    return proofs, reasons, unattributed


def publication_audit(sim_log, trace, session, proofs, reasons, unattributed):
    origin = {group: {} for group in GROUPS}
    rejected = {group: Counter() for group in GROUPS}
    seen_proofs = set()
    mismatch = Counter()
    with sim_log.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            record = LINE.search(line)
            if not record:
                continue
            group, owner, _, command, proof, published, reason, _ = record.groups()
            command, proof = int(command), int(proof)
            if published == "0":
                rejected[group][reason] += 1
                continue
            payload = proofs.get(proof)
            if (owner != session or reason != "0" or proof in seen_proofs or command in origin[group]
                    or payload is None or payload[0] != group):
                mismatch["native_publication_without_unique_group_permit"] += 1
                continue
            seen_proofs.add(proof)
            origin[group][command] = payload[1]
    observed = {group: Counter() for group in GROUPS}
    for row in trace:
        if row["kind"] != "output" or row.get("topic") not in TOPICS:
            continue
        group, message = TOPICS[row["topic"]], row["value"]
        command = message.get("sequence")
        observed[group][command] += 1
        if observed[group][command] > 1:
            mismatch["duplicate_ros_observation"] += 1
        q = message.get("positions" if group == "arm" else "position")
        if (type(command) is not int or command not in origin[group] or q != origin[group].get(command)
                or message.get("joint_names") != JOINT_NAMES[group]
                or (group == "arm" and message.get("session_id") != session)
                or (group != "arm" and message.get("side") != group)):
            mismatch["observed_topic_without_matching_session_group_and_payload"] += 1
    groups = {}
    for group in GROUPS:
        valid = sum(count for key, count in reasons[group].items() if key in GUARD_VALID)
        groups[group] = {
            "guard_service_reasons": dict(reasons[group]),
            "native_rejection_codes": dict(rejected[group]),
            "native_publications_with_permit": len(origin[group]),
            "observed_ros_messages": sum(observed[group].values()),
            "native_publications_not_observed": len(origin[group].keys() - observed[group].keys()),
            "valid_guard_requests": valid,
            "guard_permits": reasons[group]["PERMIT"],
            "permit_rate_over_valid_guard_requests": reasons[group]["PERMIT"] / valid if valid else None,
        }
    return {"groups": groups, "unattributed_guard_reasons": dict(unattributed), "publication_mismatches": dict(mismatch)}


def parse_attempt(directory, number, offset):
    result_path = directory / "result.json"
    result = read(result_path)
    if (result.get("root") != ROOT or type(result.get("center_offset_mm")) is not int
            or result["center_offset_mm"] != offset or result.get("scope") != SCOPE
            or result.get("formal_coverage_credit") is not False or result.get("attempt", number) != number):
        raise ValueError(f"Wrong diagnostic root, offset, scope or attempt ordinal in {result_path}")
    trace_path = directory / "evidence/trace.jsonl"
    sim_path = directory / "sim.log"
    sweep_path = directory / "sweep-trace.jsonl"
    trace_sha = digest(trace_path)
    if result.get("trace_sha256", trace_sha) != trace_sha:
        raise ValueError(f"Trace differs from result identity in {result_path}")
    input_path = directory / "evidence/input.json"
    if input_path.is_file():
        receipt = read(input_path)
        if (receipt.get("center_offset_mm") != offset or receipt.get("scope") != SCOPE
                or receipt.get("formal_coverage_credit") is not False or receipt.get("residual_enabled") is not False):
            raise ValueError(f"Wrong diagnostic provenance or residual enabled in {input_path}")
        if (receipt.get("root", {"root": ROOT}).get("root") != ROOT
                or receipt.get("plan_sha256", ROOT_FIVE_PLAN_SHA) != ROOT_FIVE_PLAN_SHA):
            raise ValueError(f"Source or original frozen roster differs in {input_path}")
    trace = list(rows(trace_path))
    if any(b["mono_ns"] < a["mono_ns"] for a, b in zip(trace, trace[1:])):
        raise ValueError(f"Observer trace clock regressed in {trace_path}")
    anchors = [row for row in trace if row["kind"] == "start_anchor"]
    if len(anchors) > 1:
        raise ValueError(f"More than one START anchor in {trace_path}")
    anchor = anchors[0] if anchors else {}
    if anchor.get("center_offset_mm", offset) != offset:
        raise ValueError(f"START anchor offset differs in {trace_path}")
    start = [row["mono_ns"] for row in trace if row["stage"] == PREPOSITION and row["kind"] == "input"]
    clocks = [row["mono_ns"] for row in trace if row["kind"] == "preposition_clock"]
    if len(clocks) > 1 or (clocks and start and clocks[0] > start[0]):
        raise ValueError(f"Preposition clock is repeated or follows first input in {trace_path}")
    faults = [row for row in trace if row["kind"] == "pilot_lifecycle" and row.get("status") == "FAULT_STOP"]
    rejected = [row for row in trace if row["kind"] == "pilot_lifecycle" and row.get("status") == "ABORTED"]
    handoffs = [row for row in trace if row["kind"] == "start_handoff"]
    stops = [row for row in trace if row["kind"] == "pilot_stop"]
    home_proofs = [row["statuses"] for row in trace if row["kind"] == "internal_home_guard"]
    if len(stops) > 1 or len(handoffs) > 1:
        raise ValueError(f"Ambiguous START or STOP outcome in {trace_path}")
    origin = next(iter(clocks or start), None)
    preposition = [row["mono_ns"] for row in trace if row["stage"] == PREPOSITION]
    terminal = min((row["mono_ns"] for row in faults + handoffs if row["stage"] == PREPOSITION), default=max(preposition, default=0))
    bins, measured = feedback_rows(trace, origin, terminal) if origin is not None and terminal >= origin else ({}, [])
    proofs, reasons, unattributed = sweep_audit(sweep_path)
    audit = publication_audit(sim_path, trace, anchor.get("session"), proofs, reasons, unattributed)
    fault_outcomes = [{"elapsed_s": (row["mono_ns"] - origin) / 1e9 if origin is not None else None,
                       "stage": row["stage"], **row.get("failure", {})} for row in faults]
    safety = Counter(row["value"].get("code", "UNKNOWN") for row in trace
                     if row["kind"] == "output" and row.get("topic") == SAFETY_TOPIC and row["value"].get("safe") is False)
    summary = {
        "attempt": number, "center_offset_mm": offset, "phase": result.get("phase"),
        "process_exit": result.get("replay_exit"),
        "start_status": handoffs[0]["status"] if handoffs else "NOT_OBSERVED",
        "preposition_origin": "preposition_clock" if clocks else "first_preposition_input" if start else None,
        "preposition_duration_s": (terminal - origin) / 1e9 if origin is not None and terminal >= origin else None,
        "stop_ack_with_no_late_observed_command": bool(stops) and stops[0]["late_observed_command_count"] == 0,
        "domain_empty_after": result.get("same_domain_empty_after") is True,
        "owned_groups_stopped": result.get("owned_groups_stopped") is True,
        "internal_home_proofs": home_proofs,
        "recorded_history_inputs": sum(row["kind"] == "input" and row["stage"] == HISTORY for row in trace),
        "preposition_input_packets": len(start),
        "fault_stops": fault_outcomes,
        "other_abort_reports": [{"stage": row["stage"], "error": row.get("error")} for row in rejected],
        "unsafe_safety_topic_messages_by_code": dict(safety),
        "native_joint_limit_watchdog_events": sum(row.get("code") == "JOINT_LIMIT" for row in fault_outcomes),
        "all_supported_measurements_descriptive_only": {
            name: error_summary([metric[name] for metric in measured if metric[name]]) for name in METRICS},
        "common_support_metrics": {}, **audit,
        "input_identity": {"result_sha256": digest(result_path), "native_trace_sha256": trace_sha,
                           "guard_trace_sha256": digest(sweep_path), "sim_log_sha256": digest(sim_path),
                           "input_receipt_sha256": digest(input_path) if input_path.is_file() else None},
    }
    return summary, bins, max(0, (terminal - origin) // STEP_NS) if origin is not None else 0, anchor.get("inputs")


def risk_key(attempt):
    faults = attempt["fault_stops"]
    expected = {
        ("/tianji/robot_state", "DRIVER_NOT_COMMANDABLE", "position held; home required before next enable"),
        (SAFETY_TOPIC, "JOINT_LIMIT", "both arms held: joint command violates uniform joint limit margin"),
    }
    other_faults = sum((row.get("source"), row.get("code"), row.get("reason")) not in expected for row in faults)
    severe_guard = sum(row["guard_service_reasons"].get(reason, 0)
                       for row in attempt["groups"].values() for reason in ("COLLISION", "LIMIT_OR_COUPLING"))
    return (other_faults, attempt["native_joint_limit_watchdog_events"], severe_guard)


def summarize(run):
    ledger_path = run / "run_receipt.json"
    if ledger_path.is_file():
        ledger = read(ledger_path)
        if (ledger.get("scope") != SCOPE or ledger.get("formal_coverage_credit") is not False
                or ledger.get("plan_sha256") != ROOT_FIVE_PLAN_SHA
                or ledger.get("comparison") != "ROOT_FIVE_COMMON_CENTER_OFFSETS"
                or ledger.get("preregistered_center_offsets_mm") != list(OFFSETS)
                or ledger.get("preregistered_roots") != [ROOT] * len(OFFSETS)):
            raise ValueError(f"Wrong five-offset diagnostic roster in {ledger_path}")
    inspected = [parse_attempt(run / f"attempt-{index:02d}", index, offset) for index, offset in enumerate(OFFSETS, 1)]
    attempts, bins, ends, anchors = zip(*inspected)
    present_anchors = [anchor for anchor in anchors if anchor is not None]
    if present_anchors and any(anchor != present_anchors[0] for anchor in present_anchors[1:]):
        raise ValueError("Five offsets must use the same original reconstructed START input records")
    latest = min(ends)
    common = [index for index in range(1, latest + 1) if all(index in series for series in bins)]
    for attempt, series in zip(attempts, bins):
        for name in METRICS:
            values = [series[index][name] for index in common if all(other[index][name] for other in bins)]
            attempt["common_support_metrics"][name] = error_summary(values)
        attempt["risk_key"] = risk_key(attempt)
    baseline = attempts[0]
    for attempt in attempts:
        metrics = attempt["common_support_metrics"]["engineering_tcp"]
        baseline_metrics = baseline["common_support_metrics"]["engineering_tcp"]
        attempt["versus_zero_worst_wrist_p95_delta"] = {
            "position_mm": metrics["worst_wrist_position_p95_mm"] - baseline_metrics["worst_wrist_position_p95_mm"],
            "so3_deg": metrics["worst_wrist_so3_p95_deg"] - baseline_metrics["worst_wrist_so3_p95_deg"],
        } if metrics and baseline_metrics else None
    eligible = [attempt for attempt in attempts if not attempt["publication_mismatches"] and attempt["internal_home_proofs"] == [["PERMIT", "PERMIT"]]
                and attempt["stop_ack_with_no_late_observed_command"] and attempt["domain_empty_after"] and attempt["owned_groups_stopped"]
                and attempt["groups"]["arm"]["observed_ros_messages"] > 0
                and (attempt["start_status"] in {"START_REACHED", "START_NOT_REACHED"} or attempt["fault_stops"])
                and (not attempt["recorded_history_inputs"] or attempt["start_status"] == "START_REACHED")]
    secondary = ("unshifted_engineering_tcp" if all((attempt["common_support_metrics"]["unshifted_engineering_tcp"] or {}).get("samples", 0) >= MIN_COMMON_BINS
                                                    for attempt in attempts)
                 else "registered_wrist_marker")

    def key(attempt):
        primary = attempt["common_support_metrics"]["engineering_tcp"]
        extra = attempt["common_support_metrics"][secondary]
        offset = attempt["center_offset_mm"]
        return (*attempt["risk_key"], attempt["start_status"] != "START_REACHED",
                primary["worst_wrist_position_p95_mm"], primary["worst_wrist_so3_p95_deg"],
                extra["worst_wrist_position_p95_mm"] if extra else math.inf,
                extra["worst_wrist_so3_p95_deg"] if extra else math.inf,
                -attempt["preposition_duration_s"], abs(offset), offset)

    ranked = sorted(eligible, key=key) if len(common) >= MIN_COMMON_BINS and baseline in eligible else []
    best = ranked[0] if ranked else None
    return {"schema": "s05.teleop.dcr009_center_offset_comparison.v1", "scope": SCOPE, "root": ROOT,
            "formal_coverage_credit": False, "stage_credit": False, "physical_qualification": False,
            "frozen_source_plan_sha256": ROOT_FIVE_PLAN_SHA,
            "run_receipt_sha256": digest(ledger_path) if ledger_path.is_file() else None,
            "residual_enabled": False, "offset_axis_convention": "common world X; positive toward left base, negative toward right base",
            "baseline_offset_mm": 0, "attempts": attempts,
            "equal_time_support": {"bin_ms": STEP_NS / 1e6, "common_terminal_cutoff_ms": latest * STEP_NS / 1e6,
                                   "common_bin_end_ms": [index * STEP_NS / 1e6 for index in common], "eligible_bins": len(common),
                                   "minimum_bins_for_diagnostic_ranking": MIN_COMMON_BINS,
                                   "rule": "Use last distinct causal measurement inside each right-closed bin for all five; no cross-bin fill or post-stop sample"},
            "selection": {"status": "BEST_DIAGNOSTIC_ONLY" if best else "INCONCLUSIVE",
                          "best_center_offset_mm": best["center_offset_mm"] if best else None,
                          "ranking_offsets_mm": [attempt["center_offset_mm"] for attempt in ranked],
                          "tie_break_secondary_target": secondary,
                          "rule": "Require a joined real arm publication, internal Home proofs, a terminal START/fault event and clean STOP; then rank fewer other faults, fewer native JOINT_LIMIT stops, fewer guard collision/limit rejections, strict START if observed, lower worst-wrist position and SO3 P95, secondary target P95, longer survival, smaller absolute offset, signed offset",
                          "any_strict_start_observed": any(attempt["start_status"] == "START_REACHED" for attempt in attempts)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.run)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(summary["selection"])


if __name__ == "__main__":
    main()
