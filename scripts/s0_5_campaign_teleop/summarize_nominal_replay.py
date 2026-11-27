#!/usr/bin/env python3
"""Summarize observed native nominal replay without running ROS or scoring offline intent."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
from math import degrees, isfinite
from pathlib import Path


HISTORICAL = "historical_human_inputs"
WORKFLOW = "/whole_body_teleop/workflow_state"
PHASE = "/whole_body_teleop/control_phase"
ACTIVE = {WORKFLOW: ("state_name", "ACTIVE"), PHASE: ("phase", "TASK")}
ARM = "/tianji/joint_command"
COMMANDS = {ARM: 14, "/left_hand/command": 20, "/right_hand/command": 20}
FEEDBACK = {f"/{side}_{part}/{field}": n for side in ("left", "right")
            for part, field, n in (("arm", "joint_states", 7), ("hand", "state", 20))}
DIAGNOSTICS = {f"tianji/{side}_arm/teleop": side for side in ("left", "right")}
FIELDS = {
    "arm_command_accepted", "ik_target_accepted", "rate_limited", "solve_mode", "primary_reason",
    "servo_active_constraints", "servo_boundary_active", "servo_reason", "trajectory_ik_branch_guarded",
    "trajectory_reason", "tracker_invalid_count",
}
NS = 1_000_000_000


def quantiles(values):
    ordered = sorted(values)
    if not ordered:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "max": None}
    result = {"n": len(ordered), "max": ordered[-1]}
    for label, p in (("p50", .5), ("p95", .95), ("p99", .99)):
        index = p * (len(ordered) - 1)
        a = int(index)
        result[label] = ordered[a] + (index - a) * (ordered[min(a + 1, len(ordered) - 1)] - ordered[a])
    return result


def stamp(value):
    if not isinstance(value, dict):
        return None
    sec, nano = value.get("sec"), value.get("nanosec")
    if type(sec) is int and type(nano) is int and sec >= 0 and 0 <= nano < NS:
        return sec * NS + nano or None
    return None


def joint_values(msg, count):
    names = msg.get("joint_names", msg.get("name"))
    positions = msg.get("positions", msg.get("position"))
    if (not isinstance(names, list) or not isinstance(positions, list) or len(names) != count
            or len(positions) != count or not all(isinstance(name, str) and name for name in names)
            or len(set(names)) != count or not all(type(q) in (float, int) and isfinite(q) for q in positions)):
        return None
    return dict(zip(names, positions, strict=True))


def sequence_stats(records):
    groups = defaultdict(list)
    for _, _, segment, value, _ in records:
        sequence = value.get("sequence")
        if type(sequence) is int:
            groups[(segment, value.get("source", ""), value.get("session_id", ""))].append(sequence)
    output = []
    for (segment, source, session), numbers in sorted(groups.items()):
        deltas = [right - left for left, right in zip(numbers, numbers[1:])]
        output.append({"active_segment": segment, "source": source, "session_id": session, "count": len(numbers),
                       "first": numbers[0], "last": numbers[-1], "consecutive": deltas.count(1),
                       "duplicate": deltas.count(0), "backwards": sum(d < 0 for d in deltas),
                       "observed_missing_sequence_numbers": sum(d - 1 for d in deltas if d > 1)})
    return output


def joint_ranges(frames):
    ranges = {}
    for frame in frames:
        for name, value in frame.items():
            low, high = ranges.get(name, (value, value))
            ranges[name] = min(low, value), max(high, value)
    return {name: {"min_rad": low, "max_rad": high, "range_deg": degrees(high - low)}
            for name, (low, high) in sorted(ranges.items())}


def diagnostic_summary(fields, observed):
    rates = {}
    for name in ("arm_command_accepted", "ik_target_accepted", "rate_limited"):
        counts = Counter()
        for key, count in fields.get(name, {}).items():
            counts[str(key).lower()] += count
        usable = counts["true"] + counts["false"]
        rates[name] = {"true": counts["true"], "false": counts["false"],
                       "unobserved_or_non_boolean": observed - usable,
                       "true_fraction_of_boolean_diagnostics": counts["true"] / usable if usable else None}
    return {"samples": observed, "fields": fields, "rates": rates,
            "unobserved_requested_fields": sorted(FIELDS - fields.keys())}


def compare_feedback(commands, feedback):
    index = {}
    for key, commands_for_topic in commands.items():
        ordered = sorted(commands_for_topic, key=lambda row: (row[0], row[1]))
        index[key] = ([row[0] for row in ordered], ordered)
    outcomes, errors, ages = Counter(), defaultdict(list), defaultdict(list)
    for (segment, measured_stamp), (arrival, measured, group_names) in feedback:
        issued, age_by_topic = {}, {}
        for topic in COMMANDS:
            stamps, ordered = index.get((segment, topic), ((), ()))
            i = bisect_right(stamps, measured_stamp) - 1
            while i >= 0 and ordered[i][1] >= arrival:
                i -= 1
            if i < 0:
                outcomes["no_causal_issued_command"] += 1
                break
            issued_stamp, _, _, msg, q = ordered[i]
            deadline = stamp(msg.get("valid_until"))
            if deadline is not None and deadline < measured_stamp:
                outcomes["issued_command_expired"] += 1
                break
            if q is None or issued.keys() & q.keys():
                outcomes["invalid_or_overlapping_issued_joint_group"] += 1
                break
            issued.update(q)
            age_by_topic[topic] = (measured_stamp - issued_stamp) / 1e6
        else:
            if issued.keys() != measured.keys():
                outcomes["issued_feedback_joint_manifest_mismatch"] += 1
                continue
            outcomes["paired"] += 1
            for topic, age in age_by_topic.items():
                ages[topic].append(age)
            errors["all_54"].append(max(abs(measured[name] - issued[name]) for name in measured))
            for topic, names in group_names.items():
                errors[topic.split("/")[1]].append(max(abs(measured[name] - issued[name]) for name in names))
    return {"method": "latest issued header at or before feedback, received before all four feedback groups",
            "interpretation": "Issued-target comparison without actuator acknowledgment or lag fitting",
            "frame_outcomes": dict(outcomes), "max_abs_joint_error_rad": {k: quantiles(v) for k, v in errors.items()},
            "issued_stamps_age_ms": {k: quantiles(v) for k, v in ages.items()}}


def read_records(path, digest, flags):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as source:
        for line in source:
            digest.update(line)
            if not line.endswith(b"\n"):
                flags["partial_tail"] = True
                break
            yield json.loads(line)


def summarize(path):
    digest, flags = sha256(), {}
    first = last = None
    timebases = []
    stop = False
    previous_mono = None
    first_log = last_log = None
    log_backsteps = 0
    for row in read_records(path, digest, flags):
        if previous_mono is not None and row["mono_ns"] < previous_mono:
            raise ValueError("Trace monotonic receipt time went backwards")
        previous_mono = row["mono_ns"]
        if row["stage"] == HISTORICAL and row["kind"] == "input":
            scheduled = row.get("original_schedule_ns", row["original_log_ns"])
            if last is not None:
                if scheduled < last.get("original_schedule_ns", last["original_log_ns"]):
                    raise ValueError("Historical replay schedule went backwards")
                log_backsteps += row["original_log_ns"] < last["original_log_ns"]
            first_log = row["original_log_ns"] if first_log is None else min(first_log, row["original_log_ns"])
            last_log = row["original_log_ns"] if last_log is None else max(last_log, row["original_log_ns"])
            first = first or row
            last = row
        if row["stage"] == HISTORICAL and row["kind"] == "timebase":
            timebases.append(row)
        if row["kind"] == "service" and row.get("service") == "/whole_body_teleop/stop":
            response = row["response"]
            stop = stop or bool(response.get("success", response.get("accepted", False)))
    if first is None or len(timebases) != 1:
        raise ValueError("Expected one historical timebase and at least one replayed input")
    timebase = timebases[0]
    origin = first["original_log_ns"]
    schedule_origin = timebase.get("origin_schedule_ns", origin)
    end = origin + last.get("original_schedule_ns", last["original_log_ns"]) - schedule_origin
    if timebase["origin_log_ns"] != origin or first.get("original_schedule_ns", origin) != schedule_origin:
        raise ValueError("Historical input origin differs from the observed timebase")
    delta = timebase["epoch_delta_ns"]
    phases, phase_counts = {}, defaultdict(Counter)
    seen, diagnostics = Counter(), defaultdict(lambda: defaultdict(Counter))
    input_lag, active_lag = defaultdict(list), defaultdict(list)
    commands, feedback = defaultdict(list), defaultdict(dict)
    segment = 0
    segments_seen = set()
    verify_digest = sha256()
    for order, row in enumerate(read_records(path, verify_digest, {})):
        topic, value = row.get("topic"), row.get("value", {})
        header = stamp(value.get("header", {}).get("stamp"))
        active = all(phases.get(t, (None,))[0] == rule[1] for t, rule in ACTIVE.items())
        if topic in ACTIVE and row["kind"] == "output":
            key, _ = ACTIVE[topic]
            label = value.get(key, "UNKNOWN")
            phases[topic] = label, header
            phase_counts[topic][label] += 1
            if not active and all(phases.get(t, (None,))[0] == rule[1] for t, rule in ACTIVE.items()):
                segment += 1
        if (row["stage"] != HISTORICAL or not first["mono_ns"] <= row["mono_ns"] <= last["mono_ns"]
                or row["kind"] not in ("input", "output")):
            continue
        if row["kind"] == "input":
            scheduled = row.get("original_schedule_ns", row["original_log_ns"])
            due_mono_ns = timebase["origin_mono_ns"] + scheduled - schedule_origin
            input_lag[topic].append((row["mono_ns"] - due_mono_ns) / 1e6)
            message_time = row["log_est_ns"] + delta
        else:
            if header is None or not origin <= header - delta <= end:
                seen["output_outside_historical_header_or_missing_stamp"] += 1
                continue
            message_time = header
        if not active or any(s is None or s > message_time for _, s in phases.values()):
            seen["outside_observed_active_task"] += 1
            continue
        segments_seen.add(segment)
        if row["kind"] == "input":
            active_lag[topic].append(input_lag[topic][-1])
        elif topic in COMMANDS:
            commands[segment, topic].append((header, (row["mono_ns"], order), segment, value,
                                               joint_values(value, COMMANDS[topic])))
        elif topic in FEEDBACK:
            q = joint_values(value, FEEDBACK[topic])
            if "/state" in topic and (value.get("connected") is not True
                    or value.get("position_valid") != [True] * FEEDBACK[topic]):
                q = None
            group = feedback[segment, header]
            group[topic] = (None, None) if topic in group else ((row["mono_ns"], order), q)
        for status in value.get("status", []) if isinstance(value.get("status"), list) else ():
            side = DIAGNOSTICS.get(status.get("name")) if isinstance(status, dict) else None
            if side:
                seen[f"{side}_diagnostic_samples"] += 1
                for key, field_value in {pair["key"]: pair["value"] for pair in status.get("values", [])}.items():
                    if key in FIELDS or (("fallback" in key or "limit_reason" in key)
                                         and not key.endswith(("_xyz_m", "_rad"))):
                        diagnostics[side][key][field_value] += 1
    if digest.digest() != verify_digest.digest():
        raise ValueError("Replay trace was modified during analysis")

    coherent = []
    for key, groups in feedback.items():
        if groups.keys() != FEEDBACK.keys() or any(arrival is None or q is None for arrival, q in groups.values()):
            seen["incomplete_invalid_or_duplicate_feedback_stamp"] += 1
            continue
        q = {n: p for _, positions in groups.values() for n, p in positions.items()}
        if len(q) != 54:
            seen["overlapping_feedback_joint_names"] += 1
            continue
        names = {topic: positions.keys() for topic, (_, positions) in groups.items()}
        coherent.append((key, (min(a for a, _ in groups.values()), q, names)))
    by_topic = {topic: sorted((row for (_, name), group in commands.items() if topic == name for row in group),
                              key=lambda row: row[1]) for topic in COMMANDS}
    return {
        "scope": "OBSERVED_NATIVE_NOMINAL_REPLAY_DIAGNOSTIC",
        "trace_sha256": digest.hexdigest(), "partial_line_discarded": flags.get("partial_tail", False),
        "stop_ack_observed": stop,
        "phase_requirements": {key: value[1] for key, value in ACTIVE.items()},
        "observed_workflow_counts": phase_counts, "historical_log_bounds_ns": [first_log, last_log],
        "historical_log_backsteps_in_schedule_order": log_backsteps,
        "schedule_mapped_header_bounds_ns": [origin + delta, end + delta],
        "eligible_active_task_segment_count": len(segments_seen),
        "input_publish_lag_ms": {key: quantiles(values) for key, values in input_lag.items()},
        "active_task_input_publish_lag_ms": {key: quantiles(values) for key, values in active_lag.items()},
        "online_arm_diagnostics_sample_counts": {
            side: diagnostic_summary(diagnostics[side], seen[f"{side}_diagnostic_samples"])
            for side in ("left", "right")
        },
        "commands": {topic: {"observed": len(values), "valid_named_payloads": sum(row[-1] is not None for row in values),
                             "sequence_groups": sequence_stats(values),
                             "joint_ranges": joint_ranges(row[-1] for row in values if row[-1] is not None)}
                     for topic, values in by_topic.items()},
        "feedback": {"observed_stamp_count": len(feedback), "coherent_54d_stamps": len(coherent),
                     "joint_ranges": joint_ranges(q for _, (_, q, _) in coherent)},
        "feedback_vs_issued": compare_feedback(commands, coherent), "exclusion_counts": dict(seen),
        "offline_m1_intent_scored": False, "offline_swept_path_evaluated": False,
        "qualification": "NONE",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.trace)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
