#!/usr/bin/env python3
"""Audit the single root-five START→history diagnostic from sealed offline evidence."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from hmb_cohort import INPUT_ROLES
from summarize_nominal_replay import ACTIVE, COMMANDS, DIAGNOSTICS, FEEDBACK, HISTORICAL, joint_values, quantiles, stamp


ROOT = "capture:b63bc72d2e8b2ee395e966f6"
PLAN_SHA = "3c0268291bdb8916a9a520d3fa73a1340fffd7f9e8a7b7364a1ef9950f9a0004"
COHORT_SHA = "7d6c4702229aefd0d564f76dfae5a2a7b62a0340b9c4cae43fea512cc2d531c2"
LIVE_SHA = "be69f387f5bc25c8011a992a88ac2c2449c2c0419e09318b740fcdcd1402871d"
MODEL_RECEIPT_SHA = "1d67a6dad1831574af10a1fdf6349638097af6457e9d8527c9bd77e150e876fe"
FRESH_NS = 40_000_000
HUMAN = {topic for topic, _, _ in INPUT_ROLES.values()}
SAFETY = "/tianji/safety"
FEEDBACK_TOPICS = frozenset(FEEDBACK)
TRACE_MARKERS = {"model", "start_anchor", "start_handoff", "timebase", "historical_window_intent",
                 "historical_input_seal", "historical_abort", "history_halt", "pilot_lifecycle",
                 "pilot_stop", "service", "guard_stop"}
DIAG_FIELDS = {"session", "mapping_policy", "arm_command_accepted", "pilot_arm_status", "primary_reason",
               "pilot_guarded_publish", "pilot_guard_sequence", "pilot_generation", "sequence",
               "tracker_source_stamp_sec", "target_input_raw_available", "target_input_raw_xyz_m",
               "target_input_raw_quat_xyzw", "solve_mode", "rate_limited"}


def read_rows(path):
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                raise ValueError(f"Unterminated evidence line {number}: {path}")
            yield number, json.loads(line)


def plans(path):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != PLAN_SHA:
        raise ValueError("Expected the original locked root-five plan")
    plan = json.loads(raw)
    selected, = [row for row in plan["selected_roots"] if row["root"] == ROOT and row["split"] == "Train"]
    return selected


def read_trace(path):
    markers, packets, feedback, stage_counts = defaultdict(list), [], defaultdict(dict), Counter()
    changes, phases, health, command_counts = [], {}, [], Counter()
    latest_diag, last = {}, -1
    for lineno, row in read_rows(path):
        time_ns, kind, stage = row["mono_ns"], row["kind"], row["stage"]
        if type(time_ns) is not int or time_ns < last:
            raise ValueError(f"Noncausal observer record at line {lineno}")
        last = time_ns
        if kind in TRACE_MARKERS:
            markers[kind].append(row)
        if kind == "input":
            stage_counts[stage] += 1
            if stage == HISTORICAL:
                packets.append(row)
        if kind != "output":
            continue
        topic, value = row["topic"], row["value"]
        header = stamp(value.get("header", {}).get("stamp"))
        if topic in ACTIVE:
            field, expected = ACTIVE[topic]
            phases[topic] = (value.get("session_id"), value.get(field) == expected, header)
            left, right = (phases.get(name) for name in ACTIVE)
            active = (left[0] if left and right and left[0] and left[:2] == right[:2] and left[1]
                      and left[2] is not None and right[2] is not None else None)
            if not changes or changes[-1][1] != active:
                opened = max(left[2], right[2]) if active else None
                changes.append((time_ns, active, opened))
        if stage != HISTORICAL:
            continue
        if topic in COMMANDS:
            command_counts[topic] += 1
        if topic == SAFETY and value.get("safe") is False:
            health.append({"mono_ns": time_ns, "header_ns": header, "topic": topic,
                           "code": value.get("code"), "reason": value.get("reason")})
        if topic == "/tianji/robot_state" and value.get("command_allowed") is False:
            health.append({"mono_ns": time_ns, "header_ns": header, "topic": topic,
                           "code": "DRIVER_NOT_COMMANDABLE", "reason": value.get("reason")})
        if header is None:
            continue
        if topic in FEEDBACK:
            named = joint_values(value, FEEDBACK[topic])
            if topic.endswith("/state") and (value.get("connected") is not True
                    or value.get("online") != [True] * 20 or value.get("position_valid") != [True] * 20
                    or value.get("stale") != [False] * 20):
                named = None
            group = feedback[header]
            group[topic] = (time_ns, named) if topic not in group else (time_ns, None)
        if topic == "/tianji/teleop_diagnostics":
            for item in value.get("status", ()):
                side = DIAGNOSTICS.get(item["name"])
                if side:
                    latest_diag[side] = {"mono_ns": time_ns, "ros_header_ns": header, "fields": {
                        kv["key"]: kv["value"] for kv in item["values"] if kv["key"] in DIAG_FIELDS}}
    return {"markers": markers, "packets": packets, "feedback": feedback, "changes": changes,
            "health": health, "latest_arm_diagnostics": latest_diag, "last_trace_mono_ns": last,
            "input_counts_all_stages": dict(stage_counts), "history_observed_command_counts": dict(command_counts)}


def singleton(markers, name):
    values = markers.get(name, [])
    if len(values) > 1:
        raise ValueError(f"Multiple {name} markers in a single cold start")
    return values[0] if values else None


def expected_packets(cohort, window):
    from nominal_live_replay import HUMAN as NATIVE_TOPICS, deserialize_message, signed_events
    from rosidl_runtime_py.utilities import get_message

    if sha256(cohort.read_bytes()).hexdigest() != COHORT_SHA:
        raise ValueError("The independently read capture cohort SHA differs")
    native, receipt = signed_events(cohort, ROOT, None)
    low = window["source_cutoff_start_ns"]
    high = window["source_cutoff_stop_exclusive_ns"]
    types = {topic: get_message(name) for topic, name in NATIVE_TOPICS.items()}
    packets = []
    for event in native:
        if not low <= event.schedule_ns < high:
            continue
        raw = deserialize_message(event.data, types[event.topic])
        source = raw.header.stamp
        packets.append({"topic": event.topic, "record_index": event.record_index, "original_log_ns": event.log_ns,
                        "original_schedule_ns": event.schedule_ns, "original_stamp_ns": source.sec * 10**9 + source.nanosec,
                        "cdr_sha256": sha256(event.data).hexdigest()})
    if set(row["topic"] for row in packets) != HUMAN:
        raise ValueError("Original history window lacks the frozen four source streams")
    return packets, receipt["root"]


def packet_audit(expected, observed, timebase, seal, window):
    first_mismatch, matched, verified_hashes = None, 0, 0
    lag = []
    by_topic_expected = Counter(row["topic"] for row in expected)
    by_topic_observed = Counter(row["topic"] for row in observed)
    if observed and (not timebase or timebase["origin_schedule_ns"] != window["source_cutoff_start_ns"]):
        raise ValueError("Recorded history did not bind exactly to the frozen source cutoff")
    keys = ("topic", "record_index", "original_log_ns", "original_schedule_ns", "original_stamp_ns")
    for index, (wanted, actual) in enumerate(zip(expected, observed)):
        if timebase is None:
            raise ValueError("Historical publication lacks timebase evidence")
        original = wanted["original_stamp_ns"]
        before = {key: wanted[key] for key in keys}
        after = {key: actual.get(key) for key in keys}
        epoch_ok = actual.get("published_stamp_ns") == (original + timebase["epoch_delta_ns"] if original else 0)
        recorded_hash = actual.get("cdr_sha256", actual.get("original_cdr_sha256"))
        cdr_ok = recorded_hash is None or recorded_hash == wanted["cdr_sha256"]
        if before != after or not epoch_ok or not cdr_ok:
            first_mismatch = {"ordinal": index, "expected": before, "observed": after,
                              "ros_epoch_match": epoch_ok, "cdr_hash_match": cdr_ok}
            break
        matched += 1
        verified_hashes += int(recorded_hash is not None)
        due = timebase["origin_mono_ns"] + wanted["original_schedule_ns"] - timebase["origin_schedule_ns"]
        lag.append((actual["mono_ns"] - due) / 1e6)
    if matched < len(observed) and first_mismatch is None:
        first_mismatch = {"ordinal": matched, "reason": "EXTRA_OBSERVED_PACKET"}
    ordered_sha = sha256(b"".join(bytes.fromhex(row["cdr_sha256"]) for row in expected)).hexdigest()
    seal_matches = bool(seal and seal.get("messages") == len(expected) == matched
                        and seal.get("expected_cdr_sha256") == seal.get("published_cdr_sha256") == ordered_sha)
    return {
        "expected_packets": len(expected), "observed_publication_records": len(observed), "matched_source_prefix_packets": matched,
        "expected_by_topic": dict(by_topic_expected), "observed_by_topic": dict(by_topic_observed),
        "unobserved_original_suffix_packets": len(expected) - matched,
        "packet_identity_status": "COMPLETE" if len(expected) == matched and not first_mismatch
            else ("MISMATCH" if first_mismatch else ("PARTIAL_PREFIX" if matched else "NO_HISTORY_PUBLISHED")),
        "first_mismatch": first_mismatch, "verified_per_packet_original_cdr_hashes_in_prefix": verified_hashes,
        "expected_ordered_cdr_digest": ordered_sha, "complete_replayer_cdr_seal_matches": seal_matches,
        "publication_observer_lag_ms": quantiles(lag), "early_publication_observations": sum(item < 0 for item in lag),
        "last_matched": observed[matched - 1] if matched else None,
        "first_unmatched_original": expected[matched] if matched < len(expected) else None,
    }


def write_packet_rows(path, expected, observed, timebase, matched):
    """Keep exact nanosecond evidence for each packet and every unsent original."""
    with path.open("x", encoding="utf-8") as stream:
        for index in range(max(len(expected), len(observed))):
            original = expected[index] if index < len(expected) else None
            actual = observed[index] if index < len(observed) else None
            due = (timebase["origin_mono_ns"] + original["original_schedule_ns"] - timebase["origin_schedule_ns"]
                   if timebase and original else None)
            comparison = {"ordinal": index, "original": original, "publication_record": actual,
                          "scheduled_replay_mono_ns": due,
                          "observation_lag_ns": actual["mono_ns"] - due if actual and due is not None else None,
                          "verified_ordered_prefix": index < matched}
            stream.write(json.dumps(comparison, separators=(",", ":"), allow_nan=False) + "\n")


def open_session_map(path, anchor):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != anchor["session_map_sha256"]:
        raise ValueError("Session frame map changed after the measured Home/start anchor")
    header, blocks = raw.decode("ascii").split("\n", 1)
    if header != f"S05_DCR009_START_MAP_V1 {anchor['session']}":
        raise ValueError("The stored frame map belongs to another native session")
    values = np.asarray([float(item) for item in blocks.split()], dtype=float)
    frames = np.repeat(np.eye(4)[None], 7, axis=0)
    frames[:, :3] = values.reshape(7, 3, 4)
    rotation = frames[:, :3, :3]
    if (not np.isfinite(frames).all() or not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-8, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1.0, atol=1e-8, rtol=0)
            or not np.array_equal(frames[0], np.asarray(anchor["shared_transform"]))):
        raise ValueError("The frozen session transform is inconsistent or improper")
    return frames


def hmb_ticks(reader, root, plan):
    from home_anchored_start_anchor import hmb_five_rows

    window = plan["historical_window"]
    shift = window["source_cutoff_start_ns"] - window["hmb_cutoff_start_ns"]
    hmb_plan = {row["partition"]: row for row in window["partitions"]}
    totals, chosen = Counter(), []
    previous = -1
    for part in root["hmb_partitions"]:
        key = part["key"]
        data, meta, _, _ = reader.read(key)
        identity = meta["identity"]
        time = data["cutoff_ns"]
        if (identity["root_episode_id"] != ROOT or identity["development_split"] != "Train"
                or identity["source_manifest_sha256"] != root["manifest_sha256"]
                or len(time) != part["planned_ticks"] or not np.array_equal(time, data["tick_ns"])
                or time[0] <= previous or np.any(np.diff(time) <= 0)):
            raise ValueError(f"Signed HMB timeline disagrees: {key}")
        previous = int(time[-1])
        joint, five = data["joint_valid"], hmb_five_rows(data)
        totals.update(partitions=1, plan_ticks=len(time), hmb_joint_valid_ticks=int(joint.sum()),
                      hmb_five_frame_candidate_ticks=int(five.sum()))
        wanted = np.flatnonzero((time >= window["hmb_cutoff_start_ns"]) & (time < window["hmb_cutoff_stop_exclusive_ns"]))
        if not wanted.size:
            continue
        frozen = hmb_plan[key]
        computed = [int(wanted[0]), int(wanted[-1]), len(wanted), int(joint[wanted].sum()), int(five[wanted].sum())]
        if computed != [frozen[field] for field in ("first_tick", "last_tick", "plan_ticks", "hmb_joint_valid_ticks",
                                                    "hmb_five_frame_candidate_ticks")]:
            raise ValueError(f"Original frozen historical support differs: {key}")
        for i in wanted:
            valid = bool(joint[i] and data["safety_permission"][i] and data["pose_valid"][i, :4].all())
            pose = data["pose"][i, 2:4, 0].copy() if valid else None
            chosen.append((key, int(i), int(time[i]), shift + int(time[i]), bool(joint[i]), bool(five[i]), pose))
    if dict(totals) != plan["full_root_denominators"] or len(chosen) != window["plan_ticks"]:
        raise ValueError("Recomputed exact HMB plan did not preserve the frozen root denominators")
    return chosen


def feedback_frames(values, names):
    frames = []
    for header, group in values.items():
        if group.keys() != FEEDBACK_TOPICS or any(named is None for _, named in group.values()):
            continue
        arrival = max(when for when, _ in group.values())
        named = {key: atom for _, axes in group.values() for key, atom in axes.items()}
        if len(named) == len(names) == 54 and named.keys() == set(names):
            frames.append((arrival, header, np.asarray([named[name] for name in names])))
    return sorted(frames, key=lambda frame: (frame[0], frame[1]))


def _poses(xyz, rotations):
    out = np.repeat(np.eye(4)[None], 2, axis=0)
    out[:, :3, :3], out[:, :3, 3] = rotations, xyz
    return out


def tcp_metrics(robot, measured, poses, frames, palm_from_tcp):
    if poses.shape != (2, 7) or not np.isfinite(poses).all() or not np.allclose(
        np.linalg.norm(poses[:, 3:], axis=1), 1, atol=1e-3, rtol=0
    ):
        raise ValueError("Signed HMB registered-wrist pose is not valid")
    trackers = _poses(poses[:, :3], Rotation.from_quat(poses[:, 3:]).as_matrix())
    goal = frames[0] @ trackers @ frames[3:5] @ palm_from_tcp
    geometry = robot.geometry(measured)
    actual = _poses(geometry.wrists, geometry.rotations)
    relative = goal[:, :3, :3].transpose((0, 2, 1)) @ actual[:, :3, :3]
    errors = (actual[:, :3, 3] - goal[:, :3, 3]) * 1000
    angles = np.degrees(Rotation.from_matrix(relative).magnitude())
    return [{"side": side, "xyz_error_mm": errors[i].tolist(), "position_mm": float(np.linalg.norm(errors[i])),
             "orientation_deg": float(angles[i]), "target_world_xyz_m": goal[i, :3, 3].tolist(),
             "actual_world_xyz_m": actual[i, :3, 3].tolist()} for i, side in enumerate(("left", "right"))]


def score_ticks(ticks, trace, audit, timebase, session, frames, robot, halt_mono):
    from pair_nominal_hmb import signed_feedback_names

    statuses, scored = Counter(), []
    feedback = feedback_frames(trace["feedback"], signed_feedback_names(robot))
    arrivals = [row[0] for row in feedback]
    changes = trace["changes"]
    phase_times = [row[0] for row in changes]
    home = robot.geometry(robot.home)
    palm_from_tcp = np.linalg.inv(_poses(home.palm, home.palm_rotation)) @ _poses(home.wrists, home.rotations)
    for part, tick, hmb_ns, source_ns, joint, five, poses in ticks:
        point = dict(partition=part, partition_tick=tick, hmb_cutoff_ns=hmb_ns, original_schedule_ns=source_ns,
                     original_hmb_joint_valid=joint, original_five_frame_candidate=five)
        last = audit["last_matched"]
        if not timebase or not last or source_ns > last["original_schedule_ns"]:
            status = "ORIGINAL_INPUT_PREFIX_NOT_REACHED"
        elif poses is None:
            status = "ORIGINAL_HMB_MASKED"
        else:
            elapsed = source_ns - timebase["origin_schedule_ns"]
            due_mono, due_ros = (timebase["origin_mono_ns"] + elapsed,
                                timebase["origin_log_ns"] + timebase["epoch_delta_ns"] + elapsed)
            point.update(replay_cutoff_mono_ns=due_mono, replay_cutoff_ros_ns=due_ros)
            phase_idx = bisect_right(phase_times, due_mono) - 1
            if halt_mono is not None and due_mono >= halt_mono:
                status = "AT_OR_AFTER_HALT"
            elif phase_idx < 0 or changes[phase_idx][1] != session:
                status = "NO_ACTIVE_TASK_SESSION"
            else:
                visit = bisect_right(arrivals, due_mono)
                pick = max((row for row in feedback[bisect_right(arrivals, due_mono - FRESH_NS - 1):visit]
                            if changes[phase_idx][2] <= row[1] <= due_ros and due_ros - row[1] <= FRESH_NS),
                           key=lambda row: (row[1], row[0]), default=None)
                if pick is None:
                    status = "NO_FRESH_CAUSAL_54_AXIS_FEEDBACK"
                else:
                    status = "OBSERVED_TCP"
                    point.update(feedback_arrival_mono_ns=pick[0], feedback_header_ros_ns=pick[1],
                                 wrists=tcp_metrics(robot, pick[2], poses, frames, palm_from_tcp))
        statuses[status] += 1
        point["status"] = status
        scored.append(point)
    return scored, dict(statuses), len(feedback)


def guard_audit(path, start_ns, stop_ns):
    counters, relevant = Counter(), {}
    if not path.is_file():
        return {"status": "MISSING", "by_group_reason": {}, "latest_denial_by_group": {}}
    for _, row in read_rows(path):
        if start_ns is None or row["receipt_ns"] < start_ns or (stop_ns is not None and row["receipt_ns"] > stop_ns):
            continue
        group, reason = row.get("group"), row["reason"]
        counters[f"{group}:{reason}"] += 1
        if reason != "PERMIT":
            relevant[str(group)] = {key: row[key] for key in ("sequence", "reason", "receipt_ns", "proof_complete_ns")}
    return {"status": "OBSERVED", "by_group_reason": dict(counters), "latest_denial_by_group": relevant}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("attempt", "plan", "cohort", "model-receipt", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--live-model", "--model", type=Path, required=True)
    parser.add_argument("--hmb-products", type=Path)
    parser.add_argument("--packets-out", type=Path, help="Optional complete original and observed packet audit JSONL")
    parser.add_argument("--docs", type=Path, default=Path(__file__).resolve().parents[2] / "docs/s0-5")
    args = parser.parse_args()
    from home_anchored_start_anchor import train_reader
    from pair_nominal_hmb import digest
    from swept_live_guard import verified_robot

    plan = plans(args.plan)
    trace_path = args.attempt / "evidence/trace.jsonl"
    input_path = args.attempt / "evidence/input.json"
    input_receipt = json.loads(input_path.read_bytes())
    if (input_receipt["root"]["root"] != ROOT or input_receipt["root"]["split"] != "Train"
            or input_receipt["plan_sha256"] != PLAN_SHA or input_receipt["cohort_receipt_sha256"] != COHORT_SHA
            or input_receipt.get("historical_window") != plan["historical_window"]):
        raise ValueError("Attempt provenance differs from original locked root-five window")
    trace = read_trace(trace_path)
    markers = trace["markers"]
    anchor, timebase, handoff, seal, intent, model, explicit_halt = (
        singleton(markers, label) for label in ("start_anchor", "timebase", "start_handoff", "historical_input_seal",
                                               "historical_window_intent", "model", "history_halt"))
    if trace["packets"] and (not (handoff and handoff.get("status") == "START_REACHED" and anchor and model)
                             or model["sha256"] != LIVE_SHA or not timebase or timebase["mono_ns"] <= handoff["mono_ns"]):
        raise ValueError("Historical publication without same-attempt measured staged START and actual-model evidence")
    expected, root = expected_packets(args.cohort, plan["historical_window"])
    if root != input_receipt["root"]:
        raise ValueError("Original capture differs from the attempt input source")
    packet = packet_audit(expected, trace["packets"], timebase, seal, plan["historical_window"])
    _, reader, _ = train_reader(args.cohort, args.docs, args.hmb_products)
    ticks = hmb_ticks(reader, root, plan)
    model_raw = args.model_receipt.read_bytes()
    if sha256(model_raw).hexdigest() != MODEL_RECEIPT_SHA:
        raise ValueError("Original exact model dependencies receipt differs")
    live = verified_robot(args.live_model, LIVE_SHA, json.loads(model_raw)["model_identity"]["live"])
    aborts = [row for row in markers["pilot_lifecycle"] if row.get("status") == "ABORTED"]
    other_halt = markers["historical_abort"] or markers["guard_stop"] or aborts
    halt = explicit_halt or (other_halt[0] if other_halt else None)
    actual_halt = (halt["mono_ns"] if halt and halt["stage"] == HISTORICAL else None)
    frames = open_session_map(args.attempt / "session-map.txt", anchor) if anchor else None
    tick_rows, statuses, complete_states = score_ticks(ticks, trace, packet, timebase,
        anchor["session"] if anchor else None, frames, live, actual_halt)
    measurements = [row for row in tick_rows if row["status"] == "OBSERVED_TCP"]
    wrists = {side: {field: quantiles([point["wrists"][i][field] for point in measurements])
                     for field in ("position_mm", "orientation_deg")} for i, side in enumerate(("left", "right"))}
    sweep = args.attempt / "sweep-trace.jsonl"
    result = {
        "scope": "DIAGNOSTIC_ONLY_ROOT_FIVE_HISTORY", "root": ROOT, "formal_coverage_credit": False,
        "original_full_root_denominators": plan["full_root_denominators"], "frozen_window": plan["historical_window"],
        "source_files_sha256": {"native_trace": digest(trace_path), "input_receipt": digest(input_path),
                                "cohort": COHORT_SHA, "plan": PLAN_SHA, "live_model": LIVE_SHA,
                                "swept_trace": digest(sweep) if sweep.is_file() else None},
        "start_handoff": handoff, "historical_window_intent": intent, "recorded_timebase": timebase,
        "input_counts_all_stages": trace["input_counts_all_stages"], "matched_historical_packets": packet,
        "observed_historical_command_messages": trace["history_observed_command_counts"],
        "feedback_complete_named_history_frames": complete_states,
        "hmb_tick_status_counts": statuses, "wrist_tcp_error": wrists,
        "last_scored_hmb_tick": measurements[-1] if measurements else None,
        "halt": halt, "historical_safety_observations": trace["health"],
        "latest_arm_diagnostics": trace["latest_arm_diagnostics"],
        "sweep": guard_audit(sweep, timebase["origin_mono_ns"] if timebase else None, actual_halt),
        "target_basis": "Frozen yaw-aligned engineering map of signed HMB registered wrists through the historical "
                        "palm mount and exact live-model palm-to-scoring-TCP link. Independent anatomical and "
                        "physical accuracy are unqualified. No feedback interpolation or packet-time realignment.",
        "tick_results": tick_rows,
    }
    if args.packets_out:
        write_packet_rows(args.packets_out, expected, trace["packets"], timebase, packet["matched_source_prefix_packets"])
        result["full_packet_audit"] = {"path": args.packets_out.name, "sha256": digest(args.packets_out)}
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
