#!/usr/bin/env python3
"""Audit a STOP boundary from native publication, exact sweep and observer evidence."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
import re

from audit_dcr009_guarded_diagnostic import GROUPS, LINE, TOPICS, read, rows
from compare_start_handoff import valid_feedback
from replay_dcr009_guarded_diagnostic import LIVE_SHA, SIGNED_SHA
from run_start_handoff_cohort import digest
from summarize_dcr009_center_offsets import JOINT_NAMES, JOINT_SLICES
from summarize_nominal_replay import FEEDBACK, stamp


BARRIER = re.compile(r"S05_DCR009_STOP_BARRIER session=(\S+) generation=(\d+) last_arm=(\d+) last_left=(\d+) last_right=(\d+)(?:\s|$)")
BARRIER_CLOCK = re.compile(r"\bmono_ns=(\d+)(?:\s|$)")
PUBLISH_HEADER = re.compile(r"\bcommand_header_ns=(\d+)(?:\s|$)")
LOGGER_TIME = re.compile(r"\[INFO\]\s+\[(\d+)\.(\d{9})\]")
STOP_SERVICE = "/whole_body_teleop/stop"
SCOPE = "DIAGNOSTIC_ONLY"
JOINT_ORDER = [joint for group in GROUPS for joint in JOINT_NAMES[group]]
HAND_FEEDBACK = {"/left_hand/state": "left", "/right_hand/state": "right"}
FRESH_NS = 40_000_000
PASS = "STOP_BOUNDARY_PASS"
INCONCLUSIVE = "INCONCLUSIVE"
VIOLATION = "POST_BARRIER_PUBLICATION"


@dataclass(frozen=True)
class Publication:
    group: str
    session: str
    generation: int
    sequence: int
    proof: int
    feedback_ns: int
    line: int
    ros_log_ns: int | None
    command_header_ns: int | None


def native_log(path):
    publications, barriers, conflicts = [], [], []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if "S05_DCR009_STOP_BARRIER" in line:
                match = BARRIER.search(line)
                if not match or not line.endswith("\n"):
                    conflicts.append(f"unreadable native barrier record at line {number}")
                    continue
                session, generation, *bounds = match.groups()
                clock = BARRIER_CLOCK.search(line)
                barriers.append({"session": session, "generation": int(generation), "line": number,
                                 "mono_ns": int(clock[1]) if clock else None,
                                 "last": dict(zip(GROUPS, map(int, bounds), strict=True))})
            if "S05_DCR009_GUARD " not in line:
                continue
            match = LINE.search(line)
            if not match or not line.endswith("\n"):
                conflicts.append(f"unreadable native guard record at line {number}")
                continue
            group, session, generation, sequence, proof, published, reason, feedback = match.groups()
            if published == "1":
                when = LOGGER_TIME.search(line)
                header = PUBLISH_HEADER.search(line)
                logger_ns = (int(when[1]) * 1_000_000_000 + int(when[2])) if when else None
                if reason != "0" or min(map(int, (generation, sequence, proof, feedback))) <= 0:
                    conflicts.append(f"conflicting positive publication at line {number}")
                publications.append(Publication(group, session, int(generation), int(sequence), int(proof),
                                                int(feedback), number, logger_ns, int(header[1]) if header else None))
    return publications, barriers, conflicts


def proof_index(path):
    index = defaultdict(list)
    for row in rows(path):
        index[row["sequence"]].append(row)
    return index


def exact_permit(publication, index):
    candidates = index.get(publication.proof, [])
    if len(candidates) != 1:
        return None
    proof = candidates[0]
    states = proof.get("issued_measured_proposed_rad")
    if (proof.get("reason") != "PERMIT" or proof.get("group") != GROUPS.index(publication.group)
            or not isinstance(states, list) or len(states) != 3
            or any(not isinstance(vector, list) or len(vector) != 54 or
                   any(type(q) not in (float, int) or not isfinite(q) for q in vector) for vector in states)):
        return None
    begin, end = proof.get("receipt_ns"), proof.get("proof_complete_ns")
    if type(begin) is not int or type(end) is not int or not 0 < begin <= end:
        return None
    issued, _, proposed = states
    group_slice = JOINT_SLICES[publication.group]
    if issued[:group_slice.start] != proposed[:group_slice.start] or issued[group_slice.stop:] != proposed[group_slice.stop:]:
        return None
    return proof


def feedback_at(trace):
    grouped = defaultdict(dict)
    for row in trace:
        topic = row.get("topic")
        if row.get("kind") != "output" or topic not in FEEDBACK:
            continue
        value = row["value"]
        header = stamp(value.get("header", {}).get("stamp"))
        if header is None:
            continue
        data = valid_feedback(topic, value)
        if topic in HAND_FEEDBACK and (value.get("side") != HAND_FEEDBACK[topic] or value.get("feedback_source") != "mujoco"
                                        or value.get("source_clock") != "simulation"
                                        or value.get("source_timestamp_valid") is not True):
            data = None
        if topic in grouped[header] and data != grouped[header][topic]:
            grouped[header][topic] = None
        else:
            grouped[header][topic] = data
    return grouped


def exact_feedback(grouped, publication, proof):
    parts = grouped.get(publication.feedback_ns, {})
    if parts.keys() != FEEDBACK.keys() or any(part is None for part in parts.values()):
        return False
    named = {name: value for part in parts.values() for name, value in part.items()}
    return set(named) == set(JOINT_ORDER) and [named[name] for name in JOINT_ORDER] == proof["issued_measured_proposed_rad"][1]


def command_matches(row, publication, proof):
    value = row["value"]
    session = value.get("session_id", publication.session if publication.group != "arm" else None)
    if (session != publication.session or value.get("generation", publication.generation) != publication.generation
            or value.get("joint_names") != JOINT_NAMES[publication.group]
            or stamp(value.get("header", {}).get("stamp")) is None):
        return False
    if publication.group != "arm" and value.get("side") != publication.group:
        return False
    positions = value.get("positions" if publication.group == "arm" else "position")
    return positions == proof["issued_measured_proposed_rad"][2][JOINT_SLICES[publication.group]]


def audit_streams(log_path, trace_path, proof_path, *, archive=False):
    publications, barriers, conflicts = native_log(log_path)
    trace = list(rows(trace_path))
    proofs = proof_index(proof_path)
    feedback = feedback_at(trace)
    causes = []
    stamps = [row.get("mono_ns") for row in trace]
    if (not stamps or any(type(value) is not int or value <= 0 for value in stamps)
            or any(after < before for before, after in zip(stamps, stamps[1:]))):
        causes.append("invalid_observer_order")
    ack_rows = [row for row in trace if row.get("kind") == "service" and row.get("service") == STOP_SERVICE
                and row.get("response", {}).get("accepted") is True]
    service_ack = ack_rows[0]["mono_ns"] if len(ack_rows) == 1 else None
    markers = [row for row in trace if row.get("kind") == "stop_observation_boundary"]
    boundary = markers[0]["mono_ns"] if len(markers) == 1 else None
    ack = markers[0].get("ack_mono_ns") if len(markers) == 1 else service_ack if archive else None
    barrier = barriers[0] if len(barriers) == 1 else None
    stops = [row for row in trace if row.get("kind") == "pilot_stop"]
    if barrier is None:
        causes.append("missing_or_nonunique_native_barrier")
    if type(ack) is not int or service_ack is None or boundary is None or not service_ack <= ack <= boundary:
        causes.append("missing_or_unordered_observer_boundary")
    if barrier and (type(barrier["mono_ns"]) is not int or type(ack) is not int or not 0 < barrier["mono_ns"] <= ack):
        causes.append("missing_or_unordered_native_barrier_clock")
    if len(stops) != 1 or (boundary is not None and stops[0]["mono_ns"] <= boundary):
        causes.append("missing_or_unordered_observation_tail")
    if conflicts:
        causes.append("ambiguous_native_records")

    native_after = [publication for publication in publications if barriers and publication.line > barriers[0]["line"]]
    if barrier:
        old = (barrier["session"], barrier["generation"])
        previous = {group: [] for group in GROUPS}
        for publication in publications:
            if publication.line >= barrier["line"]:
                continue
            if (publication.session, publication.generation) != old:
                causes.append("another_publication_generation_before_single_session_barrier")
            previous[publication.group].append(publication.sequence)
        if any(values != sorted(set(values)) or max(values, default=0) != barrier["last"][group] for group, values in previous.items()):
            causes.append("barrier_sequence_fence_disagrees_with_native_stream")
        if (len(stops) == 1 and stops[0].get("session") != barrier["session"]
                or len(markers) == 1 and markers[0].get("session") != barrier["session"]):
            causes.append("observer_and_native_boundary_sessions_differ")

    by_group_sequence = defaultdict(list)
    proof_uses = Counter(publication.proof for publication in publications)
    for publication in publications:
        by_group_sequence[publication.group, publication.sequence].append(publication)
    output = [row for row in trace if row.get("kind") == "output" and row.get("topic") in TOPICS]
    if service_ack is not None and boundary is not None and any(service_ack <= row["mono_ns"] <= boundary for row in output):
        causes.append("command_in_observer_ack_boundary_gap")
    cutoff = boundary if type(boundary) is int else service_ack
    post = [row for row in output if cutoff is not None and row["mono_ns"] > cutoff]
    if len(stops) == 1 and cutoff is not None:
        expected = stops[0].get("late_observed_command_count")
        if expected is not None and expected != len(post):
            causes.append("observer_tail_count_disagrees_with_outputs")
    details = []
    observed_counts = Counter((TOPICS[row["topic"]], row["value"].get("sequence")) for row in output)
    for row in post:
        group = TOPICS[row["topic"]]
        sequence = row["value"].get("sequence")
        candidates = by_group_sequence.get((group, sequence), [])
        publication = candidates[0] if len(candidates) == 1 and type(sequence) is int and sequence > 0 else None
        proof = exact_permit(publication, proofs) if publication else None
        payload_join = bool(proof and command_matches(row, publication, proof) and proof_uses[publication.proof] == 1)
        header_join = bool(publication and publication.command_header_ns is not None and
                           publication.command_header_ns == stamp(row["value"].get("header", {}).get("stamp")))
        feedback_join = bool(proof and exact_feedback(feedback, publication, proof))
        native_fresh = bool(publication and publication.ros_log_ns is not None and
                            0 <= publication.ros_log_ns - publication.feedback_ns <= FRESH_NS)
        clock_order = bool(proof and type(ack) is int and proof["proof_complete_ns"] <= ack)
        label = "UNKNOWN"
        if barrier and publication and publication.line > barrier["line"] and payload_join and header_join:
            label = "NEW_POST_BARRIER"
        elif (barrier and not causes and observed_counts[group, sequence] == 1 and payload_join and header_join and feedback_join
              and native_fresh and clock_order and proof["proof_complete_ns"] <= barrier["mono_ns"] and publication.line < barrier["line"]
              and (publication.session, publication.generation) == (barrier["session"], barrier["generation"])
              and sequence <= barrier["last"][group]):
            label = "DELAYED_PRE_BARRIER"
        details.append({"classification": label, "topic": row["topic"], "command_sequence": sequence,
                        "receipt_mono_ns": row["mono_ns"], "after_ack_ms": (row["mono_ns"] - ack) / 1e6 if type(ack) is int else None,
                        "native_publication": asdict(publication) if publication else None,
                        "exact_command_proof_join": payload_join, "exact_native_command_header_join": header_join,
                        "exact_coherent_feedback_join": feedback_join,
                        "native_feedback_fresh": native_fresh,
                        "proof_complete_before_ack": clock_order,
                        "proof_complete_mono_ns": proof["proof_complete_ns"] if proof else None})
    unknown = sum(row["classification"] == "UNKNOWN" for row in details)
    return {"scope": SCOPE, "formal_coverage_credit": False,
            "status": VIOLATION if native_after else INCONCLUSIVE if causes or unknown else PASS,
            "causes": sorted(set(causes)), "native_record_conflicts": conflicts, "native_barrier": barrier,
            "accepted_stop_receipt_mono_ns": service_ack, "observer_ack_mono_ns": ack, "observer_boundary_mono_ns": boundary,
            "native_post_barrier_publications": [asdict(item) for item in native_after],
            "observer_post_boundary": details, "driver_application_qualified": False,
            "source_sha256": {"native_log": digest(log_path), "observer_trace": digest(trace_path), "sweep_trace": digest(proof_path)}}


def audit_attempt(attempt_dir):
    receipt_path = attempt_dir / "result.json"
    receipt = read(receipt_path)
    trace = attempt_dir / "evidence/trace.jsonl"
    if receipt.get("scope") != SCOPE or receipt.get("formal_coverage_credit") is not False:
        raise ValueError("STOP audit requires the isolated diagnostic attempt receipt")
    if receipt.get("trace_sha256") != digest(trace):
        raise ValueError("Observer trace differs from the original attempt receipt")
    report = audit_streams(attempt_dir / "sim.log", trace, attempt_dir / "sweep-trace.jsonl")
    ready = attempt_dir / "sweep-ready.json"
    if ready.is_file():
        guard = read(ready)
        unchanged = (digest(ready) == receipt.get("guard_ready_sha256")
                     and guard.get("guard_script_sha256") == digest(Path(__file__).with_name("swept_live_guard.py"))
                     and (guard.get("live_model_sha256"), guard.get("signed_model_sha256")) == (LIVE_SHA, SIGNED_SHA))
    else:
        unchanged = False
    if not unchanged:
        report["causes"].append("unbound_sweep_guard_readiness")
    if receipt.get("owned_groups_stopped") is not True or receipt.get("same_domain_empty_after") is not True:
        report["causes"].append("incomplete_owned_process_teardown")
    if report["causes"] and report["status"] == PASS:
        report["status"] = INCONCLUSIVE
    report["recorded_phase"] = receipt.get("phase")
    report["source_sha256"]["attempt_receipt"] = digest(receipt_path)
    return report


def audit_archive(directory):
    receipt_path = directory / "run_receipt.json"
    receipt = read(receipt_path)
    if (receipt.get("scope") != SCOPE or receipt.get("formal_coverage_credit") is not False
            or len(receipt.get("attempts", [])) != 1):
        raise ValueError("Expected one archived diagnostic attempt")
    report = audit_streams(directory / "ring-and-last-arm-native.txt", directory / "lifecycle.jsonl",
                           directory / "selected-sweep-proofs.jsonl", archive=True)
    report["recorded_phase"] = receipt["attempts"][0]["phase"]
    report["source_sha256"]["archive_run_receipt"] = digest(receipt_path)
    report["archive_observation_only"] = True
    report["archive_scope"] = "Retained extracts lack the new native barrier and an independently joined 54-axis observation"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--attempt", type=Path)
    mode.add_argument("--archived-receipt", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit_attempt(args.attempt) if args.attempt is not None else audit_archive(args.archived_receipt)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(report["status"])
    return {PASS: 0, INCONCLUSIVE: 2, VIOLATION: 3}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
