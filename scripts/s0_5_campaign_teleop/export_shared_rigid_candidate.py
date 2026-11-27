"""Export native shared-START commands and measured MuJoCo state on the signed HMB grid."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import yaml

from pair_nominal_hmb import digest, one
from summarize_nominal_replay import ACTIVE, ARM, COMMANDS, DIAGNOSTICS, HISTORICAL, joint_values, stamp


MAPPING = "shared_rigid_diagnostic_v1"
MAX_AGE_NS = 40_000_000
TICK_NS = 20_000_000
GROUPS = tuple(COMMANDS)
SLICES = (slice(0, 14), slice(14, 34), slice(34, 54))
ARM_STATES = (
    "UNKNOWN", "ACTIVE_REPORTED", "LIMITED_REPORTED", "HOLD_REPORTED", "FALLBACK_REPORTED",
    "REJECTED_REPORTED", "INACTIVE_OR_OUTSIDE_REPLAY",
)


class Observations:
    """Retain trace receipt order and source stamps for double-causal as-of queries."""

    def __init__(self):
        self.keys = []
        self.rows = []

    def add(self, arrival, header, value):
        if header is not None:
            self.keys.append(arrival)
            self.rows.append((arrival, header, value))

    def get(self, arrival, header):
        end = bisect_right(self.keys, arrival)
        start = bisect_right(self.keys, arrival - MAX_AGE_NS - 1, hi=end)
        eligible = (row for row in self.rows[start:end] if 0 <= header - row[1] <= MAX_AGE_NS)
        return max(eligible, key=lambda row: (row[1], row[0]), default=None)


def read_native(path):
    phases, sessions, phase_headers, streams = {}, {}, {}, defaultdict(Observations)
    barrier_times, barrier_ids, mapping_modes, timebases = [], [], set(), []
    first_input = last_input = None
    last_arrival = -1
    epoch = 0
    active = False
    started_header = None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as trace:
        for line in trace:
            if not line.endswith(b"\n"):
                raise ValueError("Incomplete native observation trace")
            row = json.loads(line)
            arrival, kind, topic = row["mono_ns"], row["kind"], row.get("topic")
            if arrival < last_arrival:
                raise ValueError("Native trace is out of receipt order")
            last_arrival = arrival
            if kind == "timebase":
                timebases.append(row)
            if row["stage"] == HISTORICAL and kind == "input":
                schedule = row["original_schedule_ns"]
                if last_input is not None and schedule < last_input:
                    raise ValueError("Original input schedule moved backwards")
                first_input = schedule if first_input is None else first_input
                last_input = schedule
            if kind != "output":
                continue
            value = row["value"]
            header = stamp(value.get("header", {}).get("stamp"))
            if topic in ACTIVE:
                key, expected = ACTIVE[topic]
                session = value.get("session_id")
                sessions[topic] = session
                phases[topic] = value.get(key) == expected
                phase_headers[topic] = header
                ready = bool(session) and all(
                    phases.get(name) and sessions.get(name) == session and phase_headers.get(name) is not None
                    for name in ACTIVE
                )
                if ready != active or (ready and barrier_ids[-1][1] != session):
                    epoch += 1
                    active = ready
                    started_header = max(phase_headers.values()) if ready else None
                    barrier_times.append(arrival)
                    barrier_ids.append((epoch, session, started_header) if ready else (0, None, None))
            if row["stage"] != HISTORICAL or not active or header is None or header < started_header:
                continue
            if topic in COMMANDS:
                streams[epoch, topic].add(arrival, header, value)
            for status in value.get("status", ()):
                side = DIAGNOSTICS.get(status.get("name"))
                if side:
                    fields = {entry["key"]: entry["value"] for entry in status["values"]}
                    mapping_modes.add(fields.get("mapping_policy"))
                    streams[epoch, side].add(arrival, header, fields)
    base = one(timebases, "native replay timebase")
    if first_input is None or first_input != base["origin_schedule_ns"] or mapping_modes != {MAPPING}:
        raise ValueError(f"Native trace cannot attest shared START on scheduled inputs: {mapping_modes}")
    return base, last_input, barrier_times, barrier_ids, streams


def qualified_command(record, topic, session, ros_ns, names):
    if record is None:
        return None
    arrival, header, command = record
    sequence = command.get("sequence")
    if type(sequence) is not int or sequence < 0:
        return None
    if topic == ARM:
        expiry = stamp(command.get("valid_until"))
        if command.get("session_id") != session or expiry is None or expiry < ros_ns:
            return None
    elif command.get("side") != topic.removeprefix("/").split("_", 1)[0]:
        return None
    values = joint_values(command, len(names))
    if values is None or values.keys() != set(names):
        return None
    return np.array([values[name] for name in names]), sequence, arrival, header


def arm_report(record, session, issued, side):
    if record is None or record[2].get("mapping_policy") != MAPPING:
        return 0, False
    _, header, fields = record
    if fields.get("session", fields.get("session_id")) != session:
        return 0, False
    mode = fields.get("solve_mode", "").lower()
    recovery = fields.get("recovery_action", "").lower()
    reason = fields.get("primary_reason", "").lower()
    if "fallback" in mode or "fallback" in recovery:
        return 4, False
    if "hold" in mode or "hold" in recovery or "retain_previous" in mode or "retain_previous" in reason:
        return 3, False
    if fields.get("arm_command_accepted") == "false" or fields.get("ik_target_accepted") == "false":
        return 5, False
    if issued is None or fields.get("sequence") != str(issued[1]) or abs(header - issued[3]) > 4_000_000:
        return 0, False
    try:
        q = np.array([float(item) for item in fields.get("q_cmd_rad", "").split(",")])
    except ValueError:
        return 0, False
    if q.shape != (7,) or not np.allclose(q, issued[0][side * 7:(side + 1) * 7], rtol=0, atol=1e-5):
        return 0, False
    if fields.get("arm_command_accepted") != "true" or fields.get("ik_target_accepted") != "true":
        return 0, False
    rate_limited = fields.get("rate_limited")
    if rate_limited == "true":
        return 2, True
    return (1, True) if rate_limited == "false" else (0, False)


def export(args):
    cohort = json.loads(args.cohort.read_bytes())
    root = one([item for item in cohort["selected_roots"] if item["root"] == args.root], "signed source root")
    summary_path = args.pair / "pair_summary.json"
    summary = json.loads(summary_path.read_bytes())
    pair_path = args.pair / "paired_ticks.jsonl"
    trace_sha = digest(args.trace)
    if (not cohort["native_assets_sha256_checked"] or summary["root"] != args.root
            or summary["cohort_sha256"] != digest(args.cohort) or summary["replay_trace_sha256"] != trace_sha
            or summary["output_ticks_sha256"] != digest(pair_path) or summary["max_age_ms"] != MAX_AGE_NS / 1e6):
        raise ValueError("Native evidence, paired rows or cohort identity differ")
    manifest = (Path(root["canonical_episode_uri"]) / "manifest.yaml").read_bytes()
    if sha256(manifest).hexdigest() != root["manifest_sha256"]:
        raise ValueError("Original capture manifest identity differs")
    clock = yaml.safe_load(manifest)["start_monotonic_ns"]
    base, end, times, barriers, streams = read_native(args.trace)
    signed, wire = map(tuple, zip(*summary["feedback_joint_pairs_in_signed_order"], strict=True))
    if len(signed) != 54 or len(set(signed)) != 54 or len(set(wire)) != 54:
        raise ValueError("Candidate must bind 54 distinct physical and wire axes")
    count = root["planned_ticks"]
    cutoff = np.empty(count, dtype=np.int64)
    hmb_valid = np.empty(count, dtype=bool)
    feedback = np.empty(count, dtype="U64")
    issued, velocity, measured = [np.full((count, 54), np.nan, dtype=np.float64) for _ in range(3)]
    group_valid = np.zeros((count, 3), dtype=bool)
    stamps, receive, seq = [np.full((count, 3), -1, dtype=np.int64) for _ in range(3)]
    epochs = np.zeros(count, dtype=np.int32)
    arm_state = np.zeros((count, 2), dtype=np.uint8)
    arm_joined = np.zeros((count, 2), dtype=bool)
    partitions = []
    with pair_path.open(encoding="utf-8") as source:
        i = 0
        for part in root["hmb_partitions"]:
            partitions.append({"partition": part["key"], "start": i, "count": part["planned_ticks"]})
            for tick in range(part["planned_ticks"]):
                row = json.loads(next(source))
                if (row["root"], row["split"], row["partition"], row["tick"]) != (
                    root["root"], root["split"], part["key"], tick
                ):
                    raise ValueError("Signed root/partition/tick changed in paired rows")
                cutoff[i], hmb_valid[i], feedback[i] = (
                    row["hmb_cutoff_ns"], row["hmb_joint_valid"], row["replay_feedback"],
                )
                q = row["replay_measured_q_rad"]
                if q is not None:
                    if len(q) != 54 or not np.isfinite(q).all() or feedback[i] != "COHERENT_MEASURED":
                        raise ValueError("Invalid measured state in paired rows")
                    measured[i] = q
                planned = clock + int(cutoff[i])
                t = planned - base["origin_schedule_ns"]
                mono_ns = base["origin_mono_ns"] + t
                ros_ns = base["origin_log_ns"] + base["epoch_delta_ns"] + t
                barrier = bisect_right(times, mono_ns) - 1
                epoch, session, armed_at = (
                    barriers[barrier] if barrier >= 0 and t >= 0 and planned <= end else (0, None, None)
                )
                if armed_at is not None and ros_ns < armed_at:
                    epoch = 0
                epochs[i] = epoch
                if not epoch:
                    if q is not None:
                        measured[i] = np.nan
                        feedback[i] = "COHERENT_PAIRED_BUT_EPOCH_UNCONFIRMED"
                    arm_state[i] = 6
                    i += 1
                    continue
                arm = None
                for g, (topic, columns) in enumerate(zip(GROUPS, SLICES, strict=True)):
                    native = streams[epoch, topic].get(mono_ns, ros_ns)
                    current = qualified_command(native, topic, session, ros_ns, wire[columns])
                    if current is not None:
                        issued[i, columns], seq[i, g], receive[i, g], stamps[i, g] = current
                        group_valid[i, g] = True
                    if topic == ARM:
                        arm = current
                for j, side in enumerate(("left", "right")):
                    status = streams[epoch, side].get(mono_ns, ros_ns)
                    arm_state[i, j], arm_joined[i, j] = arm_report(status, session, arm, j)
                if tick and epoch == epochs[i - 1] and cutoff[i] - cutoff[i - 1] == TICK_NS:
                    velocity[i] = (issued[i] - issued[i - 1]) * (1e9 / TICK_NS)
                i += 1
        if i != count or next(source, None) is not None:
            raise ValueError("Paired row count differs from signed denominator")
    args.out.mkdir(parents=True, exist_ok=False)
    data = args.out / "commands_feedback.npz"
    arrays = {
        "cutoff_ns": cutoff, "hmb_joint_valid": hmb_valid, "active_epoch": epochs,
        "q_issued_rad": issued, "qd_issued_backward_rad_s": velocity, "q_measured_rad": measured,
        "feedback_status": feedback, "issued_group_valid": group_valid, "issued_header_ros_ns": stamps,
        "issued_receipt_mono_ns": receive, "issued_sequence": seq, "arm_diagnostic_status": arm_state,
        "arm_diagnostic_matches_issued": arm_joined,
    }
    np.savez_compressed(data, **arrays)
    metrics = {}
    for label, mask in (("planned", np.ones(count, dtype=bool)), ("hmb_joint_valid", hmb_valid)):
        metrics[label] = {
            "rows": int(mask.sum()), "all_54_issued": int((mask & group_valid.all(axis=1)).sum()),
            "all_54_measured": int((mask & np.isfinite(measured).all(axis=1)).sum()),
            "both_arms_reported_active_or_limited": int((mask & arm_joined.all(axis=1)).sum()),
            "arm_status": {side: dict(Counter(ARM_STATES[x] for x in arm_state[mask, j]))
                           for j, side in enumerate(("left", "right"))},
        }
    receipt = {
        "schema": "s05.teleop.shared_rigid_candidate.v1", "status": "DIAGNOSTIC_CANDIDATE",
        "training_admission": False, "formal_active_coverage_evaluated": False,
        "root": args.root, "split": root["split"], "mapping": MAPPING,
        "cohort_sha256": summary["cohort_sha256"], "capture_manifest_sha256": root["manifest_sha256"],
        "trace_sha256": trace_sha, "pair_summary_sha256": digest(summary_path), "npz_sha256": digest(data),
        "paired_ticks_sha256": summary["output_ticks_sha256"], "signed_joint_names": signed,
        "wire_joint_names": wire, "command_group_topics": GROUPS, "arm_status_labels": ARM_STATES,
        "schema_arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                          for key, value in arrays.items()},
        "command_semantics": "Three separately observed command groups as of a signed cutoff, without a simultaneous actuator acknowledgment.",
        "arm_status_semantics": "Controller flags with exact session and sequence and matched joint payload for active/limited. Active can include backoff or other constraints. No swept-path or IK-exactness authority.",
        "velocity_semantics": "Causal backward difference within one partition and active epoch; NaN where not observed.",
        "partition_rows": partitions, "metrics": metrics,
    }
    with (args.out / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "trace", "pair", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--root", required=True)
    export(parser.parse_args())
