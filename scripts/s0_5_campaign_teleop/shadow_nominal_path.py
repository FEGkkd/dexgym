"""Apply the signed DCR-008 static guard to observed issued-command snapshots."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

import numpy as np

from pair_nominal_hmb import digest, normalized_assets, trace_evidence
from summarize_nominal_replay import ACTIVE, COMMANDS, HISTORICAL, joint_values, stamp


MODEL_SHA = "939b3651e2426e596eed482e70a68c2382720df1fd55b124aa5f0893b640e92d"
ASSETS_SHA = "92c44cabe65b278ab8261c2659c2cfa8c670fad07fd7d3a640e8629bc5c06a0d"
TICK_NS = 20_000_000
MAX_SAMPLES = 400
PLAN_SCHEMA = "s05.teleop.issued_shadow_plan.v1"


def plan_for(path, expected_sha, actual_root):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Preregistered shadow plan SHA-256 differs")
    plan = json.loads(raw)
    if (set(plan) != {"schema", "root", "start_tick", "count", "stride_ticks", "max_age_ms"}
            or plan["schema"] != PLAN_SCHEMA or plan["root"] != actual_root):
        raise ValueError("Shadow plan does not bind the signed replay root")
    start, count, stride, age = (plan[k] for k in ("start_tick", "count", "stride_ticks", "max_age_ms"))
    if (any(type(v) is not int for v in (start, count, stride)) or start < 0 or not 1 <= count <= MAX_SAMPLES
            or stride < 1 or type(age) not in (int, float) or not isfinite(age) or not 0 < age <= 1000):
        raise ValueError("Shadow plan must be fixed and bounded to 400 control samples")
    return plan


def model_for(path, asset_manifest):
    from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel

    if digest(path) != MODEL_SHA or digest(asset_manifest) != ASSETS_SHA:
        raise ValueError("Use the frozen DCR-008 P02 XML and asset manifest")
    expected = {name.split("/generated/")[-1]: value for name, value in json.loads(asset_manifest.read_bytes()).items()}
    if normalized_assets(path.resolve()) != expected:
        raise ValueError("An independent signed P02 asset differs")
    return RobotModel(path)


def aliases_for(robot):
    aliases = {name: i for i, name in enumerate(robot.names)}
    aliases.update({f"{side}_j{i + 1}": 7 * s + i for s, side in enumerate(("left", "right")) for i in range(7)})
    aliases.update({actuator.name: i for i, actuator in enumerate(robot.actuators) if i >= 14})
    return aliases


def canonical(pieces, aliases):
    values = {}
    for q in pieces:
        if q is None:
            return None
        for joint, angle in q.items():
            index = aliases.get(joint)
            if index is None or index in values:
                return None
            values[index] = angle
    return np.asarray([values[i] for i in range(54)], dtype=float) if len(values) == 54 else None


class Past:
    def __init__(self, rows):
        self.rows = rows
        self.receipts = [row[0] for row in rows]

    def get(self, before, header, max_age):
        i = bisect_left(self.receipts, before) - 1
        while i >= 0:
            row = self.rows[i]
            if before[0] - row[0][0] > max_age:
                break
            if row[1] <= header:
                return row if header - row[1] <= max_age else None
            i -= 1
        return None


def command_streams(path, context_at):
    rows = defaultdict(list)
    sessions = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as trace:
        for index, line in enumerate(trace):
            row = json.loads(line)
            topic = row.get("topic")
            if row["kind"] == "output" and topic in ACTIVE:
                segment = context_at(row["mono_ns"])
                session = row["value"].get("session_id")
                if segment and session and sessions.setdefault((segment, topic), session) != session:
                    raise ValueError("Observed workflow session changed without an ACTIVE/TASK phase barrier")
            if row["kind"] != "output" or row["stage"] != HISTORICAL or topic not in COMMANDS:
                continue
            receipt = row["mono_ns"]
            segment = context_at(receipt)
            value = row["value"]
            header = stamp(value.get("header", {}).get("stamp"))
            if segment and header is not None:
                rows[segment, topic].append(((receipt, index), header, joint_values(value, COMMANDS[topic]),
                                             topic, value.get("sequence"), value.get("session_id"),
                                             stamp(value.get("valid_until"))))
    return {key: Past(records) for key, records in rows.items()}


def observed_issued(streams, segment, before, header, age):
    pieces = {}
    for topic in COMMANDS:
        stream = streams.get((segment, topic))
        record = None if stream is None else stream.get(before, header, age)
        if record is not None:
            pieces[topic] = record
    return pieces


def sample(context, trace, plan, aliases):
    phase_times = [t for t, _ in context["segments"]]
    phase_ids = [s for _, s in context["segments"]]

    def context_at(when):
        i = bisect_right(phase_times, when) - 1
        return phase_ids[i] if i >= 0 else 0

    commands = command_streams(trace, context_at)
    frames = defaultdict(list)
    for receipt, header, segment, joints in context["frames"]:
        frames[segment].append(((receipt, 0), header, joints))
    measured = {segment: Past(records) for segment, records in frames.items()}
    timebase = context["timebase"]
    age = round(plan["max_age_ms"] * 1e6)
    for i in range(plan["count"]):
        tick = plan["start_tick"] + i * plan["stride_ticks"]
        offset = tick * TICK_NS
        scheduled = timebase["origin_schedule_ns"] + offset
        cutoff = timebase["origin_mono_ns"] + offset
        cutoff_header = timebase["origin_log_ns"] + timebase["epoch_delta_ns"] + offset
        segment = context_at(cutoff)
        row = {"tick": tick, "schedule_ns": scheduled, "active_segment": segment}
        if not context["first_source_schedule"] <= scheduled <= context["last_source_schedule"]:
            yield dict(row, status="OUTSIDE_REPLAYED_WINDOW")
            continue
        if not segment:
            yield dict(row, status="OUTSIDE_OBSERVED_ACTIVE_TASK")
            continue
        through_tick = observed_issued(commands, segment, (cutoff, 2**63), cutoff_header, age)
        if through_tick.keys() != COMMANDS.keys():
            yield dict(row, status="INCOMPLETE_ISSUED_COMMAND_AT_CUTOFF")
            continue
        anchor = max(through_tick.values(), key=lambda record: record[0])
        receipt, header, _, topic, sequence, session, _ = anchor
        previous = observed_issued(commands, segment, receipt, header, age)
        frame_index = measured.get(segment)
        frame = None if frame_index is None else frame_index.get((receipt[0], 0), header, age)
        q_measured = canonical([frame[2]], aliases) if frame is not None else None
        row.update(anchor_topic=topic, anchor_sequence=sequence, anchor_session=session,
                   anchor_header_ns=header, anchor_receipt_mono_ns=receipt[0], anchor_trace_row=receipt[1],
                   measured_header_ns=frame[1] if frame is not None else None,
                   measured_receipt_mono_ns=frame[0][0] if frame is not None else None,
                   q_from_measured_rad=q_measured)
        q_before = canonical((r[2] for r in previous.values()), aliases) if previous.keys() == COMMANDS.keys() else None
        updated = dict(previous)
        updated[topic] = anchor
        q_end = canonical((r[2] for r in updated.values()), aliases)
        same_topic = previous.get(topic)
        contiguous = (same_topic is not None and type(sequence) is int and type(same_topic[4]) is int
                      and sequence == same_topic[4] + 1
                      and session == same_topic[5])
        row.update(
            q_from_last_observed_issued_rad=q_before, q_observed_issued_proposal_rad=q_end,
            start_max_abs_difference_rad=(float(np.abs(q_before - q_measured).max())
                                          if q_before is not None and q_measured is not None else None),
            anchor_topic_sequence_contiguous=contiguous,
            group_header_ns={name: record[1] for name, record in updated.items()},
            group_receipt_mono_ns={name: record[0][0] for name, record in updated.items()},
        )
        if any(r[6] is not None and r[6] < header for r in updated.values()):
            yield dict(row, status="EXPIRED_GROUP_AT_ANCHOR")
        elif q_end is None:
            yield dict(row, status="INVALID_OR_UNMAPPED_COMMAND_GROUP")
        else:
            yield dict(row, status="EVALUABLE")


def shadow(robot, sweep, rows):
    endpoint_cache, path_cache = {}, {}
    for row in rows:
        counts = {}
        if row["status"] == "EVALUABLE":
            end = row["q_observed_issued_proposal_rad"]
            key = end.tobytes()
            if key not in endpoint_cache:
                endpoint_cache[key] = dict(robot.legality(end))
                endpoint_cache[key]["feasible"] = bool(endpoint_cache[key]["feasible"])
            endpoint = endpoint_cache[key]
            row["endpoint_signed_safety"] = endpoint
            for label in ("last_observed_issued", "measured"):
                start = row[f"q_from_{label}_rad"]
                if start is None:
                    counts[label] = "START_UNOBSERVED"
                elif not endpoint["feasible"]:
                    counts[label] = "ENDPOINT_ILLEGAL"
                else:
                    pair = start.tobytes(), key
                    if pair not in path_cache:
                        path_cache[pair] = bool(sweep.permits(start, end))
                    counts[label] = "SWEPT_PROVEN" if path_cache[pair] else "SWEPT_UNPROVEN"
        row["guard"] = counts
        yield {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in row.items()}


def main():
    from dexgym.tasks.direct.wholebody_s05.v1.task import native_sweep

    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("trace", "model", "plan", "out"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--assets", type=Path, default=Path("docs/s0-5/evidence/P02/refinement/model_assets_sha256.json"))
    args = parser.parse_args()
    receipt_path = args.trace.with_name("input.json")
    raw_receipt = receipt_path.read_bytes()
    receipt = json.loads(raw_receipt)
    if "cohort_receipt_sha256" not in receipt or not isinstance(receipt.get("root"), dict):
        raise ValueError("Expected a signed HMB cohort replay receipt")
    plan = plan_for(args.plan, args.plan_sha256, receipt["root"]["root"])
    trace_sha = digest(args.trace)
    robot = model_for(args.model, args.assets)
    sweep = native_sweep.NativeSweep(robot)
    context = trace_evidence(args.trace)
    rows = list(shadow(robot, sweep, sample(context, args.trace, plan, aliases_for(robot))))
    if trace_sha != digest(args.trace):
        raise ValueError("Replay trace changed during shadow analysis")
    report = {
        "scope": "RETROSPECTIVE_OBSERVED_ISSUED_COMMAND_SHADOW_DIAGNOSTIC", "qualification": "NONE",
        "plan": plan, "plan_sha256": args.plan_sha256, "trace_sha256": trace_sha,
        "replay_input_sha256": sha256(raw_receipt).hexdigest(), "live_observed_xml": context["model"],
        "signed_guard": {"xml_sha256": MODEL_SHA, "asset_manifest_sha256": ASSETS_SHA,
                         "clearance_m": sweep.clearance, "max_depth": sweep.max_depth,
                         "max_queries": sweep.max_queries, "collision_pair_count": len(robot.geom_a),
                         "sweep_source_sha256": digest(Path(native_sweep.__file__)),
                         "mujoco_version": native_sweep.mujoco.__version__,
                         "endpoint_then_swept": True, "recheck_identical_unqualified_starts": True},
        "limits": [
            "Three asynchronously observed publisher messages are a causal assembled issued snapshot.",
            "Last observed issued and last coherent measured starts are separate in the current ACTIVE/TASK generation.",
            "No initial HOME substitution, rejected prepublication proposal, actuator receipt, or Servo path observed.",
            "A failed proof can represent bounded geometric uncertainty. No attribution to online or DCR-008 rejection.",
            "The signed collision model is used even if the native live XML differs.",
        ],
        "counts": dict(Counter(row["status"] for row in rows)),
        "guard_counts": {name: dict(Counter(r["guard"].get(name, "NOT_EVALUATED") for r in rows))
                         for name in ("last_observed_issued", "measured")},
        "rows": rows,
    }
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
