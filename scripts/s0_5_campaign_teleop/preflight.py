"""Identity-bound readback of the two frozen Teleop regression recordings."""

import argparse
from dataclasses import asdict, dataclass
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path


BAGS = {
    "s0_sim_20261119_172809_0.mcap": "ffd5353918e38e6527b534aa3263baa20d7cb3f8a868d91f33b8644cbd437817",
    "s0_sim_20261117_181022_0.mcap": "cbac62bb6d58ce2fa159d547408cf3534a29c32ca8b0bba1cddc8c4bc99ba1f5",
}
HUMAN = {
    "/manus_glove_left": "ManusGlove",
    "/manus_glove_right": "ManusGlove",
    "/vive/tracker/left_wrist": "TrackerPose",
    "/vive/tracker/right_wrist": "TrackerPose",
}


def topic_role(topic):
    if topic in HUMAN:
        return "HUMAN_INPUT_CANDIDATE"
    name = topic.casefold()
    if any(part in name for part in ("command", "/cmd", "setpoint", "actuator")):
        return "ROBOT_COMMAND_AUDIT_ONLY"
    if any(part in name for part in ("joint_state", "feedback", "robot_state", "servo_state")):
        return "HISTORICAL_FEEDBACK_AUDIT_ONLY"
    if any(part in name for part in ("workflow", "lifecycle", "control_phase", "start", "teleop_mode")):
        return "EVENT_CANDIDATE_REQUIRES_DECODE"
    return "DIAGNOSTIC_OR_UNKNOWN"


@dataclass
class TopicSample:
    topic: str
    role: str
    message_encoding: str
    schema_name: str | None
    schema_encoding: str | None
    schema_sha256: str | None
    metadata: dict
    count: int = 0
    log_first_ns: int | None = None
    log_last_ns: int | None = None
    publish_first_ns: int | None = None
    publish_last_ns: int | None = None
    log_backsteps: int = 0
    same_log_time: int = 0


def inventory_messages(records):
    """Read MCAP records in file order, without interpreting payload clocks."""
    channels = {}
    last_log = {}
    for schema, channel, message in records:
        topic = channel.topic
        identity = (
            channel.message_encoding,
            schema.name if schema else None,
            schema.encoding if schema else None,
            sha256(schema.data).hexdigest() if schema else None,
        )
        if channel.id not in channels:
            channels[channel.id] = TopicSample(
                topic, topic_role(topic), *identity, dict(channel.metadata)
            )
        sample = channels[channel.id]
        if sample.topic != topic or identity != (
            sample.message_encoding, sample.schema_name, sample.schema_encoding, sample.schema_sha256
        ):
            raise ValueError(f"Inconsistent declaration for channel {channel.id}")
        prev = last_log.get(channel.id)
        sample.log_backsteps += int(prev is not None and message.log_time < prev)
        sample.same_log_time += int(prev == message.log_time)
        last_log[channel.id] = message.log_time
        sample.count += 1
        for field, stamp in (("log", message.log_time), ("publish", message.publish_time)):
            first, last = f"{field}_first_ns", f"{field}_last_ns"
            before, after = getattr(sample, first), getattr(sample, last)
            setattr(sample, first, stamp if before is None else min(before, stamp))
            setattr(sample, last, stamp if after is None else max(after, stamp))
    failures = []
    for topic, expected_type in HUMAN.items():
        matching = [s for s in channels.values() if s.topic == topic]
        if not matching:
            failures.append(f"Missing human input candidate {topic}")
            continue
        variants = {(s.schema_name, s.schema_encoding, s.schema_sha256, s.message_encoding) for s in matching}
        if len(variants) != 1:
            failures.append(f"Conflicting embedded declarations for {topic}")
        for row in matching:
            if row.message_encoding != "cdr" or not row.schema_name or not row.schema_encoding or not row.schema_sha256:
                failures.append(f"Missing embedded ROS CDR schema for {topic}")
            elif row.schema_name.rsplit("/", 1)[-1] != expected_type:
                failures.append(f"Unexpected schema type for {topic}: {row.schema_name}")
    return {
        "channels": [asdict(s) for s in channels.values()],
        "file_order_message_count": sum(s.count for s in channels.values()),
        "identity_or_schema_findings": failures,
        "event_topics": sorted({s.topic for s in channels.values() if s.role == "EVENT_CANDIDATE_REQUIRES_DECODE"}),
        "start_event_status": "UNKNOWN_PAYLOAD_NOT_DECODED",
        "mode_and_initial_state_status": "UNKNOWN_PAYLOAD_NOT_DECODED",
        "replay_whitelist_approved": False,
        "replay_qualification": "NOT_EVALUATED",
    }


def audit_bag(path):
    from mcap.reader import make_reader

    expected = BAGS[path.name]
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        actual = file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for frozen recording {path.name}")
        stream.seek(0)
        reader = make_reader(stream, validate_crcs=True)
        header = reader.get_header()
        inventory = inventory_messages(reader.iter_messages(log_time_order=False))
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise ValueError(f"Recording changed during inventory: {path.name}")
    return {
        "name": path.name, "path": str(path.resolve()), "sha256": actual,
        "bytes": after.st_size, "mcap_profile": header.profile, "recorder_library": header.library,
        "purpose": "TELEOP_DEPLOYMENT_REGRESSION",
        **inventory,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.bag) != 2 or set(path.name for path in args.bag) != BAGS.keys():
        parser.error("Supply both unique frozen recording filenames using --bag")
    if args.out.exists():
        parser.error("Refusing to overwrite evidence")
    records = []
    for path in args.bag:
        try:
            records.append(audit_bag(path))
        except Exception as exc:
            records.append({"name": path.name, "error": f"{type(exc).__name__}: {exc}", "replay_qualification": "BLOCKED"})
    failed = any(row.get("error") or row.get("identity_or_schema_findings") for row in records)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump({
            "scope": "READ_ONLY_MCAP_IDENTITY_SCHEMA_AND_TIME_INVENTORY",
            "inventory": "BLOCKED" if failed else "CHECKED",
            "replay_qualification": "NOT_EVALUATED",
            "recorded_robot_commands_publishable": False,
            "historical_robot_feedback_publishable": False,
            "bags": records,
        }, out, ensure_ascii=False, indent=2)
        out.write("\n")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
