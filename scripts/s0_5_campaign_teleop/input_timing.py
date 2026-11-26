"""Build a deterministic human-input timing witness without starting a ROS publisher."""

import argparse
from dataclasses import dataclass
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import re

from preflight import BAGS, HUMAN


DIGEST = re.compile(r"[0-9a-f]{64}")
CLOCK_TAG = "REVIEWED_RECORDED_LOG_TIME"
TELEOP_REVISION = "6d0b0b5b2601618f1b0052d93ef2403eea9b064a"
MAX_NS = (1 << 64) - 1
ROS_TYPES = {
    "/manus_glove_left": "manus_ros2_msgs/msg/ManusGlove",
    "/manus_glove_right": "manus_ros2_msgs/msg/ManusGlove",
    "/vive/tracker/left_wrist": "tianji_interfaces/msg/TrackerPose",
    "/vive/tracker/right_wrist": "tianji_interfaces/msg/TrackerPose",
}


@dataclass(frozen=True)
class Source:
    channel_id: int
    schema_name: str
    schema_encoding: str
    schema_sha256: str


@dataclass(frozen=True)
class InputTiming:
    record_index: int
    topic: str
    recorded_log_ns: int
    relative_deadline_ns: int
    payload_sha256: str


def valid_digest(value):
    return isinstance(value, str) and DIGEST.fullmatch(value) is not None


def resolve_review(review, bag_sha):
    if not isinstance(review, dict) or set(review) != {
        "schema", "teleop_revision", "bag_sha256", "clock", "clock_evidence_sha256", "tracker_basis", "channels",
    }:
        raise ValueError("Unexpected or missing review identity field")
    bindings = (
        review["schema"] == "s05.teleop_timing_witness.review.v1",
        review["teleop_revision"] == TELEOP_REVISION,
        valid_digest(bag_sha) and review["bag_sha256"] == bag_sha,
        review["clock"] == CLOCK_TAG and valid_digest(review["clock_evidence_sha256"]),
        review["tracker_basis"] in ("RAW_VIVE", "REGISTERED_INPUT"),
    )
    if not all(bindings):
        raise ValueError("Review identity, clock qualification, or tracker basis does not match")
    rows = review["channels"]
    if not isinstance(rows, dict) or rows.keys() != HUMAN.keys():
        raise ValueError("The review must select exactly the four human input topics")
    channels = {}
    ids = set()
    for topic, raw in rows.items():
        if not isinstance(raw, dict) or set(raw) != set(Source.__dataclass_fields__):
            raise ValueError(f"Absent or unexpected channel fields for {topic}")
        source = Source(**raw)
        if (
            type(source.channel_id) is not int or not 0 < source.channel_id < 65536
            or source.channel_id in ids
            or source.schema_name != ROS_TYPES[topic]
            or source.schema_encoding != "ros2msg"
            or not valid_digest(source.schema_sha256)
        ):
            raise ValueError(f"Invalid or ambiguous source declaration for {topic}")
        channels[topic] = source
        ids.add(source.channel_id)
    return channels


def iter_input_timing(records, channels):
    """Retain exactly reviewed human streams in causal log-time and file order."""
    previous = origin = None
    counts = dict.fromkeys(HUMAN, 0)
    schema_identity = {}
    for index, (schema, channel, message) in enumerate(records):
        topic = channel.topic
        if topic not in HUMAN:
            continue
        source = channels[topic]
        if channel.id != source.channel_id or schema is None or channel.message_encoding != "cdr":
            raise ValueError(f"Unreviewed channel or CDR encoding for {topic}")
        if schema.id not in schema_identity:
            schema_identity[schema.id] = (
                schema.name, schema.encoding, schema.data, sha256(schema.data).hexdigest()
            )
        name, encoding, data, schema_sha = schema_identity[schema.id]
        if (
            (schema.name, schema.encoding, schema.data) != (name, encoding, data)
            or (name, encoding, schema_sha) != (source.schema_name, source.schema_encoding, source.schema_sha256)
        ):
            raise ValueError(f"Embedded declaration changed for {topic}")
        stamp = message.log_time
        if type(stamp) is not int or not 0 <= stamp <= MAX_NS or (previous is not None and stamp < previous):
            raise ValueError(f"Noncausal selected log time at record index {index}")
        if origin is None:
            origin = stamp
        previous = stamp
        counts[topic] += 1
        yield InputTiming(index, topic, stamp, stamp - origin, sha256(message.data).hexdigest())
    if not all(counts.values()):
        absent = ", ".join(topic for topic, count in counts.items() if not count)
        raise ValueError(f"Selected human input absent: {absent}")


def timing_witness(bag, review_path, expected_review_sha):
    from mcap.reader import make_reader

    review_bytes = review_path.read_bytes()
    review_sha = sha256(review_bytes).hexdigest()
    if review_sha != expected_review_sha:
        raise ValueError("Timing review bytes do not match the externally supplied SHA-256")
    review = json.loads(review_bytes)
    bag_sha = BAGS[bag.name]
    channels = resolve_review(review, bag_sha)
    digest = sha256()
    counts = dict.fromkeys(HUMAN, 0)
    origin = last_deadline = None
    with bag.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if file_digest(handle, "sha256").hexdigest() != bag_sha:
            raise ValueError(f"SHA-256 mismatch for {bag.name}")
        handle.seek(0)
        raw = make_reader(handle, validate_crcs=True).iter_messages(log_time_order=False)
        for entry in iter_input_timing(raw, channels):
            digest.update(
                json.dumps(
                    [entry.record_index, entry.topic, entry.recorded_log_ns, entry.relative_deadline_ns, entry.payload_sha256],
                    separators=(",", ":"),
                ).encode("ascii") + b"\n"
            )
            counts[entry.topic] += 1
            if origin is None:
                origin = entry.recorded_log_ns
            last_deadline = entry.relative_deadline_ns
        after = os.fstat(handle.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size, after.st_mtime_ns, after.st_ctime_ns
    ):
        raise ValueError("Recording changed during the timing witness")
    return {
        "scope": "OFFLINE_HUMAN_INPUT_TIMING_WITNESS_ONLY",
        "teleop_revision": TELEOP_REVISION,
        "bag": bag.name,
        "bag_sha256": bag_sha,
        "review_sha256": review_sha,
        "tracker_basis": review["tracker_basis"],
        "selected_counts": counts,
        "selected_input_timing_sha256": digest.hexdigest(),
        "log_time_origin_ns": origin,
        "last_relative_deadline_ns": last_deadline,
        "event_delivery": "NOT_RUN",
        "replay_qualification": "NOT_EVALUATED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.bag.name not in BAGS or not valid_digest(args.review_sha256):
        parser.error("Use one frozen regression filename and a lowercase SHA-256 review pin")
    if args.out.exists():
        parser.error("Refusing to overwrite evidence")
    try:
        result = timing_witness(args.bag, args.review, args.review_sha256)
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(2, f"Timing witness blocked: {exc}\n")
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
