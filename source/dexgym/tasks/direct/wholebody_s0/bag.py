"""Read dex-native-v2 episodes and audit the recorded streams."""

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from hashlib import file_digest
from pathlib import Path
from zoneinfo import ZoneInfo
from zlib import crc32

import numpy as np
import yaml

from . import SIDES

TOPICS = {
    "manus_left": "/manus_glove_left",
    "manus_right": "/manus_glove_right",
    "tracker_left": "/vive/tracker/left_wrist",
    "tracker_right": "/vive/tracker/right_wrist",
    "tracker_left_raw": "/vive/raw/left_wrist",
    "tracker_right_raw": "/vive/raw/right_wrist",
    "tracker_safety": "/vive/tracker/safety_status",
    "arm_left_state": "/left_arm/joint_states",
    "arm_right_state": "/right_arm/joint_states",
    "arm_command": "/tianji/joint_command",
    "hand_left_state": "/left_hand/state",
    "hand_right_state": "/right_hand/state",
    "hand_left_command": "/left_hand/command",
    "hand_right_command": "/right_hand/command",
    "robot_state": "/tianji/robot_state",
    "teleop_state": "/tianji/teleop_state",
}
TOPIC_KEYS = {topic: key for key, topic in TOPICS.items()}
MAIN_STREAMS = ("manus_left", "manus_right", "tracker_left", "tracker_right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
EPOCH_NS = 1_000_000_000
MS_NS = 1_000_000


def digest(path):
    with Path(path).open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def select_episodes(root, day, timezone):
    zone = ZoneInfo(timezone)
    episodes = []
    for path in sorted(root.glob("*/manifest.yaml")):
        manifest = load_yaml(path)
        start_ns = manifest.get("start_realtime_ns")
        if not start_ns:
            continue
        stamp = datetime.fromtimestamp(start_ns // EPOCH_NS, zone)
        if stamp.date() == day:
            if manifest.get("episode_id") != path.parent.name:
                raise ValueError(f"Episode identifier does not match its directory: {path}")
            episodes.append((path.parent, manifest, stamp.isoformat()))
    return sorted(episodes, key=lambda x: x[1]["start_realtime_ns"])


def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return dict(zip(
        ("min", "p50", "p95", "p99", "max"),
        np.percentile(values, [0, 50, 95, 99, 100]).tolist(),
    ))


def timing(stamps_ns):
    stamps = np.asarray(stamps_ns, dtype=np.int64)
    stamps = stamps[stamps > 0]
    dt = np.diff(stamps)
    positive = dt[dt > 0]
    period = float(np.median(positive)) if len(positive) else None
    gaps = dt > 1.8 * period if period else np.zeros(len(dt), dtype=bool)
    elapsed = (int(stamps[-1]) - int(stamps[0])) / EPOCH_NS if len(stamps) > 1 else 0
    return {
        "observed_hz": (len(stamps) - 1) / elapsed if elapsed > 0 else None,
        "median_interval_hz": EPOCH_NS / period if period else None,
        "interval_ms": stats(positive / MS_NS),
        "duplicate_stamp": int(np.count_nonzero(dt == 0)),
        "backward_stamp": int(np.count_nonzero(dt < 0)),
        "gap_count": int(gaps.sum()),
        "gap_estimated_samples": int(np.sum(np.rint(dt[gaps] / period) - 1)) if period else None,
    }


def stamp_ns(stamp):
    return int(stamp.sec) * EPOCH_NS + int(stamp.nanosec)


def xyz(point):
    return point.x, point.y, point.z


def quaternion(rotation):
    return rotation.x, rotation.y, rotation.z, rotation.w


def pose(message):
    return (*xyz(message.position), *quaternion(message.orientation))


def hand_topology(nodes):
    return tuple((n.node_id, n.parent_node_id, n.chain_type, n.joint_type) for n in nodes)


def manus_keypoint_order(topology):
    """Match RetargetNode::build_keypoint_map in whole_body_teleop."""
    semantic = {(chain.lower(), joint.upper()): i for i, (_, _, chain, joint) in enumerate(topology)}
    if len(semantic) != len(topology):
        raise ValueError("Duplicate MANUS semantic node")
    wrist = semantic.get(("hand", "INVALID"), semantic.get(("wrist", "INVALID")))
    if wrist is None:
        raise ValueError("MANUS skeleton has no wrist")
    order = [wrist]
    for finger in FINGERS:
        if finger != "thumb" and (finger, "IP") in semantic:
            joints = ("PIP", "IP", "DIP", "TIP")
        else:
            joints = ("MCP", "PIP", "DIP", "TIP")
        order.extend(semantic[finger, joint] for joint in joints)
    ids = {row[0] for row in topology}
    if len(ids) != len(topology) or any(parent not in ids for _, parent, _, _ in topology):
        raise ValueError("MANUS skeleton has invalid parent IDs")
    return np.asarray(order, dtype=np.int64)


@dataclass
class Track:
    arrays: dict
    info: dict

    @property
    def header(self):
        return self.arrays["header_ns"]

    @property
    def valid(self):
        return self.arrays["valid"]


class Samples:
    """Extract only fields used for QA, offline alignment and reference generation."""

    def __init__(self, key, expected_frame):
        self.key = key
        self.expected_frame = expected_frame
        self.rows = defaultdict(list)
        self.topology = None
        self.names = None
        self.ergonomics_names = None
        self.raw_sensor_width = None
        self.invalid = Counter()

    def add(self, msg, log_ns, publish_ns, sequence):
        key = self.key
        data = {}
        problem = ""
        header = getattr(msg, "header", None)
        data["header_ns"] = stamp_ns(header.stamp) if header is not None else 0
        data["frame_id"] = header.frame_id if header is not None else ""
        if key.startswith("manus_"):
            side = key.split("_")[1]
            nodes = sorted(msg.raw_nodes, key=lambda node: node.node_id)
            topology = hand_topology(nodes)
            if self.topology is None and len(nodes):
                try:
                    manus_keypoint_order(topology)
                    self.topology = topology
                except (ValueError, KeyError):
                    pass
            compatible = topology == self.topology and len(nodes) == msg.raw_node_count
            ergo = tuple(item.type for item in msg.ergonomics)
            sensors = [pose(sensor) for sensor in msg.raw_sensor]
            if self.ergonomics_names is None and ergo and len(ergo) == len(set(ergo)):
                self.ergonomics_names = ergo
            if self.raw_sensor_width is None and sensors:
                self.raw_sensor_width = len(sensors)
            ergo_ok = ergo == self.ergonomics_names and len(ergo) == msg.ergonomics_count
            sensor_ok = len(sensors) == self.raw_sensor_width == msg.raw_sensor_count
            data.update(
                pose=[pose(node.pose) for node in nodes] if compatible else None,
                source_stamp=msg.source_timestamp_us,
                sdk_receive_us=msg.receive_timestamp_us,
                glove_id=msg.glove_id,
                ergonomics_count=msg.ergonomics_count,
                ergonomics_values=[item.value for item in msg.ergonomics] if ergo_ok else None,
                raw_sensor_count=msg.raw_sensor_count,
                raw_sensor_pose=sensors if sensor_ok else None,
                raw_sensor_orientation=quaternion(msg.raw_sensor_orientation),
                auxiliary_valid=(
                    ergo_ok and sensor_ok
                    and all(np.isfinite(item.value) for item in msg.ergonomics)
                    and np.isfinite(quaternion(msg.raw_sensor_orientation)).all()
                    and np.isfinite(sensors).all()
                ),
            )
            if not compatible:
                problem = "skeleton_topology"
            elif msg.side.lower() != side:
                problem = "wrong_side"
            elif not msg.source_timestamp_us:
                problem = "missing_source_stamp"
        elif key == "tracker_safety":
            data.update(safe=msg.safe, safety_code=msg.code)
            if not msg.safe:
                problem = msg.code or "tracker_unsafe"
        elif key.startswith("tracker_"):
            data["pose"] = pose(msg.pose)
            if header.frame_id != self.expected_frame:
                problem = "frame_id"
            if not key.endswith("_raw"):
                data["quality"] = msg.quality
                expected_role = f"{key.split('_')[1]}_wrist"
                if not msg.valid or not np.isfinite(msg.quality) or msg.quality <= 0:
                    problem = "invalid_tracker"
                elif msg.role != expected_role:
                    problem = "wrong_role"
        elif key == "robot_state":
            data["state"] = msg.hardware_state_name
            if not (msg.connected and msg.feedback_valid):
                problem = "robot_feedback_invalid"
        elif key == "teleop_state":
            data["state"] = str(msg.state)
        else:
            data, problem = self.joints(msg, data)
        if data["header_ns"] <= 0:
            problem = problem or "missing_header_stamp"
        self.invalid[problem] += bool(problem)
        for name, value in data.items():
            self.rows[name].append(value)
        self.rows["log_ns"].append(log_ns)
        self.rows["publish_ns"].append(publish_ns)
        self.rows["mcap_sequence"].append(sequence)
        self.rows["valid"].append(not problem)

    def joints(self, msg, data):
        names = tuple(getattr(msg, "joint_names", getattr(msg, "name", ())))
        positions = np.asarray(getattr(msg, "positions", getattr(msg, "position", ())), dtype=float)
        velocity = np.asarray(getattr(msg, "velocity", ()), dtype=float)
        problem = ""
        if self.names is None and names and len(names) == len(set(names)) and all(names):
            self.names = names
        width = len(self.names or ())
        if self.names is None or set(names) != set(self.names) or len(positions) != len(names):
            problem = "joint_schema"
            position_out = velocity_out = None
        else:
            order = [names.index(name) for name in self.names]
            position_out = positions[order]
            velocity_out = velocity[order] if velocity.shape == (width,) else None
            if not np.isfinite(position_out).all() or np.max(np.abs(position_out)) > 2 * np.pi:
                problem = "joint_position_value"
            if self.key.startswith("hand_") and self.key.endswith("_state"):
                if not (msg.connected and all(msg.position_valid) and not any(msg.stale)):
                    problem = problem or "joint_feedback_invalid"
                if velocity_out is not None:
                    velocity_out = np.where(np.asarray(msg.velocity_valid)[order], velocity_out, np.nan)
        data["position"] = position_out
        data["velocity"] = velocity_out
        data["sequence"] = getattr(msg, "sequence", getattr(msg, "source_sequence", 0))
        data["source_stamp"] = getattr(msg, "source_timestamp_us", 0) if getattr(msg, "source_timestamp_valid", True) else 0
        if hasattr(msg, "received_stamp"):
            data["feedback_receive_ns"] = stamp_ns(msg.received_stamp)
        if hasattr(msg, "valid_until"):
            data["valid_until_ns"] = stamp_ns(msg.valid_until)
            if data["valid_until_ns"] and data["valid_until_ns"] < data["header_ns"]:
                problem = problem or "expired_command_at_publish"
        return data, problem

    def arrays(self):
        arrays = {}
        for name, values in self.rows.items():
            if any(value is None for value in values):
                exemplar = next((v for v in values if v is not None), None)
                if exemplar is None:
                    continue
                fill = np.full(np.shape(exemplar), np.nan)
                values = [v if v is not None else fill for v in values]
            arrays[name] = np.asarray(values)
        return arrays


def vector_checks(key, arrays, topology):
    result = {}
    valid = arrays["valid"]
    if "pose" not in arrays:
        return result
    poses = arrays["pose"]
    flat = poses.reshape(len(poses), -1, 7)
    finite = np.isfinite(flat).all(axis=(1, 2))
    norm = np.linalg.norm(flat[..., 3:], axis=-1)
    norm_error = np.max(np.abs(norm - 1), axis=1)
    valid &= finite & (norm_error <= 0.02)
    result.update(nonfinite_pose_frames=int((~finite).sum()), quat_norm_outliers=int((norm_error > 0.02).sum()))
    result["quaternion_norm_error"] = stats(norm_error)
    if key.startswith("tracker_"):
        p = poses[:, :3]
        q = poses[:, 3:]
        excessive_position = np.max(np.abs(p), axis=1) > 10
        result["position_range_outliers"] = int(excessive_position.sum())
        valid &= ~excessive_position
        dt = np.diff(arrays["header_ns"]) / EPOCH_NS
        pair = valid[1:] & valid[:-1] & (dt > 0) & (dt < 0.1)
        step_speed = np.full(len(dt), np.nan)
        angular_speed = np.full(len(dt), np.nan)
        step_speed[pair] = np.linalg.norm(np.diff(p, axis=0)[pair], axis=1) / dt[pair]
        unit_q = q / np.where(norm[:, 0, None] > 0, norm[:, 0, None], np.nan)
        dot = np.abs(np.sum(unit_q[1:] * unit_q[:-1], axis=1))
        angular_speed[pair] = 2 * np.arccos(np.clip(dot[pair], 0, 1)) / dt[pair]
        jumps = (step_speed > 8.0) | (angular_speed > 35.0)
        valid[1:] &= ~jumps
        result.update(position_speed_m_s=stats(step_speed), angular_speed_rad_s=stats(angular_speed),
                      jump_frames=int(jumps.sum()))
    elif topology:
        by_id = {row[0]: i for i, row in enumerate(topology)}
        edges = [(i, by_id[row[1]]) for i, row in enumerate(topology) if row[0] != row[1]]
        lengths = np.linalg.norm(poses[:, [a for a, _ in edges], :3] - poses[:, [b for _, b in edges], :3], axis=-1)
        if valid.any():
            median = np.median(lengths[valid], axis=0)
            changed = (np.abs(lengths - median) > np.maximum(0.008, 0.15 * median)).any(axis=1)
            valid &= ~changed
            result["bone_length_outliers"] = int(changed.sum())
    return result


def read_stream(folder, stream_id, spec, key, expected_frame):
    from mcap.reader import NonSeekingReader, make_reader
    from mcap.records import Chunk, DataEnd, Footer
    from mcap.stream_reader import StreamReader
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    root = (folder / spec["path"]).resolve()
    if not root.is_relative_to(folder.resolve()):
        raise ValueError(f"Stream path leaves episode: {root}")
    metadata = load_yaml(root / "metadata.yaml")["rosbag2_bagfile_information"]
    samples = Samples(key, expected_frame)
    by_time = defaultdict(deque)
    files = []
    counts = Counter()
    topic, observation = spec["topic"], spec["observation_topic"]
    types = {}
    for relative in metadata["relative_file_paths"]:
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"MCAP path leaves stream: {path}")
        file_counts = Counter()
        with path.open("rb") as file:
            summary = make_reader(file, validate_crcs=True).get_summary()
            if summary is None or summary.statistics is None:
                raise ValueError(f"Missing MCAP summary: {path}")
            file.seek(0)
            reader = NonSeekingReader(file, validate_crcs=True)
            for schema, channel, record in reader.iter_messages(log_time_order=False):
                name = channel.topic
                file_counts[name] += 1
                if name not in (topic, observation):
                    continue
                expected_type = spec["type"] if name == topic else "dex_collection_interfaces/msg/SampleObservation"
                if schema is None or schema.name != expected_type or channel.message_encoding != "cdr":
                    raise ValueError(f"Unexpected CDR schema for {name}: {schema}")
                if expected_type not in types:
                    types[expected_type] = get_message(expected_type)
                msg = deserialize_message(record.data, types[expected_type])
                if name == topic:
                    samples.add(msg, record.log_time, record.publish_time, record.sequence)
                else:
                    if msg.stream_id != stream_id or msg.host_realtime_ns != record.log_time:
                        raise ValueError(f"Misidentified collector observation in {path}")
                    by_time[record.log_time].append((
                        msg.episode_sample_index, msg.source_sequence_id, msg.host_observed_monotonic_ns,
                        msg.host_realtime_ns, msg.receive_ros_timestamp_ns, msg.source_instance_id,
                    ))
            actual = Counter()
            for channel_id, count in summary.statistics.channel_message_counts.items():
                actual[summary.channels[channel_id].topic] += count
            if file_counts != actual or sum(file_counts.values()) != summary.statistics.message_count:
                raise ValueError(f"MCAP summary counts disagree with decoded records: {path}")
            file.seek(0)
            chunk_crcs = 0
            data_crc, summary_crc = False, False
            for record in StreamReader(file, emit_chunks=True).records:
                if isinstance(record, Chunk):
                    chunk_crcs += bool(record.uncompressed_crc)
                elif isinstance(record, DataEnd):
                    data_crc = bool(record.data_section_crc)
                elif isinstance(record, Footer) and record.summary_crc:
                    # The checksum covers the summary and Footer through summary_offset_start.
                    length = path.stat().st_size - 8 - 4 - record.summary_start
                    file.seek(record.summary_start)
                    if length <= 0 or crc32(file.read(length)) != record.summary_crc:
                        raise ValueError(f"MCAP summary CRC mismatch: {path}")
                    summary_crc = True
                    break
        counts.update(file_counts)
        files.append({
            "path": str(path.relative_to(folder.resolve())),
            "sha256": digest(path),
            "bytes": path.stat().st_size,
            "records": sum(file_counts.values()),
            "indexed_chunks": len(summary.chunk_indexes),
            "verified_chunk_crcs": chunk_crcs,
            "verified_data_crc": data_crc,
            "verified_summary_crc": summary_crc,
        })
    expected = {item["topic_metadata"]["name"]: item["message_count"] for item in metadata["topics_with_message_count"]}
    if any(counts[name] != count for name, count in expected.items()) or counts[topic] != spec["written"]:
        raise ValueError(f"Manifest/ROS metadata counts disagree with MCAP: {root}")
    arrays = samples.arrays()
    n = counts[topic]
    if n == 0:
        return Track(arrays, {"topic": topic, "status": "empty", "count": 0, "files": files})
    if counts[observation] != n:
        raise ValueError(f"Payload/observation count mismatch: {root}")
    paired = [by_time[int(log_ns)].popleft() if by_time[int(log_ns)] else None for log_ns in arrays["log_ns"]]
    if any(row is None for row in paired) or any(by_time.values()):
        raise ValueError(f"Unmatched payload/observation timestamps: {root}")
    for index, name in enumerate(("episode_index", "adapter_sequence", "monotonic_ns", "realtime_ns", "receive_ros_ns", "source_instance")):
        arrays[name] = np.asarray([row[index] for row in paired])
    if not np.array_equal(arrays["episode_index"], np.arange(n)):
        raise ValueError(f"Noncontiguous or misordered collector episode indexes: {root}")
    if any(s != spec["source_instance_id"] for s in arrays["source_instance"]):
        raise ValueError(f"Collector source instance differs from manifest: {root}")
    arrays["valid"] &= np.r_[True, np.diff(arrays["header_ns"]) > 0]
    anomaly = vector_checks(key, arrays, samples.topology)
    source_stamp = arrays.get("source_stamp", np.zeros(n, dtype=np.int64)).astype(np.int64)
    source_diff = np.diff(source_stamp)
    source_pairs = (source_stamp[1:] > 0) & (source_stamp[:-1] > 0)
    duplicate_source = np.r_[False, source_pairs & (source_diff == 0)]
    source_regression = np.r_[False, source_pairs & (source_diff < 0)]
    arrays["duplicate_source"] = duplicate_source
    if key.startswith("manus_"):
        arrays["valid"] &= ~source_regression
    header = arrays["header_ns"]
    latency = (arrays["receive_ros_ns"][header > 0] - header[header > 0]) / MS_NS
    info = {
        "status": "recorded", "topic": topic, "message_type": spec["type"], "count": n, "files": files,
        "manifest_written": spec["written"], "collector_dropped": int(spec.get("dropped", 0)),
        "collector_write_errors": int(spec.get("write_errors", 0)),
        "collector_unwritten": int(spec.get("unwritten", 0)),
        "adapter_sequence_gaps": int(np.maximum(np.diff(arrays["adapter_sequence"].astype(np.int64)) - 1, 0).sum()),
        "adapter_sequence_backward_or_duplicate": int(np.count_nonzero(np.diff(arrays["adapter_sequence"].astype(np.int64)) <= 0)),
        "device_sequence_available": bool(spec.get("source_sequence_available")),
        "header_clock": timing(header), "arrival_clock": timing(arrays["monotonic_ns"]),
        "first_header_ns": int(header[0]), "last_header_ns": int(header[-1]),
        "header_to_receive_ms": stats(latency),
        "missing_header": int(np.count_nonzero(header <= 0)),
        "publish_log_differ": int(np.count_nonzero(arrays["publish_ns"] != arrays["log_ns"])),
        "host_clock_offset_change_ms": float(np.ptp(arrays["realtime_ns"] - arrays["monotonic_ns"]) / MS_NS),
        "source_stamps": int(np.count_nonzero(source_stamp)),
        "source_duplicates": int(duplicate_source.sum()), "source_backwards": int(source_regression.sum()),
        "unique_source_rate_hz": timing(header[arrays["valid"] & ~duplicate_source])["observed_hz"] if source_stamp.any() else None,
        "invalid_frames": int(np.count_nonzero(~arrays["valid"])),
        "invalid_reasons": {k: v for k, v in samples.invalid.items() if k},
        "frames": dict(Counter(arrays["frame_id"])),
        "anomalies": anomaly,
    }
    if samples.topology:
        info["topology"] = samples.topology
        info["ergonomics_names"] = samples.ergonomics_names
        info["glove_ids"] = [int(x) for x in np.unique(arrays["glove_id"])]
        info["auxiliary_invalid_frames"] = int(np.count_nonzero(~arrays["auxiliary_valid"]))
        info["sdk_callback_to_observation_ms"] = stats((arrays["monotonic_ns"] - 1000 * arrays["sdk_receive_us"].astype(np.int64)) / MS_NS)
    if samples.names:
        info["joint_names"] = samples.names
    if "safety_code" in arrays:
        info["safety_codes"] = dict(Counter(arrays["safety_code"]))
    if "state" in arrays:
        info["states"] = dict(Counter(arrays["state"]))
    return Track(arrays, info)


def read_episode(folder, manifest, expected_frame):
    if manifest.get("format") != "dex-native-v2":
        raise ValueError(f"Unsupported recorder format in {folder}")
    audit = {key: {"status": "not_recorded", "topic": topic, "count": 0} for key, topic in TOPICS.items()}
    tracks = {}
    for stream_id, spec in manifest["streams"].items():
        key = TOPIC_KEYS.get(spec["topic"])
        if key is None and stream_id in TOPICS:
            key = stream_id
        if key is None or not spec.get("recorded"):
            continue
        try:
            track = read_stream(folder, stream_id, spec, key, expected_frame)
            audit[key] = track.info
            if track.info["status"] == "recorded":
                clock = track.arrays["monotonic_ns"]
                outside = (clock < manifest["start_monotonic_ns"]) | (clock >= manifest["stop_monotonic_ns"])
                track.info["outside_admission_interval"] = int(outside.sum())
                track.info["capture_start_to_first_ms"] = (int(clock.min()) - manifest["start_monotonic_ns"]) / MS_NS
                track.info["capture_last_to_stop_ms"] = (manifest["stop_monotonic_ns"] - int(clock.max())) / MS_NS
                track.valid[outside] = False
                track.info["invalid_frames"] = int(np.count_nonzero(~track.valid))
                tracks[key] = track
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
            audit[key] = {"status": "read_error", "topic": spec["topic"], "count": 0, "error": str(exc)}
        except Exception as exc:
            from mcap.exceptions import McapError

            if not isinstance(exc, (McapError, EOFError)):
                raise
            audit[key] = {"status": "read_error", "topic": spec["topic"], "count": 0, "error": str(exc)}
    return tracks, audit
