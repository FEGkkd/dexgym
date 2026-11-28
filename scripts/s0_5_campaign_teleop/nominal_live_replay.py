#!/usr/bin/env python3
"""Nominal-only human input replay into an isolated existing ROS 2 MuJoCo graph."""

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import NamedTuple

import rclpy
import rosbag2_py
from rcl_interfaces.srv import GetParameters
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.utilities import get_message, get_service


HUMAN = {
    "/vive/tracker/left_wrist": "tianji_interfaces/msg/TrackerPose",
    "/vive/tracker/right_wrist": "tianji_interfaces/msg/TrackerPose",
    "/manus_glove_left": "manus_ros2_msgs/msg/ManusGlove",
    "/manus_glove_right": "manus_ros2_msgs/msg/ManusGlove",
}
ACTUATORS = {"/tianji/joint_command", "/left_hand/command", "/right_hand/command"}
FEEDBACK = {
    "/left_arm/joint_states", "/right_arm/joint_states",
    "/left_hand/state", "/right_hand/state", "/tianji/robot_state",
}
WORKFLOW = {"/whole_body_teleop/workflow_state", "/whole_body_teleop/control_phase"}
NS = 1_000_000_000
CONTROLLER = "whole_body_teleop"
SIMULATOR = "whole_body_mujoco_driver"


class Event(NamedTuple):
    schedule_ns: int
    log_ns: int
    record_index: int
    topic: str
    data: bytes


def input_events(paths, seconds):
    events, receipts = [], []
    for path in paths:
        with path.open("rb") as stream:
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        receipts.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256})
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
                    rosbag2_py.ConverterOptions("", ""))
        types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
        topics = HUMAN.keys() & types.keys()
        for topic in topics:
            if types[topic] != HUMAN[topic]:
                raise ValueError(f"Unexpected schema {types[topic]} for {topic}")
        reader.set_filter(rosbag2_py.StorageFilter(topics=sorted(topics | WORKFLOW & types.keys())))
        while reader.has_next():
            topic, data, log_ns = reader.read_next()
            if topic in HUMAN:
                events.append(Event(log_ns, log_ns, len(events), topic, data))
            else:
                msg = deserialize_message(data, get_message(types[topic]))
                receipts[-1].setdefault("historical_workflow", []).append(
                    {"log_ns": log_ns, "topic": topic, "value": message_to_ordereddict(msg)}
                )
    return clip_and_check(events, seconds), receipts


def signed_events(cohort_path, root_id, seconds):
    from mcap.reader import NonSeekingReader
    from mcap_ros2.decoder import DecoderFactory

    receipt = json.loads(cohort_path.read_text())
    if receipt.get("schema") != "s05.teleop.hmb_source_cohort.v1" or not receipt["native_assets_sha256_checked"]:
        raise ValueError("Expected a byte-verified signed HMB source cohort receipt")
    selected = [root for root in receipt["selected_roots"] if root["root"] == root_id]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one signed HMB source for {root_id}")
    root = selected[0]
    start, stop = root["interval_ns"]
    events = []
    if {asset["topic"]: asset["ros_type"] for asset in root["inputs"]} != HUMAN or len(root["inputs"]) != 4:
        raise ValueError(f"Input-role discrepancy for {root_id}")
    for asset in root["inputs"]:
        path = Path(asset["canonical_uri"])
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            if checksum != asset["sha256"] or path.stat().st_size != asset["bytes"]:
                raise ValueError(f"Native asset identity changed: {path}")
            stream.seek(0)
            reader = NonSeekingReader(stream, validate_crcs=True, record_size_limit=32 * 2**20)
            decoder = DecoderFactory()
            raw = None
            count, prev_sequence, prev_mono = 0, -1, -1
            for index, (schema, channel, message) in enumerate(reader.iter_messages(log_time_order=False)):
                is_raw = channel.topic == asset["topic"]
                expected_topic = asset["topic"] if is_raw else asset["observation_topic"]
                expected_sha = asset["schema_sha256"] if is_raw else asset["observation_schema_sha256"]
                if (channel.topic != expected_topic or channel.message_encoding != "cdr"
                        or hashlib.sha256(schema.data).hexdigest() != expected_sha):
                    raise ValueError(f"Unexpected channel or embedded schema: {path}:{index}")
                if is_raw:
                    if raw is not None or schema.name != asset["ros_type"]:
                        raise ValueError(f"Missing adjacent observation or wrong input type: {path}:{index}")
                    raw = message
                    continue
                observed = decoder.decoder_for("cdr", schema)(message.data)
                mono = observed.host_observed_monotonic_ns
                sequence = observed.source_sequence_id
                if (raw is None or raw.log_time != message.log_time
                        or raw.log_time != observed.host_realtime_ns or observed.stream_id != asset["role"]
                        or observed.source_instance_id != asset["independent_raw_witness"]["source_instance"]
                        or observed.episode_sample_index != count or mono <= prev_mono or sequence <= prev_sequence):
                    raise ValueError(f"Native writer/time provenance discrepancy: {path}:{index}")
                if start <= raw.log_time < stop:
                    events.append(Event(mono, raw.log_time, index - 1, asset["topic"], raw.data))
                count += 1
                prev_sequence, prev_mono = sequence, mono
                raw = None
            if raw is not None or not count:
                raise ValueError(f"Unpaired or empty file: {path}")
    return clip_and_check(events, seconds), {
        "root": root, "cohort_receipt_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        "schedule": "paired collector monotonic", "window_ns": [start, stop],
        "start": "synthetic simulator START from first eligible human sample of each input",
    }


def clip_and_check(events, seconds):
    events.sort(key=lambda event: (event.schedule_ns, event.topic, event.record_index))
    if seconds is not None and events:
        origin = events[0].schedule_ns
        events = [event for event in events if event.schedule_ns - origin < seconds * NS]
    if {event.topic for event in events} != HUMAN.keys():
        raise ValueError("Expected all four human source streams in selected window")
    return events


class Replay:
    def __init__(self, node, trace):
        self.node = node
        self.trace = trace
        self.stage = "setup"
        self.origin_mono_ns = None
        self.origin_log_ns = None
        self.session = f"s05-nominal-{uuid.uuid4()}"
        self.messages = {topic: get_message(type_name) for topic, type_name in HUMAN.items()}
        self.subscriptions = []

    def write(self, kind, **payload):
        mono_ns = time.monotonic_ns()
        log_est_ns = (None if self.origin_mono_ns is None else
                      self.origin_log_ns + mono_ns - self.origin_mono_ns)
        record = dict(kind=kind, stage=self.stage, mono_ns=mono_ns, log_est_ns=log_est_ns, **payload)
        self.trace.write(json.dumps(record, separators=(",", ":")) + "\n")

    def service(self, operation, timeout=40):
        path = "/whole_body_teleop/" + operation
        services = dict(self.node.get_service_names_and_types())
        if path not in services:
            raise RuntimeError(f"Missing workflow service {path}: {sorted(services)}")
        service_type = get_service(services[path][0])
        request = service_type.Request()
        if hasattr(request, "session_id"):
            request.session_id = self.session
        client = self.node.create_client(service_type, path)
        if not client.wait_for_service(timeout_sec=3):
            raise RuntimeError(f"Unavailable workflow service {path}")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
        response = future.result() if future.done() else None
        outcome = message_to_ordereddict(response) if response is not None else {}
        self.write("service", service=path, response=outcome)
        print(path, dict(outcome), flush=True)
        self.node.destroy_client(client)
        if response is None or not outcome.get("success", outcome.get("accepted", False)):
            raise RuntimeError(f"Workflow failed {path}: {outcome}")

    def parameters(self, node_name, names):
        client = self.node.create_client(GetParameters, f"/{node_name}/get_parameters")
        if not client.wait_for_service(timeout_sec=3):
            raise RuntimeError(f"Effective parameters unavailable for {node_name}")
        future = client.call_async(GetParameters.Request(names=names))
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=3)
        response = future.result() if future.done() else None
        self.node.destroy_client(client)
        if response is None:
            raise RuntimeError(f"Effective parameter query failed for {node_name}")
        values = dict(zip(names, response.values, strict=True))
        self.write("parameters", node_name=node_name,
                   values={name: message_to_ordereddict(value) for name, value in values.items()})
        return values

    def wait(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=max(0, deadline - time.monotonic()))

    def verify_graph(self, *, expect_replay_writer):
        graph = sorted(self.node.get_node_names_and_namespaces())
        expected = sorted((name, "/") for name in (self.node.get_name(), SIMULATOR, CONTROLLER))
        if graph != expected:
            raise RuntimeError(f"Expected the two isolated simulator nodes, observed {graph}")
        topics = dict(self.node.get_topic_names_and_types())
        for topic in HUMAN:
            publishers = self.node.get_publishers_info_by_topic(topic)
            subscribers = self.node.get_subscriptions_info_by_topic(topic)
            writers = [(info.node_name, info.node_namespace) for info in publishers]
            expected_writers = [(self.node.get_name(), "/")] if expect_replay_writer else []
            if writers != expected_writers or (CONTROLLER, "/") not in {
                (info.node_name, info.node_namespace) for info in subscribers
            }:
                raise RuntimeError(f"Unexpected provenance for human input {topic}")
        for group, writer in ((ACTUATORS, CONTROLLER), (FEEDBACK, SIMULATOR)):
            for topic in group:
                publishers = self.node.get_publishers_info_by_topic(topic)
                if [(info.node_name, info.node_namespace) for info in publishers] != [(writer, "/")]:
                    raise RuntimeError(f"Unexpected provenance for {topic}: {publishers}")
        services = dict(self.node.get_service_names_and_types())
        if any("real_driver" in service for service in services):
            raise RuntimeError("Physical driver service discovered")
        self.write("graph", nodes=graph, topics=topics, services=services)
        return topics

    def verify_and_subscribe(self):
        deadline = time.monotonic() + 9.0
        while True:
            self.wait(0.25)
            try:
                topics = self.verify_graph(expect_replay_writer=False)
                break
            except RuntimeError as error:
                if any(name not in {self.node.get_name(), SIMULATOR, CONTROLLER} or ns != "/"
                       for name, ns in self.node.get_node_names_and_namespaces()):
                    raise
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Isolated graph did not become ready: {error}") from error
        for topic, type_names in topics.items():
            if (topic in ACTUATORS | FEEDBACK | WORKFLOW
                    or "diagnostic_msgs/msg/DiagnosticArray" in type_names
                    or ("end_effector" in topic and "geometry_msgs/msg/PoseStamped" in type_names)):
                def receive(message, name=topic):
                    value = message_to_ordereddict(message)
                    self.write("output", topic=name, value=value)
                    self.observe_output(name, value)
                self.subscriptions.append(self.node.create_subscription(
                    get_message(type_names[0]), topic, receive, qos_profile_sensor_data
                ))
        self.publishers = {
            topic: self.node.create_publisher(msg_type, topic, 16)
            for topic, msg_type in self.messages.items()
        }
        self.wait(0.6)
        if any(pub.get_subscription_count() < 1 for pub in self.publishers.values()):
            raise RuntimeError("Human inputs did not match the controller subscribers")

    def send(self, event, epoch_delta_ns=None):
        msg = deserialize_message(event.data, self.messages[event.topic])
        original = msg.header.stamp.sec * NS + msg.header.stamp.nanosec
        if original:
            adjusted = (self.node.get_clock().now().nanoseconds if epoch_delta_ns is None
                        else original + epoch_delta_ns)
            msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(adjusted, NS)
        self.publishers[event.topic].publish(msg)
        packet_identity = {"cdr_sha256": hashlib.sha256(event.data).hexdigest()} if epoch_delta_ns is not None else {}
        self.write("input", topic=event.topic, original_log_ns=event.log_ns,
                   original_schedule_ns=event.schedule_ns, record_index=event.record_index,
                   original_stamp_ns=original,
                   published_stamp_ns=msg.header.stamp.sec * NS + msg.header.stamp.nanosec, **packet_identity)
        return msg.header.stamp.sec * NS + msg.header.stamp.nanosec

    def observe_output(self, topic, value):
        pass

    def start_inputs(self, events):
        warmup = {}
        for event in events:
            warmup.setdefault(event.topic, event)
            if warmup.keys() == HUMAN.keys():
                break
        return warmup

    def before_history(self, warmup):
        pass

    def run(self, events):
        self.verify_and_subscribe()
        controller = self.parameters(CONTROLLER, [
            "workflow.driver_prefix", "arm.kinematics_backend", "arm.control_pipeline",
            "arm.analytic_diagnostics_full_rate",
        ])
        simulator = self.parameters(SIMULATOR, ["model_path", "publish_rate_hz", "update_rate_hz"])
        if controller["workflow.driver_prefix"].string_value != "/tianji_mujoco_driver":
            raise RuntimeError("Workflow does not bind to the MuJoCo driver")
        if controller["arm.kinematics_backend"].string_value != "marvin":
            raise RuntimeError("Unexpected arm controller backend")
        model = Path(simulator["model_path"].string_value)
        with model.open("rb") as stream:
            self.loaded_model_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        self.write("model", path=str(model.resolve()), sha256=self.loaded_model_sha256)
        self.service("connect")
        self.service("home")
        warmup = self.start_inputs(events)
        self.stage = "synthetic_start_warmup"
        for _ in range(25):
            for event in warmup.values():
                self.send(event)
            self.wait(0.02)
        try:
            self.verify_graph(expect_replay_writer=True)
            self.service("start")
            for event in warmup.values():
                self.send(event)
            self.service("enable")
            self.before_history(warmup)
            self.replay(events, warmup)
        finally:
            self.stage = "tail"
            self.service("stop")

    def replay(self, events, warmup):
        # Set the initial epoch far enough ahead that warmup does not invalidate the first raw tracker stamps.
        first = events[0]
        pad_ns = 30_000_000
        for event in warmup.values():
            if "/tracker/" in event.topic:
                stamp = deserialize_message(event.data, self.messages[event.topic]).header.stamp
                source_ns = stamp.sec * NS + stamp.nanosec
                pad_ns = max(pad_ns, first.log_ns - source_ns + 5_000_000)
        if pad_ns >= 100_000_000:
            raise ValueError(f"Cannot preserve first human timestamps and freshness after START: {pad_ns}ns")
        self.origin_log_ns = first.log_ns
        self.origin_mono_ns = time.monotonic_ns() + pad_ns
        epoch_delta = self.node.get_clock().now().nanoseconds + pad_ns - self.origin_log_ns
        self.wait((self.origin_mono_ns - time.monotonic_ns()) / NS)
        self.stage = "historical_human_inputs"
        self.write("timebase", epoch_delta_ns=epoch_delta, origin_log_ns=self.origin_log_ns,
                   origin_mono_ns=self.origin_mono_ns, origin_schedule_ns=first.schedule_ns)
        for event in events:
            remaining = (self.origin_mono_ns + event.schedule_ns - first.schedule_ns - time.monotonic_ns()) / NS
            self.wait(remaining)
            self.send(event, epoch_delta)
            rclpy.spin_once(self.node, timeout_sec=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--bag", type=Path, nargs="+", help="Explicit diagnostics only; not the signed HMB cohort")
    inputs.add_argument("--cohort", type=Path, help="Byte-verified signed HMB receipt from hmb_cohort.py")
    parser.add_argument("--root", help="Single signed HMB root to replay")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    if (int(os.environ.get("ROS_DOMAIN_ID", "0")) <= 0
            or os.environ.get("ROS_LOCALHOST_ONLY") != "1"
            or os.environ.get("ROS_STATIC_PEERS")
            or os.environ.get("CYCLONEDDS_URI")
            or os.environ.get("FASTDDS_DEFAULT_PROFILES_FILE")
            or os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE")):
        parser.error("Launch in a dedicated nonzero localhost-only ROS domain")
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive")
    if bool(args.cohort) != bool(args.root):
        parser.error("--cohort and --root must be provided together")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.cohort:
        events, receipts = signed_events(args.cohort, args.root, args.seconds)
    else:
        events, receipts = input_events(args.bag, args.seconds)
    (args.output / "input.json").write_text(json.dumps(receipts, indent=2))
    rclpy.init()
    node = rclpy.create_node("s05_nominal_replay")
    with (args.output / "trace.jsonl").open("w", buffering=1) as trace:
        try:
            Replay(node, trace).run(events)
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
