"""Compare immediate replay and a measured START pre-position in the isolated native simulator."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from nominal_live_replay import HUMAN, NS, Replay, deserialize_message, rclpy, signed_events
from summarize_nominal_replay import ACTIVE, DIAGNOSTICS, FEEDBACK, joint_values, stamp


HANDS = {"left": "/left_hand/state", "right": "/right_hand/state"}
FRESH_NS = 40_000_000
RAMP_SEC = 1.5
DWELL_SEC = 0.2
TIMEOUT_SEC = 10.0
WRIST_M = 0.010
WRIST_DEG = 5.0
HAND_RAD = 0.10
SETTLED_RAD_S = 0.16
HOME_RAD = np.deg2rad(1.0)
MEASUREMENT_JOIN_RAD = 0.01


def array(fields, key):
    return np.fromstring(fields[key], sep=",")


def pose(fields, prefix):
    result = np.eye(4)
    result[:3, 3] = array(fields, prefix + "_xyz_m")
    result[:3, :3] = Rotation.from_quat(array(fields, prefix + "_quat_xyzw")).as_matrix()
    return result


def pose_error(goal, observed):
    return [float(np.linalg.norm(goal[:3, 3] - observed[:3, 3])),
            float(np.rad2deg(Rotation.from_matrix(goal[:3, :3].T @ observed[:3, :3]).magnitude()))]


def shared_targets(frames, diagnostic):
    """Reconstruct the unprojected endpoint from the native START latch and calibrated frames."""
    registration, left_base, right_base, left_mount, right_mount, left_tool, right_tool = frames
    bases, mounts, tools = (left_base, right_base), (left_mount, right_mount), (left_tool, right_tool)
    home, human = [], []
    for side, base, mount, tool in zip(HANDS, bases, mounts, tools, strict=True):
        fields = diagnostic[side]
        home.append(base @ pose(fields, "robot_start") @ np.linalg.inv(tool))
        human.append(pose(fields, "tracker_start") @ mount)
    shared = registration.copy()
    shared[:3, 3] += (np.mean(home, axis=0) - registration @ np.mean(human, axis=0))[:3, 3]
    target = [np.linalg.inv(base) @ shared @ wrist @ tool
              for base, wrist, tool in zip(bases, human, tools, strict=True)]
    return shared, target, home, [shared @ wrist for wrist in human]


class HandoffReplay(Replay):
    def __init__(self, node, trace, anchor, frames, hold, diagnostic_after_miss, home_proof):
        super().__init__(node, trace)
        self.anchor, self.frames, self.hold = anchor, frames, hold
        self.diagnostic_after_miss = diagnostic_after_miss
        self.phases, self.diagnostics, self.hands = {}, {}, {}
        self.pending = defaultdict(dict)
        self.physical = None
        self.last_synthetic = {}
        self.home_q = None
        self.home_proof = home_proof

    def observe_output(self, topic, value):
        if self.stage == "historical_human_inputs" or self.stage == "tail":
            return
        header = stamp(value.get("header", {}).get("stamp"))
        if topic in ACTIVE:
            self.phases[topic] = value
        if header is None:
            return
        if topic in FEEDBACK:
            q = joint_values(value, FEEDBACK[topic])
            if topic in HANDS.values() and (value.get("connected") is not True
                    or value.get("position_valid") != [True] * 20
                    or value.get("online") != [True] * 20 or value.get("stale") != [False] * 20):
                q = None
            group = self.pending[header]
            group[topic] = q if topic not in group else None
            if group.keys() == FEEDBACK.keys() and all(item is not None for item in group.values()):
                angles = {key: x for item in group.values() for key, x in item.items()}
                previous = self.physical
                speed = None
                if previous is not None and 0 < header - previous[0] <= FRESH_NS and angles.keys() == previous[1].keys():
                    speed = max(abs(x - previous[1][key]) for key, x in angles.items()) * NS / (header - previous[0])
                if len(angles) == 54 and (previous is None or header > previous[0]):
                    self.physical = header, angles, speed
            for stale in [key for key in self.pending if key < header - FRESH_NS]:
                del self.pending[stale]
        for side in HANDS:
            if topic == f"/{side}_hand/command" and value.get("side") == side:
                self.hands[side] = header, joint_values(value, 20)
        for status in value.get("status", ()):
            side = DIAGNOSTICS.get(status["name"])
            if side:
                fields = {entry["key"]: entry["value"] for entry in status["values"]}
                if fields.get("mapping_policy") == "shared_rigid_diagnostic_v1" and fields.get("session") == self.session:
                    self.diagnostics[side] = header, fields

    def send(self, event, epoch_delta_ns=None):
        header = super().send(event, epoch_delta_ns)
        if epoch_delta_ns is None:
            self.last_synthetic[event.topic] = header
        return header

    def start_inputs(self, events):
        home_service_done = self.node.get_clock().now().nanoseconds
        self.wait(0.15)
        now = self.node.get_clock().now().nanoseconds
        if (self.physical is None or not 0 <= now - self.physical[0] <= FRESH_NS
                or self.physical[0] < home_service_done or self.physical[2] is None
                or self.physical[2] > SETTLED_RAD_S):
            raise RuntimeError("Home lacks fresh, settled 54-axis MuJoCo feedback after the service")
        self.home_q = self.physical[1].copy()
        home_error = None
        if self.home_proof is not None:
            reference = self.home_proof["named_q_rad"]
            if (self.loaded_model_sha256 != self.home_proof["model_sha256"] or self.home_q.keys() != reference.keys()
                    or max(self.home_proof["palm_normal_angle_to_world_down_deg"]) > 5.0):
                raise RuntimeError("Home reference and loaded simulator do not share the verified palms-down frame")
            home_error = max(abs(q - reference[name]) for name, q in self.home_q.items())
            if home_error > HOME_RAD:
                raise RuntimeError(f"Fresh measured Home differs from the palms-down reference: {home_error:.6f} rad")
        self.write("home_feedback", named_q_rad=self.home_q, measured_header_ns=self.physical[0],
                   max_measured_speed_rad_s=self.physical[2], max_error_to_home_proof_rad=home_error)
        witnessed = self.anchor["inputs"]
        selected = {event.topic: event for event in events
                    if event.record_index == witnessed[event.topic]["record_index"]}
        if selected.keys() != HUMAN.keys() or any(
            sha256(event.data).hexdigest() != witnessed[topic]["cdr_sha256"]
            or event.schedule_ns > self.anchor["source_cutoff_ns"] for topic, event in selected.items()
        ):
            raise ValueError("Frozen START samples do not match the native causal records")
        self.write("start_anchor", source_cutoff_ns=self.anchor["source_cutoff_ns"], inputs=witnessed,
                   preanchor_input_messages=sum(event.schedule_ns < self.anchor["source_cutoff_ns"] for event in events))
        return selected

    def measure(self):
        now = self.node.get_clock().now().nanoseconds
        if (self.physical is None or not 0 <= now - self.physical[0] <= FRESH_NS
                or self.diagnostics.keys() != HANDS.keys()
                or any(not 0 <= now - item[0] <= FRESH_NS for item in self.diagnostics.values())):
            return None
        diag = {side: self.diagnostics[side][1] for side in HANDS}
        if any(fields.get("measured_fk_available") != "true" for fields in diag.values()):
            return None
        shared, goal, home, wrists = shared_targets(self.frames, diag)
        errors = [pose_error(target, pose(diag[side], "measured_fk"))
                  for side, target in zip(HANDS, goal, strict=True)]
        q = self.physical[1]
        diagnostic_q = {side: array(diag[side], "q_meas_rad") for side in HANDS}
        if any(angles.shape != (7,) or not np.isfinite(angles).all() for angles in diagnostic_q.values()):
            return None
        max_diagnostic_q_difference = max(
            max(abs(angle - q[f"{side}_j{joint}"]) for joint, angle in enumerate(angles, 1))
            for side, angles in diagnostic_q.items()
        )
        hand_error = []
        for side in HANDS:
            header, command = self.hands.get(side, (0, None))
            error = (max(abs(angle - q[name]) for name, angle in command.items())
                     if command is not None and command.keys() <= q.keys() and 0 <= now - header <= FRESH_NS else None)
            hand_error.append(error)
        active = all(self.phases.get(topic, {}).get(key) == label
                     and self.phases[topic].get("session_id") == self.session
                     for topic, (key, label) in ACTIVE.items())
        return {"wrist_error_m_deg": errors, "hand_error_to_issued_rad": hand_error,
                "measured_speed_rad_s": self.physical[2], "named_q_rad": q, "active_task": active,
                "feedback_header_ns": self.physical[0], "max_diagnostic_q_difference_rad": max_diagnostic_q_difference,
                "arms_accepted": all(x.get("arm_command_accepted") == "true" and x.get("ik_target_accepted") == "true"
                                     for x in diag.values()),
                "control_reasons": {side: {key: fields.get(key) for key in
                                            ("solve_mode", "recovery_action", "primary_reason", "rate_limited")}
                                    for side, fields in diag.items()},
                "shared_transform": shared.tolist(), "native_goal": [x.tolist() for x in goal],
                "home_world_wuji": [x.tolist() for x in home], "start_world_wuji": [x.tolist() for x in wrists]}

    def before_history(self, warmup):
        self.stage = "preposition_start"
        if not self.hold:
            self.write("start_handoff", status="IMMEDIATE_CONTROL", measurement=self.measure())
            return
        start = time.monotonic()
        ready_since = last_ready_stamp = None
        status, last = "START_NOT_REACHED", None
        while time.monotonic() - start < TIMEOUT_SEC:
            for event in warmup.values():
                self.send(event)
            self.wait(0.02)
            elapsed = time.monotonic() - start
            current = self.measure()
            ready = (current is not None and elapsed >= RAMP_SEC and current["active_task"] and current["arms_accepted"]
                     and current["measured_speed_rad_s"] is not None and current["measured_speed_rad_s"] <= SETTLED_RAD_S
                     and current["max_diagnostic_q_difference_rad"] <= MEASUREMENT_JOIN_RAD
                     and all(d <= WRIST_M and a <= WRIST_DEG for d, a in current["wrist_error_m_deg"])
                     and all(x is not None and x <= HAND_RAD for x in current["hand_error_to_issued_rad"]))
            if current is not None:
                last = current
                self.write("start_measurement", elapsed_s=elapsed, ready=ready, value=current)
            if not ready:
                ready_since = last_ready_stamp = None
                continue
            header = current["feedback_header_ns"]
            if last_ready_stamp is None or header - last_ready_stamp > FRESH_NS:
                ready_since = header
            if last_ready_stamp is None or header > last_ready_stamp:
                last_ready_stamp = header
                if header - ready_since >= DWELL_SEC * NS:
                    status = "START_REACHED"
                    break
        self.write("start_handoff", status=status, measurement=last, elapsed_s=time.monotonic() - start,
                   continue_after_miss=self.diagnostic_after_miss and status != "START_REACHED")
        print(status, None if last is None else last["wrist_error_m_deg"], flush=True)
        if status != "START_REACHED" and not (self.diagnostic_after_miss and last and last["active_task"]):
            raise RuntimeError("Fixed START target did not meet the pre-position criteria")

    def replay(self, events, warmup):
        origin = self.anchor["source_cutoff_ns"]
        historical = [event for event in events if event.schedule_ns >= origin]
        firsts = {}
        for event in historical:
            firsts.setdefault(event.topic, event)
            if firsts.keys() == HUMAN.keys():
                break
        if firsts.keys() != HUMAN.keys():
            raise ValueError("Incomplete native input after the frozen START cutoff")
        first = historical[0]
        self.origin_log_ns = first.log_ns - (first.schedule_ns - origin)
        now_ros = self.node.get_clock().now().nanoseconds
        source_headers = {topic: stamp({"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec})
                          for topic, msg in ((topic, deserialize_message(event.data, self.messages[topic]))
                                             for topic, event in firsts.items())}
        if any(value is None for value in source_headers.values()):
            raise ValueError("Missing source header at recorded handoff")
        pad = max(30_000_000, *(self.origin_log_ns + self.last_synthetic[topic] - now_ros - header + 5_000_000
                                for topic, header in source_headers.items()))
        if pad >= 100_000_000 or any(not 0 <= firsts[topic].schedule_ns - origin + self.origin_log_ns - source < 100_000_000
                                     for topic, source in source_headers.items()):
            raise ValueError("Recorded START handoff cannot preserve causal source age within the native freshness window")
        self.origin_mono_ns = time.monotonic_ns() + pad
        delta = now_ros + pad - self.origin_log_ns
        self.wait((self.origin_mono_ns - time.monotonic_ns()) / NS)
        self.stage = "historical_human_inputs"
        self.write("timebase", epoch_delta_ns=delta, origin_log_ns=self.origin_log_ns,
                   origin_mono_ns=self.origin_mono_ns, origin_schedule_ns=origin)
        expected, sent = sha256(), sha256()
        for event in historical:
            expected.update(sha256(event.data).digest())
        for event in historical:
            self.wait((self.origin_mono_ns + event.schedule_ns - origin - time.monotonic_ns()) / NS)
            self.send(event, delta)
            sent.update(sha256(event.data).digest())
            rclpy.spin_once(self.node, timeout_sec=0)
        self.write("historical_input_seal", messages=len(historical), expected_cdr_sha256=expected.hexdigest(),
                   published_cdr_sha256=sent.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cohort", "anchor", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--condition", choices=("immediate", "preposition"), required=True)
    parser.add_argument("--diagnostic-after-miss", action="store_true")
    args = parser.parse_args()
    if (os.environ.get("ROS_DOMAIN_ID") not in {str(x) for x in range(206, 214)}
            or os.environ.get("ROS_LOCALHOST_ONLY") != "1"
            or any(os.environ.get(key) for key in ("ROS_STATIC_PEERS", "CYCLONEDDS_URI",
                                                  "FASTDDS_DEFAULT_PROFILES_FILE", "FASTRTPS_DEFAULT_PROFILES_FILE"))):
        parser.error("Expected the dedicated isolated shared-map simulation domains")
    path = Path(os.environ["S05_SHARED_RIGID_MAP"])
    proof_path = os.environ.get("S05_START_HOME_PROOF")
    home_proof = json.loads(Path(proof_path).read_bytes()) if proof_path is not None else None
    frames = np.repeat(np.eye(4)[None, :, :], 7, axis=0)
    frames[:, :3, :] = np.loadtxt(path).reshape(7, 3, 4)
    anchor = json.loads(args.anchor.read_bytes())
    if sha256(args.cohort.read_bytes()).hexdigest() != anchor["cohort_sha256"]:
        raise ValueError("START anchor belongs to another cohort")
    events, receipt = signed_events(args.cohort, anchor["root"], None)
    receipt.update(anchor_sha256=sha256(args.anchor.read_bytes()).hexdigest(), condition=args.condition,
                   shared_frames_sha256=sha256(path.read_bytes()).hexdigest())
    args.output.mkdir(exist_ok=False)
    (args.output / "input.json").write_text(json.dumps(receipt, indent=2) + "\n")
    rclpy.init()
    node = rclpy.create_node("s05_nominal_replay")
    with (args.output / "trace.jsonl").open("x", buffering=1) as trace:
        try:
            HandoffReplay(node, trace, anchor, frames, args.condition == "preposition",
                          args.diagnostic_after_miss, home_proof).run(events)
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
