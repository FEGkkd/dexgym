#!/usr/bin/env python3
"""Run one preregistered, zero-residual DCR-009 START Pilot on the isolated native graph."""

import argparse
from dataclasses import dataclass
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import re
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from hmb_cohort import IDENTITIES, INPUT_ROLES


DESIGN = "6808935d7588834ad875789a2ca6bbb779d2caa9"
RUN_ID = "HOME-ANCHORED-PILOT-DCR009-01"
AUTHORITY_SCHEMA = "s05.teleop.home_anchored_pilot_authority.v1"
PLAN_SCHEMA = "s05.teleop.home_anchored_pilot_plan.v1"
ANCHOR_SCHEMA = "s05.teleop.home_anchored_start_anchor.v1"
HISTORICAL = "historical_human_inputs"
FRESH_NS = 40_000_000
DWELL_NS = 200_000_000
ROOT = re.compile(r"capture:[0-9a-f]{24}")
HEX = re.compile(r"[0-9a-f]{64}")
WRISTS = ("/vive/tracker/left_wrist", "/vive/tracker/right_wrist")
SIDES = ("left", "right")
ARM_TOPIC = "/tianji/joint_command"
FRAME_ORDER = ("world_from_tracker", "world_from_base_left", "world_from_base_right",
               "tracker_from_anatomical_palm_left", "tracker_from_anatomical_palm_right",
               "anatomical_palm_from_native_tcp_left", "anatomical_palm_from_native_tcp_right")
BANNED_ENV = ("ROS_STATIC_PEERS", "ROS_DISCOVERY_SERVER", "CYCLONEDDS_URI", "FASTRTPS_DEFAULT_PROFILES_FILE",
              "FASTDDS_DEFAULT_PROFILES_FILE", "S05_SHARED_RIGID_MAP")
SAFETY_TOPIC = "/tianji/safety"
HARD_SAFETY_CODES = frozenset({"TRACKER_LOST", "ROBOT_FEEDBACK_UNAVAILABLE", "ROBOT_FEEDBACK_STALE",
                              "JOINT_FEEDBACK_INVALID", "JOINT_FEEDBACK_STALE", "HARDWARE_NOT_COMMANDABLE",
                              "NON_FINITE_IK", "TRACKING_HARD_ERROR", "JOINT_LIMIT", "COUPLED_JOINT_LIMIT"})


class StartMiss(RuntimeError):
    """Evidence-bearing START miss that can end cleanly after verified STOP."""


def digest(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def object_at(path, expected):
    raw = Path(path).read_bytes()
    if not HEX.fullmatch(expected) or sha256(raw).hexdigest() != expected:
        raise ValueError(f"Pinned Pilot identity differs: {path}")
    return json.loads(raw)


def validate_environment(env, session_map):
    if (env.get("ROS_DOMAIN_ID") not in {str(i) for i in range(206, 214)}
            or env.get("ROS_LOCALHOST_ONLY") != "1" or env.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST"
            or any(env.get(key) for key in BANNED_ENV)
            or Path(env.get("S05_HOME_ANCHORED_SESSION_MAP", ".")).resolve() != session_map.resolve()
            or session_map.exists() or session_map.is_symlink()):
        raise ValueError("Require an unused isolated MuJoCo domain and a fresh one-session map path")


def proper(transforms, count):
    array = np.asarray(transforms, dtype=float)
    if array.shape != (count, 4, 4) or not np.isfinite(array).all():
        raise ValueError(f"Expected {count} finite 4×4 rigid transforms")
    r = array[:, :3, :3]
    if (not np.allclose(array[:, 3], [0., 0., 0., 1.], atol=1e-12, rtol=0.)
            or not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=1e-8, rtol=0.)
            or not np.allclose(np.linalg.det(r), 1., atol=1e-8, rtol=0.)):
        raise ValueError("Reflected, scaled or malformed Pilot frame")
    return array


def read_frames(path):
    loaded = np.loadtxt(path)
    if loaded.shape != (21, 4):
        raise ValueError("Require the exact seven ordered 3×4 CAL frame matrices")
    frames = np.repeat(np.eye(4)[None], 7, axis=0)
    frames[:, :3] = loaded.reshape(7, 3, 4)
    return proper(frames, 7)


def lock_home_frames(frames, home_palm_world, anatomical_start_h):
    """Fit just a common translation to the independently measured Home midpoint."""
    home = proper(home_palm_world, 2)
    start = proper(anatomical_start_h, 2)
    locked = proper(frames, 7).copy()
    locked[0, :3, 3] = home[:, :3, 3].mean(axis=0) - locked[0, :3, :3] @ start[:, :3, 3].mean(axis=0)
    expected_start = locked[0] @ start
    native_goal = np.linalg.inv(locked[1:3]) @ expected_start @ locked[5:7]
    return locked, expected_start, native_goal


def load_prerequisites(args):
    authority = object_at(args.authority, args.authority_sha256)
    if (authority.get("schema") != AUTHORITY_SCHEMA or authority.get("run_id") != RUN_ID
            or authority.get("design_commit") != DESIGN or authority.get("scope") != "MUJOCO_ONLY"
            or authority.get("pilot_execution_authorized") is not True
            or any(authority.get(key) is not False for key in
                   ("full_campaign_authorized", "training_authorized", "real_robot_authorized"))
            or not re.fullmatch(r"[0-9a-f]{40}", authority.get("implementation_commit", ""))):
        raise ValueError("DCR-009 strict six-root simulation Pilot has no verified authority")
    identities = authority["paths_sha256"]
    calibration = object_at(args.calibration, identities["calibration_receipt"])
    if (calibration.get("schema_version") != 1 or calibration.get("scope") != "MUJOCO_ONLY"
            or calibration.get("status") != "QUALIFIED"
            or calibration.get("qualification", {}).get("recorded_mcap_pilot") != "QUALIFIED"):
        raise ValueError(f"Simulation calibration has no qualification: {calibration.get('status', 'UNKNOWN')}")
    if authority.get("qualifications", {}).get("independent_whole_body_publish_guard") != "VERIFIED":
        raise ValueError("No independently verified signed 54-axis swept-path publish guard")
    if args.controller_executable.resolve() == args.simulator_executable.resolve():
        raise ValueError("Controller and simulator must have independently bound executables")
    for key, file in ({"cohort": args.cohort, "pilot_plan": args.plan, "calibration_receipt": args.calibration,
                       "home_reference": args.home_reference, "live_model": args.model, "shared_frames": args.shared_frames,
                       "controller_executable": args.controller_executable, "simulator_executable": args.simulator_executable,
                       "runtime_files": args.runtime_files}).items():
        if digest(file) != identities[key]:
            raise ValueError(f"Approved Pilot runtime changed: {key}")
    runtime_files = object_at(args.runtime_files, identities["runtime_files"])
    paths = runtime_files["files"]
    for name in (Path(__file__).name, "nominal_live_replay.py", "replay_start_handoff.py", "prepare_home_anchored_pilot_overlay.py",
                 "hmb_cohort.py", "summarize_nominal_replay.py", "pair_nominal_hmb.py"):
        if digest(Path(__file__).with_name(name)) != paths[name]:
            raise ValueError(f"Pilot runner source differs from deployment manifest: {name}")
    if (calibration["transforms"]["sha256"] != identities["shared_frames"]
            or tuple(calibration["transforms"]["order"]) != FRAME_ORDER
            or calibration["model"]["xml_sha256"] != identities["live_model"]):
        raise ValueError("CAL must bind the ordered frames and the actual live MJCF")
    model_palm_from_anatomical = proper(calibration["model"]["model_palm_from_anatomical"], 2)
    frames = read_frames(args.shared_frames)
    cohort = object_at(args.cohort, identities["cohort"])
    plan = object_at(args.plan, identities["pilot_plan"])
    if (cohort["schema"] != "s05.teleop.hmb_source_cohort.v1" or cohort["native_assets_sha256_checked"] is not True
            or cohort["identity_sha256"] != {key: value[1] for key, value in IDENTITIES.items()}
            or plan["schema"] != PLAN_SCHEMA or plan["run_id"] != RUN_ID
            or plan["selection_status"] != "PILOT_PLAN_READY_FOR_REVIEW"
            or plan["cohort_sha256"] != identities["cohort"] or plan["shared_frames_sha256"] != identities["shared_frames"]
            or plan["calibration_receipt_sha256"] != identities["calibration_receipt"]):
        raise ValueError("Pilot plan does not bind the qualified model and signed HMB cohort")
    source_roots = {root["root"]: root for root in cohort["selected_roots"] if root["split"] == "Train"}
    rows = {item["root"]: item for item in plan["selected_roots"]}
    if (len(rows) != 6 or len(plan["selected_roots"]) != 6 or not rows.keys() <= source_roots.keys()
            or not all(ROOT.fullmatch(name) for name in rows)
            or not all(type(row["repeat_cold_start"]) is bool for row in rows.values())
            or len(plan["repeat_roots"]) != 2 or set(plan["repeat_roots"]) != {
                name for name, row in rows.items() if row["repeat_cold_start"] is True
            }):
        raise ValueError("Pilot roster must bind six distinct signed Train sources and at most two cold repeats")
    matching = [(name, row) for name, row in rows.items()
                if row["anchor_path"] == f"anchors/{name.split(':')[1]}.json"
                and (args.plan.parent / row["anchor_path"]).resolve() == args.anchor.resolve()]
    if len(matching) != 1:
        raise ValueError("Anchor is outside the preregistered six-root roster")
    name, selection = matching[0]
    if name not in calibration.get("qualification", {}).get("qualified_recorded_roots", []):
        raise ValueError("CAL does not cover this recording")
    if args.attempt == "repeat" and selection["repeat_cold_start"] is not True:
        raise ValueError("No registered cold-repeat attempt for this source")
    root = source_roots[name]
    anchor = object_at(args.anchor, selection["anchor_sha256"])
    if (anchor["schema"] != ANCHOR_SCHEMA or anchor["root"] != name or anchor["split"] != "Train"
            or anchor["source_manifest_sha256"] != root["manifest_sha256"] or anchor["cohort_sha256"] != identities["cohort"]
            or anchor["shared_frames_sha256"] != identities["shared_frames"]
            or anchor["stable_primary"]["status"] not in {"EXPLICIT_START", "RECONSTRUCTED_START"}
            or anchor["historical_window"] != selection["historical_window"]):
        raise ValueError("Anchor has no causal stable START qualification under the preregistered rule")
    start = proper(anchor["anatomical_start_h"], 2)
    window = anchor["historical_window"]
    if (window["source_cutoff_start_ns"] != anchor["source_cutoff_ns"]
            or window["hmb_cutoff_start_ns"] != anchor["hmb_cutoff_ns"]
            or not 0 < window["source_cutoff_stop_exclusive_ns"] - window["source_cutoff_start_ns"] <= 120_000_000_000
            or type(window["plan_ticks"]) is not int or not 0 < window["plan_ticks"] <= root["planned_ticks"]):
        raise ValueError("Pilot time window is outside the preregistered bounded route")
    home = object_at(args.home_reference, identities["home_reference"])
    if (home["model_sha256"] != identities["live_model"] or len(home["named_q_rad"]) != 54
            or not np.isfinite(list(home["named_q_rad"].values())).all()):
        raise ValueError("Named Home reference does not match the qualified live model")
    return authority, calibration, root, anchor, frames, start, home, model_palm_from_anatomical


def map_file(path, session, frames):
    """Persist one immutable session map before the native START call."""
    matrix = proper(frames, 7)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(f"S05_DCR009_START_MAP_V1 {session}\n")
        np.savetxt(stream, matrix[:, :3].reshape(21, 4), fmt="%.17g")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    return digest(path)


@dataclass
class FeedbackDwell:
    first_ns: int | None = None
    last_ns: int | None = None

    def observe(self, header_ns, ready):
        if not ready or header_ns is None:
            self.first_ns = self.last_ns = None
            return False
        if self.last_ns is None or not 0 <= header_ns - self.last_ns <= FRESH_NS:
            self.first_ns = header_ns
        self.last_ns = header_ns
        return header_ns - self.first_ns >= DWELL_NS


def verify_owned_processes(launch_pid, controller, simulator):
    pending, seen, active = [launch_pid], set(), {controller.resolve(): 0, simulator.resolve(): 0}
    while pending:
        pid = pending.pop()
        if pid in seen or not Path(f"/proc/{pid}").is_dir():
            continue
        seen.add(pid)
        entry = Path(f"/proc/{pid}")
        try:
            pending.extend(int(child) for task in (entry / "task").iterdir()
                           for child in (task / "children").read_text().split())
            binary = (entry / "exe").resolve(strict=True)
        except (FileNotFoundError, ProcessLookupError):
            continue
        if binary in active:
            env = dict(item.split(b"=", 1) for item in (entry / "environ").read_bytes().split(b"\0") if b"=" in item)
            if (env.get(b"ROS_DOMAIN_ID") != os.environ["ROS_DOMAIN_ID"].encode()
                    or env.get(b"ROS_LOCALHOST_ONLY") != b"1" or any(env.get(key.encode()) for key in BANNED_ENV)):
                raise RuntimeError("Owned Teleop or MuJoCo node has a non-isolated effective environment")
            if (binary == controller.resolve()
                    and env.get(b"S05_HOME_ANCHORED_SESSION_MAP") != os.environ["S05_HOME_ANCHORED_SESSION_MAP"].encode()):
                raise RuntimeError("Controller does not load the exact owned one-session map path")
            active[binary] += 1
    if set(active.values()) != {1}:
        raise RuntimeError("Could not bind one actual controller and one actual simulator executable to the owned launch")


def pose_from_xyz_quat(xyz, xyzw):
    pose = np.eye(4)
    pose[:3, 3] = xyz
    if not np.isfinite(np.r_[xyz, xyzw]).all() or abs(np.linalg.norm(xyzw) - 1.) > 1e-4:
        raise ValueError("Invalid replay tracker pose")
    pose[:3, :3] = Rotation.from_quat(xyzw).as_matrix()
    return pose


def palm_poses(geometry):
    return np.stack([pose_from_xyz_quat(pos, Rotation.from_matrix(rot).as_quat())
                     for pos, rot in zip(geometry.palm, geometry.palm_rotation, strict=True)])


def angle_from_down(geometry):
    knuckles = geometry.landmarks[:, 1:, 0]
    forward = knuckles.mean(axis=1) - geometry.palm
    forward /= np.linalg.norm(forward, axis=1)[:, None]
    across = knuckles[:, 0] - knuckles[:, -1]
    normal = np.cross(forward, across) * np.array([1., -1.])[:, None]
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    return np.degrees(np.arccos(np.clip(-normal[:, 2], -1., 1.)))


def expected_arm_publish(fields, session, command):
    """Bind native classifications to exactly one observed command, not nearest diagnostics."""
    if command is None:
        return False
    header, payload = command
    sequence = payload["sequence"]
    return (type(sequence) is int and sequence > 0 and payload["session_id"] == session
            and fields.get("session") == session and fields.get("pilot_publish_advanced") == "true"
            and fields.get("pilot_arm_status") in {"ACCEPTED", "LIMITED"}
            and fields.get("sequence") == str(sequence) and fields.get("pilot_previous_sequence") == str(sequence - 1)
            and fields.get("pilot_command_header_ns") == str(header))


def replay_class():
    from nominal_live_replay import ACTUATORS, CONTROLLER, HUMAN, SIMULATOR, deserialize_message
    from pair_nominal_hmb import signed_feedback_names
    from replay_start_handoff import HandoffReplay, pose, pose_error, rclpy
    from rclpy.qos import qos_profile_sensor_data
    from rosidl_runtime_py.convert import message_to_ordereddict
    from rosidl_runtime_py.utilities import get_message
    from summarize_nominal_replay import ACTIVE, DIAGNOSTICS, FEEDBACK, joint_values, stamp

    class PilotReplay(HandoffReplay):
        def __init__(self, node, trace, args, anchor, frames, anatomical_start, home, model, model_palm_from_anatomical):
            super().__init__(node, trace, anchor, frames, True, False, home)
            self.args = args
            self.session = f"s05-dcr009-{uuid.uuid4()}"
            self.robot, self.plant_names = model, signed_feedback_names(model)
            self.anatomical_start = anatomical_start
            self.model_palm_from_anatomical = model_palm_from_anatomical
            self.start_ready = False
            self.verified_commands, self.pilot_diagnostics = {}, {}
            self.home_snapshot, self.shared, self.goal = None, None, None
            self.stop_accept_mono_ns = None
            self.late_command_count = 0
            self.session_map_sha = None
            self.hard_fault = None

        def observe_output(self, topic, value):
            header = stamp(value.get("header", {}).get("stamp"))
            if topic in ACTUATORS and self.stop_accept_mono_ns is not None:
                self.late_command_count += 1
            if topic in ACTIVE:
                self.phases[topic] = value
            if header is None:
                return
            if topic in FEEDBACK:
                self.observe_feedback(topic, header, value)
            for side in SIDES:
                if topic == f"/{side}_hand/command" and value.get("side") == side:
                    self.hands[side] = header, joint_values(value, 20)
            if (self.stage in {"preposition_start", HISTORICAL} and topic == "/tianji/robot_state"
                    and value.get("command_allowed") is False):
                self.hard_fault = {"source": topic, "code": "DRIVER_NOT_COMMANDABLE", "reason": value.get("reason"),
                                   "header_ns": header}
            if (self.stage in {"preposition_start", HISTORICAL} and topic == SAFETY_TOPIC
                    and value.get("safe") is False and value.get("code", "").upper() in HARD_SAFETY_CODES):
                self.hard_fault = {"source": topic, "code": value["code"], "reason": value.get("reason"),
                                   "header_ns": header}
            if topic == ARM_TOPIC and value.get("session_id") == self.session and header is not None:
                sequence = value.get("sequence")
                if type(sequence) is int and sequence > 0 and joint_values(value, 14) is not None:
                    key = (header, sequence)
                    self.verified_commands[key] = (header, value) if key not in self.verified_commands else None
            if header is not None:
                for status in value.get("status", ()):
                    side = DIAGNOSTICS.get(status["name"])
                    if side:
                        fields = {item["key"]: item["value"] for item in status["values"]}
                        if fields.get("session") == self.session and fields.get("mapping_policy") == "home_anchored_pilot_v1":
                            self.pilot_diagnostics[side] = header, fields
                cutoff = header - 2 * FRESH_NS
                self.verified_commands = {key: command for key, command in self.verified_commands.items() if key[0] >= cutoff}

        def observe_feedback(self, topic, header, value):
            position = joint_values(value, FEEDBACK[topic])
            if topic.endswith("/state") and (value.get("connected") is not True or value.get("online") != [True] * 20
                    or value.get("position_valid") != [True] * 20 or value.get("stale") != [False] * 20):
                position = None
            group = self.pending[header]
            group[topic] = position if topic not in group else None
            if group.keys() == FEEDBACK.keys():
                prev = self.physical
                if any(item is None for item in group.values()):
                    if prev is not None and header >= prev[0]:
                        self.physical = None
                elif prev is None or header > prev[0]:
                    q = {name: angle for angles in group.values() for name, angle in angles.items()}
                    if set(q) != set(self.plant_names):
                        self.physical = None
                    else:
                        speed = None
                        if prev is not None and header - prev[0] <= FRESH_NS:
                            speed = max(abs(q[name] - prev[1][name]) for name in q) * 1e9 / (header - prev[0])
                        self.physical = header, q, speed
            for past in [key for key in self.pending if key < header - FRESH_NS]:
                del self.pending[past]

        def verify_graph(self, *, expect_replay_writer):
            topics = super().verify_graph(expect_replay_writer=expect_replay_writer)
            watched = dict.fromkeys(ACTUATORS, CONTROLLER)
            watched.update(dict.fromkeys(ACTIVE, CONTROLLER))
            watched["/tianji/teleop_diagnostics"] = CONTROLLER
            watched[SAFETY_TOPIC] = CONTROLLER
            for topic, owner in watched.items():
                publishers = [(info.node_name, info.node_namespace)
                              for info in self.node.get_publishers_info_by_topic(topic)]
                if publishers != [(owner, "/")]:
                    raise RuntimeError(f"Unverified graph writer for {topic}")
            for topic in ACTUATORS:
                readers = {(info.node_name, info.node_namespace)
                           for info in self.node.get_subscriptions_info_by_topic(topic)}
                if not readers <= {(self.node.get_name(), "/"), (SIMULATOR, "/")} or (SIMULATOR, "/") not in readers:
                    raise RuntimeError(f"Unverified actuator subscriber for {topic}")
            verify_owned_processes(self.args.launch_pid, self.args.controller_executable, self.args.simulator_executable)
            return topics

        def watch_native_safety(self):
            def on_message(message):
                value = message_to_ordereddict(message)
                self.write("output", topic=SAFETY_TOPIC, value=value)
                self.observe_output(SAFETY_TOPIC, value)

            self.subscriptions.append(self.node.create_subscription(
                get_message("tianji_interfaces/msg/SafetyStatus"), SAFETY_TOPIC, on_message, qos_profile_sensor_data))

        def require_no_hard_fault(self):
            if self.hard_fault is not None:
                self.write("pilot_lifecycle", status="FAULT_STOP", failure=self.hard_fault)
                raise RuntimeError(f"Pilot watchdog observed native hard fault: {self.hard_fault['code']}")

        def verify_loaded_model(self):
            controller = self.parameters(CONTROLLER, ["workflow.driver_prefix", "arm.kinematics_backend",
                "arm.control_pipeline", "arm.analytic_diagnostics_full_rate", "head.enabled", "enable_hands"])
            if (controller["workflow.driver_prefix"].string_value != "/tianji_mujoco_driver"
                    or controller["arm.kinematics_backend"].string_value != "marvin"
                    or controller["arm.control_pipeline"].string_value != "manimux_analytic_ik_position"
                    or not controller["arm.analytic_diagnostics_full_rate"].bool_value
                    or controller["head.enabled"].bool_value or not controller["enable_hands"].bool_value):
                raise RuntimeError("Actual Teleop graph lacks the checked Marvin/Wuji simulation profile")
            sim = self.parameters(SIMULATOR, ["model_path", "update_rate_hz", "publish_rate_hz"])
            model = Path(sim["model_path"].string_value).resolve()
            if model != self.robot.path or digest(model) != digest(self.args.model):
                raise RuntimeError("Actual loaded MuJoCo asset is not the authorized model")
            self.write("model", path=str(model), sha256=digest(model))

        def qualify_home(self):
            dwell = FeedbackDwell()
            home_done = self.node.get_clock().now().nanoseconds
            reference = self.home_proof["named_q_rad"]
            if set(reference) != set(self.plant_names):
                raise ValueError("Home reference has a different named joint manifest")
            self.stage = "home_measured"
            deadline = time.monotonic() + 3.
            while time.monotonic() < deadline:
                self.wait(.015)
                header, q, speed = self.physical if self.physical is not None else (None, {}, None)
                now = self.node.get_clock().now().nanoseconds
                valid = (header is not None and header >= home_done and 0 <= now - header <= FRESH_NS
                         and speed is not None and speed <= .16 and set(q) == set(reference))
                if not valid:
                    dwell.observe(None, False)
                    continue
                error = max(abs(q[name] - reference[name]) for name in self.plant_names)
                geometry = self.robot.geometry(np.asarray([q[name] for name in self.plant_names]))
                down = angle_from_down(geometry)
                ready = error <= np.deg2rad(1.) and np.isfinite(down).all() and max(down) <= 5.
                self.write("home_measurement", measured_header_ns=header, error_rad=error,
                           speed_rad_s=speed, palm_normal_to_down_deg=down.tolist(), ready=bool(ready))
                if dwell.observe(header, ready):
                    self.home_snapshot = palm_poses(geometry) @ self.model_palm_from_anatomical
                    self.write("home_feedback", named_q_rad=q, measured_header_ns=header, max_measured_speed_rad_s=speed,
                               max_error_to_home_proof_rad=error, dwell_ns=header - dwell.first_ns,
                               anatomical_palm_world=self.home_snapshot.tolist())
                    return
            raise RuntimeError("HOME_UNQUALIFIED: missing a fresh, settled 54-axis palms-down Home dwell")

        def start_inputs(self, events):
            selected = {}
            witnessed = self.anchor["inputs"]
            if set(witnessed) != set(HUMAN):
                raise ValueError("Stable anchor must bind the four original human streams")
            for event in events:
                proof = witnessed[event.topic]
                if event.record_index == proof["record_index"]:
                    if (event.topic in selected or event.schedule_ns > self.anchor["source_cutoff_ns"]
                            or sha256(event.data).hexdigest() != proof["cdr_sha256"]):
                        raise ValueError("Stable START payload differs from the signed causal endpoint")
                    selected[event.topic] = event
            if selected.keys() != HUMAN.keys():
                raise ValueError("A causal START source could not be verified")
            actual_start = []
            for i, topic in enumerate(WRISTS):
                msg = deserialize_message(selected[topic].data, self.messages[topic])
                pose_msg = msg.pose
                actual_start.append(pose_from_xyz_quat([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z],
                    [pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w]) @ self.frames[i + 3])
            if not np.allclose(np.asarray(actual_start), self.anatomical_start, atol=1e-5, rtol=0.):
                raise ValueError("Signed native Tracker endpoints do not agree with the anatomical START receipt")
            locked, start_world, self.goal = lock_home_frames(self.frames, self.home_snapshot, self.anatomical_start)
            self.shared = locked[0]
            self.session_map_sha = map_file(self.args.session_map, self.session, locked)
            self.write("start_anchor", source_cutoff_ns=self.anchor["source_cutoff_ns"], inputs=witnessed,
                       preanchor_input_messages=sum(e.schedule_ns < self.anchor["source_cutoff_ns"] for e in events),
                       stable_status=self.anchor["stable_primary"]["status"], session=self.session,
                       session_map_sha256=self.session_map_sha, shared_transform=self.shared.tolist(),
                       start_world_anatomical_palm=start_world.tolist())
            return selected

        def active_task(self):
            return all(self.phases.get(topic, {}).get(key) == value
                       and self.phases[topic].get("session_id") == self.session
                       for topic, (key, value) in ACTIVE.items())

        def measure(self):
            now = self.node.get_clock().now().nanoseconds
            if (self.physical is None or not 0 <= now - self.physical[0] <= FRESH_NS or not self.active_task()
                    or self.pilot_diagnostics.keys() != set(SIDES)):
                return None
            commands, reasons, wrist_errors, hand_errors = [], [], [], []
            header, q, speed = self.physical
            if set(q) != set(self.plant_names):
                return None
            geometry = self.robot.geometry(np.asarray([q[name] for name in self.plant_names]))
            actual_native = np.linalg.inv(self.frames[1:3]) @ palm_poses(geometry) @ self.model_palm_from_anatomical @ self.frames[5:7]
            for i, side in enumerate(SIDES):
                diag_stamp, fields = self.pilot_diagnostics[side]
                if not 0 <= now - diag_stamp <= FRESH_NS:
                    return None
                if fields.get("target_input_raw_available") != "true":
                    return None
                raw_pos_error, raw_rot_error_deg = pose_error(self.goal[i], pose(fields, "target_input_raw"))
                if raw_pos_error > .0001 or raw_rot_error_deg > np.rad2deg(.001):
                    raise RuntimeError("Native raw START mapping differs from the frozen measured Home anchor")
                try:
                    key = int(fields["pilot_command_header_ns"]), int(fields["sequence"])
                except (ValueError, KeyError):
                    return None
                command = self.verified_commands.get(key)
                commands.append(expected_arm_publish(fields, self.session, command))
                if command is not None:
                    expiration = stamp(command[1].get("valid_until"))
                    commands[-1] &= (expiration is not None and now < expiration and 0 <= now - key[0] <= FRESH_NS)
                reasons.append(fields.get("pilot_arm_status"))
                wrist_errors.append(pose_error(self.goal[i], actual_native[i]))
                stamp_ns, hand = self.hands.get(side, (0, None))
                names = self.plant_names[14 + 20 * i:34 + 20 * i]
                hand_errors.append(max(abs(hand[name] - q[name]) for name in names)
                    if hand is not None and set(hand) == set(names) and 0 <= now - stamp_ns <= FRESH_NS else None)
            return {"wrist_error_m_deg": wrist_errors, "hand_error_to_issued_rad": hand_errors,
                    "measured_speed_rad_s": speed, "named_q_rad": q, "active_task": True,
                    "feedback_header_ns": header, "arms_accepted": all(commands), "arm_publish_status": reasons,
                    "shared_transform": self.shared.tolist(), "native_goal": self.goal.tolist(),
                    "session_map_sha256": self.session_map_sha}

        def additional_start_checks(self, measurement):
            return True

        def before_history(self, warmup):
            self.stage = "preposition_start"
            begin = time.monotonic()
            dwell, last = FeedbackDwell(), None
            while time.monotonic() - begin < 10.0:
                for event in warmup.values():
                    self.send(event)
                self.wait(.020)
                self.require_no_hard_fault()
                elapsed = time.monotonic() - begin
                current = self.measure()
                ready = (current is not None and elapsed >= 1.5 and current["arms_accepted"]
                         and current["measured_speed_rad_s"] is not None and current["measured_speed_rad_s"] <= .16
                         and all(d <= .010 and a <= 5. for d, a in current["wrist_error_m_deg"])
                         and all(x is not None and x <= .10 for x in current["hand_error_to_issued_rad"])
                         and self.additional_start_checks(current))
                if current is not None:
                    last = current
                    self.write("start_measurement", elapsed_s=elapsed, ready=ready, value=current)
                if dwell.observe(current["feedback_header_ns"] if current is not None else None, ready):
                    self.start_ready = True
                    break
            self.write("start_handoff", status="START_REACHED" if self.start_ready else "START_NOT_REACHED",
                       measurement=last, elapsed_s=time.monotonic() - begin, continue_after_miss=False)
            if not self.start_ready:
                raise StartMiss("START_NOT_REACHED: no complete native measured START dwell")

        def send(self, event, epoch_delta_ns=None):
            self.require_no_hard_fault()
            if self.stage == HISTORICAL and (not self.start_ready or not self.active_task()):
                raise RuntimeError("Historical input denied outside the successful strict START and same ACTIVE/TASK session")
            return super().send(event, epoch_delta_ns)

        def run(self, events):
            self.write("pilot_lifecycle", status="SIM_PREFLIGHT", session=self.session)
            self.verify_and_subscribe()
            self.watch_native_safety()
            self.verify_loaded_model()
            attempted_connection = False
            try:
                attempted_connection = True
                self.service("connect")
                self.service("home")
                self.qualify_home()
                warmup = self.start_inputs(events)
                self.stage = "synthetic_start_warmup"
                for _ in range(25):
                    for event in warmup.values():
                        self.send(event)
                    self.wait(.020)
                self.verify_graph(expect_replay_writer=True)
                if digest(self.args.session_map) != self.session_map_sha:
                    raise RuntimeError("Sealed Home→START map changed before START")
                self.service("start")
                for event in warmup.values():
                    self.send(event)
                self.service("enable")
                self.before_history(warmup)
                self.verify_graph(expect_replay_writer=True)
                selected_end = self.anchor["historical_window"]["source_cutoff_stop_exclusive_ns"]
                selected = [event for event in events if event.schedule_ns < selected_end]
                if not self.start_ready:
                    raise RuntimeError("Historical replay attempted without START_REACHED")
                self.replay(selected, warmup)
                self.require_no_hard_fault()
            except StartMiss:
                self.write("pilot_lifecycle", status="START_NOT_REACHED", session=self.session)
            finally:
                if attempted_connection:
                    self.stage = "tail"
                    self.service("stop")
                    self.stop_accept_mono_ns = time.monotonic_ns()
                    self.wait(.20)
                    self.write("pilot_stop", late_observed_command_count=self.late_command_count, session=self.session)
                    if self.late_command_count:
                        raise RuntimeError("Observed a command after native STOP acknowledgement")

    return PilotReplay, rclpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("authority", "calibration", "plan", "anchor", "cohort", "shared-frames", "home-reference", "model",
                "session-map", "output", "runtime-files", "controller-executable", "simulator-executable"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--launch-pid", type=int, required=True)
    parser.add_argument("--attempt", choices=("primary", "repeat"), default="primary")
    args = parser.parse_args()
    if args.launch_pid <= 1:
        parser.error("Supply the owned isolated launch process PID")
    validate_environment(os.environ, args.session_map)
    authority, calibration, root, anchor, frames, start, home, model_palm_from_anatomical = load_prerequisites(args)
    from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel, asset_files
    from nominal_live_replay import signed_events

    model = RobotModel(args.model)
    assets = {Path(name).relative_to(model.path.parent).as_posix(): sha for name, sha in asset_files(model.path).items()}
    if assets != calibration["model"]["assets_sha256"]:
        raise ValueError("Live MuJoCo dependent model assets differ from CAL qualification")
    events, receipt = signed_events(args.cohort, root["root"], None)
    if receipt["root"] != root or receipt["cohort_receipt_sha256"] != authority["paths_sha256"]["cohort"]:
        raise ValueError("Raw input verifier did not return the selected signed Pilot source")
    args.output.mkdir(exist_ok=False)
    receipt.update(scope="DCR009_STAGE_II_PILOT_SIMULATION_ONLY", authority_sha256=args.authority_sha256,
                   calibration_sha256=digest(args.calibration), plan_sha256=digest(args.plan), anchor_sha256=digest(args.anchor),
                   shared_frames_sha256=digest(args.shared_frames), condition="preposition", attempt=args.attempt,
                   historical_window=anchor["historical_window"], residual_enabled=False)
    (args.output / "input.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    PilotReplay, rclpy = replay_class()
    rclpy.init()
    node = rclpy.create_node("s05_nominal_replay")
    try:
        with (args.output / "trace.jsonl").open("x", buffering=1, encoding="utf-8") as trace:
            runner = PilotReplay(node, trace, args, anchor, frames, start, home, model, model_palm_from_anatomical)
            try:
                runner.run(events)
            except (RuntimeError, ValueError) as error:
                runner.write("pilot_lifecycle", status="ABORTED", error=str(error))
                raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
