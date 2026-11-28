#!/usr/bin/env python3
"""One zero-residual, guarded diagnostic replay. This entry point confers no Pilot qualification."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from replay_home_anchored_pilot import (
    FRESH_NS, WRISTS, digest, map_file, object_at, palm_poses, pose_from_xyz_quat, proper, read_frames,
    replay_class, validate_environment,
)


RUN_ID = "HOME-ANCHORED-DIAGNOSTIC-DCR009-01"
SCOPE = "DIAGNOSTIC_ONLY"
VIRTUAL_R = np.asarray([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
VIRTUAL_SHA = "4798dfbdaf80178b265a0c325bce1b96b09221d35b4b11c86d35c650c5f3782a"
LEGACY_BRIDGE_SHA = "33cbc93c4a2db925986f596b4b1730c7eaca026a9facfdf26aa4aaa8032c8f94"
COHORT_SHA = "7d6c4702229aefd0d564f76dfae5a2a7b62a0340b9c4cae43fea512cc2d531c2"
LIVE_SHA = "be69f387f5bc25c8011a992a88ac2c2449c2c0419e09318b740fcdcd1402871d"
SIGNED_SHA = "939b3651e2426e596eed482e70a68c2382720df1fd55b124aa5f0893b640e92d"
ROOT_FIVE = "capture:b63bc72d2e8b2ee395e966f6"
ROOT_FIVE_PLAN_SHA = "3c0268291bdb8916a9a520d3fa73a1340fffd7f9e8a7b7364a1ef9950f9a0004"
# +X in W goes toward the named left base. -X is virtual operator right.
CENTER_OFFSET_SEQUENCE_MM = (0, 10, 20, -10, -20)


def check_scope(plan):
    if (plan["run_id"] != RUN_ID or plan["scope"] != SCOPE
            or plan.get("formal_coverage_credit") is not False or plan.get("stage_credit") is not False
            or plan.get("attempt_limit") != 6 or plan.get("repeat_roots") != []):
        raise ValueError("Require the isolated six-source diagnostic plan without formal qualification")
    rows = plan["selected_roots"]
    if len(rows) != 6 or len({row["root"] for row in rows}) != 6 or any(row["split"] != "Train" for row in rows):
        raise ValueError("Diagnostic plan must contain exactly six unique Train sources")
    return rows


def registered_targets(frames, measured_home, raw_registered, center_offset_mm=0):
    """Lock translation on registered wrist origins; engineering attachment never rewrites that target."""
    locked = proper(frames, 7).copy()
    home = proper(measured_home, 2)
    raw = proper(raw_registered, 2)
    locked[0, :3, 3] = home[:, :3, 3].mean(axis=0) - locked[0, :3, :3] @ raw[:, :3, 3].mean(axis=0)
    locked[0, 0, 3] += center_offset_mm / 1000
    registered_world = locked[0] @ raw
    native = np.linalg.inv(locked[1:3]) @ registered_world @ locked[3:5] @ locked[5:7]
    return locked, registered_world, native


def align_pair_yaw(locked, measured_home, registered_world):
    """Rotate the whole intent about its midpoint, matching the signed left-to-right heading."""
    home_line = np.diff(measured_home[:, :2, 3], axis=0)[0]
    intent_line = np.diff(registered_world[:, :2, 3], axis=0)[0]
    if min(np.linalg.norm(home_line), np.linalg.norm(intent_line)) < 1e-3:
        raise ValueError("Cannot align START yaw for a degenerate horizontal left-to-right line")
    cross = intent_line[0] * home_line[1] - intent_line[1] * home_line[0]
    yaw = np.arctan2(cross, intent_line @ home_line)
    pivot = registered_world[:, :3, 3].mean(axis=0)
    turn = np.eye(4)
    turn[:3, :3] = Rotation.from_rotvec([0., 0., yaw]).as_matrix()
    turn[:3, 3] = pivot - turn[:3, :3] @ pivot
    aligned = turn @ registered_world
    frames = locked.copy()
    frames[0] = turn @ locked[0]
    return frames, aligned, {"yaw_deg": float(np.degrees(yaw)), "pivot_world_m": pivot.tolist(),
                             "world_correction": turn.tolist()}


def load(args):
    validate_environment(os.environ, args.session_map)
    if (args.launch_pid <= 1 or Path(os.environ.get("S05_DCR009_GUARD_SOCKET", ".")).resolve() != args.guard_socket
            or not args.guard_socket.is_socket() or digest(args.model) != LIVE_SHA):
        raise ValueError("Require isolated command guard and exact live MuJoCo model")
    object_at(args.virtual_axes, VIRTUAL_SHA)
    cohort = object_at(args.cohort, COHORT_SHA)
    legacy = read_frames(args.legacy_frames)
    if digest(args.legacy_frames) != LEGACY_BRIDGE_SHA:
        raise ValueError("Existing engineering wrist bridge identity differs")
    frames = read_frames(args.frames)
    if (not np.array_equal(frames[0, :3, :3], VIRTUAL_R) or np.any(frames[0, :3, 3])
            or not np.array_equal(frames[1:], legacy[1:])):
        raise ValueError("Only the common qualified virtual rotation may replace the old bridge")
    plan = object_at(args.plan, args.plan_sha256)
    rows = check_scope(plan)
    if args.center_offset_mm is not None and (args.root != ROOT_FIVE or args.plan_sha256 != ROOT_FIVE_PLAN_SHA):
        raise ValueError("Center offset comparison is authorized only for root five in the frozen diagnostic plan")
    if args.start_only_profile is not None:
        if args.root != ROOT_FIVE or args.center_offset_mm != 0 or args.plan_sha256 != ROOT_FIVE_PLAN_SHA:
            raise ValueError("The isolated START diagnosis requires root five at the original center")
        state_file = os.environ.get("S05_DCR009_SEGMENTED_START_STATE")
        if args.start_only_profile == "segmented":
            if not state_file or Path(state_file).read_bytes() != b"PREPOSITION\n":
                raise ValueError("Opt-in segmented controller profile was not armed for preposition")
        elif state_file:
            raise ValueError("The original START audit must use the original unsegmented controller")
    if args.align_start_yaw_to_home and args.start_only_profile != "segmented":
        raise ValueError("Common yaw alignment is confined to the segmented root-five START-only run")
    if plan["cohort_sha256"] != COHORT_SHA or plan["shared_frames_sha256"] != digest(args.frames):
        raise ValueError("Plan does not bind the original cohort and exact runtime frames")
    selection = next(row for row in rows if row["root"] == args.root)
    path = args.plan.parent / selection["anchor_path"]
    if args.plan.parent.resolve() not in path.resolve().parents:
        raise ValueError("Diagnostic anchor escapes the plan directory")
    anchor = object_at(path, selection["anchor_sha256"])
    if (anchor["scope"] != SCOPE or anchor["root"] != args.root or anchor["split"] != "Train"
            or anchor["anatomical_start_h"] is not None
            or anchor["cohort_sha256"] != COHORT_SHA
            or anchor["shared_frames_sha256"] != digest(args.frames)):
        raise ValueError("Invalid bound diagnostic anchor or invented anatomical transform")
    raw = proper(anchor["registered_wrist_start_h"], 2)
    root = next(row for row in cohort["selected_roots"] if row["root"] == args.root)
    if (root["split"] != "Train" or root["manifest_sha256"] != anchor["source_manifest_sha256"]
            or not 0 < selection["replay_planned_ticks"] <= min(root["planned_ticks"], 6000)):
        raise ValueError("Anchor is outside its original Train identity or replay budget")
    home = json.loads(args.home_reference.read_bytes())
    if home["model_sha256"] != LIVE_SHA:
        raise ValueError("Home receipt does not use the actual simulation model")
    return root, anchor, frames, raw, home


def diagnostic_class():
    base, rclpy = replay_class()
    from nominal_live_replay import HUMAN, deserialize_message
    from summarize_nominal_replay import joint_values, stamp

    class DiagnosticReplay(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.segmented_diag = {}
            self.segmented_commands = {}

        def observe_output(self, topic, value):
            super().observe_output(topic, value)
            if self.args.start_only_profile != "segmented":
                return
            header = stamp(value.get("header", {}).get("stamp"))
            if header is None:
                return
            now = self.node.get_clock().now().nanoseconds
            if not 0 <= now - header <= FRESH_NS:
                return
            self.segmented_commands = {key: item for key, item in self.segmented_commands.items() if now - key[1] <= FRESH_NS}
            for side in ("left", "right"):
                if topic == f"/{side}_hand/command" and value.get("side") == side:
                    sequence = value.get("sequence")
                    if type(sequence) is int and sequence > 0:
                        key = side, header, sequence
                        hand = joint_values(value, 20)
                        self.segmented_commands[key] = hand if key not in self.segmented_commands else None
            if topic != "/tianji/teleop_diagnostics":
                return
            for item in value.get("status", []):
                for side in ("left", "right"):
                    if (item.get("name") == f"tianji/{side}_hand/dcr009_segmented_target"
                            and item.get("hardware_id") == "dcr009_isolated_mujoco"):
                        fields = {field["key"]: field["value"] for field in item["values"]}
                        if fields.get("session") == self.session:
                            self.segmented_diag[side] = header, fields

        def segmented_hand_errors(self, q):
            now = self.node.get_clock().now().nanoseconds
            generation = {fields.get("pilot_generation") for _, fields in self.pilot_diagnostics.values()}
            errors, proof_ids = [], []
            for index, side in enumerate(("left", "right")):
                observed = self.segmented_diag.get(side)
                proof, error = None, None
                if observed is not None and 0 <= now - observed[0] <= FRESH_NS:
                    _, fields = observed
                    try:
                        header = int(fields["command_header_ns"])
                        sequence = int(fields["command_sequence"])
                        guard = int(fields["guard_sequence"])
                        raw = np.asarray([float(value) for value in fields["raw_wuji_target_rad"].split(",")])
                        issued = np.asarray([float(value) for value in fields["issued_target_rad"].split(",")])
                        names = self.plant_names[14 + 20 * index:34 + 20 * index]
                        actual = self.segmented_commands.get((side, header, sequence))
                        if (actual is not None and set(actual) == set(names) and 0 <= now - header <= FRESH_NS
                                and guard > 0 and generation == {fields["generation"]}
                                and raw.shape == issued.shape == (20,) and np.isfinite(raw).all()
                                and np.array_equal(issued, [actual[name] for name in names])):
                            error = float(np.max(np.abs(raw - [q[name] for name in names])))
                            proof = guard
                    except (KeyError, ValueError):
                        pass
                errors.append(error)
                proof_ids.append(proof)
            return errors, proof_ids

        def additional_start_checks(self, measurement):
            if self.args.start_only_profile != "segmented":
                return True
            return all(error is not None and error <= .10 for error in measurement["hand_error_to_unsegmented_wuji_rad"])

        def replay(self, events, warmup):
            if self.args.start_only_profile is not None:
                self.write("start_only_completed", profile=self.args.start_only_profile,
                           original_history_started=False)
                return
            return super().replay(events, warmup)

        def before_history(self, warmup):
            self.stage = "preposition_start"
            self.write("preposition_clock", center_offset_mm=self.args.center_offset_mm)
            return super().before_history(warmup)

        def measure(self):
            value = super().measure()
            if value is not None:
                diagnostics = [fields for _, fields in self.pilot_diagnostics.values()]
                sequences = {fields.get("pilot_guard_sequence") for fields in diagnostics}
                guarded = (all(fields.get("pilot_guarded_publish") == "true" for fields in diagnostics)
                           and len(sequences) == 1 and next(iter(sequences), None) not in {None, "", "0"})
                value["arms_accepted"] = value["arms_accepted"] and guarded
                value["arm_guard_proof"] = next(iter(sequences)) if guarded else None
                if self.args.start_only_profile == "segmented":
                    errors, proofs = self.segmented_hand_errors(value["named_q_rad"])
                    value["hand_error_to_unsegmented_wuji_rad"] = errors
                    value["hand_segment_guard_proofs"] = proofs
                if self.args.center_offset_mm is not None:
                    geometry = self.robot.geometry()
                    native = (np.linalg.inv(self.frames[1:3]) @ palm_poses(geometry)
                              @ self.model_palm_from_anatomical @ self.frames[5:7])
                    value["wrist_original_error_m_deg"] = pair_errors(self.original_goal, native)
                    scoring_wrist = np.repeat(np.eye(4)[None], 2, axis=0)
                    scoring_wrist[:, :3, :3] = geometry.rotations
                    scoring_wrist[:, :3, 3] = geometry.wrists
                    value["registered_wrist_error_m_deg"] = pair_errors(self.registered_goal, scoring_wrist)
            return value

        def service(self, operation, timeout=40):
            if operation == "home":
                from swept_live_guard import Reason

                self.wait(.10)
                if self.physical is None:
                    raise RuntimeError("No complete simulator state for the internal Home precheck")
                header, q, speed = self.physical
                now = self.node.get_clock().now().nanoseconds
                if not 0 <= now - header <= 40_000_000 or speed is None or speed > .16:
                    raise RuntimeError("Cold Home initialization is stale or moving")
                actual = np.asarray([q[name] for name in self.plant_names])
                desired = self.robot.home
                bounds = np.minimum(actual, desired), np.maximum(actual, desired)
                if np.max(np.abs(actual - desired)) > .01 or not self.home_guard._legal_box(*bounds):
                    raise RuntimeError("No tight legal startup Home envelope")
                statuses = [item.prove(*bounds, time.monotonic_ns() + 50_000_000)
                            for item in self.home_guard.envelopes]
                self.write("internal_home_guard", statuses=[status.name for status in statuses],
                           q_measured=actual.tolist(), q_home=desired.tolist(),
                           max_error_rad=float(np.max(np.abs(actual - desired))))
                if any(status != Reason.PERMIT for status in statuses):
                    raise RuntimeError("The internal Home target envelope is not proven in both models")
            return super().service(operation, timeout)

        def start_inputs(self, events):
            selected = {}
            witnessed = self.anchor["inputs"]
            if set(witnessed) != set(HUMAN):
                raise ValueError("Require four exact original human streams")
            for event in events:
                proof = witnessed[event.topic]
                if event.record_index != proof["record_index"]:
                    continue
                if (event.topic in selected or event.schedule_ns > self.anchor["source_cutoff_ns"]
                        or sha256(event.data).hexdigest() != proof["cdr_sha256"]):
                    raise ValueError("Diagnostic anchor input is duplicated, late, or differs in bytes")
                selected[event.topic] = event
            if set(selected) != set(HUMAN):
                raise ValueError("Diagnostic anchor source is incomplete")
            actual = []
            for topic in WRISTS:
                message = deserialize_message(selected[topic].data, self.messages[topic])
                if message.header.frame_id != "openvr_world":
                    raise ValueError("Registered wrist does not identify the validated virtual input frame")
                pose = message.pose
                actual.append(pose_from_xyz_quat([pose.position.x, pose.position.y, pose.position.z],
                    [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]))
            if not np.allclose(actual, self.anatomical_start, atol=1e-5, rtol=0):
                raise ValueError("Original ROS wrist poses disagree with verified HMB registered poses")
            offset = self.args.center_offset_mm or 0
            locked, registered, self.goal = registered_targets(self.frames, self.home_snapshot, np.asarray(actual), offset)
            yaw_evidence = {}
            if self.args.align_start_yaw_to_home:
                original_registered, original_native = registered, self.goal
                locked, registered, yaw = align_pair_yaw(locked, self.home_snapshot, registered)
                self.goal = np.linalg.inv(locked[1:3]) @ registered @ locked[3:5] @ locked[5:7]
                yaw_evidence = {"yaw_alignment": yaw, "unaligned_registered_wrist": original_registered.tolist(),
                                "unaligned_native_goal": original_native.tolist()}
            self.registered_goal = registered
            if self.args.center_offset_mm is not None:
                _, _, self.original_goal = registered_targets(self.frames, self.home_snapshot, np.asarray(actual))
            self.shared = locked[0]
            self.session_map_sha = map_file(self.args.session_map, self.session, locked)
            self.write("start_anchor", source_cutoff_ns=self.anchor["source_cutoff_ns"], inputs=witnessed,
                       selected_anchor_status=self.anchor["selected_anchor_status"], session=self.session,
                       session_map_sha256=self.session_map_sha, shared_transform=self.shared.tolist(),
                       start_world_registered_wrist=registered.tolist(), runtime_native_goal=self.goal.tolist(),
                       center_offset_mm=self.args.center_offset_mm, center_offset_world_m=[offset / 1000, 0., 0.],
                       scope=SCOPE, anatomical_mount_status="UNKNOWN", formal_coverage_credit=False, **yaw_evidence)
            return selected

    return DiagnosticReplay, rclpy


def pair_errors(target, actual):
    """Euclidean position then SO(3) geodesic angle, per named left/right side."""
    positions = np.linalg.norm(target[:, :3, 3] - actual[:, :3, 3], axis=1)
    relative = target[:, :3, :3].transpose(0, 2, 1) @ actual[:, :3, :3]
    angle = np.degrees(Rotation.from_matrix(relative).magnitude())
    return np.column_stack((positions, angle)).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "cohort", "virtual-axes", "legacy-frames", "frames", "home-reference", "model", "session-map",
                "guard-socket", "output", "controller-executable", "simulator-executable"):
        parser.add_argument(f"--{key}", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--signed-model", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--model-receipt", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--launch-pid", type=int, required=True)
    parser.add_argument("--center-offset-mm", type=int, choices=CENTER_OFFSET_SEQUENCE_MM)
    parser.add_argument("--start-only-profile", choices=("original", "segmented"))
    parser.add_argument("--align-start-yaw-to-home", action="store_true")
    args = parser.parse_args()
    root, anchor, frames, raw, home = load(args)
    from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel
    from nominal_live_replay import signed_events
    from swept_live_guard import EXACT_MODEL_RECEIPT_SHA, Guard, verified_robot

    events, receipt = signed_events(args.cohort, root["root"], None)
    if receipt["root"] != root or receipt["cohort_receipt_sha256"] != COHORT_SHA:
        raise ValueError("Raw input verifier selected a different source")
    model = RobotModel(args.model)
    model_audit = object_at(args.model_receipt, EXACT_MODEL_RECEIPT_SHA)["model_identity"]
    home_guard = Guard((model, verified_robot(args.signed_model, SIGNED_SHA, model_audit["signed"])))
    args.output.mkdir(exist_ok=False)
    receipt.update(scope=SCOPE, formal_coverage_credit=False, anatomical_mount_status="UNKNOWN",
                   cumulative_campaign_credit=False, residual_enabled=False, plan_sha256=args.plan_sha256,
                   shared_frames_sha256=digest(args.frames), historical_window=anchor["historical_window"])
    receipt["center_offset_mm"] = args.center_offset_mm
    receipt["start_only_profile"] = args.start_only_profile
    if args.align_start_yaw_to_home:
        receipt["align_start_yaw_to_home"] = True
    (args.output / "input.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    Runner, rclpy = diagnostic_class()
    rclpy.init()
    node = rclpy.create_node("s05_nominal_replay")
    try:
        with (args.output / "trace.jsonl").open("x", encoding="utf-8", buffering=1) as trace:
            runner = Runner(node, trace, args, anchor, frames, raw, home, model, np.repeat(np.eye(4)[None], 2, axis=0))
            runner.home_guard = home_guard
            runner.write("diagnostic_scope", scope=SCOPE, formal_coverage_credit=False, anatomical_mount_status="UNKNOWN")
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
