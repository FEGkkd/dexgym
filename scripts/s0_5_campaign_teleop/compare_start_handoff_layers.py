#!/usr/bin/env python3
"""Export one full signed HMB root with causal native START replay wrist layers."""

import argparse
from bisect import bisect_right
from collections import defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from compare_start_handoff import ordered_q, valid_feedback
from dexgym.tasks.direct.wholebody_s05.hmb_reader import HMBReader, SIMULATED
from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel, WRISTS
from dexgym.tasks.direct.wholebody_s05.v1.rmb.reader import RobotReferenceBank
from export_shared_rigid_candidate import MAPPING, MAX_AGE_NS, Observations, qualified_command
from hmb_cohort import IDENTITIES, validate_partitions
from pair_nominal_hmb import DCR_COMPARISON_SHA, digest, one, signed_feedback_names
from summarize_nominal_replay import ACTIVE, COMMANDS, DIAGNOSTICS, FEEDBACK, HISTORICAL, stamp


LAYERS = {
    "m1": "Unchanged signed independent M1 wrist intent",
    "shared_hmb_raw": "Native START rigid map applied to signed causal HMB tracker samples",
    "native_shared_raw": "Native target_input_raw on controller-received unfiltered Tracker",
    "native_filtered": "Native target_retargeted, alias target_raw, on filtered Tracker",
    "native_rate_limited": "Native solver step target when an IK attempt exists",
    "native_ik_requested": "Native last solver attempt target",
    "issued_fk": "Live MuJoCo FK of named published arm command",
    "measured_fk": "Live MuJoCo FK of same-stamp bilateral arm feedback",
}
LAYER_INDEX = {name: i for i, name in enumerate(LAYERS)}
SIDES = ("left", "right")
SLICES = (slice(0, 14), slice(14, 34), slice(34, 54))
FEEDBACK_GROUPS = (("/left_arm/joint_states", "/right_arm/joint_states"), ("/left_hand/state",), ("/right_hand/state",))
STATUS = {name: i for i, name in enumerate(("UNAVAILABLE", "VALID", "HMB_MASKED", "NO_NATIVE_TRACE",
    "OUTSIDE_REPLAY", "NO_ACTIVE_TASK_SESSION", "NO_FRESH_DIAGNOSTIC", "DIAGNOSTIC_UNQUALIFIED",
    "NO_START_LATCH", "NO_FRESH_ISSUED_ARMS", "NO_FRESH_MEASURED_ARMS"))}
DIAG_LAYERS = (("native_shared_raw", "target_input_raw", "target_input_raw_available"),
               ("native_filtered", "target_retargeted", "target_available"),
               ("native_rate_limited", "target_rate_limited", "attempt_target_available"),
               ("native_ik_requested", "target_ik_requested", "attempt_target_available"))
REASON_FIELDS = ("solve_mode", "recovery_action", "primary_reason", "rate_limited")
RAW_PARITY_TOLERANCE_M_RAD = (1e-4, 1e-3)


def se3(xyz, quat):
    if len(xyz) != 3 or len(quat) != 4 or not np.isfinite([*xyz, *quat]).all() or abs(np.linalg.norm(quat) - 1) > 1e-3:
        raise ValueError("Malformed reference or diagnostic SE(3)")
    result = np.eye(4)
    result[:3, 3] = xyz
    result[:3, :3] = Rotation.from_quat(quat).as_matrix()
    return result


def rigid(values):
    rotations = values[..., :3, :3]
    return (np.isfinite(values).all() and np.allclose(values[..., 3, :], [0, 0, 0, 1], atol=1e-9)
            and np.allclose(rotations @ np.swapaxes(rotations, -1, -2), np.eye(3), atol=1e-8)
            and np.allclose(np.linalg.det(rotations), 1, atol=1e-8))


def numeric(fields, key):
    return np.fromstring(fields[key], sep=",")


def diag_pose(fields, key):
    return se3(numeric(fields, key + "_xyz_m"), numeric(fields, key + "_quat_xyzw"))


def packed(value):
    return np.r_[value[:3, 3], Rotation.from_matrix(value[:3, :3]).as_quat()]


def read_trace(path, model_sha):
    streams, feedback = defaultdict(Observations), defaultdict(dict)
    phases, barriers = {}, []
    model, anchor, anchor_inputs, handoff, home, timebase, start, first, end = (None,) * 9
    session, epoch, opened, previous, source_count, duplicate_feedback = None, 0, None, -1, 0, 0
    trace_sha = sha256()
    small_fields = set(REASON_FIELDS) | {
        "session", "session_id", "mapping_policy", "tracker_source_stamp_sec", "tracker_cached_valid",
        "target_workspace_xyz_m", "workspace_target_available", "workspace_projected",
        "tracker_position_raw_xyz_m", "tracker_orientation_raw_quat_xyzw",
        "arm_command_accepted", "ik_target_accepted",
    }
    for _, prefix, valid in DIAG_LAYERS:
        small_fields.update((valid, prefix + "_xyz_m", prefix + "_quat_xyzw"))
    for prefix in ("robot_start", "tracker_start"):
        small_fields.update((prefix + "_xyz_m", prefix + "_quat_xyzw"))
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                raise ValueError("Incomplete native trace line")
            trace_sha.update(line)
            row = json.loads(line)
            arrival, kind, stage = row["mono_ns"], row["kind"], row["stage"]
            if arrival < previous:
                raise ValueError("Nonmonotonic native observer arrival")
            previous = arrival
            if kind == "model":
                if model is not None or row["sha256"] != model_sha:
                    raise ValueError("Unexpected simulator XML identity")
                model = row["sha256"]
            elif kind == "start_anchor":
                if anchor is not None:
                    raise ValueError("Duplicate START anchor")
                anchor = row["source_cutoff_ns"]
                anchor_inputs = row["inputs"]
            elif kind == "start_handoff":
                if handoff is not None:
                    raise ValueError("Duplicate START handoff")
                handoff = row
            elif kind == "home_feedback":
                if home is not None:
                    raise ValueError("Duplicate measured Home receipt")
                home = row
            elif kind == "timebase":
                if timebase is not None:
                    raise ValueError("Duplicate historical replay timebase")
                timebase = row
            elif kind == "service" and row["service"] == "/whole_body_teleop/start":
                if start is not None:
                    raise ValueError("Repeated native START")
                start = row["response"].get("success", row["response"].get("accepted", False))
            elif kind == "input" and stage == HISTORICAL:
                current = row["original_schedule_ns"]
                if end is not None and current < end:
                    raise ValueError("Historical source schedule recedes")
                first = current if first is None else first
                end, source_count = current, source_count + 1
            if kind != "output":
                continue
            topic, value = row["topic"], row["value"]
            header = stamp(value.get("header", {}).get("stamp"))
            if topic in ACTIVE:
                key, expected = ACTIVE[topic]
                candidate = value.get("session_id")
                phases[topic] = value.get(key) == expected, candidate, header
                ready = bool(candidate) and all(
                    name in phases and phases[name][0] and phases[name][1] == candidate and phases[name][2] is not None
                    for name in ACTIVE)
                new_session = candidate if ready else None
                if new_session != session:
                    session = new_session
                    opened = max(phases[name][2] for name in ACTIVE) if ready else None
                    epoch += 1
                    barriers.append((arrival, epoch if ready else 0, session, opened))
            if stage != HISTORICAL or session is None or header is None or header < opened:
                continue
            if topic in COMMANDS:
                streams[epoch, topic].add(arrival, header, value)
            if topic in FEEDBACK:
                group = feedback[epoch, header]
                old = group.get(topic)
                if old is None:
                    group[topic] = arrival, valid_feedback(topic, value), None
                else:
                    duplicate_feedback += 1
                    if old[2] is None:
                        group[topic] = old[0], old[1], arrival
            for status in value.get("status", ()):
                side = DIAGNOSTICS.get(status["name"])
                if side:
                    fields = {item["key"]: item["value"] for item in status["values"] if item["key"] in small_fields}
                    if fields.get("session", fields.get("session_id")) == session:
                        if fields.get("mapping_policy") != MAPPING:
                            raise ValueError("Historical diagnostics have a different native mapping policy")
                        streams[epoch, side].add(arrival, header, fields)
    if source_count and (model is None or anchor is None or start is not True or timebase is None
                         or timebase["origin_schedule_ns"] != anchor or first < anchor):
        raise ValueError("Recorded history does not have causal START/anchor/model lineage")
    frames = []
    for (group_epoch, header), group in feedback.items():
        if all(topic in group and group[topic][1] is not None for topic in FEEDBACK_GROUPS[0]):
            frames.append((max(group[topic][0] for topic in FEEDBACK_GROUPS[0]), header, group_epoch, group))
    for completed, header, group_epoch, group in sorted(frames, key=lambda item: item[:3]):
        streams[group_epoch, "measured"].add(completed, header, group)
    handoff = handoff or {}
    measurement = handoff.get("measurement")
    shared = np.asarray(measurement["shared_transform"], dtype=float) if measurement else None
    if shared is not None and (shared.shape != (4, 4) or not rigid(shared)):
        raise ValueError("Native START latch has an invalid shared transform")
    return dict(streams=streams, barriers=barriers, barrier_times=[row[0] for row in barriers],
                timebase=timebase, last_input=end, anchor=anchor, shared=shared, start_accepted=start is True,
                anchor_inputs=anchor_inputs, home=home, duplicate_feedback_records=duplicate_feedback,
                start_status=handoff.get("status", "NO_START_HANDOFF_RECEIPT"), input_count=source_count,
                continue_after_miss=handoff.get("continue_after_miss", False), trace_sha256=trace_sha.hexdigest())


def at_cutoff(trace, cutoff):
    if trace is None:
        return "NO_NATIVE_TRACE", None
    base, end = trace["timebase"], trace["last_input"]
    if base is None or end is None or not base["origin_schedule_ns"] <= cutoff <= end:
        return "OUTSIDE_REPLAY", None
    delta = cutoff - base["origin_schedule_ns"]
    mono = base["origin_mono_ns"] + delta
    ros = base["origin_log_ns"] + base["epoch_delta_ns"] + delta
    slot = bisect_right(trace["barrier_times"], mono) - 1
    if slot < 0:
        return "NO_ACTIVE_TASK_SESSION", None
    _, epoch, session, opened = trace["barriers"][slot]
    if not epoch or ros < opened:
        return "NO_ACTIVE_TASK_SESSION", None
    return "VALID", (mono, ros, epoch, session)


def start_from_diagnostic(frames, diagnostic):
    if any(item is None for item in diagnostic) or diagnostic[0][1] != diagnostic[1][1]:
        return None
    registered, bases, mounts, tool_inverse, _ = frames
    homes, hands = [], []
    for side, (_, _, fields) in enumerate(diagnostic):
        homes.append((bases[side] @ diag_pose(fields, "robot_start") @ tool_inverse[side])[:3, 3])
        hands.append((registered @ diag_pose(fields, "tracker_start") @ mounts[side])[:3, 3])
    shared = registered.copy()
    shared[:3, 3] += np.mean(homes, axis=0) - np.mean(hands, axis=0)
    return shared


def arrays_for(n):
    shape = n, len(LAYERS), 2
    out = {key: np.full(n, -1, dtype=np.int64) for key in (
        "cutoff_ns", "source_cutoff_ns", "partition_index", "partition_tick", "measured_header_ns",
        "replay_cutoff_header_ns", "replay_cutoff_arrival_ns", "active_epoch",
    )}
    out.update({key: np.full((n, width), -1, dtype=np.int64) for key, width in (
        ("hmb_tracker_arrival_hi_ns", 2), ("native_diag_header_ns", 2), ("native_diag_arrival_ns", 2),
        ("native_tracker_source_header_ns", 2), ("issued_header_ns", 3), ("issued_arrival_ns", 3),
        ("issued_sequence", 3), ("measured_arrival_ns", 3),
    )})
    out.update(hmb_joint_valid=np.zeros(n, bool), hmb_safety_permission=np.zeros(n, bool),
               wrist_pose_world=np.full((*shape, 7), np.nan), wrist_valid=np.zeros(shape, bool),
               wrist_status=np.full(shape, STATUS["UNAVAILABLE"], dtype=np.uint8),
               q_issued_rad=np.full((n, 54), np.nan), q_issued_valid=np.zeros((n, 3), bool),
               q_measured_rad=np.full((n, 54), np.nan), q_measured_valid=np.zeros((n, 3), bool),
               native_workspace_xyz_world=np.full((n, 2, 3), np.nan),
               native_workspace_valid=np.zeros((n, 2), bool), native_workspace_projected=np.zeros((n, 2), bool),
               raw_shared_reconstruction_error_m_rad=np.full((n, 2, 2), np.nan),
               native_reasons=np.full((n, 2, len(REASON_FIELDS)), "", dtype="U80"))
    return out


def mark(out, i, layer, side, status):
    out["wrist_status"][i, LAYER_INDEX[layer], side] = STATUS[status]


def emit(out, i, layer, side, pose):
    out["wrist_pose_world"][i, LAYER_INDEX[layer], side] = pose
    out["wrist_valid"][i, LAYER_INDEX[layer], side] = True
    mark(out, i, layer, side, "VALID")


def native_wrist(out, i, side, diagnostic, shared, frames):
    _, bases, mounts, tool_inverse, palm_to_site = frames
    if diagnostic is None:
        for layer, _, _ in DIAG_LAYERS:
            mark(out, i, layer, side, "NO_FRESH_DIAGNOSTIC")
        return
    arrival, header, fields = diagnostic
    out["native_diag_header_ns"][i, side] = header
    out["native_diag_arrival_ns"][i, side] = arrival
    out["native_reasons"][i, side] = [fields.get(key, "") for key in REASON_FIELDS]
    source = fields.get("tracker_source_stamp_sec")
    if source and np.isfinite(float(source)) and float(source) > 0:
        out["native_tracker_source_header_ns"][i, side] = round(float(source) * 1e9)
    local_from_tcp = tool_inverse[side] @ palm_to_site[side]
    raw_target = None
    for layer, prefix, validity in DIAG_LAYERS:
        if fields.get(validity) != "true":
            mark(out, i, layer, side, "DIAGNOSTIC_UNQUALIFIED")
            continue
        target = bases[side] @ diag_pose(fields, prefix)
        emit(out, i, layer, side, packed(target @ local_from_tcp))
        if layer == "native_shared_raw":
            raw_target = target
    if fields.get("workspace_target_available") == "true":
        point = numeric(fields, "target_workspace_xyz_m")
        if len(point) != 3 or not np.isfinite(point).all():
            raise ValueError("Invalid qualified native workspace target")
        out["native_workspace_xyz_world"][i, side] = (bases[side] @ np.r_[point, 1])[:3]
        out["native_workspace_valid"][i, side] = True
        out["native_workspace_projected"][i, side] = fields.get("workspace_projected") == "true"
    if raw_target is not None and shared is not None and fields.get("tracker_cached_valid") == "true":
        tracker = se3(numeric(fields, "tracker_position_raw_xyz_m"), numeric(fields, "tracker_orientation_raw_quat_xyzw"))
        predicted = shared @ tracker @ mounts[side] @ np.linalg.inv(tool_inverse[side])
        xyz = np.linalg.norm(predicted[:3, 3] - raw_target[:3, 3])
        angle = Rotation.from_matrix(predicted[:3, :3].T @ raw_target[:3, :3]).magnitude()
        out["raw_shared_reconstruction_error_m_rad"][i, side] = xyz, angle


def wrist_fk(robot, arms):
    q = robot.home.copy()
    q[:14] = arms
    robot.set(q)
    return np.stack([np.r_[robot.data.site(name).xpos,
                           Rotation.from_matrix(robot.data.site(name).xmat.reshape(3, 3)).as_quat()] for name in WRISTS])


def feedback_available(item, mono):
    return (item is not None and item[1] is not None and 0 <= mono - item[0] <= MAX_AGE_NS
            and (item[2] is None or mono < item[2]))


def measured_at(stream, mono, ros):
    """A duplicate affects a stamped group only after that duplicate reaches the observer."""
    stop = bisect_right(stream.keys, mono)
    start = bisect_right(stream.keys, mono - MAX_AGE_NS - 1, hi=stop)
    return max((record for record in stream.rows[start:stop] if 0 <= ros - record[1] <= MAX_AGE_NS
                and all(feedback_available(record[2][name], mono) for name in FEEDBACK_GROUPS[0])),
               key=lambda row: (row[1], row[0]), default=None)


def joints(out, i, timeline, trace, names, robot):
    mono, ros, epoch, session = timeline
    streams = trace["streams"]
    for group, (topic, part) in enumerate(zip(COMMANDS, SLICES, strict=True)):
        command = qualified_command(streams[epoch, topic].get(mono, ros), topic, session, ros, names[part])
        if command is not None:
            q, seq, arrival, header = command
            out["q_issued_rad"][i, part], out["q_issued_valid"][i, group] = q, True
            out["issued_sequence"][i, group], out["issued_arrival_ns"][i, group], out["issued_header_ns"][i, group] = seq, arrival, header
    measured = measured_at(streams[epoch, "measured"], mono, ros)
    if measured is not None:
        _, header, pieces = measured
        out["measured_header_ns"][i] = header
        for group, (topics, part) in enumerate(zip(FEEDBACK_GROUPS, SLICES, strict=True)):
            items = [pieces.get(topic) for topic in topics]
            if all(feedback_available(item, mono) for item in items):
                named = {name: q for _, item, _ in items for name, q in item.items()}
                if set(named) == set(names[part]):
                    out["q_measured_rad"][i, part] = [named[name] for name in names[part]]
                    out["measured_arrival_ns"][i, group] = max(item[0] for item in items)
                    out["q_measured_valid"][i, group] = True
    for key in ("issued", "measured"):
        if out[f"q_{key}_valid"][i, 0]:
            emit(out, i, f"{key}_fk", slice(None), wrist_fk(robot, out[f"q_{key}_rad"][i, :14]))
        else:
            mark(out, i, f"{key}_fk", slice(None), f"NO_FRESH_{key.upper()}_ARMS")


def fill(out, part, i, trace, raw, exact, clock, names, robot, frames):
    times = raw["cutoff_ns"]
    if (len(times) != part["planned_ticks"] or not np.array_equal(times, raw["tick_ns"])
            or not np.array_equal(times, exact["tick_ns"]) or np.any(np.diff(times) <= 0)
            or (i and times[0] <= out["cutoff_ns"][i - 1])):
        raise ValueError("HMB and signed M1 timetables disagree")
    if any(data[key].dtype != np.dtype(bool) for data, key in (
        (raw, "joint_valid"), (raw, "safety_permission"),
        (exact, "mapped_wrist_xyz_mask"), (exact, "mapped_wrist_xyzw_mask"),
    )):
        raise ValueError("Signed validity and safety masks must be boolean")
    _, _, mounts, _, palm_to_site = frames
    for tick, relative in enumerate(times):
        cutoff = clock + int(relative)
        out["cutoff_ns"][i], out["source_cutoff_ns"][i] = relative, cutoff
        out["partition_index"][i], out["partition_tick"][i] = part["index"], tick
        valid, permitted = raw["joint_valid"][tick], raw["safety_permission"][tick]
        out["hmb_joint_valid"][i], out["hmb_safety_permission"][i] = valid, permitted
        arrival = raw["arrival_bounds_ns"][tick, 2:4, 1]
        out["hmb_tracker_arrival_hi_ns"][i] = np.where(arrival >= 0, clock + arrival, -1)
        shared = trace["shared"] if trace is not None else None
        reason, timeline = at_cutoff(trace, cutoff)
        if timeline is None:
            out["wrist_status"][i, 2:] = STATUS[reason]
        else:
            mono, ros, epoch, _ = timeline
            out["replay_cutoff_header_ns"][i], out["replay_cutoff_arrival_ns"][i], out["active_epoch"][i] = ros, mono, epoch
            diagnostic = [trace["streams"][epoch, side].get(mono, ros) for side in SIDES]
            if shared is None:
                shared = start_from_diagnostic(frames, diagnostic)
            for side, item in enumerate(diagnostic):
                native_wrist(out, i, side, item, shared, frames)
            joints(out, i, timeline, trace, names, robot)
        for side in range(2):
            if exact["mapped_wrist_xyz_mask"][tick, side] and exact["mapped_wrist_xyzw_mask"][tick, side]:
                xyz, quat = exact["mapped_wrist_xyz"][tick, side], exact["mapped_wrist_xyzw"][tick, side]
                se3(xyz, quat)
                emit(out, i, "m1", side, np.r_[xyz, quat])
            else:
                mark(out, i, "m1", side, "HMB_MASKED")
            if not valid or not permitted:
                mark(out, i, "shared_hmb_raw", side, "HMB_MASKED")
            elif shared is None or trace["anchor"] is None or cutoff < trace["anchor"]:
                mark(out, i, "shared_hmb_raw", side, "NO_START_LATCH")
            else:
                pose = raw["pose"][tick, side + 2, 0]
                mapped = shared @ se3(pose[:3], pose[3:]) @ mounts[side] @ palm_to_site[side]
                emit(out, i, "shared_hmb_raw", side, packed(mapped))
        i += 1
    return i


def counts(out):
    valid = out["hmb_joint_valid"]
    report = {}
    for key, mask in (("planned", np.ones_like(valid)), ("hmb_joint_valid", valid)):
        errors = out["raw_shared_reconstruction_error_m_rad"][mask]
        present = np.isfinite(errors).all(axis=-1)
        report[key] = {"rows": int(mask.sum()), "wrist_both_by_layer": dict(zip(
            LAYERS, out["wrist_valid"][mask].all(axis=-1).sum(axis=0).tolist(), strict=True)),
            "native_raw_parity_available_side_rows": int(present.sum()),
            "native_raw_parity_within_tolerance_side_rows": int((errors[present] <= RAW_PARITY_TOLERANCE_M_RAD).all(axis=-1).sum())}
        for data in ("issued", "measured"):
            report[key][f"{data}_arms"] = int(out[f"q_{data}_valid"][mask, 0].sum())
            report[key][f"{data}_all_54"] = int(out[f"q_{data}_valid"][mask].all(axis=1).sum())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cohort", "frames", "direct-bank", "dcr-comparison", "model", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--root", help="Signed root ID, required when no native trace exists")
    parser.add_argument("--anchor", type=Path, help="Optional exact START anchor JSON for receipt and source cross-check")
    parser.add_argument("--docs", type=Path, default=Path(__file__).resolve().parents[2] / "docs/s0-5")
    parser.add_argument("--hmb-products", type=Path)
    args = parser.parse_args()
    if args.trace is None and args.root is None:
        parser.error("Provide a native trace or the signed ID of the missing-runtime root")
    cohort = json.loads(args.cohort.read_bytes())
    if (cohort["schema"] != "s05.teleop.hmb_source_cohort.v1" or cohort["native_assets_sha256_checked"] is not True
            or cohort["identity_sha256"] != {key: item[1] for key, item in IDENTITIES.items()}):
        raise ValueError("Expected the approved byte-verified native HMB cohort")
    receipt = json.loads(args.trace.with_name("input.json").read_bytes()) if args.trace else None
    root_id = args.root or receipt["root"]["root"]
    root = one([row for row in cohort["selected_roots"] if row["root"] == root_id], "signed root")
    validate_partitions(root, root["hmb_partitions"])
    cohort_sha, frames_sha = digest(args.cohort), digest(args.frames)
    if receipt and (receipt["root"] != root or receipt["cohort_receipt_sha256"] != cohort_sha
                    or receipt["shared_frames_sha256"] != frames_sha or receipt["condition"] != "preposition"):
        raise ValueError("Native trace receipt does not bind the signed root, bridge and preposition condition")
    anchor_sha = receipt["anchor_sha256"] if receipt else None
    anchor = None
    if args.anchor:
        anchor_data = args.anchor.read_bytes()
        anchor = json.loads(anchor_data)
        explicit_sha = sha256(anchor_data).hexdigest()
        if (anchor["root"] != root_id or anchor["cohort_sha256"] != cohort_sha
                or (anchor_sha is not None and explicit_sha != anchor_sha)):
            raise ValueError("Explicit START anchor differs from the signed root or native receipt")
        anchor_sha = explicit_sha
    manifest = (Path(root["canonical_episode_uri"]) / "manifest.yaml").read_bytes()
    if sha256(manifest).hexdigest() != root["manifest_sha256"]:
        raise ValueError("Signed capture manifest differs")
    clock = yaml.safe_load(manifest)["start_monotonic_ns"]
    if type(clock) is not int or clock <= 0:
        raise ValueError("Invalid recorder monotonic origin")
    if anchor and anchor["source_cutoff_ns"] != clock + anchor["hmb_cutoff_ns"]:
        raise ValueError("Explicit START anchor differs from signed recorder timebase")
    if digest(args.dcr_comparison) != DCR_COMPARISON_SHA:
        raise ValueError("Expected the signed DCR-008 comparison")
    frozen = json.loads(args.dcr_comparison.read_bytes())
    bank = RobotReferenceBank(args.direct_bank, expected_sha=frozen["identity_sha256"]["direct_bank"],
                              verify=True, require_qualified=False)
    evidence = args.docs / "evidence"
    hmb = HMBReader(evidence / IDENTITIES["hmb_manifest"][0], evidence / IDENTITIES["plan"][0],
                    evidence / IDENTITIES["config"][0], args.hmb_products or cohort["signed_hmb_ubuntu_products"],
                    root["split"], SIMULATED)
    robot = RobotModel(args.model)
    names = signed_feedback_names(robot)
    model_sha = digest(robot.path)
    bridge = np.repeat(np.eye(4)[None], 7, axis=0)
    bridge[:, :3] = np.loadtxt(args.frames).reshape(7, 3, 4)
    if not rigid(bridge):
        raise ValueError("Invalid shared rigid coordinate bridge")
    home = robot.geometry(robot.home)
    palm_to_site = [np.linalg.solve(se3(home.palm[s], Rotation.from_matrix(home.palm_rotation[s]).as_quat()),
                                  se3(home.wrists[s], Rotation.from_matrix(home.rotations[s]).as_quat())) for s in range(2)]
    frames = bridge[0], bridge[1:3], bridge[3:5], np.linalg.inv(bridge[5:7]), palm_to_site
    trace = read_trace(args.trace, model_sha) if args.trace else None
    if anchor and trace and trace["anchor"] is not None and (
        trace["anchor"] != anchor["source_cutoff_ns"] or trace["anchor_inputs"] != anchor["inputs"]
    ):
        raise ValueError("Observed START anchor differs from explicit signed input records")
    home_proof = None
    if trace and trace["home"] is not None:
        observed = trace["home"]
        home_q = ordered_q(observed["named_q_rad"], names)
        home_proof = {"q_rad": home_q.tolist(), "measured_header_ns": observed["measured_header_ns"],
                      "observed_arrival_ns": observed["mono_ns"],
                      "max_measured_speed_rad_s": observed.get("max_measured_speed_rad_s"),
                      "wrist_pose_world": wrist_fk(robot, home_q[:14]).tolist()}
    out = arrays_for(root["planned_ticks"])
    i = 0
    for part in root["hmb_partitions"]:
        raw, meta, _, _ = hmb.read(part["key"])
        direct = bank.read(part["key"], root["split"])
        if meta["identity"]["root_episode_id"] != root_id or direct.root != root_id:
            raise ValueError("HMB and direct M1 root disagree")
        i = fill(out, part, i, trace, raw, direct.arrays, clock, names, robot, frames)
    if i != root["planned_ticks"]:
        raise ValueError("Missing signed planned rows")
    report = {
        "schema": "s05.teleop.start_handoff_wrist_layers.v1", "root": root_id, "split": root["split"],
        "wrist_layers": list(LAYERS), "wrist_layer_descriptions": LAYERS, "wrist_status_codes": STATUS,
        "joint_groups": ["bilateral_arm14", "left_hand20", "right_hand20"], "model_joint_names": list(robot.names),
        "native_joint_names": list(names), "frame": "MuJoCo world tcp_L/tcp_R site, XYZ metres and XYZW",
        "partitions": [{key: part[key] for key in ("index", "key", "planned_ticks")} for part in root["hmb_partitions"]],
        "cohort_sha256": cohort_sha, "signed_dcr_comparison_sha256": DCR_COMPARISON_SHA,
        "model_sha256": model_sha, "shared_frames_sha256": frames_sha, "source_manifest_sha256": root["manifest_sha256"],
        "trace_sha256": digest(args.trace) if args.trace else None,
        "native_trace_sha256_uncompressed": trace["trace_sha256"] if trace else None,
        "anchor_sha256": anchor_sha,
        "home_feedback": home_proof, "native_raw_parity_tolerance_m_rad": RAW_PARITY_TOLERANCE_M_RAD,
        "duplicate_feedback_records": trace["duplicate_feedback_records"] if trace else None,
        "native_start": {key: trace[key] for key in ("anchor", "start_status", "start_accepted", "input_count",
                                                    "continue_after_miss")} if trace else {"start_status": "NOT_RUN"},
        "counts": counts(out), "freshness_ns": MAX_AGE_NS,
        "notes": [
            "Original M1 wrist target comes from the pinned direct bank.",
            "HMB collector clocks, native ROS headers and native observer receipt clocks have separate columns.",
            "HMB shared target and native raw shared diagnostic use physically distinct as-of source choices.",
            "Native raw target parity residual independently reconstructs on diagnostic Tracker data when available.",
            "Native workspace position is the native TCP origin. Its quaternion is absent from diagnostics.",
            "Measured hands require the selected bilateral arm header stamp and causal receipt; each group is masked.",
            "Unavailable poses are NaN. This diagnostic supplies no Final Test, real robot or training qualification.",
        ],
    }
    args.out.mkdir(exist_ok=False, parents=True)
    data = args.out / "wrist_layers.npz"
    np.savez_compressed(data, **out)
    report["wrist_layers_npz_sha256"] = digest(data)
    report["npz_sha256"] = report["wrist_layers_npz_sha256"]
    with (args.out / "wrist_layer_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
