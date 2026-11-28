#!/usr/bin/env python3
"""Compare native immediate replay and held START using actual MuJoCo feedback."""

import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel
from hmb_cohort import IDENTITIES, INPUT_ROLES, TICK_NS, checked
from pair_nominal_hmb import PastMeasured, digest, normalized_assets, one, signed_feedback_names
from summarize_nominal_replay import ACTIVE, DIAGNOSTICS, FEEDBACK, HISTORICAL, joint_values, stamp


ROOT_03 = "capture:a34b6b9544d8f84e440282be"
FIXED_HMB_CUTOFF_NS = 56_720_000_069
MAX_AGE_NS = 40_000_000
SIDE_LABELS = ("left", "right")
HUMAN_TOPICS = {item[0] for item in INPUT_ROLES.values()}
MEASUREMENTS = ("wrist_error_m_deg", "hand_error_to_issued_rad", "measured_speed_rad_s",
                "active_task", "arms_accepted", "control_reasons")
METRICS = ("palm_origin_gap_mm", "right_minus_left_palm_height_mm", "right_minus_left_palm_foreaft_mm",
           "acute_palm_plane_deg", "directed_palm_normals_deg", "closest_cross_hand_tip_mm")
START_CRITERIA = {"ramp_sec": 1.5, "dwell_sec": 0.2, "timeout_sec": 10.0, "wrist_position_m": 0.010,
                  "wrist_rotation_deg": 5.0, "hand_error_to_issued_rad": 0.10, "joint_speed_rad_s": 0.16}


def read_receipt(trace, anchor, anchor_sha, condition):
    receipt = json.loads(trace.with_name("input.json").read_bytes())
    if (receipt["condition"] != condition or receipt["anchor_sha256"] != anchor_sha
            or receipt["cohort_receipt_sha256"] != anchor["cohort_sha256"]
            or receipt["root"]["root"] != anchor["root"]):
        raise ValueError(f"{condition} input receipt does not bind the requested signed anchor")
    return receipt


def ordered_q(named, names):
    if set(named) != set(names) or len(named) != 54:
        raise ValueError("Observed state cannot bind all 54 signed joint names")
    values = np.asarray([named[name] for name in names], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Invalid measured joint value")
    return values


def unit(vectors):
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(lengths <= 1e-12):
        raise ValueError("Degenerate palm geometry")
    return vectors / lengths


def pose_geometry(model, q):
    """Apply the historical 03 palm landmark, centre, and mirrored normal definitions."""
    measured = model.geometry(q)
    palms, fingers = measured.palm, measured.landmarks
    knuckles = fingers[:, 1:, 0]
    centres = (palms + knuckles.sum(axis=1)) / 5
    forward = unit(knuckles.mean(axis=1) - palms)
    across = knuckles[:, 0] - knuckles[:, -1]
    lateral = unit(across - (across * forward).sum(axis=1, keepdims=True) * forward)
    normals = np.cross(forward, lateral) * np.array([1., -1.])[:, None]
    alignment = float(np.clip(normals[0] @ normals[1], -1, 1))
    delta = centres[1] - centres[0]
    tips = fingers[:, :, -1]
    nearest = np.linalg.norm(tips[0, :, None] - tips[1, None, :], axis=-1).min()
    metrics = (np.linalg.norm(palms[0] - palms[1]) * 1000, delta[2] * 1000, -delta[1] * 1000,
               np.degrees(np.arccos(abs(alignment))), np.degrees(np.arccos(alignment)), nearest * 1000)
    right_in_left = measured.rotations[0].T @ (measured.wrists[1] - measured.wrists[0])
    wrist_angle = Rotation.from_matrix(measured.rotations[0].T @ measured.rotations[1]).magnitude()
    return {"palm_origins_xyz_m": palms.tolist(), "finger_landmarks_xyz_m": fingers.tolist(),
            "palm_centres_xyz_m": centres.tolist(), "palm_normals_world": normals.tolist(),
            "palm_normal_angle_to_world_down_deg": np.degrees(np.arccos(np.clip(-normals[:, 2], -1, 1))).tolist(),
            "wrist_tcp_xyz_m": measured.wrists.tolist(), "right_wrist_in_left_mm": (right_in_left * 1000).tolist(),
            "relative_wrist_so3_deg": float(np.degrees(wrist_angle)),
            "palm_metrics": {key: float(value) for key, value in zip(METRICS, metrics, strict=True)}}


def valid_feedback(topic, message):
    if topic.endswith("/state") and (
        message.get("connected") is not True or message.get("online") != [True] * FEEDBACK[topic]
        or message.get("stale") != [False] * FEEDBACK[topic]
        or message.get("position_valid") != [True] * FEEDBACK[topic]
    ):
        return None
    return joint_values(message, FEEDBACK[topic])


def read_trace(path, anchor):
    models, anchors, homes, handoffs, timebases, samples = [], [], [], [], [], []
    parameters, services, phases, groups = {}, defaultdict(list), {}, defaultdict(dict)
    input_times, segments, historical_diagnostics = [], [], defaultdict(list)
    mapping_modes = set()
    input_digest, trace_digest = sha256(), sha256()
    epoch, active_session, barrier_header, last_arrival = 0, None, None, -1
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                raise ValueError(f"Incomplete trace line in {path}")
            trace_digest.update(line)
            row = json.loads(line)
            arrival, kind, stage = row["mono_ns"], row["kind"], row["stage"]
            if arrival < last_arrival:
                raise ValueError(f"Trace arrival runs backwards in {path}")
            last_arrival = arrival
            if kind == "model":
                models.append(row["sha256"])
            elif kind == "parameters":
                if row["node_name"] in parameters:
                    raise ValueError("Duplicate effective parameter receipt")
                parameters[row["node_name"]] = row["values"]
            elif kind == "service":
                services[row["service"]].append(row["response"])
            elif kind == "start_anchor":
                anchors.append(row)
            elif kind == "home_feedback":
                homes.append(row)
            elif kind == "start_measurement":
                samples.append(row)
            elif kind == "start_handoff":
                handoffs.append(row)
            elif kind == "timebase":
                timebases.append(row)
            elif kind == "input" and stage == HISTORICAL:
                keys = ("topic", "record_index", "original_schedule_ns", "original_log_ns", "original_stamp_ns")
                original = row["original_schedule_ns"]
                if (row["topic"] not in HUMAN_TOPICS or original < anchor["source_cutoff_ns"]
                        or (input_times and original < input_times[-1])):
                    raise ValueError("The historical input sequence is outside the signed forward timeline")
                input_times.append(original)
                input_digest.update(json.dumps([row[key] for key in keys]).encode() + b"\n")
            if kind != "output":
                continue
            topic, value = row["topic"], row["value"]
            header = stamp(value.get("header", {}).get("stamp"))
            if topic in ACTIVE:
                key, required = ACTIVE[topic]
                phases[topic] = value.get(key) == required, value.get("session_id"), header
                session = value.get("session_id")
                armed = bool(session) and all(name in phases and phases[name][0] and phases[name][1] == session
                                              and phases[name][2] is not None for name in ACTIVE)
                changed = (session if armed else None) != active_session
                if changed:
                    active_session = session if armed else None
                    epoch += 1
                    barrier_header = max(phases[name][2] for name in ACTIVE) if armed else None
                    segments.append((arrival, epoch if armed else 0))
            if stage != HISTORICAL or active_session is None or header is None or header < barrier_header:
                continue
            if topic in FEEDBACK:
                group = groups[epoch, header]
                group[topic] = (arrival, valid_feedback(topic, value)) if topic not in group else (arrival, None)
            for item in value.get("status", ()):
                side = DIAGNOSTICS.get(item["name"])
                if side:
                    fields = {entry["key"]: entry["value"] for entry in item["values"]}
                    if fields.get("session") == active_session:
                        mapping_modes.add(fields.get("mapping_policy"))
                        historical_diagnostics[epoch, side].append((arrival, header, fields))
    selected = one(anchors, "live signed START anchor")
    if selected["source_cutoff_ns"] != anchor["source_cutoff_ns"] or selected["inputs"] != anchor["inputs"]:
        raise ValueError("Native trace START selection differs from the signed record")
    if len(timebases) > 1 or (input_times and len(timebases) != 1):
        raise ValueError("Missing or repeated native historical replay timebase")
    timebase = timebases[0] if timebases else None
    if timebase and timebase["origin_schedule_ns"] != anchor["source_cutoff_ns"]:
        raise ValueError("Historical replay did not start at the signed anchor clock")
    if mapping_modes and mapping_modes != {"shared_rigid_diagnostic_v1"}:
        raise ValueError("Observed a different runtime mapping during the historical replay")
    frames = []
    for (session, header), group in groups.items():
        if group.keys() == FEEDBACK.keys() and all(q is not None for _, q in group.values()):
            named = {name: value for _, q in group.values() for name, value in q.items()}
            if len(named) == 54:
                frames.append((max(at for at, _ in group.values()), header, session, named))
    return {"model": one(models, "live model receipt"), "home": one(homes, "measured Home state"),
            "handoff": one(handoffs, "START handoff result") if handoffs else None,
            "samples": samples, "parameters": parameters, "services": dict(services),
            "trace_sha256_uncompressed": trace_digest.hexdigest(), "input_count": len(input_times),
            "input_digest": input_digest.hexdigest() if input_times else None,
            "last_source_schedule": input_times[-1] if input_times else None, "timebase": timebase,
            "segments": segments, "frames": sorted(frames, key=lambda item: item[:3]),
            "diagnostics": historical_diagnostics, "preanchor_input_messages": selected["preanchor_input_messages"]}


def observed_start(trace, model, names):
    home_q = ordered_q(trace["home"]["named_q_rad"], names)
    home = pose_geometry(model, home_q)
    handoff = trace["handoff"] or {}
    measurement = handoff.get("measurement")
    samples = trace["samples"]
    if any(a["elapsed_s"] >= b["elapsed_s"] for a, b in zip(samples, samples[1:])):
        raise ValueError("Measured START trajectory is not chronological")
    trajectory = [{"elapsed_s": row["elapsed_s"], "ready": row["ready"],
                   **{key: row["value"][key] for key in MEASUREMENTS}} for row in samples]
    ready = [row["elapsed_s"] for row in samples if row["ready"]]
    tail_ready = 0.0
    for row in reversed(samples):
        if not row["ready"]:
            break
        tail_ready = samples[-1]["elapsed_s"] - row["elapsed_s"]
    start = trace["services"].get("/whole_body_teleop/start", [])
    enable = trace["services"].get("/whole_body_teleop/enable", [])
    stop = trace["services"].get("/whole_body_teleop/stop", [])
    start_passes = sum(item.get("success", item.get("accepted", False)) is True for item in start)
    enable_passes = sum(item.get("success", item.get("accepted", False)) is True for item in enable)
    result = {"status": handoff.get("status", "NO_HANDOFF_RECEIPT"), "elapsed_s": handoff.get("elapsed_s"),
              "continue_after_miss": handoff.get("continue_after_miss", False),
              "start_calls": len(start), "start_calls_accepted": start_passes, "enable_calls": len(enable),
              "enable_calls_accepted": enable_passes, "stop_responses": stop,
              "home_measured": home, "home_q_rad": home_q.tolist(),
              "preanchor_input_messages": trace["preanchor_input_messages"],
              "first_ready_elapsed_s": ready[0] if ready else None, "last_ready_dwell_observed_s": tail_ready,
              "last_measurement": None, "trajectory": trajectory}
    if measurement is not None:
        q = ordered_q(measurement["named_q_rad"], names)
        result["last_measurement"] = {**{key: measurement[key] for key in MEASUREMENTS},
                                      "max_q_change_from_home_rad": float(np.abs(q - home_q).max()),
                                      "measured": pose_geometry(model, q),
                                      "native_goal": measurement["native_goal"],
                                      "shared_transform": measurement["shared_transform"]}
    return result


def diagnostic_at(trace, source_cutoff):
    base = trace["timebase"]
    offset = source_cutoff - base["origin_schedule_ns"]
    cutoff_mono = base["origin_mono_ns"] + offset
    cutoff_ros = base["epoch_delta_ns"] + base["origin_log_ns"] + offset
    epoch = next((epoch for at, epoch in reversed(trace["segments"]) if at <= cutoff_mono), 0)
    result = {}
    for side in SIDE_LABELS:
        candidates = ((header, arrival, fields) for arrival, header, fields in trace["diagnostics"][epoch, side]
                      if 0 <= cutoff_ros - header <= MAX_AGE_NS and 0 <= cutoff_mono - arrival <= MAX_AGE_NS)
        found = max(candidates, key=lambda record: record[:2], default=None)
        result[side] = {name: found[2].get(name) for name in (
            "mapping_policy", "arm_command_accepted", "ik_target_accepted", "rate_limited", "solve_mode",
            "primary_reason", "recovery_action", "servo_boundary_active", "servo_reason",
        )} if found else None
    return result


def snapshot(at, traces, aligners, model):
    results = {}
    for condition, aligner in aligners.items():
        status, q, age, completion = aligner.get(at)
        item = {"status": status, "header_age_ns": age, "receive_age_ns": completion}
        if q is not None:
            item.update(pose_geometry(model, q), measured_q_rad=q.tolist(), diagnostics=diagnostic_at(traces[condition], at))
        results[condition] = item
    return {"source_cutoff_ns": at, "common_active_measured": all(x["status"] == "COHERENT_MEASURED"
                                                                   for x in results.values()), **results}


def snapshots(anchor, traces, model, names, plan_path):
    anchor_offset = anchor["hmb_cutoff_ns"]
    clock = anchor["source_cutoff_ns"] - anchor_offset
    fixed = clock + FIXED_HMB_CUTOFF_NS
    plan = json.loads(checked(plan_path, IDENTITIES["plan"][1]))
    part = one([row for row in plan["partitions"] if row["key"] == anchor["partition"]
                and row["parent"] == anchor["root"] and row["split"] == anchor["split"]], "signed START partition")
    if not 0 <= anchor["partition_tick"] < part["planned_ticks"]:
        raise ValueError("START anchor tick lies outside the signed shard")
    limit = part["planned_ticks"] - anchor["partition_tick"]
    aligners = {condition: PastMeasured(trace, model.names, MAX_AGE_NS, names) for condition, trace in traces.items()}
    first, statuses = None, Counter()
    for i in range(limit):
        candidate = snapshot(anchor["source_cutoff_ns"] + i * TICK_NS, traces, aligners, model)
        statuses.update((candidate["immediate"]["status"], candidate["preposition"]["status"]))
        if candidate["common_active_measured"]:
            first = candidate
            first.update(hmb_cutoff_ns=candidate["source_cutoff_ns"] - clock,
                         seconds_after_anchor=(candidate["source_cutoff_ns"] - anchor["source_cutoff_ns"]) / 1e9)
            break
    fixed_aligners = {condition: PastMeasured(trace, model.names, MAX_AGE_NS, names)
                      for condition, trace in traces.items()}
    fixed_sample = snapshot(fixed, traces, fixed_aligners, model)
    fixed_sample.update(hmb_cutoff_ns=FIXED_HMB_CUTOFF_NS, prior_clip_seconds=7.941974178,
                        seconds_after_anchor=(fixed - anchor["source_cutoff_ns"]) / 1e9)
    return {"first_common_in_signed_anchor_partition": first, "first_search_status_counts": dict(statuses),
            "prior_03_7p94_snapshot": fixed_sample}


def add_reference(report, path, anchor):
    raw = path.read_bytes()
    ref = json.loads(raw)
    if ref["identity"] != anchor["root"] or ref["target_source_cutoff_ns"] != FIXED_HMB_CUTOFF_NS:
        raise ValueError("Prior 03 snapshot is from a different signed cutoff")
    report["prior_reference_7p94"] = {"sha256": sha256(raw).hexdigest(),
                                    "human_calibration_scope": ref["human_calibration_scope"],
                                    "metrics_format": ref["metrics_format"], "human": ref["human"]["metrics"],
                                    "historical_B": ref["B"]["geom"]["metrics"]}


def plots(report, directory):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    paths = []
    available = [(title, report["snapshots"].get(key)) for title, key in (
        ("First common active signed cutoff", "first_common_in_signed_anchor_partition"),
        ("03 at original video 7.94 s", "prior_03_7p94_snapshot"),
    )]
    for page, (title, item) in enumerate(available):
        if not item or not item["common_active_measured"]:
            continue
        figure = plt.figure(figsize=(10, 5))
        points = []
        for key in ("immediate", "preposition"):
            row = item[key]
            center = np.asarray(row["palm_centres_xyz_m"])
            offset = center.mean(axis=0)
            axes = np.array([1., -1., 1.])
            points.append(((np.asarray(row["palm_origins_xyz_m"]) - offset) * axes,
                           (np.asarray(row["finger_landmarks_xyz_m"]) - offset) * axes,
                           (center - offset) * axes, np.asarray(row["palm_normals_world"]) * axes))
        extent = max(.12, *(float(np.max(np.abs(fingers))) + .06 for _, fingers, _, _ in points))
        for col, (label, p) in enumerate(zip(("Immediate shared START", "Pre-position shared START"), points, strict=True), 1):
            axis = figure.add_subplot(1, 2, col, projection="3d")
            axis.set_proj_type("ortho")
            axis.view_init(elev=24, azim=-55)
            axis.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent), title=label)
            axis.set_box_aspect((1, 1, 1))
            axis.set_axis_off()
            palms, fingers, centers, normals = p
            for side, color in enumerate(("#1774a7", "#ac4e28")):
                for finger in fingers[side]:
                    axis.plot(*np.vstack((palms[side], finger)).T, color=color)
                axis.quiver(*centers[side], *normals[side], length=.07, color=color, normalize=True)
        figure.suptitle(title)
        figure.text(.5, .04, "Live MuJoCo joint feedback | Blue left, red right | Shared scale and camera", ha="center")
        name = ("first_common_palms.png", "same_cutoff_7p94_palms.png")[page]
        figure.savefig(directory / name, dpi=155)
        plt.close(figure)
        paths.append(name)
    trajectory = report["start"]["preposition"]["trajectory"]
    if trajectory:
        figure, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, layout="constrained")
        times = [row["elapsed_s"] for row in trajectory]
        for side, label in enumerate(SIDE_LABELS):
            axes[0, 0].plot(times, [row["wrist_error_m_deg"][side][0] * 1000 for row in trajectory], label=label)
            axes[0, 1].plot(times, [row["wrist_error_m_deg"][side][1] for row in trajectory])
            axes[1, 0].plot(times, [row["hand_error_to_issued_rad"][side] for row in trajectory])
        axes[1, 1].plot(times, [row["measured_speed_rad_s"] for row in trajectory], color="#525267")
        limits = (START_CRITERIA["wrist_position_m"] * 1000, START_CRITERIA["wrist_rotation_deg"],
                  START_CRITERIA["hand_error_to_issued_rad"], START_CRITERIA["joint_speed_rad_s"])
        for axis, threshold, label in zip(axes.flat, limits, ("Wrist error (mm)", "Wrist SO(3) error (deg)",
                                                            "Measured/issued hand error (rad)",
                                                            "Peak measured joint speed (rad/s)"), strict=True):
            axis.set(ylabel=label, xlabel="Seconds after Enable")
            axis.axhline(threshold, color="#983d22", linestyle=":", alpha=.75)
            axis.grid(alpha=.2)
        axes[0, 0].legend()
        figure.savefig(directory / "preposition_trajectory.png", dpi=155)
        plt.close(figure)
        paths.append("preposition_trajectory.png")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("immediate", "preposition", "anchor", "model", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--reference-snapshot", type=Path)
    parser.add_argument("--docs", type=Path, default=Path(__file__).resolve().parents[2] / "docs/s0-5")
    args = parser.parse_args()
    anchor_raw = args.anchor.read_bytes()
    anchor = json.loads(anchor_raw)
    anchor_sha = sha256(anchor_raw).hexdigest()
    if anchor["schema"] != "s05.teleop.preposition_start_handoff.v1" or anchor["root"] != ROOT_03:
        raise ValueError("Expected the signed 03 START comparison anchor")
    inputs = {key: getattr(args, key) for key in ("immediate", "preposition")}
    receipts = {key: read_receipt(path, anchor, anchor_sha, key) for key, path in inputs.items()}
    if set(receipts["immediate"]) != set(receipts["preposition"]) or any(
        receipts["immediate"][key] != receipts["preposition"][key] for key in receipts["immediate"] if key != "condition"
    ):
        raise ValueError("Native input provenance or shared mapping frames differ")
    traces = {key: read_trace(path, anchor) for key, path in inputs.items()}
    model = RobotModel(args.model)
    model_sha = digest(model.path)
    names = signed_feedback_names(model)
    first, second = traces.values()
    if first["model"] != model_sha or second["model"] != model_sha:
        raise ValueError("Actual simulator model XML differs from the scoring model")
    parameters_equal = first["parameters"] == second["parameters"] and len(first["parameters"]) == 2
    start = {key: observed_start(trace, model, names) for key, trace in traces.items()}
    histories = first["input_count"] > 0 and second["input_count"] > 0
    event_digest_equal = (first["input_digest"] == second["input_digest"]
                          and first["input_count"] == second["input_count"]) if histories else None
    once_each = all(item["start_calls"] == item["start_calls_accepted"] == item["enable_calls"] == item["enable_calls_accepted"] == 1
                    for item in start.values())
    snapshot_blockers = [reason for passed, reason in (
        (parameters_equal, "EFFECTIVE_PARAMETERS_DIFFER_OR_ARE_INCOMPLETE"),
        (once_each, "NATIVE_START_OR_ENABLE_RECEIPT_MISSING_OR_REPEATED"),
        (event_digest_equal, "HISTORICAL_INPUTS_DIFFER" if histories else "HISTORICAL_INPUTS_MISSING"),
    ) if not passed]
    report = {"scope": "NATIVE_PREPOSITION_START_DIAGNOSTIC", "root": ROOT_03, "anchor_sha256": anchor_sha,
              "shared_frames_sha256": receipts["immediate"]["shared_frames_sha256"],
              "model_sha256": model_sha, "model_assets_at_scoring_sha256": normalized_assets(model.path),
              "observed_parameters_equal": parameters_equal, "one_native_accepted_start_and_enable_each": once_each,
              "original_input_digest_equal": event_digest_equal,
              "historical_input_counts": {key: trace["input_count"] for key, trace in traces.items()},
              "historical_input_sha256": {key: trace["input_digest"] for key, trace in traces.items()},
              "uncompressed_trace_sha256": {key: trace["trace_sha256_uncompressed"] for key, trace in traces.items()},
              "preposition_criteria": START_CRITERIA, "snapshot_blockers": snapshot_blockers,
              "start": start, "max_abs_home_q_difference_rad": float(np.max(np.abs(
                  np.array(start["immediate"]["home_q_rad"]) - start["preposition"]["home_q_rad"]))),
              "snapshots": {}, "limits": [
                  "The first common snapshot is searched on the signed anchor partition grid at 50 Hz.",
                  "A common snapshot needs same-session ACTIVE/TASK, four same-stamp named feedback messages and 40 ms header/arrival freshness.",
                  "The historical replay may continue after an explicit START miss in diagnostic mode.",
                  "Closest fingertip separation and palm normals describe the robot geometry. Independent human fingertip error is unavailable.",
              ]}
    if not snapshot_blockers:
        plan = args.docs / "evidence" / IDENTITIES["plan"][0]
        report["snapshots"] = snapshots(anchor, traces, model, names, plan)
    if args.reference_snapshot:
        add_reference(report, args.reference_snapshot, anchor)
    args.out.mkdir(exist_ok=False)
    report["plots"] = plots(report, args.out)
    with (args.out / "comparison.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
