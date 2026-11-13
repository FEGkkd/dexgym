"""Recover rate-limited wholebody_s0 motion on a slower 50 Hz reference clock."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import SIDES
from .bag import MAIN_STREAMS, Track, digest
from .first_wave import write_library
from .library import MotionLibrary
from .pipeline import FILTER, clip_ranges, read_json, read_npz, velocity
from .prepare import write_json, write_npz
from .retarget import run_arm_tool, run_hand_sdk
from .robot import Robot
from .sync import MANUS_GAP_NS, PERIOD_NS, RATE_HZ, TRACKER_GAP_NS, causal_index, interpolate, runs, sample_indices


DILATIONS = (2, 3)
MIN_SATURATED_STEPS = 5


@dataclass(frozen=True)
class Window:
    start: int
    stop: int
    factor: int
    speed_evidence: dict


def speed_windows(aligned, reference, quality, config, occupied):
    time = aligned["time_ns"]
    rate = config["teleop"]
    human = aligned["human_valid"] & reference["hand_valid"].all(axis=1)
    blocked = np.zeros(len(time), dtype=bool)
    for reason in ("joint_bounds", "static_contact", "bimanual_start_unaligned", "nonfinite_joint"):
        if reason in quality["rejection_bit_names"]:
            bit = quality["rejection_bit_names"].index(reason)
            blocked |= (reference["rejection_bits"] & (1 << bit)) != 0
    eligible = human & ~occupied & ~blocked
    pose = reference["target_tcp_world"].astype(float)
    both = eligible[1:] & eligible[:-1] & np.isfinite(pose[1:]).all((1, 2)) & np.isfinite(pose[:-1]).all((1, 2))
    ratios = np.zeros((len(time) - 1, 2), dtype=float)
    if both.any():
        a, b = pose[:-1][both], pose[1:][both]
        translation = np.linalg.norm(b[..., :3] - a[..., :3], axis=-1) * RATE_HZ
        ra = Rotation.from_quat(a[..., 3:].reshape(-1, 4))
        rb = Rotation.from_quat(b[..., 3:].reshape(-1, 4))
        rotation = (ra.inv() * rb).magnitude().reshape(-1, 2) * RATE_HZ
        ratios[both] = np.maximum(translation / rate["arm.max_translation_rate_m_sec"],
                                  rotation / rate["arm.max_rotation_rate_rad_sec"])
    limited = reference["arm_rate_limited"][1:] & both[:, None]
    arm_fast = ((ratios > 1.0) & limited).any(axis=1)
    # Joint-space saturation can occur even while the TCP speeds stay in range.
    command_delta = np.max(np.abs(np.diff(reference["arm_q"].reshape(-1, 2, 7), axis=0)), axis=2) * RATE_HZ
    command_saturated = (command_delta >= .99 * rate["arm.max_command_rate_rad_sec"]) & limited
    mapping_exceeded = (reference["arm_position_error_m"][1:] > FILTER["max_target_position_error_m"]) | (
        np.degrees(reference["arm_orientation_error_rad"][1:]) > FILTER["max_target_orientation_error_deg"]
    )
    joint_fast = (command_saturated & mapping_exceeded).any(axis=1)
    hand_speed = np.max(np.abs(np.diff(reference["q_ref"][:, 14:], axis=0)), axis=1) * RATE_HZ
    hand_fast = (hand_speed > FILTER["max_hand_speed_rad_s"]) & both
    for a, b in runs(eligible):
        a, b = int(a), int(b)
        if b - a < 2:
            continue
        interval = slice(a, b - 1)
        counts = [int(mask[interval].sum()) for mask in (arm_fast, joint_fast, hand_fast)]
        if max(counts) < MIN_SATURATED_STEPS:
            continue
        peak = float(ratios[interval].max(initial=0))
        estimate = max(float(np.quantile(ratios[interval].max(axis=1), .95)), 1.)
        factor = DILATIONS[-1] if estimate > DILATIONS[0] else DILATIONS[0]
        if (b - a - 1) * factor + 1 < FILTER["minimum_clip_frames"]:
            continue
        yield Window(a, b, factor, {"tcp_rate_limited_intervals": counts[0], "arm_joint_saturated_intervals": counts[1],
                                    "fast_hand_intervals": counts[2], "peak_tcp_speed_ratio": peak,
                                    "p95_tcp_speed_ratio": estimate})


def arm_input(aligned, start, capture):
    """Use the original START history, then play the sampled poses at 50 Hz."""
    clock = np.arange(start + len(capture["time_ns"]), dtype=np.int64) * PERIOD_NS
    output = {"time_s": clock / 1e9}
    for side in SIDES:
        for suffix in ("pose", "valid"):
            key = f"tracker_{side}_{suffix}"
            output[key] = np.concatenate((aligned[key][:start], capture[key]))
    key = "tracker_safety_valid"
    output[key] = np.r_[aligned[key][:start], capture[key]]
    return output


def capture_at(parsed, query):
    """Read recorded poses on the source clock with the original gap and safety rules."""
    capture = {"time_ns": query}
    for key in MAIN_STREAMS:
        prefix = f"{key}__"
        arrays = {name: parsed[prefix + name] for name in ("pose", "header_ns", "valid", "receive_ros_ns")}
        track = Track(arrays, {})
        manus = key.startswith("manus_")
        max_gap = MANUS_GAP_NS if manus else TRACKER_GAP_NS
        omit = parsed[prefix + "duplicate_source"] if manus else None
        indexes = sample_indices(track.header, track.valid, query, max_gap, omit)
        capture[key + "_pose"] = interpolate(arrays["pose"], track.header, query, indexes, poses=True)
        capture[key + "_valid"] = indexes[:, 0] >= 0
        capture[key + "_brackets"] = indexes
        capture[key + "_causal_index"] = causal_index(track, query, max_gap)
    safety = "tracker_safety__"
    if safety + "header_ns" in parsed:
        track = Track({name: parsed[safety + name] for name in ("header_ns", "valid", "receive_ros_ns")}, {})
        capture["tracker_safety_valid"] = causal_index(track, query, TRACKER_GAP_NS) >= 0
    else:
        capture["tracker_safety_valid"] = np.ones(len(query), dtype=bool)
    capture["human_valid"] = np.logical_and.reduce([capture[key + "_valid"] for key in MAIN_STREAMS])
    capture["human_valid"] &= capture["tracker_safety_valid"]
    return capture


def retarget_window(window, aligned, parsed, streams, config, robot, assets):
    a, b, factor = window.start, window.stop, window.factor
    source = np.arange(a * factor, (b - 1) * factor + 1, dtype=np.int64)
    source_time = aligned["time_ns"][0] + source * PERIOD_NS // factor
    captured = capture_at(parsed, source_time)
    arms, anchor = run_arm_tool(arm_input(aligned, a, captured), config, robot, assets["offline_tool"], assets["marvin_ccs"])
    arms = {key: value[a:] for key, value in arms.items()}
    # The hand SDK still sees the recorded messages in order, with the original
    # smoothing and receive-latency checks. Its output is sampled on the source clock.
    hand, hand_valid, _ = run_hand_sdk(parsed, captured, streams, config, assets["wuji_sdk"])
    uncapped = np.column_stack((arms["arm_q"], hand))
    q = np.clip(uncapped, robot.bounds[:, 0], robot.bounds[:, 1]).astype(np.float32)
    adjustments = np.abs(q - uncapped)
    finite = np.isfinite(q).all(1)
    geo = robot.geometry(np.where(np.isfinite(q), q, robot.home))
    target = robot.world_tcp(arms["target_tcp_base"])
    native = robot.world_tcp(arms["native_tcp_base"])
    error = np.linalg.norm(native[..., :3] - geo["robot_wrist_pose"][..., :3], axis=-1)
    if not np.isfinite(error).all() or error.max(initial=0) > .002:
        raise ValueError("Retimed native arm FK disagrees with the reference MJCF")
    checks = {
        "human_gap": ~captured["human_valid"], "arm_ik_or_hold": ~arms["arm_accepted"].all(axis=1),
        "hand_sdk_unavailable": ~hand_valid.all(axis=1), "nonfinite_joint": ~finite,
        "joint_bounds": (adjustments[:, :14].max(1) > 1e-6) | (adjustments[:, 14:].max(1) > FILTER["max_hand_limit_projection_rad"]),
        "position_tracking": ~np.isfinite(arms["arm_position_error_m"]).all(1)
                             | (arms["arm_position_error_m"] > FILTER["max_target_position_error_m"]).any(1),
        "orientation_tracking": ~np.isfinite(arms["arm_orientation_error_rad"]).all(1)
                                | (np.degrees(arms["arm_orientation_error_rad"]) > FILTER["max_target_orientation_error_deg"]).any(1),
        "static_contact": geo["static_penetration_m"] > FILTER["max_static_penetration_m"],
    }
    if config["teleop"].get("arm.bimanual_registration_enabled", False):
        tolerance = config["teleop"].get("arm.bimanual_position_tolerance_m", .01)
        checks["bimanual_start_unaligned"] = np.full(len(q), anchor["relative_wrist_vector_error_m"] > tolerance)
    usable = ~np.logical_or.reduce(tuple(checks.values()))
    fast = np.r_[False, np.max(np.abs(np.diff(q[:, 14:], axis=0)), axis=1) * RATE_HZ > FILTER["max_hand_speed_rad_s"]]
    fast[1:] &= usable[:-1] & usable[1:]
    checks["hand_step_speed"] = fast
    usable &= ~fast
    # Reject intervals whose original capture has a gap at an intermediate sample.
    source_brackets = np.searchsorted(aligned["time_ns"], source_time, side="right") - 1
    if (source_brackets < a).any() or (source_brackets >= b).any():
        raise ValueError("Retiming escaped its original source window")
    fields = {
        "q_ref": q, "source_time_ns": source_time, "source_frame_index": source_brackets,
        "tracker_pose": np.stack([captured[f"tracker_{side}_pose"] for side in SIDES], axis=1),
        "target_tcp_world": target, "arm_position_error_m": arms["arm_position_error_m"],
        "arm_orientation_error_rad": arms["arm_orientation_error_rad"],
    }
    for key in MAIN_STREAMS:
        for suffix in ("brackets", "causal_index"):
            fields[f"{key}_{suffix}"] = captured[f"{key}_{suffix}"]
        if key.startswith("manus"):
            fields[key + "_pose"] = captured[key + "_pose"]
    for key in ("robot_wrist_pose", "robot_fingertip_pose", "static_penetration_m"):
        fields[key] = geo[key]
    return fields, clip_ranges(usable), {key: int(mask.sum()) for key, mask in checks.items()}


def save_candidate(output, episode, window, fields, start, stop, index, names):
    arrays = {key: val[start:stop] for key, val in fields.items()}
    n = stop - start
    clock = np.arange(n, dtype=np.int64) * PERIOD_NS
    start_ns = int(arrays["source_time_ns"][0])
    arrays.update(time_ns=start_ns + clock, time_s=clock / 1e9, qd_ref=velocity(arrays["q_ref"]),
                  joint_names=np.asarray(names), source_episode=np.asarray(episode["id"]))
    source_start = int(arrays["source_frame_index"][0])
    source_end = int(arrays["source_frame_index"][-1]) + 1
    clip_id = f"{episode['id']}_{source_start:05d}_{source_end:05d}_slow{window.factor}x_{index:03d}"
    path = f"clips/{clip_id}.npz"
    write_npz(output / path, arrays)
    return {"id": clip_id, "path": path, "sha256": digest(output / path), "source_episode": episode["id"],
            "task": episode["task"], "trial": episode["trial"], "split": "train", "frames": n,
            "source_start_frame": source_start, "source_stop_frame_exclusive": source_end,
            "source_start_time_ns": start_ns, "source_end_time_ns": int(arrays["source_time_ns"][-1]),
            "start_time_ns": start_ns, "duration_s": n / RATE_HZ, "timing": "retimed",
            "time_dilation_factor": window.factor, "retiming_method": "source_pose_interpolation_then_native_retarget",
            "source_window": [window.start, window.stop], "recording_date": episode["start_local"][:10],
            "source_manifest_sha256": episode["source_manifest_sha256"]}


def retime_episode(job):
    episode, folder, output, original_entries, assets = job
    aligned = read_npz(folder / "aligned.npz")
    reference = read_npz(folder / "reference.npz")
    quality = read_json(folder / "retarget_quality.json")
    config = read_json(folder / "configuration.json")
    occupied = np.zeros(len(aligned["time_ns"]), dtype=bool)
    for entry in original_entries:
        occupied[entry["source_start_frame"]:entry["source_stop_frame_exclusive"]] = True
    windows = list(speed_windows(aligned, reference, quality, config, occupied))
    if not windows:
        return [], {"id": episode["id"], "windows": []}
    # Only load the recorded streams for episodes with recoverable speed-limited windows.
    prefixes = (*MAIN_STREAMS, "tracker_safety")
    with np.load(folder / "parsed.npz", allow_pickle=False) as capture:
        parsed = {key: capture[key] for key in capture if key.split("__", 1)[0] in prefixes
                  and key.split("__", 1)[-1] in ("pose", "header_ns", "valid", "receive_ros_ns", "duplicate_source",
                                                  "monotonic_ns", "glove_id")}
    streams = read_json(folder / "quality.json")["streams"]
    robot = Robot(assets["robot_model"])
    entries, logs = [], []
    for window in windows:
        fields, ranges, failures = retarget_window(window, aligned, parsed, streams, config, robot, assets)
        ids = []
        for start, stop in ranges:
            entry = save_candidate(output, episode, window, fields, start, stop, len(entries), robot.names)
            entries.append(entry)
            ids.append(entry["id"])
        logs.append({"start": window.start, "stop": window.stop, "dilation": window.factor,
                     "speed_evidence": window.speed_evidence, "rejection_frames": failures, "candidate_clips": ids})
    print(f"Retimed {episode['task']} trial {episode['trial']}: {len(windows)} windows, {len(entries)} candidates", flush=True)
    return entries, {"id": episode["id"], "windows": logs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    batch = args.batch.resolve(strict=True)
    baseline = MotionLibrary.open(batch / "native_replay_original/motion_library/manifest.json")
    dataset = read_json(batch / "dataset.json")
    output = batch / "retimed_candidates"
    (output / "clips").mkdir(parents=True)
    provenance = baseline.manifest["provenance"]
    assets = {k: Path(provenance[k]["path"]) for k in ("robot_model", "offline_tool", "marvin_ccs", "wuji_sdk")}
    for key, path in assets.items():
        if digest(path) != provenance[key]["sha256"]:
            raise ValueError(f"Retiming asset changed: {path}")
    by_episode = {}
    for entry in baseline.entries("train"):
        by_episode.setdefault(entry["source_episode"], []).append(entry)
    jobs = [(row, batch / "episodes" / row["id"], output, by_episode.get(row["id"], []), assets)
            for row in dataset["episodes"] if row["split"] == "train" and row["reference"]["status"] == "built"]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        result = list(pool.map(retime_episode, jobs))
    entries = [entry for clips, _ in result for entry in clips]
    template = {k: v for k, v in baseline.manifest.items() if k != "native_replay"}
    write_library(output, template, entries, retiming={
        "scope": "training episodes, only outside accepted original-speed clips", "time_dilation_factors": list(DILATIONS),
        "minimum_speed_evidence_intervals": MIN_SATURATED_STEPS,
        "time_ns": "synthetic 50 Hz playback, anchored at the first original source timestamp",
        "source_time_ns": "interpolation query on the original recording clock",
        "source_frame_index": "preceding frame on the original aligned 50 Hz clock",
    })
    write_json(output / "report.json", {"source_episodes": len(jobs), "candidate_clips": len(entries),
                                       "by_dilation": dict(Counter(entry["time_dilation_factor"] for entry in entries)),
                                       "episodes": [report for _, report in result]})
    print(f"Built {len(entries)} slower candidates from {sum(bool(r['windows']) for _, r in result)} source episodes", flush=True)


if __name__ == "__main__":
    main()
