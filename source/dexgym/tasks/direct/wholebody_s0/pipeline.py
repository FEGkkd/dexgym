"""Build, index and verify the isolated wholebody_s0 motion library."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

import mujoco
import numpy as np

from dexgym.data.wuji_retarget import SDK_VERSION

from . import SCHEMA, TASK_NAME
from .bag import digest, stats
from .library import MotionLibrary, SPLITS
from .prepare import write_json, write_npz
from .retarget import run_arm_tool, run_hand_sdk
from .robot import DEFAULT_MODEL, Robot
from .sync import RATE_HZ, runs

FILTER = {
    "max_target_position_error_m": 0.12,
    "max_target_orientation_error_deg": 35.0,
    "max_static_penetration_m": 0.003,
    "max_hand_limit_projection_rad": 0.12,
    "max_hand_speed_rad_s": 20.0,
    "minimum_clip_frames": RATE_HZ,
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_npz(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def velocity(q):
    return np.gradient(q, 1 / RATE_HZ, axis=0).astype(np.float32)


def clip_ranges(mask):
    return [(int(a), int(b)) for a, b in runs(mask) if b - a >= FILTER["minimum_clip_frames"]]


def fill_display(q, home):
    """Hold missing components for the viewer, leaving stored reference values intact."""
    finite = np.isfinite(q)
    last = np.maximum.accumulate(np.where(finite, np.arange(len(q))[:, None], -1), axis=0)
    value = np.take_along_axis(q, np.maximum(last, 0), axis=0)
    return np.where(last >= 0, value, home).astype(np.float32)


def reference_one(job):
    episode, output, model, tool, kine, sdk = job
    folder = output / "episodes" / episode["id"]
    quality = read_json(folder / "quality.json")
    alignment_sha = digest(folder / "aligned.npz")
    if episode["integrity_status"] != "read" or episode["missing_human_streams"] or not episode["alignment"]["valid_frames"]:
        report = {"status": "unavailable", "reason": "No integrity-checked synchronized human data",
                  "aligned_sha256": alignment_sha, "clips": []}
        write_json(folder / "retarget_quality.json", report)
        return {**episode, "reference": report}

    robot = Robot(model)
    aligned = read_npz(folder / "aligned.npz")
    parsed = read_npz(folder / "parsed.npz")
    config = read_json(folder / "configuration.json")
    arms, anchor = run_arm_tool(aligned, config, robot, tool, kine)
    hand, hand_valid, hand_report = run_hand_sdk(parsed, aligned, quality["streams"], config, sdk)
    q_uncapped = np.column_stack((arms["arm_q"], hand))
    q = np.clip(q_uncapped, robot.bounds[:, 0], robot.bounds[:, 1]).astype(np.float32)
    bound_adjust = np.abs(q - q_uncapped)
    arm_bound = np.max(bound_adjust[:, :14], axis=1)
    hand_bound = np.max(bound_adjust[:, 14:], axis=1)
    geometry = robot.geometry(fill_display(q, robot.home))
    target_world = robot.world_tcp(arms["target_tcp_base"])
    native_world = robot.world_tcp(arms["native_tcp_base"])
    mjcf_error = np.linalg.norm(native_world[..., :3] - geometry["robot_wrist_pose"][..., :3], axis=-1)
    if not np.isfinite(mjcf_error).all() or mjcf_error.max() > 0.002:
        raise ValueError(f"Native and MJCF TCP positions disagree in {episode['id']}: {np.nanmax(mjcf_error):.6f} m")
    human = aligned["human_valid"]
    wrist_target_pair = target_world[:, 1, :3] - target_world[:, 0, :3]
    wrist_reference_pair = native_world[:, 1, :3] - native_world[:, 0, :3]
    pair_error = np.linalg.norm(wrist_reference_pair - wrist_target_pair, axis=1)
    reasons = {
        "human_gap": ~human,
        "arm_ik_or_hold": ~arms["arm_accepted"].all(axis=1),
        "hand_sdk_unavailable": ~hand_valid.all(axis=1),
        "joint_bounds": (arm_bound > 1e-6) | (hand_bound > FILTER["max_hand_limit_projection_rad"]),
        "position_tracking": (arms["arm_position_error_m"] > FILTER["max_target_position_error_m"]).any(axis=1),
        "orientation_tracking": (np.degrees(arms["arm_orientation_error_rad"]) > FILTER["max_target_orientation_error_deg"]).any(axis=1),
        "static_contact": geometry["static_penetration_m"] > FILTER["max_static_penetration_m"],
        "nonfinite_joint": ~np.isfinite(q).all(axis=1),
    }
    if config["teleop"].get("arm.bimanual_registration_enabled", False):
        tolerance = config["teleop"].get("arm.bimanual_position_tolerance_m", 0.01)
        reasons["bimanual_start_unaligned"] = np.full(len(q), anchor["relative_wrist_vector_error_m"] > tolerance)
    usable = ~np.logical_or.reduce(tuple(reasons.values()))
    fast = np.zeros(len(q), dtype=bool)
    for a, b in runs(usable):
        if b - a > 1:
            delta = np.max(np.abs(np.diff(q[a:b, 14:], axis=0)), axis=1) * RATE_HZ
            fast[a + 1:b] = delta > FILTER["max_hand_speed_rad_s"]
    reasons["hand_step_speed"] = fast
    usable &= ~fast
    clips = clip_ranges(usable)
    admitted = np.zeros(len(q), dtype=bool)
    for a, b in clips:
        admitted[a:b] = True
    mask = np.zeros(len(q), dtype=np.uint16)
    for index, invalid in enumerate(reasons.values()):
        mask[invalid] |= 1 << index
    hand_bound_mask = np.isfinite(bound_adjust[:, 14:]) & (bound_adjust[:, 14:] > 1e-6)
    trace = {
        **arms, **geometry, "q_ref": q, "hand_q_unclipped": hand, "hand_valid": hand_valid,
        "target_tcp_world": target_world, "native_tcp_world": native_world, "reference_usable": usable,
        "in_library": admitted, "rejection_bits": mask, "time_ns": aligned["time_ns"],
        "hand_limit_adjust_rad": hand_bound,
        "relative_wrist_vector_error_m": pair_error,
    }
    write_npz(folder / "reference.npz", trace)
    valid_arms = aligned["tracker_left_valid"] & aligned["tracker_right_valid"] & aligned["tracker_safety_valid"]
    report = {
        "status": "built", "aligned_sha256": alignment_sha, "anchor": anchor, "hand": hand_report,
        "frames": len(q), "usable_frames": int(usable.sum()), "library_frames": int(admitted.sum()),
        "arm_accepted_frames": arms["arm_accepted"].sum(axis=0).tolist(),
        "arm_rate_limited_frames": arms["arm_rate_limited"].sum(axis=0).tolist(),
        "arm_safety_codes": [dict(Counter(int(x) for x in arms["arm_safety_code"][:, i])) for i in range(2)],
        "arm_position_error_mm": [stats(arms["arm_position_error_m"][valid_arms, i] * 1000) for i in range(2)],
        "arm_orientation_error_deg": [stats(np.degrees(arms["arm_orientation_error_rad"][valid_arms, i])) for i in range(2)],
        "library_arm_position_error_mm": [stats(arms["arm_position_error_m"][admitted, i] * 1000) for i in range(2)],
        "library_arm_orientation_error_deg": [stats(np.degrees(arms["arm_orientation_error_rad"][admitted, i])) for i in range(2)],
        "library_relative_wrist_vector_error_mm": stats(pair_error[admitted] * 1000),
        "native_vs_mjcf_tcp_position_mm": stats(mjcf_error * 1000),
        "hand_clamped_values": int(hand_bound_mask.sum()),
        "hand_limit_adjust_rad": stats(hand_bound),
        "static_penetration_mm": stats(geometry["static_penetration_m"] * 1000),
        "max_adjacent_arm_speed_rad_s": stats(np.max(np.abs(np.diff(q[:, :14], axis=0)), axis=1) * RATE_HZ),
        "rejection_counts": {name: int(values.sum()) for name, values in reasons.items()},
        "rejection_bit_names": list(reasons),
        "clips": clips,
    }
    write_json(folder / "retarget_quality.json", report)
    print(f"{episode['task']:<12} {episode['trial']}  {len(clips):>3} clips  "
          f"{admitted.sum():>5}/{len(q)} frames  "
          f"anchor {anchor['method']}  max contact {geometry['static_penetration_m'].max()*1000:.2f} mm", flush=True)
    return {**episode, "reference": report}


def split_episodes(results):
    by_task = defaultdict(list)
    assignment = {row["id"]: "train" for row in results}
    for row in results:
        if row["reference"]["clips"]:
            by_task[row["task"]].append(row)
    pair = 0
    for rows in by_task.values():
        # Prefer healthy, longer episodes for the held-out sets.
        ordered = sorted(rows, key=lambda row: (row["capture_quality"] == "good", row["duration_s"], row["id"]))
        if len(ordered) >= 3:
            assignment[ordered[-1]["id"]] = "test"
            assignment[ordered[-2]["id"]] = "validation"
        elif len(ordered) == 2:
            assignment[ordered[-1]["id"]] = ("test", "validation")[pair % 2]
            pair += 1
    return assignment


def export_library(output, results, robot, identity):
    root = output / "motion_library"
    clips_dir = root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    assignment = split_episodes(results)
    entries = []
    for row in results:
        episode = row["id"]
        row["split"] = assignment[episode] if row["reference"]["clips"] else None
        if not row["reference"]["clips"]:
            continue
        folder = output / "episodes" / episode
        trace = read_npz(folder / "reference.npz")
        aligned = read_npz(folder / "aligned.npz")
        for index, (start, stop) in enumerate(row["reference"]["clips"]):
            selection = slice(start, stop)
            clip_id = f"{episode}_{start:05d}_{stop:05d}"
            relative = f"clips/{clip_id}.npz"
            q = trace["q_ref"][selection]
            fields = {
                "q_ref": q, "qd_ref": velocity(q), "time_ns": aligned["time_ns"][selection],
                "time_s": np.arange(stop - start, dtype=np.float64) / RATE_HZ,
                "joint_names": np.asarray(robot.names), "source_episode": np.asarray(episode),
                "source_frame_index": np.arange(start, stop),
                "tracker_pose": np.stack((aligned["tracker_left_pose"][selection], aligned["tracker_right_pose"][selection]), axis=1),
            }
            for key in ("manus_left", "manus_right", "tracker_left", "tracker_right"):
                for suffix in ("brackets", "causal_index"):
                    fields[f"{key}_{suffix}"] = aligned[f"{key}_{suffix}"][selection]
                if key.startswith("manus"):
                    fields[key + "_pose"] = aligned[key + "_pose"][selection]
            for key in ("target_tcp_world", "robot_wrist_pose", "robot_fingertip_pose", "static_penetration_m",
                        "arm_position_error_m", "arm_orientation_error_rad"):
                fields[key] = trace[key][selection]
            write_npz(root / relative, fields)
            entries.append({
                "id": clip_id, "path": relative, "sha256": digest(root / relative), "source_episode": episode,
                "task": row["task"], "trial": row["trial"], "split": row["split"], "frames": stop - start,
                "source_start_frame": start, "source_stop_frame_exclusive": stop, "clip_index": index,
                "start_time_ns": int(fields["time_ns"][0]), "duration_s": (stop - start) / RATE_HZ,
                "source_manifest_sha256": row["source_manifest_sha256"],
            })
    manifest = {
        "schema": SCHEMA, "task": TASK_NAME, "kind": "reference_motion_library", "rate_hz": RATE_HZ,
        "joint_names": robot.names, "firmware_joint_names": robot.firmware_names,
        "qpos_indices": robot.qpos, "actuator_indices": robot.ctrl, "joint_limits_rad": robot.bounds.tolist(),
        "joint_groups": {"left_arm": [0, 7], "right_arm": [7, 14], "left_hand": [14, 34], "right_hand": [34, 54]},
        "position_unit": "rad", "velocity_unit": "rad/s", "pose_unit": "m",
        "quaternion_order": "xyzw", "tracker_frame": sorted({row["tracker_frame"] for row in results}),
        "robot_frame": "mjcf_world", "source_frame_interval": "start inclusive, stop exclusive",
        "reference_origin": "offline_wuji_sdk_and_marvin_libkine", "provenance": identity,
        "quality_filter": FILTER, "splits": {}, "clips": entries,
    }
    for split in SPLITS:
        rows = [entry for entry in entries if entry["split"] == split]
        manifest["splits"][split] = {
            "path": f"{split}.json", "episodes": len({entry["source_episode"] for entry in rows}),
            "clips": len(rows), "frames": sum(entry["frames"] for entry in rows),
        }
        write_json(root / f"{split}.json", [entry["path"] for entry in rows])
    write_json(root / "manifest.json", manifest)
    return manifest


def build(args):
    audit = read_json(args.output / "audit.json")
    model = (args.robot_model or DEFAULT_MODEL).resolve(strict=True)
    tool = (args.ik_tool or args.output / ".build/wholebody_s0_ik").resolve(strict=True)
    teleop = args.teleop_root / "src/whole_body_teleop"
    kine = (teleop / "config/ccs_m6_40.MvKDCfg").resolve(strict=True)
    sdk = (args.teleop_root / "third_party/wuji_c_api/lib/libwuji_sdk_c.so").resolve(strict=True)
    jobs = [(row, args.output, model, tool, kine, sdk) for row in audit["episodes"]]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(reference_one, jobs))
    else:
        results = [reference_one(job) for job in jobs]
    robot = Robot(model)
    files = {
        "robot_model": model, "offline_tool": tool, "marvin_ccs": kine, "wuji_sdk": sdk,
        "marvin_libkine": teleop / "vendor/marvin/lib/libKine.so",
        "arm_mapping_and_control": teleop / "src/teleop_core.cpp",
        "hand_retargeting": teleop / "src/retarget_node.cpp",
    }
    identity = {key: {"path": str(path), "sha256": digest(path)} for key, path in files.items()}
    identity["wuji_sdk_version"] = SDK_VERSION.decode()
    identity["mujoco_version"] = mujoco.__version__
    library = export_library(args.output, results, robot, identity)
    write_json(args.output / "dataset.json", {
        **audit, "episodes": results, "joint_names": robot.names, "robot_links": robot.links,
        "robot_nodes": len(robot.nodes), "motion_library": "motion_library/manifest.json",
        "reference_origin": library["reference_origin"], "splits": library["splits"],
    })
    if not library["clips"]:
        raise RuntimeError("No motion met the declared clip filter. Inspect the per-episode retarget reports.")


def viewer(args):
    from .viewer import export_viewer

    export_viewer(args.output)


def verify(args):
    dataset = read_json(args.output / "dataset.json")
    library = MotionLibrary.open(args.output / dataset["motion_library"])
    if not library.manifest["clips"]:
        raise ValueError("Motion Library is empty")
    seen = {}
    expected = {episode["id"]: episode for episode in dataset["episodes"]}
    per_stream = defaultdict(list)
    total_raw_files, total_records, total_chunk_crcs, total_data_crcs, total_summary_crcs = 0, 0, 0, 0, 0
    for episode in expected.values():
        folder = args.output / "episodes" / episode["id"]
        if digest(folder / "aligned.npz") != episode["reference"]["aligned_sha256"]:
            raise ValueError(f"Aligned data changed after reference generation: {episode['id']}")
        quality = read_json(folder / "quality.json")
        source = Path(quality["source_episode"])
        if digest(source / "manifest.yaml") != quality["source_manifest_sha256"]:
            raise ValueError(f"Source episode manifest changed: {source}")
        for key, stream in quality["streams"].items():
            per_stream[key].append(stream)
            for entry in stream.get("files", ()):
                if digest(source / entry["path"]) != entry["sha256"]:
                    raise ValueError(f"Original MCAP changed: {source / entry['path']}")
                total_raw_files += 1
                total_records += entry["records"]
                total_chunk_crcs += entry["verified_chunk_crcs"]
                total_data_crcs += entry["verified_data_crc"]
                total_summary_crcs += entry["verified_summary_crc"]
    for split in SPLITS:
        split_list = read_json(library.root / library.manifest["splits"][split]["path"])
        if split_list != [entry["path"] for entry in library.entries(split)]:
            raise ValueError(f"Split list disagrees with manifest: {split}")
        for entry in library.entries(split):
            eid = entry["source_episode"]
            if seen.setdefault(eid, split) != split:
                raise ValueError(f"Episode {eid} leaks across splits")
            source = expected[eid]
            if source["source_manifest_sha256"] != entry["source_manifest_sha256"]:
                raise ValueError(f"Raw recording identity disagrees with manifest: {eid}")
            if digest(library.root / entry["path"]) != entry["sha256"]:
                raise ValueError(f"Clip checksum mismatch: {entry['path']}")
            clip = library.load_clip(entry)
            a, b = entry["source_start_frame"], entry["source_stop_frame_exclusive"]
            if [a, b] not in source["reference"]["clips"] or not np.array_equal(clip["source_frame_index"], np.arange(a, b)):
                raise ValueError(f"Clip source range is inconsistent: {entry['id']}")
            if np.max(clip["static_penetration_m"]) > FILTER["max_static_penetration_m"]:
                raise ValueError(f"Clip violates static contact threshold: {entry['id']}")
    result = {
        "schema": SCHEMA, "verified": True, "source_episodes": len(expected),
        "library_episodes": len(seen), "clips": len(library.manifest["clips"]), "splits": library.manifest["splits"],
    }
    write_json(args.output / "verification.json", result)
    stream_totals = {}
    for key, rows in per_stream.items():
        present = [stream for stream in rows if stream["status"] == "recorded"]
        stream_totals[key] = {
            "recorded_episodes": len(present), "messages": sum(stream["count"] for stream in present),
            "invalid_frames": sum(stream["invalid_frames"] for stream in present),
            "source_duplicates": sum(stream["source_duplicates"] for stream in present),
            "collector_dropped": sum(stream["collector_dropped"] for stream in present),
            "header_hz_across_episodes": stats([stream["header_clock"]["observed_hz"] for stream in present]),
            "max_internal_gap_ms": max((stream["header_clock"]["interval_ms"]["max"] for stream in present
                                        if stream["header_clock"]["interval_ms"]), default=None),
            "max_trailing_gap_ms": max((stream["capture_last_to_stop_ms"] for stream in present), default=None),
        }
    all_episodes = list(expected.values())
    reference_reports = [episode["reference"] for episode in all_episodes if episode["reference"]["status"] == "built"]
    calibration_sides = [side for episode in all_episodes for side in episode["calibration"].get("sides", {}).values()]
    summary = {
        **result,
        "recording_duration_s": sum(e["duration_s"] for e in all_episodes),
        "aligned_frames": sum(e["alignment"]["frames"] for e in all_episodes),
        "synchronized_human_frames": sum(e["alignment"]["valid_frames"] for e in all_episodes),
        "reference_episodes": len(reference_reports),
        "usable_reference_frames": sum(report["usable_frames"] for report in reference_reports),
        "library_frames": sum(part["frames"] for part in library.manifest["splits"].values()),
        "checked_raw_mcap_files": total_raw_files,
        "checked_raw_mcap_records": total_records,
        "verified_chunk_crcs": total_chunk_crcs,
        "verified_data_crcs": total_data_crcs,
        "verified_summary_crcs": total_summary_crcs,
        "calibration_valid_pairs": sum(side.get("compared_valid_pairs", 0) for side in calibration_sides),
        "calibration_mismatches": sum(side.get("calibration_mismatch_count", 0) for side in calibration_sides),
        "max_native_vs_mjcf_tcp_mm": max((row["native_vs_mjcf_tcp_position_mm"]["max"] for row in reference_reports), default=None),
        "streams": stream_totals,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(result, ensure_ascii=False))
