"""Build wholebody_s0 datasets from recorded dual-hand/wrist episodes."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import date
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from . import SCHEMA, TASK_NAME
from .bag import MAIN_STREAMS, digest, load_yaml, read_episode, select_episodes
from .sync import align_episode, validate_calibration

RAW_ROOT = Path.home() / "MANUS_STEAMVR_ws/dex_data_platform-cpp-collection/data/raw/episodes"
TELEOP_ROOT = Path.home() / "teleop_ws"


def write_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temp = Path(file.name)
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    temp.replace(path)


def write_npz(path, arrays):
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as file:
        temp = Path(file.name)
        np.savez_compressed(file, **arrays)
    temp.replace(path)


def snapshot_config(folder, teleop_root):
    path = folder / "run_snapshot.json"
    snapshot = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    units = {unit["unit"]: unit for unit in snapshot.get("units", ())}
    configs = {"tracker": [], "teleop": []}
    merged = {}
    calibration = None
    for unit_name in configs:
        for entry in units.get(unit_name, {}).get("startedConfigurations", ()):
            src = Path(entry["snapshot"])
            if not src.is_file():
                src = Path(entry["path"])
            if not src.is_file() or digest(src) != entry["sha256"]:
                continue
            configs[unit_name].append({"path": str(src), "sha256": entry["sha256"]})
            values = load_yaml(src)
            if unit_name == "teleop":
                merged.update(values["whole_body_teleop"]["ros__parameters"])
            elif "tracker_calibration_node" in values:
                calibration = values["tracker_calibration_node"]["ros__parameters"]
    origin = "recording_snapshot"
    if not merged:
        src = teleop_root / "src/whole_body_teleop/config/real_full.yaml"
        if src.is_file():
            merged = load_yaml(src)["whole_body_teleop"]["ros__parameters"]
            configs["teleop"].append({"path": str(src), "sha256": digest(src)})
        origin = "current_teleop_config_fallback"
    return {
        "origin": origin, "teleop": merged, "tracker_calibration": calibration,
        "verified_snapshots": configs, "capture_control_state": snapshot.get("controlObservation"),
        "run_snapshot_sha256": digest(path) if path.is_file() else None,
        "recording_sources": snapshot.get("sources", []),
    }


def audit_one(item):
    folder, manifest, local_start, output, teleop_root = item
    episode_id = manifest["episode_id"]
    path = output / "episodes" / episode_id
    path.mkdir(parents=True, exist_ok=True)
    config = snapshot_config(folder, teleop_root)
    frame = config["teleop"].get("arm.tracker_world_frame", "openvr_world")
    tracks, streams = read_episode(folder, manifest, frame)
    aligned, alignment = align_episode(tracks, manifest["start_realtime_ns"], manifest["stop_realtime_ns"])
    calibration = validate_calibration(tracks, config["tracker_calibration"])
    missing = [key for key in MAIN_STREAMS if key not in tracks]
    errors = [key for key, info in streams.items() if info["status"] == "read_error"]
    robot_fields = ("arm_left_state", "arm_right_state", "hand_left_state", "hand_right_state",
                    "arm_command", "hand_left_command", "hand_right_command")
    status = "integrity_error" if errors or manifest["state"] != "COMPLETED" else "read"
    summary = {
        "id": episode_id, "start_local": local_start, "task": manifest["metadata"]["task_name"],
        "trial": manifest["metadata"].get("trial_index"), "capture_status": manifest["state"],
        "capture_quality": manifest.get("quality_state"), "integrity_status": status,
        "duration_s": (manifest["stop_realtime_ns"] - manifest["start_realtime_ns"]) / 1e9,
        "source_episode": str(folder.resolve()), "source_manifest_sha256": digest(folder / "manifest.yaml"),
        "tracker_frame": frame, "missing_human_streams": missing, "stream_errors": errors,
        "missing_robot_streams": [key for key in robot_fields if key not in tracks],
        "alignment": alignment, "calibration": calibration,
        "capture_control_state": config["capture_control_state"],
    }
    parsed = {f"{key}__{name}": array for key, track in tracks.items() for name, array in track.arrays.items()}
    write_npz(path / "parsed.npz", parsed)
    write_npz(path / "aligned.npz", aligned)
    write_json(path / "configuration.json", config)
    write_json(path / "quality.json", {**summary, "streams": streams})
    print(f"{local_start[11:19]} {summary['task']:<12} {summary['trial']}  "
          f"{alignment['valid_frames']}/{alignment['frames']} synchronized  {status}", flush=True)
    return summary


def audit(args):
    episodes = select_episodes(args.episodes_dir, args.date, args.timezone)
    if args.episode:
        episodes = [row for row in episodes if row[0].name in args.episode]
    if not episodes:
        raise ValueError(f"No matching episodes on {args.date} in {args.episodes_dir}")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.raw_index:
        raw = args.data_root / "raw" / TASK_NAME / str(args.date)
        raw.mkdir(parents=True, exist_ok=True)
        for folder, _, _ in episodes:
            link = raw / folder.name
            if link.is_symlink() and link.resolve() == folder.resolve():
                continue
            link.symlink_to(folder.resolve(), target_is_directory=True)
    items = [(folder, manifest, start, args.output, args.teleop_root) for folder, manifest, start in episodes]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(audit_one, items))
    else:
        results = [audit_one(item) for item in items]
    write_json(args.output / "audit.json", {
        "schema": SCHEMA, "recording_date": str(args.date), "timezone": args.timezone,
        "source_root": str(args.episodes_dir.resolve()), "episodes": results,
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "build", "viewer", "verify"))
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--episodes-dir", type=Path, default=RAW_ROOT)
    parser.add_argument("--teleop-root", type=Path, default=TELEOP_ROOT)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episode", action="append", help="Select only the given episode(s) for the audit.")
    parser.add_argument("--raw-index", action="store_true", help="Link selected raw episodes below the DexGym data root.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--robot-model", type=Path)
    parser.add_argument("--ik-tool", type=Path)
    args = parser.parse_args(argv)
    args.output = (args.output or args.data_root / "processed" / TASK_NAME / str(args.date)).resolve()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.stage == "audit":
        audit(args)
    else:
        from . import pipeline

        getattr(pipeline, args.stage)(args)


if __name__ == "__main__":
    sys.exit(main())
