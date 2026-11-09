"""Enrich GRAB in-hand candidates with object-world and retargeted-finger motion."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SYMMETRIC_OBJECTS = {
    "spheresmall", "spheremedium", "spherelarge", "cylindersmall", "cylindermedium",
    "cylinderlarge", "torussmall", "torusmedium", "toruslarge",
}
FIXED_OR_NONOBJECT = {"doorknob", "hand", "body"}


def _rotation_between(rotations: Rotation, start: int, stop: int) -> float:
    beginning = rotations[start:start + 5].mean()
    ending = rotations[stop - 5:stop].mean()
    return float(np.degrees((beginning.inv() * ending).magnitude()))


def _enrich_source(rows: list[dict], raw_path: Path, processed_root: Path) -> None:
    with np.load(raw_path, allow_pickle=True) as raw:
        params = raw["object"].item()["params"]
        object_rotation = Rotation.from_rotvec(np.asarray(params["global_orient"])[::4])
        object_position = np.asarray(params["transl"])[::4]
    by_file = defaultdict(list)
    for row in rows:
        start, stop = row["start_frame"], row["stop_frame_exclusive"]
        row["world_object_rotation_deg"] = round(_rotation_between(object_rotation, start, stop), 2)
        row["world_object_translation_mm"] = round(float(np.linalg.norm(
            np.median(object_position[stop - 5:stop], axis=0)
            - np.median(object_position[start:start + 5], axis=0)
        ) * 1000), 2)
        by_file[row["processed_file"]].append(row)
    for filename, group in by_file.items():
        with np.load(processed_root / filename, allow_pickle=False) as motion:
            q_ref = np.asarray(motion["q_ref"])
            tip_ref = np.asarray(motion["tip_ref"])
        for row in group:
            start, stop = row["processed_start_frame"], row["processed_stop_frame_exclusive"]
            q, tips = q_ref[start:stop], tip_ref[start:stop]
            row["robot_joint_range_mean_rad"] = round(float(np.ptp(q, axis=0).mean()), 4)
            row["robot_tip_motion_mean_mm"] = round(float(
                np.linalg.norm(tips - tips[0], axis=-1).max(axis=0).mean() * 1000
            ), 2)


def _high_confidence(row: dict, category: str) -> bool:
    if row["object"] in FIXED_OR_NONOBJECT:
        return False
    if category == "rotation":
        return (row["rotation_candidate"] and row["object"] not in SYMMETRIC_OBJECTS
                and row["world_object_rotation_deg"] >= 20
                and row["robot_joint_range_mean_rad"] >= 0.12
                and row["robot_tip_motion_mean_mm"] >= 15)
    if category == "translation":
        return (row["translation_candidate"] and row["world_object_translation_mm"] >= 25
                and row["robot_joint_range_mean_rad"] >= 0.12
                and row["robot_tip_motion_mean_mm"] >= 20)
    if category == "regrasp":
        return (row["regrasp_candidate"] and min(row["contact_gains"], row["contact_losses"]) >= 2
                and row["robot_joint_range_mean_rad"] >= 0.18
                and row["robot_tip_motion_mean_mm"] >= 25)
    raise ValueError(category)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path,
                        default=Path("/home/ubuntu/dexgym/artifacts/hand21/grab_inhand_windows.json"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/ubuntu/dexgym/data"))
    parser.add_argument("--output", type=Path,
                        default=Path("/home/ubuntu/dexgym/artifacts/hand21/grab_inhand_enriched.json"))
    args = parser.parse_args()
    scan = json.loads(args.scan.expanduser().read_text())
    candidates = [row for row in scan["windows"] if any(row[key] for key in (
        "rotation_candidate", "translation_candidate", "regrasp_candidate"
    ))]
    by_source = defaultdict(list)
    for row in candidates:
        by_source[row["source"]].append(row)
    data_root = args.data_root.expanduser().resolve()
    processed_root = data_root / "processed/hand21_right_geometry_feasible"
    for index, (source, rows) in enumerate(sorted(by_source.items()), 1):
        _enrich_source(rows, data_root / "grab" / source, processed_root)
        if index % 100 == 0 or index == len(by_source):
            print(f"ENRICH_PROGRESS {index}/{len(by_source)}", flush=True)
    for row in candidates:
        for category in ("rotation", "translation", "regrasp"):
            row[f"high_confidence_{category}"] = _high_confidence(row, category)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "source_scan": str(args.scan.expanduser().resolve()),
        "source_count": len(by_source),
        "window_count": len(candidates),
        "criteria": {
            "rotation": "relative >=45 deg, object-world >=20 deg, non-symmetric object, robot finger motion",
            "translation": "relative 30-180 mm with <=45 deg rotation, world >=25 mm, robot finger motion",
            "regrasp": "at least two stable gains and two losses across fingers, robot finger motion",
        },
        "windows": candidates,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"ENRICH_DONE {output} windows={len(candidates)}", flush=True)


if __name__ == "__main__":
    main()
