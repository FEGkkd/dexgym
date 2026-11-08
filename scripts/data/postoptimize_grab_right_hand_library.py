"""Refine the existing Wuji SDK motion bank for hand self-collision.

Input files are read only. The output is a separate MotionBank-compatible
directory with q_ref and tip_ref recomputed from the optimized joint angles.
"""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from dexgym.data.self_collision_postopt import HandPostOptimizer, PostOptWeights
from dexgym.data.wuji_retarget import RightHandFK

_OPTIMIZER: HandPostOptimizer | None = None


def _initialize_worker(right_xml: str, weights: PostOptWeights) -> None:
    global _OPTIMIZER
    _OPTIMIZER = HandPostOptimizer(RightHandFK(right_xml), weights)


def _process_record(job: tuple[dict, Path, Path]) -> tuple[dict | None, dict | None]:
    entry, input_dir, output_dir = job
    if _OPTIMIZER is None:
        raise RuntimeError("Post-optimizer worker was not initialized")
    try:
        with np.load(input_dir / entry["file"], allow_pickle=False) as raw:
            q_sdk = np.asarray(raw["q_ref"], dtype=np.float32)
            indices = np.asarray(raw["source_indices"], dtype=np.int32)
            fps = np.asarray(raw["fps"], dtype=np.float32)
        q_star, tip_star, metrics = _OPTIMIZER.optimize_sequence(q_sdk)
        if metrics["optimization_nonfinite_failures"]:
            raise RuntimeError(f"Nonfinite optimization for {entry['source']}")
        if not np.isfinite(q_star).all() or not np.isfinite(tip_star).all():
            raise RuntimeError(f"Nonfinite output for {entry['source']}")
        target = output_dir / entry["file"]
        temporary = target.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, q_ref=q_star, tip_ref=tip_star, source_indices=indices, fps=fps)
        temporary.replace(target)
        new_entry = dict(entry)
        new_entry["postoptimization"] = metrics
        new_entry["max_tip_step_m"] = (
            float(np.linalg.norm(np.diff(tip_star, axis=0), axis=-1).max()) if len(q_star) > 1 else 0.0
        )
        new_entry["max_joint_step_rad"] = float(np.abs(np.diff(q_star, axis=0)).max()) if len(q_star) > 1 else 0.0
        return new_entry, None
    except (OSError, ValueError, RuntimeError) as error:
        return None, {"source": entry["source"], "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/hand_only_right"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/hand_only_right_collision_aware"))
    parser.add_argument(
        "--right-xml",
        type=Path,
        default=Path("~/teleop_ws/src/whole_body_mujoco_driver/model/source/wujihand2_beta2/mjcf/right.xml"),
    )
    parser.add_argument("--split", choices=("all", "train", "validation"), default="all")
    parser.add_argument(
        "--motion", action="append", default=[], help="Optional source path such as s8/flashlight_on_1.npz"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Independent motion optimizer processes")
    args = parser.parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if input_dir == output_dir:
        parser.error("Output must be a separate directory from SDK input")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    source_manifest_path = input_dir / "manifest.json"
    source_bytes = source_manifest_path.read_bytes()
    source_manifest = json.loads(source_bytes)
    right_xml = args.right_xml.expanduser().resolve()
    right_xml_sha256 = hashlib.sha256(right_xml.read_bytes()).hexdigest()
    if right_xml_sha256 != source_manifest["right_xml_sha256"]:
        parser.error("Right-hand MJCF differs from the model used for SDK library FK")
    records = [
        entry
        for entry in source_manifest["motions"]
        if (args.split == "all" or entry["split"] == args.split) and (not args.motion or entry["source"] in args.motion)
    ]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        parser.error("No motions matched the selection")
    unmatched = set(args.motion) - {entry["source"] for entry in records}
    if unmatched:
        parser.error(f"Unknown selected motions: {sorted(unmatched)}")

    weights = PostOptWeights()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {key: value for key, value in source_manifest.items() if key not in {"motions", "errors"}}
    result.update(
        {
            "format_version": 2,
            "description": "Wuji SDK targets with self-collision-aware post-optimization and FK fingertips",
            "postoptimization": {
                "method": "contact-mesh L-BFGS-B, two local temporal sweeps",
                "weights": vars(weights),
                "input_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "input_manifest": str(source_manifest_path),
            },
            "accepted": 0,
            "rejected": 0,
            "total_frames": 0,
            "motions": [],
            "errors": [],
        }
    )
    jobs = [(entry, input_dir, output_dir) for entry in records]
    if args.workers == 1:
        _initialize_worker(str(right_xml), weights)
        processed = map(_process_record, jobs)
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_initialize_worker,
            initargs=(str(right_xml), weights),
        )
        processed = executor.map(_process_record, jobs)
    for index, (new_entry, error) in enumerate(processed, 1):
        if new_entry is not None:
            metrics = new_entry["postoptimization"]
            result["motions"].append(new_entry)
            result["accepted"] += 1
            result["total_frames"] += metrics["frames"]
            print(
                f"[{index}/{len(records)}] {new_entry['source']}: "
                f"penetration {metrics['initial_max_penetration_mm']:.2f} -> "
                f"{metrics['final_max_penetration_mm']:.2f} mm, "
                f"tip change {metrics['max_tip_change_mm']:.2f} mm",
                flush=True,
            )
        else:
            result["errors"].append(error)
            result["rejected"] += 1
            print(f"[{index}/{len(records)}] {error['source']}: ERROR {error['error']}", flush=True)
    if args.workers > 1:
        executor.shutdown()
    temporary_manifest = output_dir / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(output_dir / "manifest.json")
    print(f"accepted={result['accepted']} rejected={result['rejected']} frames={result['total_frames']}")
    print(f"manifest={output_dir / 'manifest.json'}")
    if result["rejected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
