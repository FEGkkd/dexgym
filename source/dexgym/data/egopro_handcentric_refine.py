"""Refine validated hand-centric references with locally bounded tip corrections."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from .egopro_handcentric import DT, IK_DEPTH_M, Robot, Solver, native_replay, quaternion, rotation_error


def unpack(data):
    goals = {}
    for key in ("tip_pos", "wrist_pos", "elbow_pos", "shape"):
        goals[key] = data[f"mapped_{key}"]
    for key in ("wrist", "tip"):
        quat = data[f"mapped_{key}_quat_wxyz"]
        goals[f"{key}_rot"] = Rotation.from_quat(quat.reshape(-1, 4), scalar_first=True).as_matrix().reshape(
            *quat.shape[:-1], 3, 3
        )
    return goals


def measure(robot, q, goal):
    frames, depth = [], 0.
    for pose in q:
        robot.forward(pose)
        frames.append(robot.frames())
        depth = max(depth, robot.depth())
    fk = {key: np.stack([f[key] for f in frames]) for key in frames[0]}
    tip_delta = fk["tip_pos"] - goal["tip_pos"]
    cross_delta = tip_delta[:, 0, :, None] - tip_delta[:, 1, None]
    return fk, dict(
        tip_mm=float(np.linalg.norm(tip_delta, axis=-1).mean() * 1000),
        cross_tip_mm=float(np.linalg.norm(cross_delta, axis=-1).mean() * 1000),
        shape_mm=float(np.linalg.norm(fk["shape"][..., 1:, :] - goal["shape"][..., 1:, :], axis=-1).mean() * 1000),
        tip_deg=float(np.rad2deg(np.linalg.norm(rotation_error(goal["tip_rot"], fk["tip_rot"]), axis=-1).mean())),
        geometric_depth_mm=depth * 1000,
    )


def refine_arrays(data, xml, priority):
    robot = Robot(xml)
    native = Robot(xml)
    solver = Solver(robot, priority)
    q = data["qpos"].astype(np.float64)
    goal = unpack(data)
    original_fk, before = measure(native, q, goal)
    before["replay"] = native_replay(native, q, data["qvel"], original_fk["tip_pos"])
    corrections = np.zeros_like(q)
    radius = np.where(robot.arm, 0.095, 0.16)
    for frame, pose in enumerate(q):
        target = {key: value[frame] for key, value in goal.items()}
        lo, hi = np.maximum(robot.lo, pose - radius), np.minimum(robot.hi, pose + radius)
        improved = solver.solve(pose, target, pose, lo, hi, 4)
        corrections[frame] = improved - pose
    corrections = gaussian_filter1d(corrections, 1.6, axis=0, mode="nearest")
    speed = np.where(robot.arm, 0.063, 0.115)
    attempts = []
    for blend in (1., 0.75, 0.5, 0.25, 0.125):
        candidate = np.clip(q + blend * corrections, robot.lo, robot.hi)
        for frame in range(1, len(candidate)):
            np.clip(candidate[frame], candidate[frame - 1] - speed, candidate[frame - 1] + speed,
                    out=candidate[frame])
        fk, after = measure(native, candidate, goal)
        attempts.append({"blend": blend, **after})
        if (after["geometric_depth_mm"] > IK_DEPTH_M * 1000
                or after["tip_mm"] >= before["tip_mm"] or after["cross_tip_mm"] >= before["cross_tip_mm"]
                or after["shape_mm"] > before["shape_mm"] + 1.0 or after["tip_deg"] > before["tip_deg"] + 1.0):
            continue
        qd = np.gradient(candidate, DT, axis=0)
        after["replay"] = native_replay(native, candidate, qd, fk["tip_pos"])
        attempts[-1]["replay"] = after["replay"]
        if not after["replay"]["complete"] or after["replay"]["tip_p95_mm"] > min(
            85.0, before["replay"].get("tip_p95_mm", 85.0) + 4.0
        ):
            continue
        result = {**data, "qpos": candidate.astype(np.float32), "qvel": qd.astype(np.float32)}
        for key, value in fk.items():
            if key.endswith("_rot"):
                value = quaternion(value)
                key = key.replace("_rot", "_quat_wxyz")
            result[f"ref_{key}"] = value.astype(np.float32)
        return result, dict(blend=blend, before=before, after=after, frames=len(q), attempts=attempts)
    return data, dict(blend=0., before=before, after=before, frames=len(q), attempts=attempts)


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def work(path, xml, output, priority):
    data, report = refine_arrays(load(path), xml, priority)
    if not report["after"]["replay"]["complete"]:
        return {**report, "source": str(path), "path": None}
    np.savez_compressed(output, **data)
    return {**report, "source": str(path), "path": str(output)}


def stitch(paths):
    clips = [load(path) for path in paths]
    first = clips[0]
    arrays = [key for key in first if key in ("qpos", "qvel") or key.startswith(("ref_", "mapped_"))]
    for previous, current in zip(clips[:-1], clips[1:], strict=True):
        previous_ns = int(previous["source_header_start_ns"]) + round(previous["time_s"][-1] * 1e9)
        if int(current["source_header_start_ns"]) != previous_ns:
            raise ValueError("The witness must have contiguous one-frame overlapping segments")
    result = {**first, **{key: np.concatenate([first[key], *(clip[key][1:] for clip in clips[1:])]) for key in arrays}}
    result["time_s"] = np.arange(len(result["qpos"])) * DT
    result["qvel"] = np.gradient(result["qpos"], DT, axis=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--witness", type=Path)
    source.add_argument("--library", type=Path)
    parser.add_argument("--priority", type=float, default=3.5)
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.witness:
        data = stitch(sorted(args.witness.glob("*.npz")))
        result, report = refine_arrays(data, args.model, args.priority)
        np.savez_compressed(args.output / "witness_full.npz", **result)
        (args.output / "witness.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        if not report["after"]["replay"]["complete"]:
            raise RuntimeError("The full witness did not complete native MuJoCo replay")
        return
    paths = {}
    for split in ("train", "val", "test"):
        paths[split] = [Path(p) for p in json.loads((args.library / f"{split}.json").read_text())]
        (args.output / split).mkdir(exist_ok=True)
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (args.output / "progress.jsonl").open("w") as log:
        pending = {
            pool.submit(work, path, args.model, args.output / split / path.name, args.priority): split
            for split, values in paths.items() for path in values
        }
        for job in as_completed(pending):
            entry = {**job.result(), "split": pending[job]}
            reports.append(entry)
            print(json.dumps(entry), file=log, flush=True)
            if len(reports) % 25 == 0:
                print(f"Refined {len(reports)}/{len(pending)} validated clips", flush=True)
    reports.sort(key=lambda row: row["source"])
    (args.output / "refinement.json").write_text(json.dumps(reports, indent=2) + "\n")
    for split in paths:
        (args.output / f"{split}.json").write_text(json.dumps(
            [r["path"] for r in reports if r["split"] == split and r["path"]], indent=2
        ) + "\n")


if __name__ == "__main__":
    main()
