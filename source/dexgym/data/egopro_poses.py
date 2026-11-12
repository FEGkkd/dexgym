"""Decode verified EgoPro body MCAP episodes into aligned 50 Hz pose segments."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp


FPS = 50
MAX_GAP_S = 0.20
MIN_FRAMES = 101
POSE = {"/pose/body": ("arms", 22), "/pose/left_hand": ("left", 21), "/pose/right_hand": ("right", 21)}
QUALITY = {"/annotation/bad_frame/pose/upper_body": "arms", "/annotation/bad_frame/pose/hand": "hands"}


def message_class(schema):
    """Resolve the Protobuf descriptor dependencies embedded in a MCAP schema."""
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    pool = descriptor_pool.DescriptorPool()
    pending = list(descriptor_pb2.FileDescriptorSet.FromString(schema.data).file)
    while pending:
        names = {f.name for f in pending}
        ready = [f for f in pending if not names.intersection(f.dependency)]
        if not ready:
            raise ValueError(f"Unresolved embedded dependencies for {schema.name}")
        for file in ready:
            pool.Add(file)
        done = {file.name for file in ready}
        pending = [file for file in pending if file.name not in done]
    return message_factory.GetMessageClass(pool.FindMessageTypeByName(schema.name))


def read_poses(source):
    from mcap.reader import make_reader

    records = {key: {} for key, _ in POSE.values()}
    bad = {key: set() for key in QUALITY.values()}
    bad_shape = defaultdict(int)
    classes = {}
    with source.open("rb") as stream:
        for schema, channel, msg in make_reader(stream).iter_messages(topics=[*POSE, *QUALITY]):
            if schema.id not in classes:
                classes[schema.id] = message_class(schema)
            pose = classes[schema.id].FromString(msg.data)
            t = int(pose.header.ts)
            if channel.topic in QUALITY:
                if pose.is_bad:
                    bad[QUALITY[channel.topic]].add(t)
                continue
            key, count = POSE[channel.topic]
            if t <= 0 or len(pose.transforms) != count or t in records[key]:
                bad_shape[key] += 1
                continue
            joints = pose.transforms[16:22] if key == "arms" else pose.transforms
            records[key][t] = [
                (j.pos.x, j.pos.y, j.pos.z, j.quat.w, j.quat.x, j.quat.y, j.quat.z) for j in joints
            ]
    exact = np.asarray(sorted(set.intersection(*(set(rows) for rows in records.values()))), dtype=np.int64)
    if not len(exact):
        raise ValueError("No shared pose timestamps")
    raw = {key: np.asarray([rows[t] for t in exact]) for key, rows in records.items()}
    clean = ~np.isin(exact, list(bad["arms"] | bad["hands"]))
    for value in raw.values():
        norms = np.linalg.norm(value[..., 3:], axis=-1)
        clean &= np.isfinite(value).all(axis=(1, 2)) & (abs(norms - 1) < 0.05).all(axis=-1)
    arms = raw["arms"][..., :3]
    width = np.linalg.norm((arms[:, 0] - arms[:, 1]) * [1, 1, 0], axis=-1)
    clean &= (width > 0.12) & (width < 0.85)
    for key in ("left", "right"):
        fingers = raw[key][:, 1:, :3].reshape(-1, 5, 4, 3)
        bones = np.linalg.norm(np.diff(fingers, axis=2), axis=-1)
        clean &= ((bones > 0.003) & (bones < 0.16)).all(axis=(1, 2))
        forward = raw[key][:, 9, :3] - raw[key][:, 0, :3]
        across = raw[key][:, 5, :3] - raw[key][:, 17, :3]
        clean &= np.linalg.norm(np.cross(forward, across), axis=-1) > 1e-4
    stats = {
        "source_samples": {key: len(rows) for key, rows in records.items()},
        "bad_shape": dict(bad_shape), "aligned_samples": len(exact), "valid_samples": int(clean.sum()),
    }
    return exact[clean], {key: v[clean] for key, v in raw.items()}, stats


def resample(time, pose, grid):
    p = np.stack([np.interp(grid, time, x) for x in pose[..., :3].reshape(len(time), -1).T], axis=1)
    p = gaussian_filter1d(p.reshape(len(grid), -1, 3), 0.8, axis=0, mode="nearest")
    q = []
    for joint in pose[..., 3:].transpose(1, 0, 2):
        quat = Slerp(time, Rotation.from_quat(joint, scalar_first=True))(grid).as_quat(scalar_first=True)
        signs = np.where((quat[1:] * quat[:-1]).sum(axis=-1) >= 0, 1, -1)
        quat[1:] *= np.cumprod(signs)[:, None]
        quat = gaussian_filter1d(quat, 0.8, axis=0, mode="nearest")
        q.append(quat / np.linalg.norm(quat, axis=-1, keepdims=True))
    return np.concatenate((p, np.stack(q, axis=1)), axis=-1).astype(np.float32)


def extract(source, dest):
    timestamps, raw, stats = read_poses(source)
    chunks = np.split(np.arange(len(timestamps)), np.flatnonzero(np.diff(timestamps) > MAX_GAP_S * 1e9) + 1)
    segments = {key: [] for key in raw}
    lengths, starts = [], []
    for ids in chunks:
        if len(ids) < 2:
            continue
        time = (timestamps[ids] - timestamps[ids[0]]) / 1e9
        grid = np.arange(int(np.floor(time[-1] * FPS + 1e-8)) + 1) / FPS
        if len(grid) < MIN_FRAMES:
            continue
        grid[-1] = min(grid[-1], time[-1])
        for key, values in raw.items():
            segments[key].append(resample(time, values[ids], grid))
        lengths.append(len(grid))
        starts.append(int(timestamps[ids[0]]))
    stats.update(segments=len(lengths), frames=sum(lengths), seconds=(sum(lengths) - len(lengths)) / FPS)
    if lengths:
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(temp, fps=FPS, offsets=np.r_[0, np.cumsum(lengths)], header_start_ns=starts,
                            source_episode=source.stem, **{k: np.concatenate(v) for k, v in segments.items()})
        temp.replace(dest)
    return stats


def inventory(root, manifest):
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    groups = defaultdict(list)
    seen = set()
    for entry in payload["entries"]:
        source = root / entry["path"]
        if source.stem in seen or source.stat().st_size != entry["size"]:
            raise ValueError(f"Duplicate or incomplete manifest episode {source}")
        seen.add(source.stem)
        groups[source.parent.name].append((source, entry["duration_s"]))
    result = []
    for task, entries in sorted(groups.items()):
        entries.sort(key=lambda item: hashlib.sha256(item[0].stem.encode()).digest())
        n = len(entries)
        n_eval = max(1, round(n * 0.1))
        n_test = n_eval if n >= 2 else 0
        n_val = n_eval if n >= 3 else 0
        for i, (source, duration) in enumerate(entries):
            split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            result.append(dict(source=str(source), episode=source.stem, task=task, split=split, duration_s=duration))
    return result


def worker(row, output):
    dest = output / row["split"] / (row["episode"] + ".npz")
    try:
        report = extract(Path(row["source"]), dest)
    except (ValueError, OSError) as exc:
        return {**row, "status": "rejected", "reason": str(exc)}
    return {**row, **report, "status": "ok" if report["frames"] else "no_segment",
            "path": str(dest) if report["frames"] else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--witness", type=Path)
    parser.add_argument("--mcap-packages", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    sys.path.extend(map(str, args.mcap_packages))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.witness:
        report = extract(args.witness, args.output / "witness.npz")
        if report["frames"] < MIN_FRAMES:
            raise RuntimeError(f"The witness has no synchronized segment: {report}")
        (args.output / "witness.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return
    rows = inventory(args.root, args.root / "egopro_manifest.json")
    (args.output / "inventory.json").write_text(json.dumps(rows, indent=2) + "\n")
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (args.output / "progress.jsonl").open("w") as log:
        jobs = [pool.submit(worker, row, args.output) for row in rows]
        for job in as_completed(jobs):
            report = job.result()
            reports.append(report)
            print(json.dumps(report), file=log, flush=True)
            if len(reports) % 10 == 0:
                print(f"Decoded {len(reports)}/{len(rows)} episodes", flush=True)
    reports.sort(key=lambda row: row["episode"])
    (args.output / "segments.json").write_text(json.dumps(reports, indent=2) + "\n")
    if not any(r["status"] == "ok" for r in reports):
        raise RuntimeError("No usable aligned clips")


if __name__ == "__main__":
    main()
