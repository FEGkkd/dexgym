"""Keep Hand 2.1 target segments except frames with severe residual self-penetration.

This is a geometry gate, not a 10 mm position-controller replay gate. Original
GRAB, SDK targets, and collision-optimized targets are read only.
"""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from dexgym.assets.wuji_hand21 import OFFICIAL_MJCF
from dexgym.data.self_collision_postopt import HandPostOptimizer, PostOptWeights
from dexgym.data.wuji_retarget import RightHandFK

_AUDITOR = None


def _init_worker(xml: str) -> None:
    global _AUDITOR
    _AUDITOR = HandPostOptimizer(RightHandFK(xml), PostOptWeights())


def _segments(valid: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, valid, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _process(job: tuple[dict, str, str, float, int]) -> tuple[list[dict], dict]:
    entry, source_dir, output_dir, threshold_mm, min_frames = job
    assert _AUDITOR is not None
    with np.load(Path(source_dir) / entry["file"], allow_pickle=False) as data:
        q = np.asarray(data["q_ref"], dtype=np.float32)
        tips = np.asarray(data["tip_ref"], dtype=np.float32)
        indices = np.asarray(data["source_indices"], dtype=np.int32)
        fps = np.asarray(data["fps"], dtype=np.float32)
    if q.shape != (len(indices), 20) or tips.shape != (len(q), 5, 3):
        raise ValueError(f"Malformed motion: {entry['source']}")
    depth = np.array([_AUDITOR.audit(frame)["deepest_penetration_mm"] for frame in q])
    valid = depth <= threshold_mm
    rows = []
    retained = np.zeros(len(q), dtype=bool)
    for part, (start, stop) in enumerate(_segments(valid)):
        if stop - start < min_frames:
            continue
        retained[start:stop] = True
        filename = f"{Path(entry['file']).stem}_part{part:03d}.npz"
        target = Path(output_dir) / filename
        temporary = target.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, q_ref=q[start:stop], tip_ref=tips[start:stop], source_indices=indices[start:stop], fps=fps
            )
        temporary.replace(target)
        row = dict(entry)
        row.update(
            source=f"{entry['source']}#part{part:03d}",
            origin_source=entry["source"],
            file=filename,
            frames=int(stop - start),
            source_start_frame=int(start),
            source_stop_frame_exclusive=int(stop),
            max_penetration_mm=float(depth[start:stop].max()),
            max_tip_step_m=(
                float(np.linalg.norm(np.diff(tips[start:stop], axis=0), axis=-1).max()) if stop - start > 1 else 0.0
            ),
        )
        rows.append(row)
    audit = {
        "source": entry["source"],
        "source_frames": len(q),
        "severe_frames": int((~valid).sum()),
        "discarded_short_fragment_frames": int((valid & ~retained).sum()),
        "kept_frames": int(retained.sum()),
        "segments": len(rows),
        "max_penetration_mm": float(depth.max()),
    }
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/hand21_right_collision_aware"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/hand21_right_geometry_feasible"))
    parser.add_argument("--severe-penetration-mm", type=float, default=5.0)
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.severe_penetration_mm <= 0 or args.min_frames < 1 or args.workers < 1:
        parser.error("Threshold, minimum frames, and workers must be positive")
    source_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if source_dir == output_dir or output_dir.exists():
        parser.error("Output must be a new directory distinct from input")
    raw = (source_dir / "manifest.json").read_bytes()
    source = json.loads(raw)
    xml = OFFICIAL_MJCF.resolve()
    if source["right_xml_sha256"] != hashlib.sha256(xml.read_bytes()).hexdigest():
        parser.error("Input library used another hand geometry")
    output_dir.mkdir(parents=True)
    jobs = [
        (entry, str(source_dir), str(output_dir), args.severe_penetration_mm, args.min_frames)
        for entry in source["motions"]
    ]
    records, audits = [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(str(xml),)) as pool:
        for index, (rows, audit) in enumerate(pool.map(_process, jobs), 1):
            records.extend(rows)
            audits.append(audit)
            if index % 100 == 0 or index == len(jobs):
                print(f"audited={index}/{len(jobs)} segments={len(records)}", flush=True)
    manifest = {key: value for key, value in source.items() if key not in {"motions", "errors"}}
    manifest.update(
        format_version=4,
        description="Official Hand 2.1 SDK targets, collision optimized, severe self-collision frames removed",
        screening={
            "criterion": "Hand 2.1 MJCF self-penetration only; no 10 mm replay threshold",
            "severe_penetration_mm": args.severe_penetration_mm,
            "minimum_segment_frames": args.min_frames,
            "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        },
        motions=records,
        accepted=len(records),
        rejected=sum(a["segments"] == 0 for a in audits),
        total_frames=sum(row["frames"] for row in records),
        errors=[{"source": a["source"], "error": "no usable segment"} for a in audits if a["segments"] == 0],
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "collision_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    print(
        f"segments={len(records)} source_motions={len(jobs)} kept_frames={manifest['total_frames']} "
        f"severe_frames={sum(a['severe_frames'] for a in audits)}"
    )


if __name__ == "__main__":
    main()
