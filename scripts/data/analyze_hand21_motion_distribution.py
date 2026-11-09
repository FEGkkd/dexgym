"""Summarize the 30 Hz Hand 2.1 motion bank actually loaded by the RL task."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {f"p{x}": float(np.percentile(values, x)) for x in (5, 50, 90, 95, 99)}


def analyze(records: list[dict], root: Path, fps: float, joint_names: list[str], tip_names: list[str]):
    joints, joint_speeds, tips, speeds, clips = [], [], [], [], []
    jumps = {span: [] for span in (1, 4, 10, 20, 40)}
    subjects = defaultdict(lambda: {"segments": 0, "frames": 0})
    for record in records:
        with np.load(root / record["file"], allow_pickle=False) as data:
            q = np.asarray(data["q_ref"], dtype=np.float32)
            tip = np.asarray(data["tip_ref"], dtype=np.float32)
        if q.shape != (record["frames"], 20) or tip.shape != (record["frames"], 5, 3):
            raise ValueError(f"Unexpected motion shape: {record['file']}")
        joints.append(q)
        tips.append(tip)
        joint_speeds.append(np.abs(np.diff(q, axis=0)) * fps)
        speed = np.linalg.norm(np.diff(tip, axis=0), axis=-1) * fps * 1000
        speeds.append(speed)
        clips.append({"file": record["file"], "frames": len(q),
                      "duration_s": len(q) / fps,
                      "median_max_tip_speed_mm_s": float(np.median(speed.max(axis=1)))})
        subject = record["origin_source"].split("/")[0]
        subjects[subject]["segments"] += 1
        subjects[subject]["frames"] += len(q)
        for span in jumps:
            if len(tip) > span:
                jump = np.linalg.norm(tip[span:] - tip[:-span], axis=-1).max(axis=-1) * 1000
                jumps[span].append(jump)
    q = np.concatenate(joints)
    qspeed = np.concatenate(joint_speeds)
    tip = np.concatenate(tips)
    speed = np.concatenate(speeds)
    active = (speed > 50).sum(axis=1)
    max_speed = speed.max(axis=1)
    summary = {
        "segments": len(records), "frames": len(q), "duration_s": len(q) / fps,
        "clip_duration_s": quantiles(np.asarray([x["duration_s"] for x in clips])),
        "clip_median_max_tip_speed_mm_s": quantiles(
            np.asarray([x["median_max_tip_speed_mm_s"] for x in clips])
        ),
        "subjects": dict(sorted(subjects.items())),
        "joint": {name: {
            "angle_rad": quantiles(q[:, i]),
            "central_90_range_rad": float(np.percentile(q[:, i], 95) - np.percentile(q[:, i], 5)),
            "abs_velocity_rad_s": quantiles(qspeed[:, i]),
        } for i, name in enumerate(joint_names)},
        "tip": {name: {
            "speed_mm_s": quantiles(speed[:, i]),
            "fraction_above_50_mm_s": float(np.mean(speed[:, i] > 50)),
            "fraction_above_100_mm_s": float(np.mean(speed[:, i] > 100)),
            "total_path_m": float(np.sum(speed[:, i]) / fps / 1000),
            "position_m": {axis: quantiles(tip[:, i, j]) for j, axis in enumerate("xyz")},
        } for i, name in enumerate(tip_names)},
        "max_tip_speed_mm_s": quantiles(max_speed),
        "fraction_all_tips_below_20_mm_s": float(np.mean(max_speed < 20)),
        "fraction_any_tip_above_100_mm_s": float(np.mean(max_speed > 100)),
        "active_finger_count_above_50_mm_s": {
            str(i): float(np.mean(active == i)) for i in range(6)
        },
        "goal_jump_max_tip_mm": {
            str(span): {"pairs": int(sum(len(x) for x in arrays)),
                        "distance_mm": quantiles(np.concatenate(arrays)),
                        "fraction_within_15_mm": float(np.mean(np.concatenate(arrays) < 15))}
            for span, arrays in jumps.items()
        },
    }
    return summary


def plot(train: dict, validation: dict, joint_names: list[str], tip_names: list[str], output: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    labels = ["Thumb", "Index", "Middle", "Ring", "Little"]
    x = np.arange(5)
    for split, data, offset in (("Train", train, -0.18), ("Validation", validation, 0.18)):
        values = [data["tip"][name]["speed_mm_s"]["p90"] for name in tip_names]
        axes[0, 0].bar(x + offset, values, width=0.35, label=split)
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel("90th percentile tip speed (mm/s)")
    axes[0, 0].legend()

    spans = np.array([1, 4, 10, 20, 40])
    for split, data, marker in (("Train", train, "o"), ("Validation", validation, "s")):
        for percentile, style in (("p50", "-"), ("p90", "--")):
            values = [data["goal_jump_max_tip_mm"][str(s)]["distance_mm"][percentile] for s in spans]
            axes[0, 1].plot(spans, values, style, marker=marker, label=f"{split} {percentile}")
    axes[0, 1].axhline(15, color="black", linestyle=":", label="15 mm tracking tolerance (scale only)")
    axes[0, 1].set_xlabel("Reference-frame span (30 Hz)")
    axes[0, 1].set_ylabel("Largest of five tip displacements (mm)")
    axes[0, 1].legend(fontsize=8)

    for split, data, offset in (("Train", train, -0.18), ("Validation", validation, 0.18)):
        values = [data["active_finger_count_above_50_mm_s"][str(i)] * 100 for i in range(6)]
        axes[1, 0].bar(np.arange(6) + offset, values, width=0.35, label=split)
    axes[1, 0].set_xticks(range(6))
    axes[1, 0].set_xlabel("Fingers moving faster than 50 mm/s")
    axes[1, 0].set_ylabel("Share of within-clip frame transitions (%)")
    axes[1, 0].legend()

    x = np.arange(20)
    for split, data, offset in (("Train", train, -0.18), ("Validation", validation, 0.18)):
        values = [data["joint"][name]["central_90_range_rad"] for name in joint_names]
        axes[1, 1].bar(x + offset, values, width=0.35, label=split)
    axes[1, 1].set_xticks(x, [name[2:] for name in joint_names], rotation=70, fontsize=8)
    axes[1, 1].set_ylabel("Central 90% joint range (rad)")
    axes[1, 1].legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/hand21_right_geometry_feasible/manifest.json"))
    parser.add_argument("--min-frames", type=int, default=6, help="Match the RL task's subgoal_stride + 2 gate")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    fps = float(manifest["fps"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for split in ("train", "validation"):
        records = [r for r in manifest["motions"] if r["split"] == split and r["frames"] >= args.min_frames]
        results[split] = analyze(records, args.manifest.parent, fps, manifest["joint_names"], manifest["tip_names"])
    results["metadata"] = {"manifest": str(args.manifest.resolve()), "fps": fps,
                           "min_frames": args.min_frames, "threshold_mm": 15,
                           "velocity_note": "Within-segment finite differences only; no segment-boundary velocity"}
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    plot(results["train"], results["validation"], manifest["joint_names"], manifest["tip_names"],
         args.output_dir / "motion_distribution.png")
    print(f"MOTION_DISTRIBUTION_OK {args.output_dir}")


if __name__ == "__main__":
    main()
