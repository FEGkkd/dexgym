"""Process one actual CMU BVH or the 20-clip Subject 141 pilot into M1 diagnostic outputs."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))
from dexgym.motion.bvh import human_fk, parse_bvh, resample_motion
from dexgym.motion.human_motion import build_upper_body
from dexgym.motion.orientation_audit import audit_hand_orientation

PILOT_IDS = (
    "141_04",
    "141_06",
    "141_10",
    "141_11",
    "141_12",
    "141_13",
    "141_14",
    "141_15",
    "141_16",
    "141_18",
    "141_20",
    "141_21",
    "141_22",
    "141_23",
    "141_24",
    "141_26",
    "141_28",
    "141_29",
    "141_32",
    "141_33",
)
DATA_ROOT = Path("/home/ubuntu/dexgym_data")


def process(path: Path, root: Path, assumed_stature_m: float) -> dict:
    motion = parse_bvh(path)
    # Frame zero is an author-added T-pose. Stature calibration is an explicit
    # assumption for diagnostic geometry, not a source-certified unit conversion.
    t_pose = human_fk(motion, scale_m_per_unit=1, remove_t_pose=False)
    source_stature = float(np.ptp(t_pose.position[0, :, 1]))
    if source_stature <= 0:
        raise ValueError("Invalid T-pose height")
    scale = assumed_stature_m / source_stature
    fk = human_fk(motion, scale_m_per_unit=scale, remove_t_pose=True)
    audit = audit_hand_orientation(motion, fk)
    sampled = resample_motion(fk, sample_rate_hz=50)
    arrays = build_upper_body(sampled, audit)
    motion_id = path.stem
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    state = "accepted_m1" if audit["full_6d_accepted"] else "rejected_full_6d"
    out = root / "human_motion_v1" / ("motions" if state == "accepted_m1" else "diagnostic_motions")
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / f"{motion_id}.npz"
    np.savez_compressed(npz_path, **arrays)
    metadata = {
        "schema": "HumanUpperBodyMotion v1",
        "motion_id": motion_id,
        "source_bvh": str(path.resolve()),
        "source_bvh_sha256": source_sha,
        "source_subject": path.parent.name,
        "original_frame_count_including_t_pose": len(motion.frames),
        "original_frame_time_s": motion.frame_time,
        "removed_first_frame": True,
        "first_frame_reason": "Bruce Hahne 2010 conversion inserts artificial T-pose at frame 0",
        "source_duration_after_removal_s": float(fk.time[-1]),
        "resampled_frame_count": len(sampled.time),
        "resampled_duration_s": float(sampled.time[-1]),
        "effective_sample_rate_hz": float((len(sampled.time) - 1) / sampled.time[-1]),
        "target_sample_rate_hz": 50,
        "position_unit": "m",
        "time_unit": "s",
        "angular_velocity_unit": "rad/s",
        "quaternion_order": "XYZW",
        "coordinate_frame": "BVH right-handed, Y up, positive Z forward",
        "length_scale_m_per_source_unit": scale,
        "length_scale_method": "assumed adult T-pose stature; not source-certified",
        "assumed_stature_m": assumed_stature_m,
        "source_t_pose_height_units": source_stature,
        "joint_hierarchy": [
            {
                "name": j.name,
                "parent": motion.joints[j.parent].name if j.parent >= 0 else None,
                "offset_source_units": j.offset.tolist(),
                "channels": list(j.channels),
                "channel_start": j.channel_start,
            }
            for j in motion.joints
        ],
        "pose_node_mapping": {"elbow_pose": "ForeArm joint origin", "forearm_pose": "ForeArm joint frame"},
        "orientation_available": all(h["orientation_available"] for h in audit["hands"].values()),
        "orientation_reconstruction_valid": all(h["orientation_reconstruction_valid"] for h in audit["hands"].values()),
        "orientation_source": "bvh_reconstructed",
        "orientation_confidence": min(
            (h["orientation_confidence"] for h in audit["hands"].values()),
            key=lambda x: ["none", "low", "moderate", "high"].index(x),
        ),
        "left_hand_rotation_quality": audit["hands"]["left"]["rotation_quality"],
        "right_hand_rotation_quality": audit["hands"]["right"]["rotation_quality"],
        "orientation_limitations": sorted(
            {item for h in audit["hands"].values() for item in h["orientation_limitations"]}
        ),
        "audit": audit,
        "result_status": state,
        "npz_path": str(npz_path),
    }
    metadata_dir = root / "human_motion_v1" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / f"{motion_id}.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=DATA_ROOT / "cmu/cmu-mocap")
    parser.add_argument("--output-root", type=Path, default=DATA_ROOT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--motion-id")
    group.add_argument("--pilot", action="store_true")
    parser.add_argument("--assumed-stature-m", type=float, default=1.70)
    args = parser.parse_args()
    if args.assumed_stature_m <= 0:
        parser.error("--assumed-stature-m must be positive")
    ids = PILOT_IDS if args.pilot else (args.motion_id,)
    records = []
    for motion_id in ids:
        subject = motion_id.split("_", 1)[0]
        path = args.repo / "data" / subject / f"{motion_id}.bvh"
        if not path.is_file():
            raise FileNotFoundError(path)
        record = process(path, args.output_root, args.assumed_stature_m)
        records.append(record)
        left_axes = record["audit"]["hands"]["left"]["active_local_rotation_axes"]
        right_axes = record["audit"]["hands"]["right"]["active_local_rotation_axes"]
        print(f"{motion_id}: {record['result_status']}, L={left_axes}, R={right_axes}")
    report = {
        "source_repo": str(args.repo),
        "selected_count": len(ids),
        "parsed_count": len(records),
        "full_6d_accepted_count": sum(x["result_status"] == "accepted_m1" for x in records),
        "records": [
            {k: x[k] for k in ("motion_id", "result_status", "source_bvh_sha256", "npz_path", "audit")} for x in records
        ],
    }
    output = args.output_root / (
        "reports/pilot_human_orientation_audit.json" if args.pilot else "reports/single_human_orientation_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
