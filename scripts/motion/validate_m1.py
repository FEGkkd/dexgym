"""Validate actual M1 pilot NPZ bytes and metadata before reporting the source gate."""

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/ubuntu/dexgym_data")
SCHEMA_FIELDS = {
    "time",
    "shoulder_pose",
    "upper_arm_pose",
    "elbow_pose",
    "forearm_pose",
    "hand_pose",
    "hand_linear_velocity",
    "hand_angular_velocity",
    "hand_orientation_valid",
}


def main() -> None:
    files = sorted((ROOT / "human_motion_v1/diagnostic_motions").glob("*.npz"))
    if len(files) != 20:
        raise ValueError(f"Expected 20 pilot diagnostic NPZ, found {len(files)}")
    result = []
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            if set(data.files) != SCHEMA_FIELDS:
                raise ValueError(f"Wrong schema: {file}")
            time = data["time"]
            if len(time) < 2 or time[0] != 0 or not np.all(np.diff(time) > 0):
                raise ValueError(f"Invalid time: {file}")
            for field in SCHEMA_FIELDS - {"time"}:
                if not np.isfinite(data[field]).all():
                    raise ValueError(f"Nonfinite {field}: {file}")
            for field in ("shoulder_pose", "upper_arm_pose", "elbow_pose", "forearm_pose", "hand_pose"):
                pose = data[field]
                if pose.shape != (len(time), 2, 7):
                    raise ValueError(f"Wrong pose shape {field}: {file}")
                if np.max(np.abs(np.linalg.norm(pose[..., 3:], axis=-1) - 1)) > 1e-5:
                    raise ValueError(f"Non-unit quaternion {field}: {file}")
                if np.min(np.sum(pose[1:, :, 3:] * pose[:-1, :, 3:], axis=-1)) < -1e-8:
                    raise ValueError(f"Quaternion sign discontinuity {field}: {file}")
            if data["hand_orientation_valid"].any():
                raise ValueError(f"Rejected motion marked valid: {file}")
            metadata = json.loads((ROOT / "human_motion_v1/metadata" / f"{file.stem}.json").read_text())
            if metadata["result_status"] != "rejected_full_6d" or not np.isclose(
                time[-1], metadata["source_duration_after_removal_s"]
            ):
                raise ValueError(f"Metadata mismatch: {file}")
            result.append(
                {
                    "motion_id": file.stem,
                    "frames": len(time),
                    "duration_s": float(time[-1]),
                    "effective_rate_hz": float((len(time) - 1) / time[-1]),
                }
            )
    report = {
        "validated_count": len(result),
        "total_frames": sum(x["frames"] for x in result),
        "total_duration_s": sum(x["duration_s"] for x in result),
        "full_6d_valid_count": 0,
        "status": "diagnostic_only",
        "records": result,
    }
    output = ROOT / "reports/m1_npz_validation.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
