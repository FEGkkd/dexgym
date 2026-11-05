"""Validate and register the two-arm CMU 141_16 reference without changing the one-arm entry."""

import argparse
import json
import shutil
from pathlib import Path

from retarget_marvin_wave import OUT
from validate_marvin_reference import assess

ID = "cmu_141_16_wave_bimanual"
ROOT = OUT / "bimanual"


def register(candidate: Path, root: Path = ROOT):
    result = assess(candidate, root, require_bimanual=True)
    report_path = root / "reports" / f"{ID}_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2))
    if result["status"] != "validated":
        return result
    retarget = json.loads((OUT / "reports" / f"{ID}_retarget.json").read_text())
    config = json.loads((OUT / "metadata/retarget_config.json").read_text())
    config.update(
        {
            "motion_id": ID,
            "retargeted_arms": ["left", "right"],
            "left_position_mapping": retarget["left_mapping"],
            "left_local_hand_channel_ranges_deg": retarget["left_local_hand_channel_ranges_deg"],
            "left_hand_to_tcp_quat_xyzw": retarget["left_hand_to_tcp_quat_xyzw"],
            "left_tcp_target_formula": "R_alignment @ R_human_left_hand @ R_left_hand_to_tcp",
            "left_ik": {
                "joints": [f"Joint{i}_L" for i in range(1, 8)],
                "position_scale_m": 0.012,
                "orientation_scale_rad": 0.12,
                "elbow_weight": 0.8,
                "joint_smoothing_sigma_50hz_frames": 5.0,
                "warm_start": True,
                "multi_seed": True,
            },
            "source_right_reference": retarget["source_right_reference"],
            "source_right_q_sha256": retarget["source_right_q_sha256"],
            "validation_report": str(report_path),
            "physical_replay_root": str(root / "physics_replays"),
            "candidate_paths": config["candidate_paths"] + [str(candidate)],
        }
    )
    config_path = root / "metadata" / f"{ID}_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))
    formal = OUT / "validated_motions" / f"{ID}.npz"
    formal.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, formal)
    index_path = OUT / "index.json"
    index = json.loads(index_path.read_text())
    index["motions"] = [m for m in index["motions"] if m["id"] != ID]
    index["motions"].append(
        {
            "id": ID,
            "status": "validated",
            "npz": str(formal),
            "metadata": str(config_path),
            "validation": str(report_path),
            "frames": result["frames"],
            "fps": 50,
            "duration_s": result["duration_s"],
            "retargeted_arms": ["left", "right"],
        }
    )
    index_path.write_text(json.dumps(index, indent=2))
    result["formal_npz"] = str(formal)
    report_path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path, default=OUT / "preview_motions" / f"{ID}.npz", nargs="?")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = register(args.candidate, args.root)
    print(json.dumps(result, indent=2))
    if result["status"] != "validated":
        raise SystemExit(1)
