"""Audit every materialized CMU BVH and commit-verified cross-Subject spot check."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))
from dexgym.motion.bvh import human_fk, parse_bvh
from dexgym.motion.orientation_audit import audit_hand_orientation

ROOT = Path("/home/ubuntu/dexgym_data")
REPO = ROOT / "cmu/cmu-mocap"


def main() -> None:
    files = sorted(REPO.glob("data/*/*.bvh")) + sorted((ROOT / "cmu/spot_checks").glob("*.bvh"))
    records = []
    for file in files:
        if file.parent.name == "spot_checks":
            ident = file.stem
            relative = f"data/{ident.split('_')[0].zfill(3)}/{ident}.bvh"
            expected = subprocess.check_output(
                ["git", "-C", str(REPO), "rev-parse", f"HEAD:{relative}"], text=True
            ).strip()
            actual = subprocess.check_output(["git", "hash-object", str(file)], text=True).strip()
            if actual != expected:
                raise ValueError(f"Spot-check bytes differ from pinned Git blob: {file}")
        else:
            relative = str(file.relative_to(REPO))
        motion = parse_bvh(file)
        fk = human_fk(motion, scale_m_per_unit=1, remove_t_pose=True)
        audit = audit_hand_orientation(motion, fk)
        records.append(
            {
                "motion_id": file.stem,
                "source_path": relative,
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                "frame_count_including_t_pose": len(motion.frames),
                "frame_time_s": motion.frame_time,
                "audit": audit,
            }
        )
    unique = {r["source_path"]: r for r in records}
    records = list(unique.values())
    report = {
        "source": "una-dinosauria/cmu-mocap",
        "repository_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "scanned_motion_count": len(records),
        "scanned_subject_count": len({Path(r["source_path"]).parent.name for r in records}),
        "bilateral_hand_nodes_with_rotation_channels": sum(
            all(h["orientation_available"] for h in r["audit"]["hands"].values()) for r in records
        ),
        "bilateral_world_so3_reconstruction_valid": sum(
            all(h["orientation_reconstruction_valid"] for h in r["audit"]["hands"].values()) for r in records
        ),
        "bilateral_only_one_dynamic_local_axis_count": sum(
            all(len(h["active_local_rotation_axes"]) == 1 for h in r["audit"]["hands"].values()) for r in records
        ),
        "bilateral_zy_exactly_zero_count": sum(
            all(
                h["channel_dynamic_range_deg"].get("Zrotation") == 0
                and h["channel_dynamic_range_deg"].get("Yrotation") == 0
                for h in r["audit"]["hands"].values()
            )
            for r in records
        ),
        "full_6d_accepted_count": sum(r["audit"]["full_6d_accepted"] for r in records),
        "high_speed_hand_count": sum(h["high_speed_flag"] for r in records for h in r["audit"]["hands"].values()),
        "unrepaired_jump_hand_count": sum(h["jump_count"] > 0 for r in records for h in r["audit"]["hands"].values()),
        "orientation_source": "bvh_reconstructed",
        "independent_full_palm_orientation_proven": False,
        "source_limitations": [
            "The mirror is an ASF/AMC-to-BVH conversion with empirically adjusted artificial T-pose.",
            "The source README says finger and thumb motion was not captured.",
            "Observed local Hand Z and Y channels are exactly zero in all audited clips; only X changes.",
            "World SO(3) from FK inherits forearm rotation; "
            "it does not establish independently observed full palm orientation.",
        ],
        "records": records,
    }
    out = ROOT / "reports/human_orientation_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("records", "source_limitations")}, indent=2))


if __name__ == "__main__":
    main()
