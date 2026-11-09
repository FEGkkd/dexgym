"""Audit the real S0 reference manifest without importing the simulator.

This is a data/configuration preflight, never a policy-performance result.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def audit(path: Path) -> dict:
    path = path.expanduser().resolve()
    manifest = json.loads(path.read_text())
    if manifest["fps"] != 30 or len(manifest["joint_names"]) != 20:
        raise ValueError("S0 requires 30 Hz and 20 hand joints")
    if len(manifest["tip_names"]) != 5:
        raise ValueError("S0 requires five fingertips")
    if set(manifest["train_subjects"]) & set(manifest["validation_subjects"]):
        raise ValueError("Train and validation subjects overlap")
    counts = Counter()
    frames = Counter()
    missing = []
    seen = set()
    for entry in manifest["motions"]:
        if entry["file"] in seen:
            raise ValueError("Duplicate motion file")
        seen.add(entry["file"])
        if entry["split"] not in ("train", "validation"):
            raise ValueError("Unexpected split")
        if not (path.parent / entry["file"]).is_file():
            missing.append(entry["file"])
        # Same eligibility criterion as current PPO: subgoal_stride(4)+2.
        if entry["frames"] >= 6:
            counts[entry["split"]] += 1
            frames[entry["split"]] += entry["frames"]
    if missing or min(counts.get(s, 0) for s in ("train", "validation")) < 1:
        raise ValueError(f"Missing data or empty split: {missing[:5]}")
    return {
        "artifact_kind": "data_preflight_not_training_result",
        "manifest": str(path),
        "fps": 30,
        "joints": 20,
        "fingertips": 5,
        "eligible_clips": dict(counts),
        "eligible_frames": dict(frames),
        "train_subjects": manifest["train_subjects"],
        "validation_subjects": manifest["validation_subjects"],
        "missing_files": missing,
        "policy_accuracy": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
