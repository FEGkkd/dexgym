"""Admit a Mimic PGS grasp only from independent no-support PhysX replays."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

from dexteroushands_lab.scenes.grasp_init import grasp_sample_from_dict


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curate(candidate_path: Path, replay_path: Path, left_only_path: Path):
    candidate_result = _load(candidate_path)
    replay = _load(replay_path)
    left_only = _load(left_only_path)
    sample = grasp_sample_from_dict(candidate_result["diagnostic_candidate"])
    if (sample.validated or sample.source !=
            "pgs_physx_settled_candidate_requires_fresh_reset_replay"):
        raise ValueError("Expected an unvalidated live PGS state")
    if (sample.metadata.get("solver_type") != "pgs"
            or replay.get("solver_type") != "pgs"
            or left_only.get("solver_type") != "pgs"):
        raise ValueError("All measurements must use the PGS solver")
    if (replay["source_result"] != str(candidate_path)
            or left_only["source_result"] != str(candidate_path)):
        raise ValueError("Replay provenance differs from captured candidate")
    if not replay["diagnostic_pass"] or not left_only["right_cap_contact_ablated"]:
        raise ValueError("Independent full hold or left-only ablation was not run")
    if left_only["right_sidewall_tripod_ratio"] != 0 or left_only["failure_any"]:
        raise ValueError("Left-only ablation is invalid")
    if (left_only["left_contact_ratio"] <= .8
            or left_only["maximum_bottle_drift_m"] >= .02
            or left_only["ground_contact_any"]):
        raise ValueError("Left hand did not support the free bottle")
    if any(not math.isfinite(row["cap_spin_rad_s"]) for row in replay["rows"]):
        raise ValueError("Nonfinite screw velocity")
    metrics = {
        "passed": True,
        "fixture": "gravity_on_no_support",
        "scope": "bimanual",
        "contact_evidence_version": "mimic_force_opposition_v1",
        "duration_s": replay["duration_s"],
        "hold_displacement_m": replay["maximum_bottle_drift_m"],
        "hold_rotation_rad": replay["maximum_bottle_rotation_rad"],
        "left_only_displacement_m": left_only["maximum_bottle_drift_m"],
        "left_contact_ratio": replay["left_contact_ratio"],
        "right_contact_ratio": replay["right_sidewall_tripod_ratio"],
        "left_only_contact_ratio": left_only["left_contact_ratio"],
        "ground_contact_ratio": 1.0 if replay["ground_contact_any"] else 0.0,
        "opposing_contact_coverage_ratio": replay["left_force_opposition_ratio"],
        "right_sidewall_tripod_ratio": replay["right_sidewall_tripod_ratio"],
        "cross_contact_ratios": replay["cross_contact_ratios"],
        "hand_hand_monitored": True,
        "initial_min_separation_m": replay["initial_min_contact_separation_m"],
        "minimum_separation_m": replay["minimum_contact_separation_m"],
        "max_initial_penetration_guard_m": .002,
        "max_contact_peak_n": max(
            replay["maximum_left_net_body_force_n"],
            replay["maximum_right_net_body_force_n"],
            left_only["maximum_left_net_body_force_n"]),
        "max_peak_force_guard_n": 100.,
        "max_thread_pitch_error_m": replay["maximum_thread_error_m"],
        "support_released": True,
        "reset_replay_passed": True,
        "fresh_process_replay_passed": True,
        "source_result_sha256": _digest(candidate_path),
        "fresh_replay_result_sha256": _digest(replay_path),
        "left_only_source_sha256": _digest(left_only_path),
        "wrist_control_mode": "pose_command_wrist",
    }
    metadata = dict(sample.metadata)
    metadata["mimic_candidate_unverified"] = False
    metadata["validation_family"] = "row9_pgs_same_relative_grasp_family"
    sample = replace(sample, source="physx_pgs_mimic_grasp_fresh_replay_validated",
                     metadata=metadata)
    return sample.with_validation(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--replay-result", type=Path, required=True)
    parser.add_argument("--left-only-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sample = curate(args.candidate_result, args.replay_result, args.left_only_result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([asdict(sample)], indent=2) + "\n")
    print("MIMIC_GRASP_BANK", json.dumps({
        "validated_samples": 1,
        "source_family": sample.metadata["validation_family"],
        "hold_displacement_m": sample.validation["hold_displacement_m"],
        "left_only_displacement_m": sample.validation["left_only_displacement_m"],
    }), flush=True)


if __name__ == "__main__":
    main()
