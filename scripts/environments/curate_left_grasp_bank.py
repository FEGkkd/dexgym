"""Promote only PhysX held-and-replayed left grasps into GraspInitBank.

Input is the JSON output of generate_grasp_synthesis.py. This curation step
does not run physics and cannot turn a failed release or replay into a pass.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from dexteroushands_lab.scenes.grasp_init import GraspInitBank, grasp_sample_from_dict


def validation_metrics(result: dict, output: dict, digest: str) -> dict:
    replay = result.get("reset_replay")
    if result.get("release_passed") is not True or not replay or replay.get("passed") is not True:
        raise ValueError("Candidate did not pass both unsupported PhysX holds")
    candidate = result.get("candidate")
    if candidate is None:
        raise ValueError("Passing result has no captured release state")
    metadata = candidate["metadata"]
    synthesis = metadata["synthesis"]
    limits = synthesis["limits"]
    for outer, inner in (("asset_commit", "asset_source_commit"),
                         ("left_usd_sha256", "left_usd_sha256"),
                         ("right_usd_sha256", "right_usd_sha256")):
        if output[outer] != metadata[inner]:
            raise ValueError(f"Asset provenance changed between result and sample: {outer}")
    if output["wrist_control_mode"] != metadata["wrist_control_mode"]:
        raise ValueError("Wrist control mode changed between result and sample")
    release = result["release_finger_contact_time_s"]
    repeated = replay["left_finger_contact_time_s"]
    times = {digit: min(release.get(digit, 0.0), repeated.get(digit, 0.0))
             for digit in set(release) | set(repeated)}
    others = sorted((time for digit, time in times.items() if digit != "thumb"),
                    reverse=True)
    ground = max(result["contact_summary"].get("bottle_ground", {}).get(
        "contact_ratio", 0.0), replay["contact_summary"].get(
        "bottle_ground", {}).get("contact_ratio", 0.0))
    return {
        "scope": "left_only", "fixture": "gravity_on_no_support", "passed": True,
        "wrist_control_mode": output["wrist_control_mode"],
        "duration_s": limits["release_s"],
        "hold_displacement_m": max(result["bottle_displacement_m"],
                                   replay["bottle_displacement_m"]),
        "hold_rotation_rad": max(result["bottle_rotation_rad"],
                                 replay["bottle_rotation_rad"]),
        "left_only_displacement_m": max(result["bottle_displacement_m"],
                                        replay["bottle_displacement_m"]),
        "left_contact_ratio": min(result["release_contact_ratio"],
                                  replay["left_contact_ratio"]),
        "left_only_contact_ratio": min(result["release_contact_ratio"],
                                       replay["left_contact_ratio"]),
        "left_thumb_contact_time_s": times.get("thumb", 0.0),
        "left_other_contact_time_s": others[0] if others else 0.0,
        "left_additional_digit_contact_time_s": others[1] if len(others) > 1 else 0.0,
        "ground_contact_ratio": ground,
        "max_contact_peak_n": max(result["maximum_contact_force_n"],
                                  replay["left_contact_peak_n"]),
        "max_peak_force_guard_n": limits["max_contact_force_n"],
        "initial_min_separation_m": min(result["release_initial_min_separation_m"],
                                        replay["initial_min_separation_m"]),
        "minimum_separation_m": min(result["minimum_separation_m"],
                                    replay["minimum_separation_m"]),
        "max_initial_penetration_guard_m": limits["max_penetration_m"],
        "opposing_contact_coverage_ratio": min(
            result["release_opposing_contact_coverage_ratio"],
            replay["opposing_contact_coverage_ratio"]),
        "frictional_opposition_ratio": min(
            result["release_frictional_opposition_ratio"],
            replay["frictional_opposition_ratio"]),
        "support_released": synthesis["support_removed_before_acceptance"] and
                            all(not row["temporary_support_joint_enabled"]
                                for row in result["trace"] if row["phase"] == "release"),
        "reset_replay_passed": True,
        "source_result_sha256": digest,
        "release_result": {
            "displacement_m": result["bottle_displacement_m"],
            "rotation_rad": result["bottle_rotation_rad"],
            "contact_ratio": result["release_contact_ratio"],
        },
        "reset_replay_result": {
            "displacement_m": replay["bottle_displacement_m"],
            "rotation_rad": replay["bottle_rotation_rad"],
            "contact_ratio": replay["left_contact_ratio"],
        },
    }


def curate(paths: list[Path], bank_path: Path) -> list[dict]:
    bank = GraspInitBank(bank_path)
    known = {item.validation["source_result_sha256"] for item in bank.load()
             if "source_result_sha256" in item.validation}
    added = []
    for path in paths:
        raw = path.read_bytes()
        digest = sha256(raw).hexdigest()
        if digest in known:
            continue
        output = json.loads(raw)
        for result in output["results"]:
            if result.get("release_passed") is not True:
                continue
            metrics = validation_metrics(result, output, digest)
            sample = grasp_sample_from_dict(result["candidate"])
            bank.add(sample.with_validation(metrics))
            added.append({"path": str(path), "seed": sample.seed,
                          "hold_displacement_m": metrics["hold_displacement_m"]})
        known.add(digest)
    return added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--bank", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"added": curate(args.results, args.bank)}, indent=2))


if __name__ == "__main__":
    main()
