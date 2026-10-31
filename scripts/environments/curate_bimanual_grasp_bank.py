"""Admit only fresh-process replayed, unsupported PhysX bimanual grasps.

Inputs are complete output JSON files from the formation and fresh-process
replay scripts. This offline curation cannot make an untested candidate pass.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from dexteroushands_lab.scenes.grasp_init import GraspInitBank, grasp_sample_from_dict


def _digit_times(holds, side):
    entries = [hold[f"{side}_finger_contact_time_s"] for hold in holds]
    thumb = min(entry.get("thumb", 0.0) for entry in entries)
    other = min(max((time for digit,time in entry.items() if digit != "thumb"),
                    default=0.0) for entry in entries)
    return thumb, other


def curate(result_path: Path, replay_path: Path, left_bank_path: Path,
           target_bank_path: Path):
    raw = result_path.read_bytes()
    result = json.loads(raw)
    fresh_raw = replay_path.read_bytes()
    fresh = json.loads(fresh_raw)
    if (result.get("passed") is not True or result.get("hand_hand_monitored") is not True
            or result.get("support_used_during_right_formation") is not False
            or result.get("candidate") is None):
        raise ValueError("Formation lacks full no-support, hand-hand-monitored PhysX evidence")
    if (fresh["summary"].get("fresh_process_replay_passed") is not True
            or fresh["summary"].get("source_result_sha256") != sha256(raw).hexdigest()):
        raise ValueError("Fresh-process replay is missing, failed, or is from another result")
    sample = grasp_sample_from_dict(result["candidate"])
    left_bank = GraspInitBank(left_bank_path).load()
    source = left_bank[result["left_sample_index"]]
    if (source.seed != sample.seed or source.validation.get("scope") != "left_only"
            or source.validation.get("wrist_control_mode") != "pose_command_wrist"
            or source.metadata["left_usd_sha256"] != sample.metadata["left_usd_sha256"]
            or source.metadata["right_usd_sha256"] != sample.metadata["right_usd_sha256"]
            or source.metadata["bottle_parameters"] != sample.metadata["bottle_parameters"]):
        raise ValueError("Captured state is not derived from the validated pose-left bank")
    if (sample.metadata.get("right_cap_synthesis", {}).get("support_used") is not False
            or sample.metadata.get("synthesis", {}).get("support") != "none"):
        raise ValueError("Bimanual candidate has unsupported formation provenance")
    holds = [result["unsupported_hold"], result["reset_replay"], fresh["replay"]]
    if any(hold.get("passed") is not True for hold in holds):
        raise ValueError("All three unsupported hold runs must pass")
    left_thumb, left_other = _digit_times(holds, "left")
    right_thumb, right_other = _digit_times(holds, "right")
    cross = {name: max(hold["cross_contact_ratios"][name] for hold in holds)
             for name in ("left_cap", "right_bottle", "hand_hand")}
    ground = max(hold["contact_summary"].get(name, {}).get("contact_ratio", 0.0)
                 for hold in holds for name in ("bottle_ground", "cap_ground"))
    minimum = min(result["formation_minimum_separation_m"],
                  *(hold["minimum_separation_m"] for hold in holds))
    initial = min(result["formation_minimum_separation_m"],
                  *(hold["initial_min_separation_m"] for hold in holds))
    metrics = {
        "scope": "bimanual", "contact_evidence_version": "radial_contact_v1",
        "fixture": "gravity_on_no_support", "passed": True,
        "wrist_control_mode": "pose_command_wrist",
        "duration_s": min(hold["duration_s"] for hold in holds),
        "hold_displacement_m": max(hold["bottle_displacement_m"] for hold in holds),
        "hold_rotation_rad": max(hold["bottle_rotation_rad"] for hold in holds),
        "left_only_displacement_m": source.validation["left_only_displacement_m"],
        "left_contact_ratio": min(hold["left_contact_ratio"] for hold in holds),
        "right_contact_ratio": min(hold["right_contact_ratio"] for hold in holds),
        "left_only_contact_ratio": source.validation["left_only_contact_ratio"],
        "left_thumb_contact_time_s": left_thumb,
        "left_other_contact_time_s": left_other,
        "right_thumb_contact_time_s": right_thumb,
        "right_other_contact_time_s": right_other,
        "right_radial_opposition_ratio": min(
            hold["right_radial_opposition_ratio"] for hold in holds),
        "right_min_radial_normal_dot": min(
            hold["right_min_radial_normal_dot"] for hold in holds),
        "max_thread_lateral_error_m": max(
            hold["max_thread_lateral_error_m"] for hold in holds),
        "cross_contact_ratios": cross,
        "ground_contact_ratio": ground,
        "max_contact_peak_n": max(result["formation_peak_contact_force_n"],
                                  *(hold["peak_contact_force_n"] for hold in holds)),
        "max_peak_force_guard_n": 100.0,
        "initial_min_separation_m": initial,
        "minimum_separation_m": minimum,
        "max_initial_penetration_guard_m": .002,
        "hand_hand_monitored": True,
        "support_released": True,
        "reset_replay_passed": True,
        "fresh_process_replay_passed": True,
        "source_result_sha256": sha256(raw).hexdigest(),
        "fresh_replay_result_sha256": sha256(fresh_raw).hexdigest(),
        "left_only_source_sha256": source.validation["source_result_sha256"],
    }
    bank = GraspInitBank(target_bank_path)
    if any(entry.validation.get("source_result_sha256") == metrics["source_result_sha256"]
           for entry in bank.load()):
        return {"added": False, "seed": sample.seed}
    bank.add(sample.with_validation(metrics))
    return {"added": True, "seed": sample.seed,
            "right_proposal_index": sample.metadata["right_cap_proposal_index"],
            "bottle_displacement_m": metrics["hold_displacement_m"],
            "right_radial_opposition_ratio": metrics["right_radial_opposition_ratio"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formation", type=Path, required=True)
    parser.add_argument("--fresh_replay", type=Path, required=True)
    parser.add_argument("--left_bank", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(curate(args.formation, args.fresh_replay,
                            args.left_bank, args.bank), indent=2))


if __name__ == "__main__":
    main()
