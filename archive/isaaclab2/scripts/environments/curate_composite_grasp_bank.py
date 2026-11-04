"""Admit a composite grasp only after three unsupported PhysX holds agree.

The output can depend on a user-supplied, uncommitted source bank. Keep such
results in ignored artifacts until all input data is versioned for reuse.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from dexteroushands_lab.scenes.grasp_init import GraspInitBank, grasp_sample_from_dict


def curate(result_path, replay_path, left_bank_path, right_bank_path, output_path):
    raw = result_path.read_bytes()
    result = json.loads(raw)
    fresh_raw = replay_path.read_bytes()
    fresh = json.loads(fresh_raw)
    left_raw = left_bank_path.read_bytes()
    right_raw = right_bank_path.read_bytes()
    left = GraspInitBank(left_bank_path).load()[0]
    right = GraspInitBank(right_bank_path).load()[0]
    sample = grasp_sample_from_dict(result["candidate"])
    if (result.get("passed") is not True or
            fresh["summary"].get("fresh_process_replay_passed") is not True or
            fresh["summary"].get("source_result_sha256") != sha256(raw).hexdigest() or
            result.get("independent_fresh_replay_required") is not True or
            sample.metadata.get("composite_left_source_sha256") != sha256(left_raw).hexdigest() or
            sample.metadata.get("composite_right_source_sha256") != sha256(right_raw).hexdigest() or
            left.validation.get("scope") != "left_only" or
            right.validation.get("right_grasp_topology") != "thumb_index_middle_tripod" or
            result.get("candidate", {}).get("validated") is not False):
        raise ValueError("Composite source or independent PhysX evidence is invalid")
    holds = [result["unsupported_hold"], result["same_process_reset_replay"],
             fresh["replay"]]
    if any(hold.get("passed") is not True for hold in holds):
        raise ValueError("Every unsupported hold must pass")
    duration = min(hold["duration_s"] for hold in holds)
    loaded = lambda hold, side, digit: hold[f"{side}_finger_contact_time_s"].get(digit, 0.0)
    left_loaded = tuple(digit for digit in ("thumb", "index", "middle", "ring", "pinky")
                        if all(loaded(hold, "left", digit) > .8*hold["duration_s"]
                               for hold in holds))
    right_loaded = tuple(digit for digit in ("thumb", "index", "middle")
                         if all(loaded(hold, "right", digit) > .8*hold["duration_s"]
                                for hold in holds))
    if ("thumb" not in left_loaded or len(set(left_loaded) - {"thumb"}) < 2 or
            right_loaded != ("thumb", "index", "middle") or
            any(hold["right_tripod_held"] is not True for hold in holds)):
        raise ValueError("Composite failed the left multi-digit/right tripod topology")
    cross = {name: max(hold["cross_contact_ratios"][name] for hold in holds)
             for name in ("left_cap", "right_bottle", "hand_hand")}
    ground = max(hold["contact_summary"].get(name, {}).get("contact_ratio", 0.0)
                 for hold in holds for name in ("bottle_ground", "cap_ground"))
    metrics = {
        "scope": "bimanual", "contact_evidence_version": "radial_contact_v1",
        "fixture": "gravity_on_no_support", "passed": True,
        "wrist_control_mode": "pose_command_wrist", "duration_s": duration,
        "hold_displacement_m": max(hold["bottle_displacement_m"] for hold in holds),
        "hold_rotation_rad": max(hold["bottle_rotation_rad"] for hold in holds),
        "left_only_displacement_m": left.validation["left_only_displacement_m"],
        "left_contact_ratio": min(hold["left_contact_ratio"] for hold in holds),
        "right_contact_ratio": min(hold["right_contact_ratio"] for hold in holds),
        "left_only_contact_ratio": left.validation["left_only_contact_ratio"],
        "left_thumb_contact_time_s": min(loaded(hold, "left", "thumb") for hold in holds),
        "left_other_contact_time_s": min(max(
            loaded(hold, "left", digit) for digit in ("index", "middle", "ring", "pinky"))
            for hold in holds),
        "right_thumb_contact_time_s": min(loaded(hold, "right", "thumb") for hold in holds),
        "right_other_contact_time_s": min(max(
            loaded(hold, "right", digit) for digit in ("index", "middle")) for hold in holds),
        "left_loaded_digits": left_loaded, "right_loaded_digits": right_loaded,
        "left_grasp_topology": "multi_digit_power",
        "right_grasp_topology": "thumb_index_middle_tripod",
        "right_radial_opposition_ratio": min(
            hold["right_radial_opposition_ratio"] for hold in holds),
        "max_thread_lateral_error_m": max(
            hold["max_thread_lateral_error_m"] for hold in holds),
        "cross_contact_ratios": cross, "ground_contact_ratio": ground,
        "max_contact_peak_n": max(hold["peak_contact_force_n"] for hold in holds),
        "max_peak_force_guard_n": 100.0,
        "initial_min_separation_m": min(hold["initial_min_separation_m"] for hold in holds),
        "minimum_separation_m": min(hold["minimum_separation_m"] for hold in holds),
        "max_initial_penetration_guard_m": .002,
        "hand_hand_monitored": True, "support_released": True,
        "reset_replay_passed": True, "fresh_process_replay_passed": True,
        "source_result_sha256": sha256(raw).hexdigest(),
        "fresh_replay_result_sha256": sha256(fresh_raw).hexdigest(),
        "left_source_bank_sha256": sha256(left_raw).hexdigest(),
        "right_source_bank_sha256": sha256(right_raw).hexdigest(),
    }
    if any(not math.isfinite(float(value)) for value in
           (metrics["hold_displacement_m"], metrics["hold_rotation_rad"])):
        raise ValueError("Non-finite composite hold metric")
    validated = sample.with_validation(metrics)
    bank = GraspInitBank(output_path)
    if any(entry.validation.get("source_result_sha256") == metrics["source_result_sha256"]
           for entry in bank.load()):
        return {"added": False, "output": str(output_path)}
    bank.add(validated)
    return {"added": True, "output": str(output_path),
            "left_loaded_digits": left_loaded,
            "right_loaded_digits": right_loaded,
            "max_hold_displacement_m": metrics["hold_displacement_m"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formation", type=Path, required=True)
    parser.add_argument("--fresh_replay", type=Path, required=True)
    parser.add_argument("--left_bank", type=Path, required=True)
    parser.add_argument("--right_bank", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(curate(args.formation, args.fresh_replay,
                            args.left_bank, args.right_bank, args.bank), indent=2))


if __name__ == "__main__":
    main()
