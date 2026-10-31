"""Compress Phase 0B raw PhysX reports without converting failures to passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(directory, name):
    path = directory / name
    return json.loads(path.read_text()) if path.exists() else None


def summarize(directory: Path):
    result = {"schema": "dexgym-phase0b-measured-summary-v1",
              "physics_backend": "Isaac Sim 6.0.1 / Isaac Lab 3.0 / PhysX",
              "units": "m,rad,s,N,Nm",
              "legacy_initial_states": {}, "diameter_scan": [],
              "synthesis_groups": {}, "reset_regression": {}}
    old = _load(directory, "phase0b_legacy_recheck.json")
    if old:
        result["asset_commit"] = old["asset_commit"]
        result["left_usd_sha256"] = old["left_asset_sha256"]
        for seed, entry in old["seeds"].items():
            hold = entry["hold"]
            initial_pairs = [
                {"pair": name, "minimum_separation_m": value["minimum_separation_m"],
                 "peak_force_n": value["peak_force_n"],
                 "latest_position_w_m": value["latest_position_w"]}
                for contact in hold["initial_contact_summary"].values()
                for name, value in contact["bodies"].items()
                if value["minimum_separation_m"] is not None]
            initial_pairs.sort(key=lambda item: item["minimum_separation_m"])
            result["legacy_initial_states"][seed] = {
                "first_step_min_separation_m": hold["initial_min_separation_m"],
                "first_step_left_peak_n": hold["initial_contact_summary"].get(
                    "left_bottle", {}).get("peak_force_n", 0.0),
                "one_second_bottle_displacement_m": hold["bottle_displacement_m"],
                "left_contact_ratio": hold["correct_contact_ratios"]["left_bottle"],
                "first_step_bottle_velocity_w": hold.get("first_step_bottle_velocity_w"),
                "first_step_left_velocity_w": hold.get("first_step_left_velocity_w"),
                "worst_initial_pairs": initial_pairs[:8],
                "worst_raw_contact_points": hold.get("initial_raw_contact_points", [])[:5],
                "passed": entry["passed"]}
        result["free_fall"] = old["no_contact_free_fall"]
    reach = _load(directory, "phase0b_reach_scan.json")
    if reach:
        result["diameter_scan"] = [{"diameter_m": 2*r["radius_m"],
                                    "samples": r["samples"],
                                    "clear_thumb_two_finger_candidates":
                                    r["nominal_clear_candidates"],
                                    "contact_digit_candidates_with_possible_penetration":
                                    r["contact_digit_candidates_including_deep_penetration"]}
                                   for r in reach]
    groups = {
        "70mm_random_20": "phase0b_synthesis_35mm.json",
        "50mm_diagnostic_5": "phase0b_synthesis_25mm_diagnostic.json",
        "70mm_tip_optimized_8": "phase0b_tip_optimized_synthesis_35mm.json",
        "70mm_fixed_support_1": "phase0b_fixed_support_probe_35mm.json",
        "70mm_slow_close_1": "phase0b_slow_close_probe_35mm.json",
        "70mm_low_damping_wrist_8": "phase0b_tuned_wrist_fixed_support_35mm.json",
        "70mm_higher_contact_torque_1": "phase0b_tuned_contact_probe_35mm.json",
        "70mm_high_authority_wrist_1": "phase0b_high_authority_probe_35mm.json",
    }
    for group, file_name in groups.items():
        data = _load(directory, file_name)
        if data is None:
            continue
        entries = []
        for item in data["results"]:
            contact = item.get("contact_summary", {}).get("left_bottle", {})
            entries.append({"proposal": item["reach_proposal_index"],
                            "aborted": item["aborted"],
                            "minimum_separation_m": item["minimum_separation_m"],
                            "maximum_contact_force_n": item["maximum_contact_force_n"],
                            "release_bottle_displacement_m": item["bottle_displacement_m"],
                            "release_contact_ratio": item.get("release_contact_ratio"),
                            "whole_attempt_contact_ratio": contact.get("contact_ratio", 0.0),
                            "release_passed": item.get("release_passed", False),
                            "reset_replay_passed": (item.get("reset_replay") or {}).get(
                                "passed", False)})
        result["synthesis_groups"][group] = {
            "attempts": len(entries),
            "support_kind": data.get("support_kind", "bounded_bottle_wrench"),
            "accepted_left_only": sum(int(x["release_passed"] and x["reset_replay_passed"])
                                      for x in entries),
            "cases": entries,
        }
    for n in (1, 2, 16):
        data = _load(directory, f"phase0b_reset_{n}.json")
        if data:
            result["reset_regression"][str(n)] = data
    contact = _load(directory, "phase0b_contact_regression.json")
    if contact:
        result["contact_regression"] = {
            "no_contact": contact.get("no_contact"),
            "cases": {key: {"env0_detected": bool(value.get("env0")),
                            "env1_clear": not bool(value.get("env1"))}
                      for key, value in contact["cases"].items()},
        }
    wrist = _load(directory, "phase0b_wrist_gravity_hold.json")
    if wrist:
        result["wrist_gravity_hold"] = wrist
    fixed = _load(directory, "phase0b_fixed_support_probe_35mm.json")
    if fixed:
        entry = fixed["results"][0]
        result["example_force_pose_trajectory"] = [
            {"t_s": x["t_s"], "phase": x["phase"],
             "bottle_position_w_m": x["bottle_pose_w"][:3],
             "left_wrist_position_w_m": x["left_pose_w"][:3],
             "left_wrist_quaternion_xyzw": x["left_pose_w"][3:7],
             "minimum_separation_m": x["minimum_separation_m"],
             "contact_peak_n": x["contact_peak_n"],
             "maximum_joint_applied_torque_nm": max(
                 abs(v) for v in x["isaaclab_applied_torque_buffer_nm"]),
             "temporary_support_joint_enabled": x["temporary_support_joint_enabled"],
             "contact_pairs": x["contact_pairs"][:3]}
            for x in entry["trace"] if x["phase"] in ("settle", "release")]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("docs/assets"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PHASE0B_SUMMARY " + json.dumps({
        "groups": list(result["synthesis_groups"]),
        "accepted_left_only": sum(x["accepted_left_only"] for x in
                                  result["synthesis_groups"].values()),
        "reset_env_counts": list(result["reset_regression"])}))


if __name__ == "__main__":
    main()
