"""Certify a captured photo-style grasp from separate PhysX release and Reset runs.

The strict five-digit power-grasp result is retained separately. This image
topology requires official captured-state geometry, thumb opposition and at
least three force-bearing long digits in >=80% of *each* unsupported hold.
Visual identity with a photograph is not inferred from the physics metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path

from dexteroushands_lab.scenes.grasp_init import GraspInitBank, grasp_sample_from_dict


def _read(path: Path):
    raw = path.read_bytes()
    return (json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw),
            sha256(raw).hexdigest())


def _state_digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"))
                  .encode("utf-8")).hexdigest()


def validation_metrics(output, result, replay_doc, geometry,
                       release_digest, replay_digest, geometry_digest):
    sample = result.get("candidate")
    replay = replay_doc.get("replay") or {}
    if not isinstance(sample, dict) or replay_doc.get("captured_state") != sample:
        raise ValueError("Independent Reset did not replay the exact captured state")
    if (replay_doc.get("status") !=
            "independent_PhysX_Reset_replay_of_unvalidated_candidate" or
            replay_doc.get("source_proposal_index") != result.get("reach_proposal_index")):
        raise ValueError("Independent replay provenance is missing or mismatched")
    if (geometry.get("status") != "official_MJCF_captured_state_geometry_only"
            or geometry.get("candidate_state_sha256") != _state_digest(sample)
            or geometry.get("geometry_qualified") is not True):
        raise ValueError("Official captured-state collision geometry did not qualify")
    metadata = sample["metadata"]
    for outer, inner in (("asset_commit", "asset_source_commit"),
                         ("left_usd_sha256", "left_usd_sha256"),
                         ("right_usd_sha256", "right_usd_sha256")):
        if output.get(outer) != metadata.get(inner):
            raise ValueError(f"Official asset provenance mismatch: {outer}")
    if (output.get("wrist_control_mode") != "fixed_wrist" or
            metadata.get("wrist_control_mode") != "fixed_wrist" or
            not math.isclose(output.get("bottle_radius_m", 0.), .035, abs_tol=1e-12)):
        raise ValueError("Image-grasp evidence must use the fixed wrist and 70 mm bottle")
    synthesis = metadata["synthesis"]
    limits = synthesis["limits"]
    duration = limits["release_s"]
    steps = math.ceil(duration / output["dt_s"])
    release_image = result.get("release_image_grasp") or {}
    replay_image = replay.get("image_grasp") or {}
    for label, item in (("release", release_image), ("replay", replay_image)):
        if (item.get("passed") is not True or item.get("step_count") != steps
                or item.get("minimum_loaded_long_digits") != 3
                or item.get("valid_topology_ratio", 0.) < .8):
            raise ValueError(f"{label} lacks sustained image-style contact")
    if (duration < 1.0 or result.get("aborted") is not None or
            not synthesis.get("support_removed_before_acceptance") or
            not any(row.get("phase") == "release" for row in result["trace"]) or
            any(row.get("temporary_support_joint_enabled") or
                row.get("support_wrench") is not None
                for row in result["trace"] if row.get("phase") == "release") or
            any(row.get("temporary_support_joint_enabled") or
                row.get("support_wrench") is not None
                for row in replay.get("trace", ()))):
        raise ValueError("The one-second holds were not free of external support")
    times_a = result["release_finger_contact_time_s"]
    times_b = replay["left_finger_contact_time_s"]
    times = {digit: min(times_a.get(digit, 0.), times_b.get(digit, 0.))
             for digit in set(times_a) | set(times_b)}
    other_times = sorted((value for digit, value in times.items()
                          if digit != "thumb"), reverse=True)
    digit_ratios = {name: min(release_image["digit_contact_ratio"][name],
                              replay_image["digit_contact_ratio"][name])
                    for name in ("thumb", "index", "middle", "ring", "pinky")}
    ground = max(result["contact_summary"].get("bottle_ground", {}).get(
        "contact_ratio", 0.), replay["contact_summary"].get(
        "bottle_ground", {}).get("contact_ratio", 0.))
    metrics = {
        "passed": True, "scope": "left_only", "fixture": "gravity_on_no_support",
        "contact_topology": "image_cylindrical_v2",
        "wrist_control_mode": "fixed_wrist", "duration_s": duration,
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
        "left_thumb_contact_time_s": times.get("thumb", 0.),
        "left_other_contact_time_s": other_times[0] if other_times else 0.,
        "left_additional_digit_contact_time_s": (other_times[1]
                                                  if len(other_times) > 1 else 0.),
        "ground_contact_ratio": ground,
        "max_contact_peak_n": max(result["maximum_contact_force_n"],
                                  replay["left_contact_peak_n"]),
        "max_peak_force_guard_n": limits["max_contact_force_n"],
        "initial_min_separation_m": min(result["release_initial_min_separation_m"],
                                         replay["initial_min_separation_m"]),
        "minimum_separation_m": min(result["minimum_separation_m"],
                                    replay["minimum_separation_m"]),
        "max_initial_penetration_guard_m": limits["max_penetration_m"],
        "frictional_opposition_ratio": min(
            result["release_frictional_opposition_ratio"],
            replay["frictional_opposition_ratio"]),
        "image_grasp_release_ratio": release_image["valid_topology_ratio"],
        "image_grasp_replay_ratio": replay_image["valid_topology_ratio"],
        "image_grasp_digit_contact_ratio": digit_ratios,
        "official_geometry_qualified": True,
        "official_geometry_four_finger_half_arc_rad": geometry["four_finger_half_arc_rad"],
        "official_geometry_thumb_opposition_rad": geometry["thumb_opposition_rad"],
        "support_released": True, "fresh_process_replay_passed": True,
        "reset_replay_passed": True,
        "release_result_sha256": release_digest,
        "replay_result_sha256": replay_digest,
        "geometry_result_sha256": geometry_digest,
        "visual_photo_match_confirmed": False,
        "strict_five_digit_power_grasp_passed": False,
    }
    # The sample contract independently checks the hold, penetration, contact
    # and image-topology thresholds; the curator also checks the raw evidence.
    candidate = grasp_sample_from_dict(sample)
    accepted = replace(candidate, metadata={**candidate.metadata,
        "accepted_grasp_topology": "image_cylindrical_v2"})
    accepted = accepted.with_validation(metrics)
    return accepted, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path)
    args = parser.parse_args()
    output, release_digest = _read(args.release)
    replay, replay_digest = _read(args.replay)
    geometry, geometry_digest = _read(args.geometry)
    results = output.get("results", ())
    if len(results) != 1:
        raise ValueError("Provide a release report with exactly one candidate")
    sample, metrics = validation_metrics(output, results[0], replay, geometry,
                                         release_digest, replay_digest, geometry_digest)
    if args.bank:
        bank = GraspInitBank(args.bank)
        if any(item.validation.get("release_result_sha256") == release_digest
               for item in bank.load()):
            raise ValueError("This release report is already in the bank")
        bank.add(sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "status": "PhysX_qualified_image_cylindrical_v2",
        "seed": sample.seed, "bank": str(args.bank) if args.bank else None,
        "validation": metrics,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"seed": sample.seed, "image_style_physics_passed": True,
                      "visual_photo_match_confirmed": False,
                      "strict_five_digit_power_grasp_passed": False,
                      "bank": str(args.bank) if args.bank else None}))


if __name__ == "__main__":
    main()
