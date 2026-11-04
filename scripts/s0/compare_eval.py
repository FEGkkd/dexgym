"""Compare checkpoints on the same immutable S0 evaluation cases."""

import argparse
import json
from pathlib import Path


def compare(baseline: dict, candidate: dict) -> dict:
    if baseline["case_set_sha256"] != candidate["case_set_sha256"]:
        raise ValueError("Evaluation case sets differ")
    if baseline["success_definition"] != candidate["success_definition"]:
        raise ValueError("Success definitions differ")
    if baseline["num_envs"] != candidate["num_envs"]:
        raise ValueError("Evaluation case counts differ")
    base_categories = baseline["categories"]
    candidate_categories = candidate["categories"]
    if set(base_categories) != set(candidate_categories):
        raise ValueError("Evaluation categories differ")
    regressions = {
        name: candidate_categories[name]["success_rate"] - values["success_rate"]
        for name, values in base_categories.items()
    }
    side_success_deltas = {
        side: candidate["left_right"][side]["success_rate"] - baseline["left_right"][side]["success_rate"]
        for side in ("left", "right")
    }
    side_p95_ratios = {
        side: {
            metric: candidate["left_right"][side][metric]["p95"]
            / max(baseline["left_right"][side][metric]["p95"], 1e-9)
            for metric in ("position_error_mm", "orientation_error_deg")
        }
        for side in ("left", "right")
    }
    contact_key = "newton_active_contacts_sampled_every_10_steps"
    base_contact = baseline.get(contact_key, {}).get("max_non_floor_pedestal_contacts")
    candidate_contact = candidate.get(contact_key, {}).get("max_non_floor_pedestal_contacts")
    checks = {
        "same_cases": True,
        "no_additional_terminations": candidate["terminated"] <= baseline["terminated"],
        "no_larger_sampled_non_floor_contact_peak": (
            base_contact is not None and candidate_contact is not None and candidate_contact <= base_contact
        ),
        "overall_success_improved": candidate["success_rate"] > baseline["success_rate"],
        "no_category_lost_more_than_10_percentage_points": all(delta >= -0.10 for delta in regressions.values()),
        "no_arm_lost_more_than_5_percentage_points": all(delta >= -0.05 for delta in side_success_deltas.values()),
        "no_arm_p95_error_increased_over_20_percent": all(
            ratio <= 1.2 for ratios in side_p95_ratios.values() for ratio in ratios.values()
        ),
        "actual_speed_not_increased_over_5_percent": (
            candidate["max_actual_arm_velocity_rad_s"] <= baseline["max_actual_arm_velocity_rad_s"] * 1.05
        ),
    }
    return {
        "baseline_checkpoint": baseline["checkpoint"],
        "candidate_checkpoint": candidate["checkpoint"],
        "case_set_sha256": baseline["case_set_sha256"],
        "success_delta": candidate["success_rate"] - baseline["success_rate"],
        "category_success_deltas": regressions,
        "side_success_deltas": side_success_deltas,
        "side_p95_error_ratios": side_p95_ratios,
        "checks": checks,
        "accept_candidate": all(checks.values()),
        "safety_certified": False,
        "safety_note": "Sampled aggregate contact counts cannot establish collision-free paths per case.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
