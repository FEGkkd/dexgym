"""Checkpoint selection must reject changed cases and large regressions."""

import pytest

from scripts.s0.compare_eval import compare


def sample_result(success: float, contacts: int = 0) -> dict:
    return {
        "checkpoint": "model.pt",
        "case_set_sha256": "fixed",
        "success_definition": "strict",
        "num_envs": 2,
        "terminated": 0,
        "max_actual_arm_velocity_rad_s": 1.0,
        "newton_active_contacts_sampled_every_10_steps": {"max_non_floor_pedestal_contacts": contacts},
        "categories": {"local": {"success_rate": success}},
        "left_right": {
            side: {
                "success_rate": success,
                "position_error_mm": {"p95": 1.0},
                "orientation_error_deg": {"p95": 1.0},
            }
            for side in ("left", "right")
        },
        "success_rate": success,
    }


def test_requires_same_cases_and_rejects_contact_regression():
    baseline = sample_result(0.5)
    candidate = sample_result(0.6, contacts=1)
    assert not compare(baseline, candidate)["accept_candidate"]
    candidate["case_set_sha256"] = "different"
    with pytest.raises(ValueError, match="case sets differ"):
        compare(baseline, candidate)


def test_accepts_strictly_better_success_without_measured_safety_regression():
    assert compare(sample_result(0.5), sample_result(0.6))["accept_candidate"]


def test_rejects_large_arm_p95_regression_despite_higher_success():
    baseline = sample_result(0.5)
    candidate = sample_result(0.6)
    candidate["left_right"]["left"]["position_error_mm"]["p95"] = 1.3
    assert not compare(baseline, candidate)["accept_candidate"]
