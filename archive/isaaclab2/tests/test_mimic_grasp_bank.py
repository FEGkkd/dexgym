"""Guard provenance and contact checks for the single-family Mimic PGS bank."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from dexteroushands_lab.scenes.grasp_init import grasp_sample_from_dict


BANK = Path(__file__).resolve().parents[1] / "docs/assets/phase1_mimic_pgs_grasp_init_bank.json"


def test_mimic_bank_is_revalidated_by_contract():
    rows = json.loads(BANK.read_text())
    assert len(rows) == 1  # one grasp family; do not imply diverse starts
    sample = grasp_sample_from_dict(rows[0])
    assert sample.validated
    assert sample.metadata["solver_type"] == "pgs"
    assert sample.with_validation(sample.validation).validated


@pytest.mark.parametrize("change", [
    {"right_sidewall_tripod_ratio": 0.0},
    {"opposing_contact_coverage_ratio": 0.0},
    {"minimum_separation_m": -0.003},
    {"cross_contact_ratios": {"left_cap": 0.0, "right_bottle": 0.0, "hand_hand": 0.5}},
    {"max_thread_pitch_error_m": 0.0001},
])
def test_mimic_bank_rejects_invalid_physics_evidence(change):
    sample = grasp_sample_from_dict(json.loads(BANK.read_text())[0])
    metrics = {**sample.validation, **change}
    with pytest.raises(ValueError):
        replace(sample, validated=False, validation=None).with_validation(metrics)
