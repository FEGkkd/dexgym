"""The photo-style bank needs tied geometry, release and Reset evidence."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.environments.curate_left_image_grasp import _read, validation_metrics
from dexteroushands_lab.scenes.grasp_init import GraspInitBank


ROOT = Path(__file__).resolve().parents[1] / "docs/assets"


def _evidence():
    report, rdigest = _read(ROOT / "phase0g_left_image_power_v2_release.json.gz")
    replay, pdigest = _read(ROOT / "phase0g_left_image_power_v2_replay.json.gz")
    geometry, gdigest = _read(ROOT / "phase0g_left_image_power_v2_geometry.json")
    return report, report["results"][0], replay, geometry, rdigest, pdigest, gdigest


def test_image_grasp_bank_contains_physx_release_and_independent_reset():
    report, result, replay, geometry, rdigest, pdigest, gdigest = _evidence()
    sample, metrics = validation_metrics(report, result, replay, geometry,
                                         rdigest, pdigest, gdigest)
    assert sample.validated
    assert sample.seed == 9781
    assert metrics["image_grasp_release_ratio"] >= .8
    assert metrics["image_grasp_replay_ratio"] >= .8
    assert metrics["visual_photo_match_confirmed"] is False
    assert metrics["strict_five_digit_power_grasp_passed"] is False
    bank = GraspInitBank(ROOT / "phase0g_left_image_power_v2_grasp_init_bank.json")
    assert bank.load()[0] == sample


def test_image_grasp_curator_rejects_state_swap_and_bad_geometry():
    report, result, replay, geometry, rdigest, pdigest, gdigest = _evidence()
    swapped = deepcopy(replay)
    swapped["captured_state"]["left"]["joint_position"][0] += .01
    with pytest.raises(ValueError, match="exact captured state"):
        validation_metrics(report, result, swapped, geometry, rdigest, pdigest, gdigest)
    wrong = {**geometry, "candidate_state_sha256": "0" * 64}
    with pytest.raises(ValueError, match="collision geometry"):
        validation_metrics(report, result, replay, wrong, rdigest, pdigest, gdigest)


def test_image_grasp_curator_rejects_support_and_missing_contact():
    report, result, replay, geometry, rdigest, pdigest, gdigest = _evidence()
    supported = deepcopy(result)
    release_row = next(row for row in supported["trace"] if row["phase"] == "release")
    release_row["temporary_support_joint_enabled"] = True
    with pytest.raises(ValueError, match="external support"):
        validation_metrics(report, supported, replay, geometry,
                           rdigest, pdigest, gdigest)
    unopposed = deepcopy(result)
    unopposed["release_image_grasp"]["valid_topology_ratio"] = .79
    with pytest.raises(ValueError, match="sustained image-style contact"):
        validation_metrics(report, unopposed, replay, geometry,
                           rdigest, pdigest, gdigest)
