"""The requested power grasp must use four fingers opposite the thumb."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import pytest

import dexteroushands_lab.scenes.antipodal_contact_analysis as analysis


def _contact(digit: str, angle_deg: float, axial_m: float = .025,
             force_n: float = 1.0) -> dict:
    angle = math.radians(angle_deg)
    return {
        "category": "left_bottle",
        "source_body": "Bottle",
        "partner_body": f"l_{digit}_middle",
        "point_w_m": (.035 * math.cos(angle), .035 * math.sin(angle),
                      .5 + axial_m),
        "normal_w": (math.cos(angle), math.sin(angle), 0.0),
        "normal_force_n": force_n,
    }


def _power_contacts() -> list[dict]:
    return [
        _contact("thumb", -100),
        _contact("index_finger", 40),
        _contact("middle_finger", 60),
        _contact("ring_finger", 80),
        _contact("pinky", 100),
    ]


def test_power_grasp_requires_four_finger_side_and_opposed_thumb():
    # A regression that counted any thumb plus one long finger would pass the
    # old bank and fail to enforce the user's cylindrical topology.
    assert hasattr(analysis, "analyze_cylindrical_power_grasp")
    pose = (0, 0, .5, 0, 0, 0, 1)
    valid = analysis.analyze_cylindrical_power_grasp(_power_contacts(), pose)
    assert valid["valid_topology"]
    assert valid["loaded_digits"] == ["thumb", "index", "middle", "ring", "pinky"]
    assert math.degrees(valid["thumb_to_four_finger_angle_rad"]) > 150

    without_pinky = analysis.analyze_cylindrical_power_grasp(
        _power_contacts()[:-1], pose)
    assert not without_pinky["valid_topology"]
    assert "pinky" not in without_pinky["loaded_digits"]

    same_side = [_contact("thumb", 55), *_power_contacts()[1:]]
    assert not analysis.analyze_cylindrical_power_grasp(
        same_side, pose)["valid_topology"]

    shoulder = [_contact("thumb", -100, .095), *_power_contacts()[1:]]
    assert not analysis.analyze_cylindrical_power_grasp(
        shoulder, pose)["valid_topology"]


def test_existing_pose_left_grasp_fails_requested_topology():
    assert hasattr(analysis, "analyze_cylindrical_power_grasp")
    path = Path(__file__).resolve().parents[1] / "docs/assets/phase0e_pose204.json.gz"
    record = json.loads(gzip.decompress(path.read_bytes()))["results"][0]
    release = next(row for row in record["trace"]
                   if row["phase"] == "release" and row["contact_pairs"])
    result = analysis.analyze_cylindrical_power_grasp(
        release["contact_pairs"], release["bottle_pose_w"])
    assert not result["valid_topology"]
    assert "pinky" not in result["loaded_digits"]


def test_power_grasp_series_requires_sustained_five_digit_opposition():
    assert hasattr(analysis, "CylindricalPowerGraspSeries")
    pose = (0, 0, .5, 0, 0, 0, 1)
    series = analysis.CylindricalPowerGraspSeries()
    for _ in range(8):
        series.add(_power_contacts(), pose)
    for _ in range(2):
        series.add(_power_contacts()[:-1], pose)
    result = series.summary()
    assert result["step_count"] == 10
    assert result["valid_topology_ratio"] == .8
    assert result["digit_contact_ratio"]["pinky"] == .8
    assert result["passed"]
    series.add(_power_contacts()[:-1], pose)
    assert not series.summary()["passed"]


def test_image_grasp_load_series_allows_one_unloaded_digit_with_geometry_guard():
    pose = (0, 0, .5, 0, 0, 0, 1)
    series = analysis.ImageCylindricalContactSeries()
    # Four long fingers can occupy the same cylinder side while one of them
    # temporarily carries no normal force. The thumb and three remaining
    # digits still need force-bearing contact in the sidewall band.
    for step in range(10):
        contacts = (_power_contacts() if step == 0 else
                    [item for item in _power_contacts()
                     if "pinky" not in item["partner_body"]])
        observation = series.add(contacts, pose)
        assert observation["valid_topology"]
    assert series.summary()["passed"]
    alternating = analysis.ImageCylindricalContactSeries()
    for step in range(10):
        missing = "pinky" if step % 2 else "middle_finger"
        alternating.add([item for item in _power_contacts()
                         if missing not in item["partner_body"]], pose)
    # Load sharing can alternate between the middle and little finger;
    # three long digits must still oppose the thumb in each valid substep.
    assert alternating.summary()["passed"]
    never_pinky = analysis.ImageCylindricalContactSeries()
    for _ in range(10):
        never_pinky.add(_power_contacts()[:-1], pose)
    assert not never_pinky.summary()["passed"]
    assert not analysis.CylindricalPowerGraspSeries().add(
        _power_contacts()[:-1], pose)["valid_topology"]
    assert not analysis.ImageCylindricalContactSeries().add(
        _power_contacts()[:-2], pose)["valid_topology"]
    same_side = [_contact("thumb", 55), *_power_contacts()[1:-1]]
    assert not analysis.ImageCylindricalContactSeries().add(
        same_side, pose)["valid_topology"]


def test_old_physx_hold_cannot_be_curated_as_power_grasp():
    from scripts.environments.curate_left_grasp_bank import validation_metrics
    path = Path(__file__).resolve().parents[1] / "docs/assets/phase0e_pose204.json.gz"
    output = json.loads(gzip.decompress(path.read_bytes()))
    output["required_grasp_topology"] = "cylindrical_power_grasp_v1"
    with pytest.raises(ValueError, match="power grasp"):
        validation_metrics(output["results"][0], output, "synthetic-test-digest")
