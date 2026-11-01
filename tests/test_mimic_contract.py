"""Geometric Reset-bank checks for the constrained screw assembly."""

import math

import pytest

from dexteroushands_lab.assets.bottle.config import BottleConfig
from dexteroushands_lab.assets.bottle.mimic_contract import mimic_reset_geometry


def test_thirty_degree_opening_has_quarter_millimetre_local_travel():
    cfg = BottleConfig()
    theta = math.pi / 6
    bottle = (0, 0, 0.5, 0, 0, 0, 1)
    cap = (0, 0, 0.5 + cfg.cap_initial_z + 0.00025,
           0, 0, math.sin(theta / 2), math.cos(theta / 2))
    result = mimic_reset_geometry(bottle, cap, theta, cfg)
    assert result["pitch_error_m"] == pytest.approx(0, abs=1e-9)
    assert result["lateral_m"] == pytest.approx(0, abs=1e-9)
    assert result["twist_error_rad"] == pytest.approx(0, abs=1e-9)


def test_tilt_uses_bottle_local_axis_and_rejects_axial_offset():
    cfg = BottleConfig()
    half = math.radians(15)
    bottle = (0, 0, 0.5, 0, math.sin(half), 0, math.cos(half))
    offset = cfg.cap_initial_z
    cap = (0.5 * offset, 0, 0.5 + math.cos(2 * half) * offset,
           *bottle[3:])
    good = mimic_reset_geometry(bottle, cap, 0, cfg)
    assert good["pitch_error_m"] == pytest.approx(0, abs=1e-9)
    assert good["lateral_m"] == pytest.approx(0, abs=1e-9)
    shifted = (cap[0] - 0.0005, cap[1], cap[2] - math.cos(2 * half) * 0.001,
               *cap[3:])
    bad = mimic_reset_geometry(bottle, shifted, 0, cfg)
    assert bad["pitch_error_m"] == pytest.approx(-0.001, abs=1e-9)


def test_nonfinite_pose_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        mimic_reset_geometry((0, 0, 0, 0, 0, 0, 1),
                             (math.nan, 0, 0, 0, 0, 0, 1), 0, BottleConfig())
