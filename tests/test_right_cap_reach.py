"""Regressions for right-cap proposal geometry and contact accounting."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from dexteroushands_lab.scenes.bimanual_grasp_synthesis import (
    _hand_digit_times, _right_radial_opposition, _right_tripod_held)


def test_digit_contact_time_keeps_longest_body_per_finger():
    summary = {"bodies": {
        "Cap|r_thumb_middle": {"samples": 200},
        "Cap|r_thumb_distal": {"samples": 350},
        "Cap|r_middle_finger_distal": {"samples": 420},
        "Cap|l_index_finger_distal": {"samples": 500},
    }}
    assert _hand_digit_times(summary, "r_", .002) == {
        "thumb": pytest.approx(.7), "middle": pytest.approx(.84)}


def test_right_tripod_requires_each_loaded_digit():
    assert not _right_tripod_held({"thumb": 1.0, "middle": 1.0}, 1.0)
    assert _right_tripod_held({"thumb": 1.0, "index": .9, "middle": .95}, 1.0)


def test_right_reach_quaternion_composition_stays_unit_length():
    pytest.importorskip("mujoco")
    script = Path(__file__).resolve().parents[1] / "scripts/environments/analyze_right_cap_reach.py"
    spec = importlib.util.spec_from_file_location("analyze_right_cap_reach", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    qx = (math.sqrt(.5), math.sqrt(.5), 0.0, 0.0)
    qy = (math.sqrt(.5), 0.0, math.sqrt(.5), 0.0)
    result = module._mul(qx, qy)
    assert math.sqrt(sum(x*x for x in result)) == pytest.approx(1.0)
    assert result == pytest.approx((.5, .5, .5, .5))


def test_cap_rim_contact_requires_opposed_radial_force():
    def point(body, position, normal):
        return {"category": "right_cap", "source_body": "Cap",
                "partner_body": body, "point_w_m": position,
                "normal_w": normal, "normal_force_n": 1.0}
    cap = (0, 0, 0, 0, 0, 0, 1)
    thumb = point("r_thumb_distal", (0, -.017, -.006), (0, -1, 0))
    radial_rim = point("r_middle_finger_middle", (0, .017, .0125),
                       (0, .86, .51))
    vertical_top = point("r_middle_finger_middle", (0, .017, .0125),
                         (0, 0, 1))
    assert _right_radial_opposition([thumb, radial_rim], cap, .025)[0]
    assert not _right_radial_opposition([thumb, vertical_top], cap, .025)[0]
