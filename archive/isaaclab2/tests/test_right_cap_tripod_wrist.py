"""Geometry objective for a physically meaningful cap tripod proposal."""

import importlib.util
import math
from pathlib import Path

import pytest


def test_cyclic_contact_gap_identifies_missing_opposition(monkeypatch):
    pytest.importorskip("mujoco")
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts/environments"))
    path = root / "scripts/environments/optimize_right_cap_tripod_wrist.py"
    spec = importlib.util.spec_from_file_location("optimize_right_cap_tripod_wrist", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.max_cyclic_gap_deg((-96, 95, 130)) == pytest.approx(191)
    assert module.max_cyclic_gap_deg((-96, 20, 95)) == pytest.approx(169)
    q = module.world_delta_quaternion((0.0, 0.0, math.pi / 2), (1, 0, 0, 0))
    assert sum(x*x for x in q) == pytest.approx(1.0)
    assert module.contact_candidate_bodies("index") == (
        "r_index_finger_middle", "r_index_finger_distal")
    assert "r_index_finger_proximal" not in module.contact_candidate_bodies("index")
    proposal = {"index": 7, "wrist_position_w_m": [1, 2, 3]}
    candidate = {"index": 7, "wrist_position_w_m": [1.01, 2, 3],
                 "tripod_wrist_search": {"wrist_world_rotation_vector_rad": [0, .2, 0]},
                 "joints_rad": {name: .1 for name in module.INDEX_JOINTS}}
    x = module.resume_vector(proposal, candidate, False)
    assert len(x) == 10
    assert x[:3].tolist() == pytest.approx([.01, 0, 0])
    bounds = module.local_bounds(x, [(-1, 1)] * len(x))
    assert bounds[0] == pytest.approx((.004, .016))
    assert bounds[6] == pytest.approx((-.1, .3))
