"""The offline tripod proposal must reach the cap without proxy collisions."""

import importlib.util
import json
import os
from pathlib import Path

import pytest


def test_index_refinement_reaches_sidewall_with_clearance(monkeypatch):
    pytest.importorskip("mujoco")
    if not os.environ.get("WUJI_DESCRIPTION_ROOT"):
        pytest.skip("Official Wuji description checkout is unavailable")
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts/environments"))
    script = root / "scripts/environments/optimize_right_cap_tripod.py"
    spec = importlib.util.spec_from_file_location("optimize_right_cap_tripod", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = json.loads((root / "docs/assets/phase0e_right_cap_sidewall_preshape45066.json").read_text())
    candidate = module.refine_index(source["best"][0], seed=0, maxiter=35)
    result = candidate["index_refinement"]
    assert "digit_cap_distances_m" not in candidate
    assert "pregrasp_joints_rad" not in candidate
    assert result["index_sidewall_distance_m"] <= .002
    assert result["index_nearest_cap_point_w_m"][2] <= .6535
    assert result["index_nearest_cap_point_w_m"][2] >= .6375
    assert result["minimum_index_bottle_clearance_m"] >= .003
    assert result["minimum_index_other_digit_clearance_m"] >= .002
    assert result["backend"] == "official_beta1_MJCF_proxy_geometry_only"
