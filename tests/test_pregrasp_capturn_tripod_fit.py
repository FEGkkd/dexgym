"""Offline official-MJCF preshape fitting keeps the right-hand tripod physical."""

from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

from dexteroushands_lab.assets.bottle import BottleConfig
from dexteroushands_lab.scenes.grasp_init import grasp_sample_from_dict, _reframe_sample
from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.reset_bank import make_hybrid_candidate
from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.mjcf_tripod_fit import fit_right_tripod
from scripts.environments.generate_pregrasp_reset_candidates import combined_model, MjcfGeometryScreen


def test_fitted_tripod_reduces_actual_site_gaps_and_preserves_opposition():
    pytest.importorskip("mujoco")
    root = Path(os.environ.get("WUJI_DESCRIPTION_ROOT", "/tmp/wuji-description-v2026.9.11"))
    if not root.exists():
        pytest.skip("Pinned official Beta 1 checkout unavailable")
    repo = Path(__file__).resolve().parents[1]
    source = grasp_sample_from_dict(json.loads((repo /
        "docs/assets/phase1_mimic_pgs_grasp_init_bank.json").read_text())[0])
    stable = _reframe_sample(source, (0., 0., .5, 0., 0., 0., 1.))
    stable = replace(stable,
        bottle=replace(stable.bottle, root_pose=(0., 0., .5, 0., 0., 0., 1.)),
        validated=True, validation=source.validation)
    natural = grasp_sample_from_dict(json.loads((repo /
        "docs/assets/phase0_pregrasp_reset_bank.json").read_text())[3])
    sample = make_hybrid_candidate(stable, natural, seed=48123,
                                    left_open_blend=.03, left_outward_m=.0002,
                                    right_inward_m=0)
    bottle = BottleConfig()
    screen = MjcfGeometryScreen(combined_model(root, bottle), bottle)
    before = screen.check(sample)
    fitted = fit_right_tripod(screen, sample,
                              site_radial_gap_targets_m=(.0003, .0030, .0035))
    after = screen.check(fitted)
    digits = ("thumb", "index_finger", "middle_finger")
    assert max(after["right_fingertip_sites"][d]["radial_gap_to_cap_sidewall_m"]
               for d in digits) < max(before["right_fingertip_sites"][d]["radial_gap_to_cap_sidewall_m"]
                                    for d in digits)
    assert after["right_thumb_tripod_site_opposition_rad"] > 2.18
    # MJCF proxy may have light contact; native USD/PGS still decides admission.
    assert min(after["per_digit"]["r"][d]["sidewall_gap_m"] for d in digits) >= -.0005
