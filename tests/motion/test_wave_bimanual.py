"""Regression checks for the generated left and right arm reference."""

import json
from pathlib import Path

import numpy as np
from dexgym.motion.reference_library import MarvinReference

ROOT = Path("/home/ubuntu/dexgym_data/marvin_motion_library_v1")


def test_both_arms_retarget_human_and_loader_exposes_both_links():
    import pytest

    path = ROOT / "validated_motions/cmu_141_16_wave_bimanual.npz"
    if not path.exists():
        pytest.skip("Generate and validate the bimanual reference first")
    r = np.load(path, allow_pickle=False)
    mask = r["phase"] == "human"
    assert r["motion_id"].item() == "cmu_141_16_wave_bimanual"
    assert r["q_arm"].shape == (len(r["time"]), 14)
    assert np.array_equal(r["time"], np.arange(len(r["time"])) * 0.02)
    assert np.all(np.ptp(r["q_arm"][:, :7], axis=0) > 0.1)
    assert np.all(np.ptp(r["q_arm"][:, 7:14], axis=0) > 0.1)
    assert np.linalg.norm(np.ptp(r["mapped_target_pos"][:, 0], axis=0)) > 0.05
    assert np.percentile(r["tcp_pos_error_mm"][mask], 95, axis=0).max() < 10
    assert np.percentile(r["tcp_rot_error_deg"][mask], 95, axis=0).max() < 5
    assert r["human_left_hand_local_channels_deg"].shape[1] == 3
    assert r["human_left_hand_relative_forearm_raw_quat_xyzw"].shape == (len(r["time"]), 4)
    loader = MarvinReference(path)
    future = loader.future_links(float(r["time"][-1]) - 0.02, count=10)
    assert future["pos"].shape == (10, 2, 3, 3)
    np.testing.assert_array_equal(future["pos"][-1], r["link_pos"][-1])


def test_bimanual_newton_full_replay_has_safe_physical_states():
    import pytest

    report_path = ROOT / "bimanual/physics_replays/newton_16env/newton_pd_report.json"
    if not report_path.exists():
        pytest.skip("Run bimanual Newton replay first")
    report = json.loads(report_path.read_text())
    assert report["complete"] and report["num_envs"] == 16
    assert report["source_geometry_contact_physics_step_env_count"] == 0
    assert report["illegal_joint_state_count"] == 0
    assert report["torque_saturation_rate"] == 0
    assert report["per_side"]["left"]["tcp_pos_mm"]["p95"] < 25
    assert report["per_side"]["right"]["tcp_pos_mm"]["p95"] < 25
