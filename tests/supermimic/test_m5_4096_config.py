"""The large-environment pilot must preserve model updates per real sample."""

import json
from pathlib import Path


def test_4096_profile_bounds_replay_and_model_update_ratio():
    profile = json.loads(Path("configs/supermimic/m5_se3_near_4096_pilot.json").read_text())
    trainer = profile["trainer"]
    assert profile["num_envs"] == 4096
    assert profile["curriculum_level"] == "near"
    assert profile["replay_capacity_steps"] >= 32
    assert profile["replay_capacity_steps"] <= 256
    assert trainer["world_window"] == 8
    assert trainer["max_imagination_horizon"] == 32
    assert trainer["mode"] == "pure_model_based"
    assert trainer["reference_pd_bootstrap_steps"] == 0
    # Count WM training targets per collected physical transition. A larger
    # environment batch must not silently reduce optimization data coverage.
    targets_per_transition = (trainer["world_updates_per_iteration"]
                              * trainer["world_batch"] * trainer["world_window"]
                              / (profile["num_envs"]
                                 * trainer["physics_steps_per_iteration"]))
    assert targets_per_transition >= 16


def test_hold_bootstrap_changes_only_training_goal_sampling():
    root = Path("configs/supermimic")
    standard = json.loads((root / "m5_se3_near_4096_pilot.json").read_text())
    hold = json.loads((root / "m5_se3_near_hold_mix_4096_pilot.json").read_text())
    assert hold["num_envs"] == 4096
    assert hold["curriculum_level"] == "near_hold_mix"
    assert hold["replay_capacity_steps"] == 128
    assert hold["scale_profile"] == "4096_env_hold_mix_v1"
    for field in ("network", "trainer", "trust", "geometry_trust", "exploration"):
        assert hold[field] == standard[field]
