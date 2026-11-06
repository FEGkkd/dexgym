"""A new M5 experiment may change capacity while keeping WM8 and BPTT32 fixed."""

import json
from pathlib import Path


def test_4096_full32_capacity_profile_has_explicit_bounded_schedule():
    path = Path(__file__).resolve().parents[2] / "configs/supermimic/m5_se3_full32_wm256_4096_v1.json"
    config = json.loads(path.read_text())
    assert config["scale_profile"] == "4096_env_full32_wm256_v1"
    assert config["num_envs"] == 4096
    assert config["iterations"] == 40
    assert config["replay_capacity_steps"] == 256
    assert config["network"]["world_hidden"] == [256, 256, 256]
    assert config["network"]["latent_dim"] == 32
    assert config["trainer"]["world_window"] == 8
    assert config["trainer"]["max_imagination_horizon"] == 32
    assert config["trainer"]["imagination_batch"] == 256
    assert config["trainer"]["critic_epochs"] == 5
    assert config["trainer"]["critic_minibatches"] == 4
    assert config["minimum_actor_horizon"] == 32


def test_4096_full32_bootstrap_profile_keeps_wm8_and_ramps_critic():
    path = Path(__file__).resolve().parents[2] / (
        "configs/supermimic/m5_se3_full32_wm256_4096_bootstrap_v2.json")
    config = json.loads(path.read_text())
    assert config["scale_profile"] == "4096_env_full32_wm256_bootstrap_v2"
    assert config["num_envs"] == 4096
    assert config["trainer"]["world_window"] == 8
    assert config["trainer"]["max_imagination_horizon"] == 32
    assert config["minimum_actor_horizon"] == 32
    assert config["trainer"]["actor_bootstrap_warmup_updates"] == 50
    assert config["trainer"]["critic_epochs"] == 5
    assert config["trainer"]["critic_minibatches"] == 4
