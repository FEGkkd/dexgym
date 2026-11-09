"""Public task loading must work independently of repository script directories."""

import importlib
import json
from pathlib import Path

import gymnasium as gym
import pytest
from dexgym.tasks import HAND21_SUPERMIMIC_ID, HAND21_TELEDEXTER_ID
from dexgym.tasks.direct.hand21_supermimic.supermimic_cfg import default_experiment, load_hand_configuration


def test_registered_hand_configuration_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg, experiment = load_hand_configuration()
    assert experiment["trainer"]["mode"] == "pure_model_based"
    assert experiment["trainer"]["world_window"] == 8
    assert experiment["trainer"]["max_imagination_horizon"] == 32
    assert cfg.goal_tolerance_m == 0.015
    assert cfg.paper_hand_dense and cfg.capture_model_transitions
    assert cfg.scene.robot.prim_path.endswith("/RightHand")
    assert "wuji_hand21" in str(cfg.scene.robot.spawn.usd_path)
    assert cfg.decimation == 16 and cfg.sim.dt == 1 / 480
    assert gym.spec(HAND21_SUPERMIMIC_ID).entry_point.endswith(":Hand21SuperMimicEnv")


def test_packaged_default_is_not_shared_mutable_state():
    first = default_experiment()
    second = default_experiment()
    first["network"]["world_hidden"][0] = 1
    first["trainer"]["max_imagination_horizon"] = 1
    assert second["network"]["world_hidden"] == [512, 512, 512]
    assert second["trainer"]["max_imagination_horizon"] == 32


def test_json_override_and_invalid_lane_split(tmp_path):
    config = default_experiment()
    config["num_envs"] = 16
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config))
    _, loaded = load_hand_configuration(override=path)
    assert loaded["num_envs"] == 16
    config["train_env_fraction"] = 1.0
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="lanes"):
        load_hand_configuration(override=path)


def test_supermimic_runner_rejects_ppo_task():
    with pytest.raises(ValueError, match="does not support"):
        load_hand_configuration(HAND21_TELEDEXTER_ID)


@pytest.mark.parametrize("module", ["train", "evaluate", "smoke", "preflight"])
def test_task_entrypoints_import_from_another_directory(module, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    entrypoint = importlib.import_module(f"dexgym.tasks.direct.hand21_supermimic.supermimic_{module}")
    assert callable(entrypoint.main)
    # Task code must not import implementation from the repository's CLI wrappers.
    assert "from scripts." not in Path(entrypoint.__file__).read_text()


def test_installed_package_does_not_claim_an_unrelated_parent_repository(tmp_path, monkeypatch):
    from dexgym.tasks.direct.hand21_supermimic import supermimic_runner

    installed = tmp_path / "site-packages/dexgym/tasks/direct/hand21_supermimic/supermimic_runner.py"
    monkeypatch.setattr(supermimic_runner, "__file__", str(installed))
    monkeypatch.setattr(
        supermimic_runner.subprocess,
        "check_output",
        lambda command, **kwargs: str(tmp_path) if command[-1] == "--show-toplevel" else "unrelated",
    )
    assert supermimic_runner.source_revision() == "unversioned-package"
