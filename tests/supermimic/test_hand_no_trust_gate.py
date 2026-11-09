"""Hand training and resume must update Actor/Critic without model-error gating."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint
from dexgym.algorithms.supermimic.core.trainer import SuperMimicTrainer
from dexgym.tasks.direct.hand21_supermimic.supermimic_cfg import default_experiment, load_hand_configuration
from dexgym.tasks.supermimic_toy import make_toy_trainer


def every_iteration_trainer():
    previous = make_toy_trainer("position", num_envs=8, world_hidden=16, world_batch=8, imagination_batch=4)
    modules = SimpleNamespace(
        world_model=previous.world_model,
        actor=previous.actor,
        critic=previous.critic,
        termination=previous.termination_head,
    )
    return SuperMimicTrainer(
        collector=previous.collector,
        replay=previous.replay,
        robot=previous.robot,
        task=previous.task,
        modules=modules,
        optimizers=previous.optimizers,
        trust_evaluator=None,
        config=replace(previous.config, actor_update_schedule="every_iteration", world_warmup_updates=0),
        train_env_range=(0, 8),
        validation_env_range=None,
    )


def test_actor_and_critic_update_immediately_without_model_validation(monkeypatch):
    torch.manual_seed(17)
    trainer = every_iteration_trainer()
    trainer.allowed_horizon = 0
    trainer.trust_report = {"32": {"physical_delta_rmse": float("inf")}}

    def forbidden_validation():
        raise AssertionError("Model validation must not control every-iteration updates")

    monkeypatch.setattr(trainer, "validate_world_model", forbidden_validation)
    result = trainer.run_iteration()
    assert result["actor"][0]["updated"]
    assert result["actor"][0]["horizon"] == 32
    assert trainer.counts["actor_updates"] == 1
    assert trainer.counts["critic_updates"] > 0
    assert "trust_failures" not in trainer.counts
    assert int(trainer.world_model.input_norm.count) == 8


def test_new_checkpoint_resume_keeps_fixed_horizon(tmp_path):
    trainer = every_iteration_trainer()
    manager = CoreCheckpoint(git_commit="test", reference_hashes={})
    path = manager.save(tmp_path / "new.pt", trainer)
    resumed = every_iteration_trainer()
    report = manager.load(path, resumed)
    assert resumed.allowed_horizon == 32
    assert report["trainer_config_migration"] == {}
    assert resumed.run_iteration()["actor"][0]["updated"]


def test_legacy_gate_checkpoint_migration_is_explicit_and_preserves_other_checks(tmp_path):
    old = make_toy_trainer("position", num_envs=8, world_hidden=16, world_batch=8, imagination_batch=4)
    manager = CoreCheckpoint(git_commit="test", reference_hashes={})
    path = manager.save(tmp_path / "legacy.pt", old)
    record = torch.load(path, weights_only=False)
    record["trainer_config"].pop("actor_update_schedule")
    torch.save(record, path)
    resumed = every_iteration_trainer()
    with pytest.raises(ValueError, match="Trainer configuration"):
        manager.load(path, resumed)
    report = manager.load(path, resumed, allow_actor_schedule_migration=True)
    assert report["trainer_config_migration"]["actor_update_schedule"] == {
        "previous": "model_trust",
        "current": "every_iteration",
    }
    assert resumed.allowed_horizon == 32
    assert "trust_failures" not in resumed.counts
    assert resumed.run_iteration()["actor"][0]["updated"]
    record["trainer_config"]["gamma"] = 0.9
    torch.save(record, path)
    with pytest.raises(ValueError, match="Trainer configuration"):
        manager.load(path, every_iteration_trainer(), allow_actor_schedule_migration=True)


def test_hand_configuration_migrates_legacy_gate_and_rejects_new_gate_keys(tmp_path):
    import json

    config = default_experiment()
    assert config["version"] == "hand21_s0_v2"
    assert config["trainer"]["actor_update_schedule"] == "every_iteration"
    assert config["trainer"]["world_warmup_updates"] == 0
    config["version"] = "hand21_s0_v1"
    config["trust"] = {"max_rmse": [1, 2, 3]}
    config["tip_mean_error_limits_m"] = [0.0015, 0.004, 0.0075]
    config["train_env_fraction"] = 0.75
    config["trainer"]["world_warmup_updates"] = 200
    config["trainer"]["trust_interval"] = 25
    config["trainer"]["validation_batch"] = 128
    path = tmp_path / "old.json"
    path.write_text(json.dumps(config))
    with pytest.warns(UserWarning, match="Migrated Hand S0 v1"):
        _, migrated = load_hand_configuration(override=path)
    assert migrated == default_experiment()
    migrated["trust"] = {}
    path.write_text(json.dumps(migrated))
    with pytest.raises(ValueError, match="no longer accepts"):
        load_hand_configuration(override=path)
