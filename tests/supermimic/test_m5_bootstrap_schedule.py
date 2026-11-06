"""A full-length Actor rollout must not trust an untrained terminal Critic."""

from dataclasses import replace

import pytest
import torch
from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint
from dexgym.algorithms.supermimic.core.trainer import SuperMimicTrainer, TrainerConfig
from dexgym.tasks.supermimic_toy import make_toy_trainer


def test_bootstrap_warmup_preserves_legacy_default_and_ramps_full32():
    trainer = SuperMimicTrainer.__new__(SuperMimicTrainer)
    trainer.config = TrainerConfig()
    trainer.counts = {"actor_updates": 0}
    trainer.allowed_horizon = 1
    assert trainer._actor_bootstrap_weight() == 0.0
    trainer.allowed_horizon = 32
    assert trainer._actor_bootstrap_weight() == 1.0

    trainer.config = replace(trainer.config, actor_bootstrap_warmup_updates=50)
    assert trainer._actor_bootstrap_weight() == 0.0
    trainer.counts["actor_updates"] = 5
    assert trainer._actor_bootstrap_weight() == pytest.approx(0.1)
    trainer.counts["actor_updates"] = 50
    assert trainer._actor_bootstrap_weight() == 1.0
    with pytest.raises(ValueError, match="actor_bootstrap_warmup_updates"):
        replace(trainer.config, actor_bootstrap_warmup_updates=-1)


def test_bootstrap_schedule_is_strict_on_resume_and_old_default_is_compatible(tmp_path):
    manager = CoreCheckpoint(git_commit="test", reference_hashes={})
    source = make_toy_trainer("position", num_envs=8)
    source.config = replace(source.config, actor_bootstrap_warmup_updates=50)
    saved = manager.save(tmp_path / "scheduled.pt", source)
    incompatible = make_toy_trainer("position", num_envs=8)
    with pytest.raises(ValueError, match="Trainer configuration differs"):
        manager.load(saved, incompatible)
    compatible = make_toy_trainer("position", num_envs=8)
    compatible.config = replace(compatible.config, actor_bootstrap_warmup_updates=50)
    manager.load(saved, compatible)

    old = torch.load(saved, map_location="cpu", weights_only=False)
    old["trainer_config"].pop("actor_bootstrap_warmup_updates")
    torch.save(old, tmp_path / "old_default.pt")
    with pytest.raises(ValueError, match="Trainer configuration differs"):
        manager.load(tmp_path / "old_default.pt", compatible)
    manager.load(tmp_path / "old_default.pt", incompatible)
