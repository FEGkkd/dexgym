from pathlib import Path

import pytest
import torch

from dexgym.algorithms.supermimic.checkpoint import CheckpointManager
from dexgym.algorithms.supermimic.replay_buffer import MotionReplay


def test_checkpoint_restores_network_optimizer_and_rng_and_rejects_contract_change(tmp_path: Path):
    actor = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    metadata = {"action_schema": "robot_fixed_v2", "state_schema": "arm_42_v1",
                "observation_schema": "proprio_10x56", "goal_schema": "links_10x90",
                "reference_hashes": {"task": "hashA"}, "robot_asset_hashes": {"mjcf": "hashB"}}
    manager = CheckpointManager(metadata)
    path = tmp_path / "checkpoint.pt"
    original = actor.weight.detach().clone()
    manager.save(path, models={"actor": actor}, optimizers={"actor": optimizer},
                 training={"iteration": 3, "real_transitions": 8, "trust": {"horizon": 1}},
                 config={"seed": 42}, seed=42)
    actor.weight.data.add_(10)
    record = manager.load(path, models={"actor": actor}, optimizers={"actor": optimizer})
    assert torch.equal(actor.weight, original)
    assert record["replay_saved"] is False and record["resume_requires_recollection"] is True
    assert record["training"]["iteration"] == 3
    with pytest.raises(ValueError, match="action_schema"):
        CheckpointManager({**metadata, "action_schema": "legacy_wave_v1"}).load(
            path, models={"actor": actor}, optimizers={"actor": optimizer})


def test_checkpoint_restores_ring_replay_without_joining_new_physics_episode(tmp_path: Path):
    metadata = {"action_schema": "robot_fixed_v3", "state_schema": "arm_42_v1",
                "observation_schema": "proprio_10x56", "goal_schema": "links_10x90",
                "reference_hashes": {"task": "hashA"}, "robot_asset_hashes": {"mjcf": "hashB"}}
    manager = CheckpointManager(metadata)
    replay = MotionReplay(3, 2, "cpu")
    for frame in range(5):
        state = torch.full((2, 42), float(frame))
        action = torch.zeros(2, 14)
        replay.append(state, action, state + 1, torch.zeros(2, 10, 56),
                      torch.full((2,), frame), torch.zeros(2, dtype=torch.bool),
                      source_id=torch.full((2,), 2, dtype=torch.int8))
    actor = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(actor.parameters())
    path = tmp_path / "replay.pt"
    manager.save(path, models={"actor": actor}, optimizers={"actor": optimizer},
                 training={"iteration": 5}, config={}, seed=5, replay=replay)
    restored = MotionReplay(3, 2, "cpu")
    record = manager.load(path, models={"actor": actor}, optimizers={"actor": optimizer},
                          replay=restored)
    assert record["replay_saved"] and not record["resume_requires_recollection"]
    assert restored.steps == 5
    torch.testing.assert_close(restored.timestamp, replay.timestamp)
    torch.testing.assert_close(restored.state, replay.state)
    assert torch.equal(restored.episode_counter, replay.episode_counter + 1)
    restored.append(torch.zeros(2, 42), torch.zeros(2, 14), torch.ones(2, 42),
                    torch.zeros(2, 10, 56), torch.zeros(2, dtype=torch.long),
                    torch.zeros(2, dtype=torch.bool))
    assert torch.equal(restored.episode[5 % 3], replay.episode_counter + 1)
    window = restored.sample_windows(16, 3, source_id=2)
    assert bool((window["mask"][:, 0] == 1).all())


def test_checkpoint_restores_learning_rate_schedule(tmp_path: Path):
    metadata = {"action_schema": "robot_fixed_v3", "state_schema": "arm_42_v1",
                "observation_schema": "proprio_10x56", "goal_schema": "links_10x90",
                "reference_hashes": {"task": "hashA"}, "robot_asset_hashes": {"mjcf": "hashB"}}
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    for _ in range(3):
        optimizer.step()
        schedule.step()
    saved_lr = optimizer.param_groups[0]["lr"]
    path = tmp_path / "schedule.pt"
    manager = CheckpointManager(metadata)
    manager.save(path, models={"model": model}, optimizers={"model": optimizer},
                 schedulers={"model": schedule}, training={"iteration": 3}, config={}, seed=3)
    new_model = torch.nn.Linear(2, 1)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.01)
    new_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=10)
    manager.load(path, models={"model": new_model}, optimizers={"model": new_optimizer},
                 schedulers={"model": new_schedule})
    assert new_schedule.last_epoch == 3
    assert new_optimizer.param_groups[0]["lr"] == pytest.approx(saved_lr)
    new_optimizer.step()
    new_schedule.step()
    optimizer.step()
    schedule.step()
    assert new_optimizer.param_groups[0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])
