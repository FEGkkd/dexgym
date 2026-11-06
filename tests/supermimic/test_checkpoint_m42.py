from pathlib import Path

import pytest
import torch

from dexgym.algorithms.supermimic.checkpoint import CheckpointManager


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
