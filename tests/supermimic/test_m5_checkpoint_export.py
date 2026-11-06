"""A published policy can omit Replay while retaining a safe fresh-physics resume."""

import pytest
import torch
from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint
from dexgym.tasks.supermimic_toy import make_toy_trainer


def test_export_fresh_physics_retains_weights_optimizer_and_contract(tmp_path):
    manager = CoreCheckpoint(git_commit="test", reference_hashes={})
    source = make_toy_trainer("position", num_envs=8)
    source.run_iteration()
    full = manager.save(tmp_path / "full.pt", source, resume_mode="replay")
    exported = manager.export_fresh_physics(full, tmp_path / "policy.pt")
    original = torch.load(full, map_location="cpu", weights_only=False)
    small = torch.load(exported, map_location="cpu", weights_only=False)
    assert original["replay"] is not None
    assert small["replay"] is None
    assert small["resume_mode"] == "fresh_physics"
    assert small["contract_sha256"] == original["contract_sha256"]
    assert small["optimizers"].keys() == original["optimizers"].keys()
    for key, tensor in original["modules"]["actor"].items():
        assert torch.equal(tensor, small["modules"]["actor"][key])
    restored = make_toy_trainer("position", num_envs=8)
    result = manager.load(exported, restored)
    assert result["requires_fresh_collection"]
    assert restored.counts == source.counts
    with pytest.raises(FileExistsError):
        manager.export_fresh_physics(full, exported)


def test_export_refuses_tampered_contract(tmp_path):
    manager = CoreCheckpoint(git_commit="test", reference_hashes={})
    source = make_toy_trainer("position", num_envs=8)
    full = manager.save(tmp_path / "full.pt", source, resume_mode="replay")
    record = torch.load(full, map_location="cpu", weights_only=False)
    record["contract_manifest"]["task_spec"]["goal_dim"] += 1
    torch.save(record, full)
    with pytest.raises(ValueError, match="SHA256"):
        manager.export_fresh_physics(full, tmp_path / "policy.pt")
