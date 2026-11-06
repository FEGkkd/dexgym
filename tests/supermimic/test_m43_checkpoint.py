import copy
import pytest
import torch

from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint
from dexgym.tasks.supermimic_toy import make_toy_trainer


def test_checkpoint_resume_with_replay_and_rng(tmp_path):
    torch.manual_seed(7)
    trainer = make_toy_trainer("position", num_envs=8)
    for _ in range(10):
        trainer.run_iteration()
    path = tmp_path / "state.pt"
    manager = CoreCheckpoint(git_commit="test", reference_hashes={})
    manager.save(path, trainer, include_replay=True)
    expected_rng = torch.get_rng_state().clone()
    expected = {key: copy.deepcopy(module.state_dict()) for key, module in
                (("actor", trainer.actor), ("world_model", trainer.world_model),
                 ("critic", trainer.critic))}
    resumed = make_toy_trainer("position", num_envs=8)
    report = manager.load(path, resumed)
    assert report["replay_restored"]
    assert resumed.iteration == trainer.iteration
    assert resumed.counts == trainer.counts
    assert resumed.replay.steps == trainer.replay.steps
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for key, module in (("actor", resumed.actor), ("world_model", resumed.world_model),
                        ("critic", resumed.critic)):
        for name, value in expected[key].items():
            assert torch.equal(module.state_dict()[name], value)
    assert resumed.collector.task_state.frame.equal(trainer.collector.task_state.frame)
    assert resumed.run_iteration()["counts"]["world_updates"] > trainer.counts["world_updates"]


def test_checkpoint_rejects_goal_schema_and_missing_replay(tmp_path):
    trainer = make_toy_trainer("position", num_envs=8)
    trainer.run_iteration()
    trainer.allowed_horizon = 32
    trainer.trust_report = {"32": {"physical_delta_rmse": 0.0}}
    manager = CoreCheckpoint(git_commit="test", reference_hashes={})
    path = tmp_path / "state.pt"
    manager.save(path, trainer, include_replay=False)
    with pytest.raises(ValueError, match="Task schema"):
        manager.load(path, make_toy_trainer("position_velocity", num_envs=8))
    fresh = make_toy_trainer("position", num_envs=8)
    report = manager.load(path, fresh)
    assert report["replay_restored"] is False
    assert fresh.replay.steps == 0
    assert report["requires_fresh_collection"]
    assert fresh.allowed_horizon == 0 and fresh.trust_report == {}
