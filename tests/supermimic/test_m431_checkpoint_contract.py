import pytest
import torch

from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint
from dexgym.tasks.supermimic_toy import make_toy_trainer


def test_same_shape_goal_activation_mismatch_is_rejected(tmp_path):
    manager = CoreCheckpoint(git_commit="test", reference_hashes={})
    source = make_toy_trainer("position", num_envs=8)
    path = manager.save(tmp_path / "state.pt", source, include_replay=False)
    target = make_toy_trainer("position", num_envs=8)
    target.actor.goal_latent_tanh = False
    with pytest.raises(ValueError, match="goal_latent_tanh"):
        manager.load(path, target)


def test_save_refuses_unrestorable_full_resume(tmp_path):
    manager = CoreCheckpoint(git_commit="test", reference_hashes={})
    trainer = make_toy_trainer("position", num_envs=8)
    trainer.collector.state_dict = None
    with pytest.raises(ValueError, match="collector"):
        manager.save(tmp_path / "state.pt", trainer, include_replay=True)
    assert not (tmp_path / "state.pt").exists()
    trainer.collector.state_dict = lambda: None
    with pytest.raises(ValueError, match="collector"):
        manager.save(tmp_path / "state.pt", trainer, resume_mode="full")
    trainer.collector.state_dict = lambda: {}
    with pytest.raises(ValueError, match="collector"):
        manager.save(tmp_path / "state.pt", trainer, resume_mode="full")


def test_world_model_normalize_false_disables_both_statistics():
    from dexgym.algorithms.supermimic.core.models import StateDeltaWorldModel
    from dexgym.algorithms.supermimic.core.specs import NetworkSpec
    from dexgym.tasks.supermimic_toy import ToyRobot

    model = StateDeltaWorldModel(ToyRobot.spec, NetworkSpec(world_hidden=(16,), normalize=False))
    robot = ToyRobot()
    state = torch.randn(5, 2)
    action = torch.randn(5, 1)
    context = torch.ones(5, 1)
    model.update_norms(state, action, context, state + 1, robot)
    assert not hasattr(model.input_norm, "count")
    assert not hasattr(model.delta_norm, "count")


@pytest.mark.parametrize("changed,path", [
    ("reward", "task_contract.position_weight"),
    ("termination", "task_contract.episode_length"),
    ("controller", "robot_contract.command"),
    ("normalizer", "networks.actor.normalizers.goal_norm.min_std"),
])
def test_semantic_parameter_mismatch_is_rejected(tmp_path, changed, path):
    manager = CoreCheckpoint(git_commit="test", reference_hashes={})
    source = make_toy_trainer("position", num_envs=8)
    saved = manager.save(tmp_path / "state.pt", source, include_replay=False)
    target = make_toy_trainer("position", num_envs=8)
    if changed == "normalizer":
        target.actor.goal_norm.min_std *= 2
    else:
        owner = target.robot if changed == "controller" else target.task
        original = owner.checkpoint_contract
        field = {"reward": "position_weight", "termination": "episode_length",
                 "controller": "command"}[changed]
        owner.checkpoint_contract = lambda: dict(original(), **{field: "changed"})
    with pytest.raises(ValueError, match=path):
        manager.load(saved, target)


def test_compatible_checkpoint_preserves_forward_output(tmp_path):
    manager = CoreCheckpoint(git_commit="test", reference_hashes={})
    source = make_toy_trainer("position", num_envs=8)
    history = torch.randn(4, 2, 3)
    goal = torch.randn(4, 1, 1)
    before = source.actor(history, goal).detach().clone()
    saved = manager.save(tmp_path / "state.pt", source, include_replay=False)
    target = make_toy_trainer("position", num_envs=8)
    manager.load(saved, target)
    assert torch.equal(before, target.actor(history, goal))


def test_replay_resume_starts_new_episode_and_full_resume_continues_one(tmp_path):
    manager = CoreCheckpoint(git_commit="test", reference_hashes={})
    source = make_toy_trainer("position", num_envs=8)
    source.run_iteration()
    saved_counter = source.replay.episode_counter.clone()
    replay_path = manager.save(tmp_path / "replay.pt", source, resume_mode="replay")
    fresh = make_toy_trainer("position", num_envs=8)
    report = manager.load(replay_path, fresh)
    assert report["requires_fresh_collection"]
    assert torch.equal(fresh.replay.episode_counter, saved_counter + 1)
    full_path = manager.save(tmp_path / "full.pt", source, resume_mode="full")
    exact = make_toy_trainer("position", num_envs=8)
    report = manager.load(full_path, exact)
    assert not report["requires_fresh_collection"]
    assert torch.equal(exact.replay.episode_counter, saved_counter)


def test_toy_full_resume_continues_identical_updates(tmp_path):
    torch.manual_seed(431)
    manager = CoreCheckpoint(git_commit="test", reference_hashes={})
    source = make_toy_trainer("position", num_envs=8)
    for _ in range(6):
        source.run_iteration()
    saved = manager.save(tmp_path / "full.pt", source, resume_mode="full")
    source.run_iteration()
    expected = {key: value.detach().clone() for key, value in source.actor.state_dict().items()}
    expected_replay = source.replay.state_dict()
    restored = make_toy_trainer("position", num_envs=8)
    manager.load(saved, restored)
    restored.run_iteration()
    for key, value in expected.items():
        assert torch.equal(restored.actor.state_dict()[key], value)
    assert torch.equal(restored.replay.state_dict()["episode_counter"],
                       expected_replay["episode_counter"])


@pytest.mark.parametrize("enabled", [False, True])
def test_normalization_switch_roundtrip_and_frozen_forward(tmp_path, enabled):
    from dexgym.algorithms.supermimic.core.models import NetworkFactory
    from dexgym.algorithms.supermimic.core.specs import NetworkSpec
    from dexgym.tasks.supermimic_toy import ToyGoalTask, ToyRobot

    robot, task = ToyRobot(), ToyGoalTask("position")
    spec = NetworkSpec(world_hidden=(16,), goal_hidden=(8,), actor_hidden=(16,),
                       critic_hidden=(8,), normalize=enabled,
                       normalizer_min_std=0.05, normalizer_max_count=8)
    first = NetworkFactory.build(robot.spec, task.spec, spec)
    history = torch.randn(4, task.spec.history_length, task.spec.observation_dim)
    goal = torch.randn(4, task.spec.goal_horizon, task.spec.goal_dim)
    state = torch.randn(4, 2)
    action = torch.randn(4, 1)
    context = torch.ones(4, 1)
    first.actor.update_norms(history, goal)
    first.world_model.update_norms(state, action, context, state + 0.2, robot)
    first.critic.update_norms(torch.randn(4, task.spec.critic_dim))
    before = (first.actor(history, goal).detach(),
              first.world_model(state, action, context, robot).detach())
    second = NetworkFactory.build(robot.spec, task.spec, spec)
    for name in ("actor", "world_model", "critic"):
        getattr(second, name).load_state_dict(getattr(first, name).state_dict(), strict=True)
    after = (second.actor(history, goal), second.world_model(state, action, context, robot))
    assert all(torch.equal(left, right) for left, right in zip(before, after))
    if enabled:
        assert float(first.actor.goal_norm.count) == 4
        assert float(first.world_model.input_norm.count) == 4
        assert float(first.world_model.delta_norm.count) == 4
        assert first.actor.goal_norm.std().min() >= 0.05
    else:
        assert not hasattr(first.actor.goal_norm, "count")
        assert not hasattr(first.world_model.input_norm, "count")


def test_policy_export_rejects_same_shape_wrong_network_semantics(tmp_path):
    from dataclasses import replace
    from dexgym.algorithms.supermimic.core.models import NetworkFactory
    from dexgym.algorithms.supermimic.core.policy import export_policy
    from dexgym.algorithms.supermimic.core.specs import NetworkSpec
    from dexgym.tasks.supermimic_toy import ToyGoalTask, ToyRobot

    robot, task = ToyRobot(), ToyGoalTask("position")
    spec = NetworkSpec(world_hidden=(16,), goal_hidden=(8,), actor_hidden=(16,))
    actor = NetworkFactory.build(robot.spec, task.spec, spec).actor
    with pytest.raises(ValueError, match="NetworkSpec"):
        export_policy(tmp_path / "wrong.pt", actor, robot.spec, task.spec,
                      replace(spec, goal_latent_tanh=False),
                      reference_hashes={}, git_commit="test")
