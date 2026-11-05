import torch

from dexgym.algorithms.supermimic.networks import GoalActor, PrivilegedCritic, RunningNorm


def test_running_norm_uses_fp32_capped_count_and_roundtrips():
    norm = RunningNorm(2, min_std=torch.tensor([0.5, 0.1]), max_count=3)
    norm.update(torch.tensor([[1.0, 0.0], [3.0, 0.0]], dtype=torch.float16))
    norm.update(torch.tensor([[5.0, 0.0], [100.0, 100.0]], dtype=torch.float16))
    torch.testing.assert_close(norm.mean, torch.tensor([3.0, 0.0]))
    torch.testing.assert_close(norm.std(), torch.tensor([1.6329932, 0.1]), atol=1e-6, rtol=1e-6)
    assert norm.count.item() == 3
    restored = RunningNorm(2, min_std=torch.tensor([0.5, 0.1]), max_count=3)
    restored.load_state_dict(norm.state_dict())
    torch.testing.assert_close(restored(torch.ones(2)), norm(torch.ones(2)))


def test_legacy_actor_and_critic_omit_new_normalizers_from_state_dict():
    actor = GoalActor(normalization_mode="legacy")
    critic = PrivilegedCritic(normalization_mode="legacy")
    assert not any("norm" in key for key in actor.state_dict())
    assert not any("norm" in key for key in critic.state_dict())
    history = torch.randn(1, 10, 56)
    goal = torch.randn(1, 10, 90)
    before = actor(history, goal)
    restored = GoalActor(normalization_mode="legacy")
    restored.load_state_dict(actor.state_dict(), strict=True)
    torch.testing.assert_close(restored(history, goal), before)


def test_paper_actor_goal_and_critic_norms_update_only_when_called():
    actor = GoalActor(normalization_mode="running")
    critic = PrivilegedCritic(normalization_mode="running")
    history = torch.randn(2, 10, 56)
    goal = torch.randn(2, 10, 90)
    actor.update_norms(history, goal)
    critic.update_norms(torch.randn(2, 42), torch.randn(2, 14), goal[:, 0])
    assert actor.observation_norm.count.item() == 20
    assert actor.goal_norm.count.item() == 20
    assert critic.input_norm.count.item() == 2
    actor(history, goal)
    assert actor.observation_norm.count.item() == 20
    assert actor.goal_norm.count.item() == 20
