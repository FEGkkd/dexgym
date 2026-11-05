"""Paper Eq. 2/3: arrival reward precedes soft termination of later rewards."""

import torch
from dexgym.algorithms.supermimic.imagination import (
    discounted_imagined_return,
    lambda_returns,
    survival_weights,
)
from dexgym.algorithms.supermimic.networks import DeltaWorldModel


def test_survival_weights_apply_next_state_termination_to_future_steps():
    probability = torch.tensor([[0.25], [0.5]])
    assert torch.allclose(survival_weights(probability)[:, 0], torch.tensor([1.0, 0.75, 0.375]))


def test_actor_objective_uses_arrival_reward_and_surviving_terminal_bootstrap():
    rewards = torch.tensor([[2.0], [3.0]])
    probability = torch.tensor([[0.25], [0.5]], requires_grad=True)
    terminal_value = torch.tensor([4.0])
    objective = discounted_imagined_return(rewards, probability, terminal_value, gamma=0.9)
    assert torch.allclose(objective, torch.tensor([5.24]), atol=1e-6)
    objective.sum().backward()
    assert probability.grad is not None
    assert (probability.grad < 0).all()


def test_td_lambda_targets_include_soft_termination_once():
    rewards = torch.tensor([[2.0], [3.0]])
    values = torch.tensor([[1.0], [4.0], [5.0]])
    probability = torch.tensor([[0.25], [0.5]])
    actual = lambda_returns(rewards, values, probability, gamma=0.9, lam=0.5)
    assert torch.allclose(actual[:, 0], torch.tensor([5.121875, 5.25]), atol=1e-6)


def test_world_model_termination_head_is_differentiable_and_bounded():
    model = DeltaWorldModel(hidden=32)
    state = torch.randn(4, 42, requires_grad=True)
    goal = torch.randn(4, 90)
    probability = model.termination_probability(state, goal)
    assert probability.shape == (4,)
    assert ((probability > 0) & (probability < 1)).all()
    probability.sum().backward()
    assert state.grad is not None and state.grad.abs().sum() > 0
    assert any(parameter.grad is not None for parameter in model.termination_head.parameters())
