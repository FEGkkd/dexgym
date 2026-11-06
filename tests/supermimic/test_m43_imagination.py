"""Task-agnostic imagination keeps exact survival math and Actor gradients."""

import pytest
import torch
from torch import nn

from dexgym.algorithms.supermimic.core.imagination import ImaginationEngine, critic_td_lambda_batch
from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.optimization import frozen_parameters
from dexgym.algorithms.supermimic.core.returns import discounted_imagined_return, lambda_returns
from dexgym.algorithms.supermimic.core.specs import NetworkSpec, RobotSpec, TaskSpec
from dexgym.algorithms.supermimic.core.transition import StartBatch, TaskState


def test_survival_and_td_lambda_match_hand_calculation():
    rewards = torch.tensor([[1.0], [2.0]])
    termination = torch.tensor([[0.2], [0.5]])
    terminal = torch.tensor([3.0])
    actual = discounted_imagined_return(rewards, termination, terminal, gamma=0.9)
    assert float(actual) == pytest.approx(1 + 0.9 * 0.8 * 2 + 0.9**2 * 0.8 * 0.5 * 3)
    values = torch.tensor([[0.5], [1.0], [3.0]])
    td = lambda_returns(rewards, values, termination, gamma=0.9, lam=0.5)
    second = 2 + 0.9 * 0.5 * 3
    first = 1 + 0.9 * 0.8 * (0.5 * 1 + 0.5 * second)
    assert float(td[1]) == pytest.approx(second)
    assert float(td[0]) == pytest.approx(first)


class ToyRobot:
    spec = RobotSpec("toy", 1, 1, 1, 0, "toy_action", "toy_state")

    @staticmethod
    def integrate(state, delta, action, context):
        return state + delta

    @staticmethod
    def physical_delta(state, next_state):
        return next_state - state


class ExactToyWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, state, action, context, robot):
        return robot.integrate(state, action * self.scale, action, context)


class GoalTask:
    termination_mode = "analytic"

    def __init__(self, goal_dim):
        self.spec = TaskSpec("toy", goal_dim + 1, 2, goal_dim, 1, goal_dim,
                             1 + goal_dim, f"goal{goal_dim}")

    @staticmethod
    def advance(task_state):
        return TaskState(task_state.task_id, task_state.motion_id,
                         task_state.frame + 1, task_state.data)

    def observation(self, state, previous_action, task_state):
        return torch.cat((state, task_state.data), dim=-1)

    def goal(self, state, task_state):
        return task_state.data[:, None, :]

    def critic_features(self, state, task_state):
        return torch.cat((state, task_state.data), dim=-1)

    def reward(self, state, next_state, action, next_task_state):
        return -((next_state[:, 0] - next_task_state.data[:, 0]) ** 2)

    def soft_termination(self, next_state, next_task_state):
        return torch.zeros(len(next_state), device=next_state.device)

    def valid_imagination_start(self, task_state, horizon):
        return torch.ones_like(task_state.frame, dtype=torch.bool)


@pytest.mark.parametrize("goal_dim", [1, 3])
def test_same_engine_backpropagates_through_two_goal_schemas(goal_dim):
    robot, task, world = ToyRobot(), GoalTask(goal_dim), ExactToyWorld()
    config = NetworkSpec(world_hidden=(8,), goal_hidden=(8,), actor_hidden=(8,),
                         critic_hidden=(8,), latent_dim=4)
    modules = NetworkFactory.build(robot.spec, task.spec, config)
    target = torch.ones(4, goal_dim)
    task_state = TaskState(torch.zeros(4, dtype=torch.long),
                           torch.zeros(4, dtype=torch.long),
                           torch.zeros(4, dtype=torch.long), target)
    start = StartBatch(torch.zeros(4, 1), torch.zeros(4, 2, goal_dim + 1),
                       torch.empty(4, 0), task_state, torch.zeros(4, 1),
                       torch.arange(4), torch.zeros(4, dtype=torch.long),
                       torch.zeros(4, dtype=torch.int8))
    engine = ImaginationEngine(gamma=0.99, lam=0.95)
    with frozen_parameters(world, modules.critic):
        rollout = engine.rollout(start, modules.actor, world, modules.critic,
                                 robot, task, horizon=4)
        rollout.actor_loss.backward()
    assert rollout.rewards.shape == (4, 4)
    assert rollout.critic_features.shape == (4, 4, 1 + goal_dim)
    assert world.scale.grad is None
    assert all(parameter.grad is None for parameter in modules.critic.parameters())
    assert float(modules.actor.decoder[-1].bias.grad.abs().sum()) > 0
    features, targets = critic_td_lambda_batch(rollout, gamma=0.99, lam=0.95)
    assert features.shape == (16, 1 + goal_dim)
    assert targets.shape == (16,)
    assert bool(torch.isfinite(targets).all())
