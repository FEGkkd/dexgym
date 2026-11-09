"""Regression checks for defects reached by the hand S0 integration."""
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dexgym.algorithms.supermimic.core.imagination import ImaginationEngine
from dexgym.algorithms.supermimic.core.trust_evaluator import ModelTrustEvaluator, TrustConfig
from dexgym.algorithms.supermimic.core.transition import StartBatch, TaskState, TaskStep
from dexgym.algorithms.supermimic.core.specs import RobotSpec, TaskSpec
from dexgym.tasks.marvin_supermimic.se3_trust import (
    SE3DualDistributionTrustEvaluator, SE3GeometryTrustConfig,
)


@pytest.mark.parametrize('invalid', [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize('field', ['physical_delta_rmse', 'zero_delta_baseline_rmse'])
def test_nonfinite_model_trust_never_admits_rollouts(invalid, field):
    report = {str(h): {'distinct_start_count': 64, 'physical_delta_rmse': .01,
                      'zero_delta_baseline_rmse': .2} for h in (1, 8, 32)}
    report['1'][field] = invalid
    assert ModelTrustEvaluator().trusted_horizon(report) == 0


def test_nonfinite_geometry_rejected_independently_of_good_noisy_model():
    evaluator = SE3DualDistributionTrustEvaluator(
        TrustConfig(metric='normalized'), SE3GeometryTrustConfig(),
        noisy_validation_range=(8, 10), deterministic_validation_range=(10, 12))
    sample = {'distinct_start_count': 64, 'normalized_delta_rmse': .01,
              'normalized_zero_delta_baseline_rmse': .2,
              'tcp_position_p95_max_m': .001, 'tcp_so3_p95_max_rad': .001,
              'joint_velocity_rmse_rad_s': .01}
    good = {str(h): sample.copy() for h in (1, 8, 32)}
    for field in ('normalized_delta_rmse', 'tcp_position_p95_max_m',
                  'tcp_so3_p95_max_rad', 'joint_velocity_rmse_rad_s'):
        bad = {k: v.copy() for k, v in good.items()}
        bad['1'][field] = math.nan
        assert evaluator.trusted_horizon({'noisy': good, 'deterministic': bad}) == 0


def test_physical_trust_works_with_normalization_disabled():
    class Model:
        delta_norm = nn.Identity()
        def __call__(self, state, action, context, robot):
            return state + action
    batch = 8
    window = SimpleNamespace(state=torch.zeros(batch, 1, 1),
        action=torch.ones(batch, 1, 1), dynamics_context=torch.empty(batch, 1, 0),
        next_state=torch.ones(batch, 1, 1),
        absolute_timestamp=torch.arange(batch)[:, None], env_id=torch.arange(batch))
    replay = SimpleNamespace(sample_windows=lambda *args, **kwargs: window)
    robot = SimpleNamespace(physical_delta=lambda a, b: b-a)
    report = ModelTrustEvaluator().evaluate_horizon(Model(), replay, robot, horizon=1)
    assert report['physical_delta_rmse'] == 0
    assert report['normalized_delta_rmse'] == 0
    assert report['zero_delta_baseline_rmse'] == 1


def test_state_dependent_goal_switch_scores_old_goal_and_keeps_bptt():
    class Task:
        spec = TaskSpec('hybrid', 1, 1, 1, 1, 1, 2, 'current_goal')
        termination_mode = 'analytic'
        def valid_imagination_start(self, state, horizon):
            return torch.ones_like(state.frame, dtype=torch.bool)
        def goal(self, state, task_state):
            return task_state.data[:, None]
        def critic_features(self, state, task_state):
            return torch.cat((state, task_state.data), -1)
        def observation(self, state, action, task_state):
            return state
        def advance(self, task_state):
            raise AssertionError('Hybrid task must use predicted physical state')
        def transition(self, state, next_state, action, task_state, previous_action):
            error = next_state[:, 0] - task_state.data[:, 0]
            reached = error.abs() < .01
            goal = torch.where(reached[:, None], task_state.data + 1, task_state.data)
            return TaskStep(TaskState(task_state.task_id, task_state.motion_id,
                task_state.frame + 1, goal), -error.square())
        def soft_termination(self, state, task_state):
            return torch.zeros(len(state))
    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.step = nn.Parameter(torch.tensor(.5))
        def forward(self, history, goal):
            return torch.ones_like(history[:, 0]) * self.step
    class World(nn.Module):
        def forward(self, state, action, context, robot):
            return state + action
    robot = SimpleNamespace(spec=RobotSpec('toy', 1, 1, 1, 0, 'a', 's'))
    zeros = torch.zeros(1, dtype=torch.long)
    start = StartBatch(torch.zeros(1, 1), torch.zeros(1, 1, 1), torch.empty(1, 0),
        TaskState(zeros, zeros, zeros, torch.ones(1, 1)), torch.zeros(1, 1),
        zeros, zeros, zeros.to(torch.int8))
    actor = Actor()
    rollout = ImaginationEngine().rollout(start, actor, World(),
        lambda features: features[:, 0] * 0, robot, Task(), horizon=3)
    # Arrival at step2 hits goal1 and earns zero squared error; next goal is2.
    torch.testing.assert_close(rollout.rewards[:, 0], torch.tensor([-.25, 0., -.25]))
    torch.testing.assert_close(rollout.critic_features[:, 0, 1], torch.tensor([1., 1., 2.]))
    rollout.actor_loss.backward()
    assert actor.step.grad is not None and actor.step.grad < 0
