from types import SimpleNamespace

import pytest
import torch

from dexgym.algorithms.supermimic.core.specs import RobotSpec, TaskSpec
from dexgym.algorithms.supermimic.core.transition import TaskState
from dexgym.tasks.marvin_supermimic.core_collector import CoreNewtonCollector


class _Robot:
    spec = RobotSpec("fake_marvin", 42, 14, 28, 1, "test_v1", "test_v1")

    @staticmethod
    def physical_state(env):
        return env.state.clone()

    @staticmethod
    def executed_command(state, action):
        return state[:, 28:42] + action

    @staticmethod
    def dynamics_context(env):
        return env.mass[:, None]


class _Task:
    spec = TaskSpec("test", 1, 1, 1, 1, 0, 1, "goal_v1")

    @staticmethod
    def physical_task_state(env):
        n = env.num_envs
        return TaskState(torch.zeros(n, dtype=torch.long), torch.full((n,), -1, dtype=torch.long),
                         env.frame.clone(), torch.empty(n, 0))

    @staticmethod
    def observation(state, previous_action, task_state):
        return state[:, :1]

    @staticmethod
    def goal(state, task_state):
        return state[:, None, :1]

    @staticmethod
    def advance(task_state):
        return TaskState(task_state.task_id, task_state.motion_id,
                         task_state.frame + 1, task_state.data)

    @staticmethod
    def reward(state, next_state, action, arrival, previous_action):
        return torch.zeros(len(state))

    @staticmethod
    def hard_termination(next_state, arrival, contacts):
        return torch.zeros(len(next_state), dtype=torch.bool)


class _Env:
    def __init__(self):
        self.unwrapped = self
        self.num_envs = 2
        self.state = torch.zeros(2, 42)
        self.mass = torch.tensor([1.0, 2.0])
        self.frame = torch.zeros(2, dtype=torch.long)
        self.last_action = torch.zeros(2, 14)
        self.command = torch.zeros(2, 14)
        self.last_failure_reason = torch.zeros(2, dtype=torch.int32)
        self.last_policy_contact_counts = torch.zeros(2, dtype=torch.int32)

    def reset(self):
        return None

    def step(self, action):
        self.frame += 1
        self.last_physical_arm_state = self.state.clone()
        return None, torch.zeros(2), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), {}


def test_collector_reads_real_per_environment_context():
    collector = CoreNewtonCollector(_Env(), _Robot(), _Task())
    transition = collector.step(lambda history, goal: torch.zeros(2, 14))
    assert torch.equal(transition.dynamics_context, torch.tensor([[1.0], [2.0]]))


def test_collector_rejects_missing_or_nonfinite_context():
    robot = _Robot()
    robot.dynamics_context = None
    collector = CoreNewtonCollector(_Env(), robot, _Task())
    with pytest.raises(ValueError, match="dynamics_context"):
        collector.step(lambda history, goal: torch.zeros(2, 14))
    robot.dynamics_context = lambda env: torch.tensor([[1.0], [float("nan")]])
    with pytest.raises(ValueError, match="dynamics_context"):
        collector.step(lambda history, goal: torch.zeros(2, 14))


def test_collector_keeps_executed_action_if_environment_mutates_input():
    class MutatingEnv(_Env):
        def step(self, action):
            actual = action.clone()
            self.frame += 1
            self.last_physical_arm_state = self.state.clone()
            self.last_physical_arm_state[:, 28:42] = actual
            action.zero_()  # Emulates reset mutating a caller-owned Tensor.
            return None, torch.zeros(2), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), {}

    collector = CoreNewtonCollector(MutatingEnv(), _Robot(), _Task())
    transition = collector.step(lambda history, goal: torch.ones(2, 14))
    assert torch.equal(transition.action, torch.ones(2, 14))
    assert torch.equal(transition.executed_command, torch.ones(2, 14))
