"""Strict episode success requires early SE(3) arrival and a full safe hold."""

from types import SimpleNamespace

import torch

from dexgym.algorithms.supermimic.core.evaluator import PhysicalEvaluator
from dexgym.tasks.marvin_supermimic.se3_reach_hold import (
    MarvinBimanualSE3ReachHold, pack_dual_tcp_pose)
from test_m5_se3_task import SimpleFKRobot


class _Actor(torch.nn.Module):
    def forward(self, history, goal):
        return torch.zeros(len(history), 14)


class _Collector:
    num_envs = 1
    action_noise_std = 0

    def __init__(self, task, arrival_step):
        self.task, self.arrival_step = task, arrival_step
        self.robot = task.robot
        self.step_count = 0
        self.goal = pack_dual_tcp_pose(
            torch.zeros(1, 2, 3),
            torch.eye(3).expand(1, 2, 3, 3))

    def step(self, actor, *, policy):
        self.step_count += 1
        state = torch.zeros(1, 42)
        if self.step_count < self.arrival_step:
            state[:, [0, 7]] = 0.04
        task_state = self.task.task_state_from_goal(
            self.goal, frame=torch.tensor([self.step_count]))
        result = SimpleNamespace(
            next_state=state, next_task_state=task_state,
            action=torch.zeros(1, 14), reward=torch.ones(1),
            terminated=torch.zeros(1, dtype=torch.bool),
            truncated=torch.tensor([self.step_count == 200]),
            failure_reason=torch.zeros(1, dtype=torch.int32))
        if self.step_count == 200:
            self.step_count = 0
        return result


def test_episode_evaluator_rejects_late_arrival_even_after_25_step_hold():
    task = MarvinBimanualSE3ReachHold(SimpleFKRobot())
    report = PhysicalEvaluator(lambda: _Collector(task, 150), task).evaluate(
        _Actor(), steps=200, one_episode_per_env=True)
    assert report["completed_episode_count"] == 1
    assert report["episodes"][0]["metrics"]["longest_hold_steps"] == 51
    assert not report["episodes"][0]["success"]
    assert report["episode_success_rate"] == 0


def test_episode_evaluator_accepts_early_arrival_and_complete_safe_hold():
    task = MarvinBimanualSE3ReachHold(SimpleFKRobot())
    report = PhysicalEvaluator(lambda: _Collector(task, 50), task).evaluate(
        _Actor(), steps=200, one_episode_per_env=True)
    assert report["episodes"][0]["success"]
    assert report["episodes"][0]["metrics"]["first_complete_se3_arrival_step"] == 50
    assert report["episodes"][0]["metrics"]["first_valid_hold_step"] == 74
