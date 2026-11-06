"""The new task consumes only two TCP poses, not witness joint angles."""

import torch

from dexgym.tasks.marvin_supermimic.se3_reach_hold import (
    MarvinBimanualSE3ReachHold, rotation_6d_from_matrix)


class SimpleFKRobot:
    class Spec:
        control_dt = 0.02

    spec = Spec()

    def __init__(self):
        self.action_map = type("Map", (), {
            "origin": torch.zeros(14), "scale": torch.ones(14),
            "lower": torch.full((14,), -1.0), "upper": torch.full((14,), 1.0)})()

    def fk(self, q):
        batch = len(q)
        pos = torch.zeros(batch, 2, 3, 3)
        pos[:, 0, 2, 0] = q[:, 0]
        pos[:, 1, 2, 0] = q[:, 7]
        rotation = torch.eye(3).expand(batch, 2, 3, 3, 3).clone()
        return pos, rotation


def test_goal_and_observation_schema_are_true_bimanual_se3():
    task = MarvinBimanualSE3ReachHold(SimpleFKRobot())
    assert (task.spec.goal_dim, task.spec.goal_horizon) == (18, 1)
    assert (task.spec.observation_dim, task.spec.history_length, task.spec.critic_dim) == (68, 10, 60)
    pose = torch.cat((torch.tensor([0.1, 0.0, 0.0]),
                      rotation_6d_from_matrix(torch.eye(3)),
                      torch.tensor([-0.1, 0.0, 0.0]),
                      rotation_6d_from_matrix(torch.eye(3))))[None]
    state = torch.zeros(1, 42)
    task_state = task.task_state_from_goal(pose)
    assert task.goal(state, task_state).shape == (1, 1, 18)
    assert task.observation(state, torch.zeros(1, 14), task_state).shape == (1, 68)
    assert task.critic_features(state, task_state).shape == (1, 60)
    assert not hasattr(task_state, "q_goal")


def test_far_initial_target_is_not_a_hard_failure():
    task = MarvinBimanualSE3ReachHold(SimpleFKRobot())
    target = torch.cat((torch.tensor([0.3, 0.0, 0.0]),
                        rotation_6d_from_matrix(torch.eye(3)),
                        torch.tensor([-0.3, 0.0, 0.0]),
                        rotation_6d_from_matrix(torch.eye(3))))[None]
    state = torch.zeros(1, 42)
    assert not bool(task.hard_termination(
        state, task.task_state_from_goal(target), torch.zeros(1, dtype=torch.int32))[0])
