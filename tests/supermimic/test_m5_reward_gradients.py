import math
from types import SimpleNamespace

import torch

from dexgym.tasks.marvin_supermimic.se3_reach_hold import (
    MarvinBimanualSE3ReachHold, pack_dual_tcp_pose, unpack_dual_tcp_pose)
from test_m5_se3_task import SimpleFKRobot
from dexgym.assets.marvin_wuji.controller import ARM_JOINT_NAMES
from dexgym.tasks.marvin_supermimic.core_plugins import MarvinCoreRobot


def test_so3_reward_has_finite_nonzero_joint_gradient_and_uses_both_arms():
    robot = SimpleFKRobot()
    task = MarvinBimanualSE3ReachHold(robot)
    start = torch.zeros(1, 42, dtype=torch.float64)
    next_state = start.clone()
    next_state[:, 0] = 0.08
    next_state[:, 7] = -0.08
    next_state = next_state.requires_grad_()
    goal_q = torch.zeros(1, 14, dtype=torch.float64)
    goal_q[:, 0] = 0.16
    goal_q[:, 7] = -0.16
    position, rotation = robot.fk(goal_q)
    goal = pack_dual_tcp_pose(position[:, :, 2], rotation[:, :, 2])
    task_state = task.task_state_from_goal(goal)
    reward = task.reward(start, next_state, torch.zeros(1, 14, dtype=torch.float64), task_state)
    gradient = torch.autograd.grad(reward.sum(), next_state)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[0, 0].abs() > 1e-8
    assert gradient[0, 7].abs() > 1e-8
    epsilon = 1e-5
    plus = next_state.detach().clone(); plus[:, 0] += epsilon
    minus = next_state.detach().clone(); minus[:, 0] -= epsilon
    fd = (task.reward(start, plus, torch.zeros(1, 14), task_state)
          - task.reward(start, minus, torch.zeros(1, 14), task_state)) / (2 * epsilon)
    assert torch.allclose(gradient[0, 0], fd[0], atol=1e-4, rtol=1e-3)


def test_goal_is_se3_not_hidden_joint_target_and_far_goal_is_not_failure():
    robot = SimpleFKRobot()
    task = MarvinBimanualSE3ReachHold(robot)
    state = torch.zeros(3, 42)
    position, rotation = robot.fk(state[:, :14])
    target = position[:, :, 2].clone() + 0.5
    goal = pack_dual_tcp_pose(target, rotation[:, :, 2])
    assert goal.shape == (3, 18)
    decoded_position, decoded_rotation = unpack_dual_tcp_pose(goal)
    assert torch.allclose(decoded_position, target)
    assert torch.allclose(decoded_rotation, rotation[:, :, 2], atol=1e-6)
    task_state = task.task_state_from_goal(goal)
    assert task_state.data.shape == (3, 18)
    assert not task.hard_termination(state, task_state, torch.zeros(3, dtype=torch.int32)).any()
    assert not any("q_goal" in name for name in vars(task_state))


def test_real_marvin_fk_orientation_reward_gradient_matches_finite_difference():
    robot = MarvinCoreRobot(SimpleNamespace(joint_names_arm=ARM_JOINT_NAMES), "cpu",
                            action_schema="arm_joint_delta_v1")
    task = MarvinBimanualSE3ReachHold(robot)
    home = robot.action_map.origin[None]
    witness = home.clone()
    witness[:, 1] += 0.08
    witness[:, 8] -= 0.07
    goal_position, goal_rotation = robot.fk(witness)
    goal = task.task_state_from_goal(pack_dual_tcp_pose(
        goal_position[:, :, 2], goal_rotation[:, :, 2]))
    start = torch.cat((home, torch.zeros_like(home), home), dim=-1)
    next_state = start.clone().requires_grad_()
    action = torch.zeros(1, 14)
    loss = task.reward(start, next_state, action, goal).sum()
    gradient = torch.autograd.grad(loss, next_state)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[:, 1].abs().max() > 1e-6
    assert gradient[:, 8].abs().max() > 1e-6
    epsilon = 1e-3
    plus, minus = start.clone(), start.clone()
    plus[:, 1] += epsilon
    minus[:, 1] -= epsilon
    finite_difference = (task.reward(start, plus, action, goal)
                         - task.reward(start, minus, action, goal)) / (2 * epsilon)
    torch.testing.assert_close(gradient[:, 1], finite_difference,
                               atol=3e-3, rtol=0.05)


def test_left_and_right_position_reward_are_symmetric():
    robot = SimpleFKRobot()
    task = MarvinBimanualSE3ReachHold(robot)
    start = torch.zeros(1, 42)
    first = start.clone(); first[:, 0] = 0.06; first[:, 7] = -0.03
    second = start.clone(); second[:, 0] = -0.03; second[:, 7] = 0.06
    rotation = torch.eye(3).expand(1, 2, 3, 3)
    target_a = torch.tensor([[[0.10, 0, 0], [-0.04, 0, 0]]], dtype=torch.float32)
    target_b = target_a.flip(1)
    reward_a = task.reward(start, first, torch.zeros(1, 14),
                           task.task_state_from_goal(pack_dual_tcp_pose(target_a, rotation)))
    reward_b = task.reward(start, second, torch.zeros(1, 14),
                           task.task_state_from_goal(pack_dual_tcp_pose(target_b, rotation)))
    torch.testing.assert_close(reward_a, reward_b, atol=1e-7, rtol=0)
