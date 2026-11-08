"""Hand-only subgoal reward and action contracts."""

import torch
from dexgym.assets.marvin_wuji.robot_cfg import JOINT_ORDER
from dexgym.tasks.direct.right_hand_tracking.logic import (
    action_toward_joint_target,
    build_right_hand_observation,
    fingertips_in_wrist_frame,
    residual_joint_target,
    right_hand_joint_indices,
    tracking_step,
)


def test_teledexter_deadzone_residual_accumulates_and_clips_joint_targets():
    command = torch.zeros((1, 20))
    limits = torch.tensor([[-0.05, 0.05]] * 20)
    action = torch.zeros_like(command)
    action[0, :7] = torch.tensor([-1.5, -0.5, -0.1, 0.0, 0.1, 0.5, 1.5])
    result = residual_joint_target(command, action, limits, scale=0.1, deadzone=0.1)
    torch.testing.assert_close(result[0, :7], torch.tensor([-0.05, -0.04, 0.0, 0.0, 0.0, 0.04, 0.05]))
    held = residual_joint_target(result, torch.full_like(command, 0.08), limits, scale=0.1, deadzone=0.1)
    torch.testing.assert_close(held, result)


def test_joint_replay_inverse_reaches_small_targets_through_deadzone():
    command = torch.zeros((1, 20))
    goal = torch.full_like(command, 0.002)
    limits = torch.tensor([[-1.0, 1.0]] * 20)
    action = action_toward_joint_target(goal, command, scale=0.1, deadzone=0.1)
    torch.testing.assert_close(action, torch.full_like(action, 0.12))
    torch.testing.assert_close(residual_joint_target(command, action, limits, scale=0.1, deadzone=0.1), goal)
    zero = action_toward_joint_target(command, command, scale=0.1, deadzone=0.1)
    torch.testing.assert_close(zero, torch.zeros_like(zero))


def test_command_error_observation_adds_the_actual_accumulated_target_gap():
    q = torch.full((2, 20), 0.2)
    qdot = torch.zeros_like(q)
    tips = torch.zeros((2, 5, 3))
    goals = torch.ones_like(tips)
    previous_action = torch.full_like(q, 0.5)
    command_q = q + 0.12
    old = build_right_hand_observation(q, qdot, tips, goals, previous_action)
    augmented = build_right_hand_observation(q, qdot, tips, goals, previous_action, command_q=command_q)
    assert old.shape == (2, 105)
    assert augmented.shape == (2, 125)
    torch.testing.assert_close(augmented[:, :105], old)
    torch.testing.assert_close(augmented[:, 105:], command_q - q)


def test_sparse_bonus_requires_every_finger_for_two_steps():
    goal = torch.zeros((2, 5, 3))
    current = goal.clone()
    current[1, 4, 0] = 0.03
    dwell = torch.zeros(2, dtype=torch.long)
    action_delta = torch.zeros((2, 20))

    first_reward, first_reached, dwell, error = tracking_step(current, goal, dwell, action_delta)
    assert first_reached.tolist() == [False, False]
    assert dwell.tolist() == [1, 0]
    assert error.tolist() == [0.0, 0.006000000052154064]

    second_reward, second_reached, dwell, _ = tracking_step(current, goal, dwell, action_delta)
    assert second_reached.tolist() == [True, False]
    assert dwell.tolist() == [2, 0]
    assert second_reward[0] > first_reward[0] + 9.0
    assert second_reward[1] < second_reward[0]


def test_action_change_is_penalized_without_affecting_success():
    goal = torch.zeros((1, 5, 3))
    dwell = torch.ones(1, dtype=torch.long)
    calm = torch.zeros((1, 20))
    moving = torch.ones((1, 20))
    calm_reward, calm_reached, _, _ = tracking_step(goal, goal, dwell, calm)
    moving_reward, moving_reached, _, _ = tracking_step(goal, goal, dwell, moving)
    assert calm_reached.item() and moving_reached.item()
    assert calm_reward.item() > moving_reward.item()


def test_right_hand_joint_selection_excludes_both_arms_and_left_hand():
    selected = right_hand_joint_indices(JOINT_ORDER)
    assert len(selected) == 20
    assert selected == list(range(34, 54))
    assert all(JOINT_ORDER[index].startswith("r_") for index in selected)


def test_fingertip_fk_converts_distal_offsets_to_wrist_frame():
    poses = torch.zeros((1, 6, 7))
    poses[..., 6] = 1.0  # XYZW identity
    poses[:, 0, :3] = torch.tensor([0.1, 0.2, 0.3])  # wrist
    poses[:, 1:, :3] = torch.tensor([0.11, 0.22, 0.33])
    offsets = torch.tensor([[0.0, 0.0, -0.02]] * 5)
    tips = fingertips_in_wrist_frame(poses, 0, [1, 2, 3, 4, 5], offsets)
    torch.testing.assert_close(tips, torch.tensor([[[0.01, 0.02, 0.01]] * 5]), atol=1e-6, rtol=0)
