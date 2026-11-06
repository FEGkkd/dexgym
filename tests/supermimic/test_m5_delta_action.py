"""A new relative joint command must not reinterpret legacy absolute actions."""

import torch
from types import SimpleNamespace

from dexgym.algorithms.supermimic.action import ArmJointDeltaActionMap
from dexgym.assets.marvin_wuji.controller import ARM_JOINT_NAMES
from dexgym.tasks.marvin_supermimic.robot_adapter import MarvinArmDynamicsAdapter


def test_joint_delta_map_uses_previous_command_and_has_action_gradient():
    home = torch.zeros(14)
    mapping = ArmJointDeltaActionMap(home, torch.full((14,), -1.0),
                                      torch.full((14,), 1.0), delta_q_max=0.01)
    previous = torch.full((2, 14), 0.3)
    action = torch.full((2, 14), torch.atanh(torch.tensor(0.5)), requires_grad=True)
    command = mapping(action, previous)
    torch.testing.assert_close(command, previous + 0.005, rtol=0, atol=1e-7)
    command.sum().backward()
    torch.testing.assert_close(action.grad, torch.full_like(action, 0.0075), rtol=0, atol=1e-6)
    assert torch.allclose(mapping(torch.zeros_like(action), previous), previous)


def test_joint_delta_respects_joint_bounds_without_changing_previous_action_origin():
    mapping = ArmJointDeltaActionMap(torch.zeros(14), torch.full((14,), -0.5),
                                      torch.full((14,), 0.5), delta_q_max=0.01)
    previous = torch.full((1, 14), 0.499)
    assert torch.equal(mapping(torch.full((1, 14), 10.0), previous),
                       torch.full((1, 14), 0.5))


def test_delta_command_uses_existing_ten_substep_newton_interpolation():
    adapter = MarvinArmDynamicsAdapter(
        SimpleNamespace(joint_names_arm=ARM_JOINT_NAMES), "cpu",
        action_schema="arm_joint_delta_v1")
    previous = adapter.action_map.origin[None]
    next_command = adapter.action_map(torch.ones_like(previous), previous)
    for step in range(10):
        interpolated = adapter.interpolate_command(previous, next_command, step)
        torch.testing.assert_close(interpolated,
                                   previous + (step + 1) / 10 * (next_command - previous))
    torch.testing.assert_close(interpolated, next_command)
