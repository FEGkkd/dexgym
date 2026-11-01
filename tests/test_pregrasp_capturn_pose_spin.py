"""True cap/bottle pose twist angle and substep unwrapping."""

import math

import torch

from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.pose_spin import (
    relative_pose_yaw, wrapped_delta,
)


def _yaw(degrees: float) -> torch.Tensor:
    half = math.radians(degrees) / 2
    return torch.tensor([[0., 0., math.sin(half), math.cos(half)]])


def test_relative_yaw_ignores_whole_bottle_world_rotation():
    a = relative_pose_yaw(_yaw(45), _yaw(75))
    b = relative_pose_yaw(_yaw(130), _yaw(160))
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(a, torch.tensor([math.pi / 6]), atol=1e-6)


def test_pose_difference_unwraps_through_pi_without_false_jump():
    before = relative_pose_yaw(_yaw(0), _yaw(179))
    after = relative_pose_yaw(_yaw(0), _yaw(-179))
    assert torch.allclose(wrapped_delta(after, before),
                          torch.tensor([math.radians(2)]), atol=1e-6)
