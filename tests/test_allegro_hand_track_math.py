from __future__ import annotations

import torch

from dexteroushands_lab.tasks.direct.allegro_hand_track.tracking_math import (
    clip_and_accumulate_residual,
    compute_object_position_error,
    compute_tracking_reward_components,
    rotation_error_xyzw,
)


def test_clip_and_accumulate_residual_clips_actions_and_limits():
    actions = torch.tensor([[2.0, -2.0, 0.5]])
    residual = torch.tensor([[0.1, -0.1, 0.0]])
    clipped, updated = clip_and_accumulate_residual(actions, residual, 1.0, 0.2, 0.2)
    torch.testing.assert_close(clipped, torch.tensor([[1.0, -1.0, 0.5]]))
    torch.testing.assert_close(updated, torch.tensor([[0.2, -0.2, 0.1]]))


def test_zero_action_preserves_residual_and_position_error_is_euclidean():
    residual = torch.tensor([[0.15, -0.05]])
    _, updated = clip_and_accumulate_residual(
        torch.zeros_like(residual), residual, 0.5, 0.1, 1.0
    )
    torch.testing.assert_close(updated, residual)
    torch.testing.assert_close(
        compute_object_position_error(torch.tensor([[1.0, 2.0, 2.0]]), torch.zeros(1, 3)),
        torch.tensor([3.0]),
    )


def test_reward_components_are_finite_and_report_success_and_loss():
    zeros = torch.zeros(2)
    components = compute_tracking_reward_components(
        hand_error=zeros,
        object_position_error=zeros,
        object_orientation_error=zeros,
        palm_distance=zeros,
        fingertip_distance=zeros,
        actions=torch.zeros(2, 3),
        residual=torch.zeros(2, 3),
        success=torch.tensor([True, False]),
        object_lost=torch.tensor([False, True]),
    )
    assert set(("tracking/hand_error", "tracking/object_pos_error", "tracking/success")).issubset(components)
    assert all(torch.isfinite(value).all() for value in components.values())
    torch.testing.assert_close(components["tracking/success"], torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(components["termination/object_lost"], torch.tensor([0.0, -1.0]))


def test_rotation_error_is_zero_for_equal_quaternions():
    quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    torch.testing.assert_close(rotation_error_xyzw(quaternion, quaternion), torch.zeros(1))
