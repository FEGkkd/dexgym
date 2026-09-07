"""Torch-only math helpers for the Allegro tracking DirectTask."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def clip_and_accumulate_residual(
    actions: torch.Tensor,
    residual: torch.Tensor,
    scale: float,
    dt: float,
    residual_limit: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip policy actions and update cumulative residual targets."""

    clipped = actions.to(dtype=residual.dtype, device=residual.device).clamp(-1.0, 1.0)
    updated = residual + scale * dt * clipped
    limit = torch.as_tensor(residual_limit, dtype=updated.dtype, device=updated.device)
    return clipped, torch.maximum(torch.minimum(updated, limit), -limit)


def compute_object_position_error(
    current: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Return Euclidean object position error per environment."""

    return torch.linalg.vector_norm(current - target, dim=-1)


def compute_tracking_reward_components(
    *,
    hand_error: torch.Tensor,
    object_position_error: torch.Tensor,
    object_orientation_error: torch.Tensor,
    palm_distance: torch.Tensor,
    fingertip_distance: torch.Tensor,
    actions: torch.Tensor,
    residual: torch.Tensor,
    success: torch.Tensor,
    object_lost: torch.Tensor,
    hand_scale: float = 1.0,
    object_position_scale: float = 2.0,
    object_orientation_scale: float = 0.25,
    contact_scale: float = 0.25,
    action_penalty_scale: float = 0.01,
    residual_penalty_scale: float = 0.01,
    success_reward: float = 2.0,
) -> Mapping[str, torch.Tensor]:
    """Build named reward terms with one value per vectorized environment."""

    action_norm = torch.linalg.vector_norm(actions, dim=-1)
    residual_norm = torch.linalg.vector_norm(residual, dim=-1)
    contact_distance = 0.5 * (palm_distance + fingertip_distance)
    return {
        "tracking/hand_error": -hand_scale * hand_error,
        "tracking/object_pos_error": -object_position_scale * object_position_error,
        "tracking/object_rot_error": -object_orientation_scale * object_orientation_error,
        "tracking/palm_distance": -contact_scale * palm_distance,
        "tracking/fingertip_distance": -contact_scale * fingertip_distance,
        "tracking/contact_distance": -contact_scale * contact_distance,
        "tracking/success": success.to(hand_error.dtype) * success_reward,
        "control/action_norm": -action_penalty_scale * action_norm,
        "control/residual_norm": -residual_penalty_scale * residual_norm,
        "termination/object_lost": -object_lost.to(hand_error.dtype),
    }


def rotation_error_xyzw(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the shortest angular distance between xyzw quaternions."""

    current = current / torch.linalg.vector_norm(current, dim=-1, keepdim=True).clamp_min(1.0e-8)
    target = target / torch.linalg.vector_norm(target, dim=-1, keepdim=True).clamp_min(1.0e-8)
    dot = torch.sum(current * target, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.arccos(dot)
