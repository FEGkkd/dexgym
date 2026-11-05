"""Differentiable six-link tracking reward shared by physics scoring and imagination."""

from dataclasses import dataclass

import torch

from .differentiable_fk import DifferentiableMarvinFK


@dataclass(frozen=True)
class RewardWeights:
    tcp_position: float = 2.0
    tcp_rotation: float = 1.5
    link_position: float = 0.7
    link_rotation: float = 0.4
    linear_velocity: float = 0.25
    angular_velocity: float = 0.15
    action_rate: float = 0.001
    joint_limit: float = 0.1


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    skew = (
        torch.stack(
            (
                rotation[..., 2, 1] - rotation[..., 1, 2],
                rotation[..., 0, 2] - rotation[..., 2, 0],
                rotation[..., 1, 0] - rotation[..., 0, 1],
            ),
            dim=-1,
        )
        * 0.5
    )
    sin = torch.linalg.vector_norm(skew, dim=-1)
    cos = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
    angle = torch.atan2(sin, cos)
    return skew * (angle / sin.clamp_min(1e-6))[..., None]


def so3_geodesic(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(so3_log(current @ target.transpose(-1, -2)), dim=-1)


def tracking_reward(
    q: torch.Tensor,
    previous_q: torch.Tensor,
    command: torch.Tensor,
    previous_command: torch.Tensor,
    target: dict[str, torch.Tensor],
    fk: DifferentiableMarvinFK,
    lower: torch.Tensor,
    upper: torch.Tensor,
    weights: RewardWeights = RewardWeights(),
    dt: float = 0.02,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pos, rot = fk(q)
    prior_pos, prior_rot = fk(previous_q)
    linear = (pos - prior_pos) / dt
    angular = so3_log(rot @ prior_rot.transpose(-1, -2)) / dt
    pos_sq = (pos - target["link_pos"]).square().sum(-1)
    rot_sq = so3_geodesic(rot, target["link_rotmat"]).square()
    vel_sq = (linear - target["link_linear_vel"]).square().sum(-1)
    omega_sq = (angular - target["link_angular_vel"]).square().sum(-1)
    tcp_pos = torch.exp(-30 * pos_sq[:, :, 2]).mean(-1)
    tcp_rot = torch.exp(-6 * rot_sq[:, :, 2]).mean(-1)
    link_pos = torch.exp(-30 * pos_sq[:, :, :2]).mean((-1, -2))
    link_rot = torch.exp(-6 * rot_sq[:, :, :2]).mean((-1, -2))
    linear_term = torch.exp(-2 * vel_sq).mean((-1, -2))
    angular_term = torch.exp(-0.05 * omega_sq).mean((-1, -2))
    rate_penalty = ((command - previous_command) / dt).square().mean(-1)
    limit_penalty = (torch.relu(lower + 0.05 - q).square() + torch.relu(q - upper + 0.05).square()).mean(-1)
    reward = dt * (
        weights.tcp_position * tcp_pos
        + weights.tcp_rotation * tcp_rot
        + weights.link_position * link_pos
        + weights.link_rotation * link_rot
        + weights.linear_velocity * linear_term
        + weights.angular_velocity * angular_term
        - weights.action_rate * rate_penalty
        - weights.joint_limit * limit_penalty
    )
    diagnostics = {
        "tcp_position_error_m": pos_sq[:, :, 2].sqrt(),
        "tcp_rotation_error_rad": rot_sq[:, :, 2].sqrt(),
        "link_position_error_m": pos_sq.sqrt(),
        "link_rotation_error_rad": rot_sq.sqrt(),
    }
    return reward, diagnostics
