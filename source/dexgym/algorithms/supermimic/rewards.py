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
    paper_multilink: bool = False


REWARD_PROFILES = {
    "openarm_style": RewardWeights(
        tcp_position=1.0,
        tcp_rotation=1.0,
        link_position=1.0,
        link_rotation=1.0,
        linear_velocity=1.0,
        angular_velocity=1.0,
    ),
    "marvin_tcp_priority": RewardWeights(),
    "paper_multilink": RewardWeights(
        tcp_position=0.0,
        tcp_rotation=0.0,
        link_position=1.0,
        link_rotation=1.0,
        linear_velocity=1.0,
        angular_velocity=1.0,
        action_rate=0.1,
        joint_limit=2.0,
        paper_multilink=True,
    ),
}


def reward_upper_bound(weights: RewardWeights, *, dt: float = 0.02) -> float:
    """Maximum possible positive tracking reward; all remaining terms are penalties."""
    if weights.paper_multilink:
        positive = (weights.link_position + weights.link_rotation
                    + weights.linear_velocity + weights.angular_velocity)
    else:
        positive = (weights.tcp_position + weights.tcp_rotation + weights.link_position
                    + weights.link_rotation + weights.linear_velocity + weights.angular_velocity)
    return dt * positive


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
    relative = current @ target.transpose(-1, -2)
    skew = (
        torch.stack(
            (
                relative[..., 2, 1] - relative[..., 1, 2],
                relative[..., 0, 2] - relative[..., 2, 0],
                relative[..., 1, 0] - relative[..., 0, 1],
            ),
            dim=-1,
        )
        * 0.5
    )
    sine = torch.linalg.vector_norm(skew, dim=-1)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
    return torch.atan2(sine, cosine)


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
    *,
    bounded_action: torch.Tensor | None = None,
    previous_bounded_action: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pos, rot = fk(q)
    prior_pos, prior_rot = fk(previous_q)
    linear = (pos - prior_pos) / dt
    angular = so3_log(rot @ prior_rot.transpose(-1, -2)) / dt
    pos_sq = (pos - target["link_pos"]).square().sum(-1)
    rot_sq = so3_geodesic(rot, target["link_rotmat"]).square()
    vel_sq = (linear - target["link_linear_vel"]).square().sum(-1)
    omega_sq = (angular - target["link_angular_vel"]).square().sum(-1)
    if weights.paper_multilink:
        if bounded_action is None or previous_bounded_action is None:
            raise ValueError("Paper reward requires current and previous bounded policy actions")
        body_pos = torch.exp(-pos_sq.mean((-1, -2)) / 0.3**2)
        body_rot = torch.exp(-rot_sq.mean((-1, -2)) / 0.4**2)
        body_linear = torch.exp(-vel_sq.mean((-1, -2)) / 1.0**2)
        body_angular = torch.exp(-omega_sq.mean((-1, -2)) / 3.14**2)
        action_smooth = (bounded_action - previous_bounded_action).square().sum(-1)
        hard_limit = (torch.relu(q - upper) + torch.relu(lower - q)).sum(-1)
        reward = dt * (
            weights.link_position * body_pos
            + weights.link_rotation * body_rot
            + weights.linear_velocity * body_linear
            + weights.angular_velocity * body_angular
            - weights.action_rate * action_smooth
            - weights.joint_limit * hard_limit
        )
    else:
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
        "link_linear_velocity_error_m_s": vel_sq.sqrt(),
        "link_angular_velocity_error_rad_s": omega_sq.sqrt(),
    }
    return reward, diagnostics
