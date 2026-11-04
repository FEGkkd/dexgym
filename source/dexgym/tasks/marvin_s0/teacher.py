"""Offline FK-supervised samples for bootstrapping the 73D S0 actor.

The generated joint goal is only used to label synthetic actions. It is never
included in actor observations and is not used by the deployed policy.
"""

import torch

from .kinematics import MarvinArmFK, orientation_error


def sample_fk_teacher_batch(
    fk: MarvinArmFK,
    home_arm: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    count: int,
    goal_radius_rad: float = 0.12,
    bimanual_fraction: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= bimanual_fraction <= 1.0:
        raise ValueError("bimanual_fraction must be in [0, 1]")
    device = home_arm.device
    current = torch.maximum(
        torch.minimum(home_arm + (2.0 * torch.rand((count, 14), device=device) - 1.0) * 0.08, upper), lower
    )
    mode = torch.randint(0, 2, (count,), device=device)
    if bimanual_fraction:
        mode = torch.where(torch.rand((count,), device=device) < bimanual_fraction, 2, mode)
    mask = torch.stack((mode != 1, mode != 0), dim=-1).float()
    active_joint_mask = mask.repeat_interleave(7, dim=1)
    goal = torch.maximum(
        torch.minimum(current + (2.0 * torch.rand_like(current) - 1.0) * goal_radius_rad * active_joint_mask, upper),
        lower,
    )
    velocity = (2.0 * torch.rand_like(current) - 1.0) * 0.3 * active_joint_mask
    previous_action = (2.0 * torch.rand_like(current) - 1.0) * 0.3 * active_joint_mask
    previous_target = torch.maximum(
        torch.minimum(current + (2.0 * torch.rand_like(current) - 1.0) * 0.02, upper), lower
    )
    current_pose = fk(current)
    goal_pose = fk(goal)
    position_error = goal_pose[..., :3] - current_pose[..., :3]
    rotation_error = orientation_error(goal_pose[..., 3:], current_pose[..., 3:])
    midpoint = (lower + upper) * 0.5
    half_range = (upper - lower) * 0.5
    mode_one_hot = torch.nn.functional.one_hot(mode, num_classes=3).float()
    obs = torch.cat(
        (
            (current - midpoint) / half_range,
            velocity / 0.8,
            position_error.reshape(count, 6),
            rotation_error.reshape(count, 6),
            previous_action,
            (previous_target - midpoint) / half_range,
            mask,
            mode_one_hot,
        ),
        dim=-1,
    )
    action = ((goal - previous_target) / 0.20 - 0.5 * velocity / 0.8).clamp(-1.0, 1.0) * active_joint_mask
    return obs, action
