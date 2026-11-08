"""Backend-independent reward and subgoal timing for hand-only tracking."""

import torch

from dexgym.data.wuji_retarget import ROBOT_RIGHT_JOINT_NAMES


def right_hand_joint_indices(joint_names: tuple[str, ...] | list[str]) -> list[int]:
    """Map SDK firmware order to the corresponding 20 robot articulation joints."""
    names = list(joint_names)
    if len(set(names)) != len(names) or not all(name in names for name in ROBOT_RIGHT_JOINT_NAMES):
        raise ValueError("Robot does not contain the expected unique right-hand joints")
    return [names.index(name) for name in ROBOT_RIGHT_JOINT_NAMES]


def build_right_hand_observation(
    q: torch.Tensor,
    qdot: torch.Tensor,
    tips: torch.Tensor,
    goal_tips: torch.Tensor,
    previous_action: torch.Tensor,
    *,
    command_q: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the original 105D policy input, optionally exposing the PD target gap."""
    if q.ndim != 2 or q.shape[-1] != 20 or qdot.shape != q.shape or previous_action.shape != q.shape:
        raise ValueError("Expected matching right-hand joint and action tensors[N,20]")
    if tips.shape != (len(q), 5, 3) or goal_tips.shape != tips.shape:
        raise ValueError("Expected matching fingertip tensors[N,5,3]")
    pieces = [q, qdot, tips.flatten(1), goal_tips.flatten(1), (goal_tips - tips).flatten(1), previous_action]
    if command_q is not None:
        if command_q.shape != q.shape:
            raise ValueError("Expected position command[N,20]")
        pieces.append(command_q - q)
    return torch.cat(pieces, dim=-1)


def fingertips_in_wrist_frame(
    body_poses_w: torch.Tensor,
    wrist_body_id: int,
    tip_body_ids: list[int],
    tip_local_offsets: torch.Tensor,
) -> torch.Tensor:
    """Evaluate five pinned Wuji fingertip sites from link poses in XYZW order.

    Source: https://isaac-sim.github.io/IsaacLab/v3.0.0/source/api/lab/isaaclab.assets.html
    """
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    if body_poses_w.ndim != 3 or body_poses_w.shape[-1] != 7 or len(tip_body_ids) != 5:
        raise ValueError("Expected body poses[N,B,7] and five fingertip body IDs")
    if tip_local_offsets.shape != (5, 3):
        raise ValueError("Expected five fingertip local offsets")
    distal = body_poses_w[:, tip_body_ids]
    wrist = body_poses_w[:, wrist_body_id]
    offsets = tip_local_offsets[None].expand(len(body_poses_w), -1, -1)
    tips_w = distal[..., :3] + quat_apply(distal[..., 3:], offsets)
    wrist_quat = wrist[:, None, 3:].expand(-1, 5, -1)
    return quat_apply_inverse(wrist_quat, tips_w - wrist[:, None, :3])


def tracking_step(
    current_tips: torch.Tensor,
    goal_tips: torch.Tensor,
    dwell: torch.Tensor,
    action_delta: torch.Tensor,
    *,
    tolerance_m: float = 0.015,
    dwell_steps: int = 2,
    sparse_bonus: float = 10.0,
    dense_scale: float = 1.0,
    dense_decay: float = 50.0,
    action_rate_penalty: float = 0.02,
    time_cost: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reward five-finger tracking and detect consecutive reached-goal dwell."""
    if current_tips.shape != goal_tips.shape or current_tips.shape[-2:] != (5, 3):
        raise ValueError("Expected matching fingertip arrays[N,5,3]")
    if action_delta.shape != (len(current_tips), 20) or dwell.shape != (len(current_tips),):
        raise ValueError("Expected action delta[N,20] and dwell[N]")
    error = torch.linalg.vector_norm(current_tips - goal_tips, dim=-1)
    within = torch.all(error < tolerance_m, dim=-1)
    next_dwell = torch.where(within, dwell + 1, torch.zeros_like(dwell))
    reached = next_dwell >= dwell_steps
    dense = torch.exp(-dense_decay * error).mean(dim=-1)
    action_cost = action_delta.square().mean(dim=-1)
    reward = sparse_bonus * reached.float() + dense_scale * dense - action_rate_penalty * action_cost - time_cost
    return reward, reached, next_dwell, error.mean(dim=-1)
