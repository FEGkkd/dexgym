"""Backend-independent reward and subgoal timing for hand-only tracking."""

import torch

from dexgym.data.wuji_retarget import ROBOT_RIGHT_JOINT_NAMES


def residual_joint_target(
    command_q: torch.Tensor,
    action: torch.Tensor,
    joint_limits: torch.Tensor,
    *,
    scale: float,
    deadzone: float,
) -> torch.Tensor:
    """TELEDEXTER Appendix C.1: accumulate a soft-deadzone action on the last command."""
    if command_q.shape != action.shape or command_q.ndim != 2 or joint_limits.shape != (command_q.shape[1], 2):
        raise ValueError("Expected commands/actions[N,D] and joint limits[D,2]")
    if scale <= 0 or not 0 <= deadzone < 1:
        raise ValueError("Action scale must be positive and deadzone in [0,1)")
    bounded = action.clamp(-1.0, 1.0)
    delta = bounded.sign() * (bounded.abs() - deadzone).clamp_min(0.0)
    return torch.clamp(command_q + scale * delta, min=joint_limits[:, 0], max=joint_limits[:, 1])


def action_toward_joint_target(
    goal_q: torch.Tensor, command_q: torch.Tensor, *, scale: float, deadzone: float
) -> torch.Tensor:
    """Invert the deadzone mapping for privileged joint-replay diagnostics."""
    if goal_q.shape != command_q.shape or goal_q.ndim != 2:
        raise ValueError("Expected matching goal and command arrays[N,D]")
    if scale <= 0 or not 0 <= deadzone < 1:
        raise ValueError("Action scale must be positive and deadzone in [0,1)")
    needed = (goal_q - command_q) / scale
    return (needed.sign() * (needed.abs() + deadzone)).clamp(-1.0, 1.0)


def curriculum_stride_upper(
    steps: torch.Tensor,
    *,
    minimum: int = 20,
    maximum: int = 40,
    anneal_steps: int = 25600,
    sigma_min: float = 0.7,
) -> torch.Tensor:
    """TELEDEXTER C.3 cubic stride schedule, scaled from 60-Hz to 30-Hz references."""
    if minimum < 1 or maximum < minimum or anneal_steps < 1 or not 0 < sigma_min < 1:
        raise ValueError("Invalid curriculum schedule")
    fraction = (steps.float() / anneal_steps).clamp(0.0, 1.0)
    sigma = 1.0 - (1.0 - sigma_min) * fraction
    upper = minimum + (maximum - minimum) * (1.0 - sigma.pow(3)) / (1.0 - sigma_min**3)
    return upper.floor().long().clamp(minimum, maximum)


def sample_goal_span(upper: torch.Tensor) -> torch.Tensor:
    """Sample each forward reference-frame jump uniformly from 1 through its current bound."""
    if upper.ndim != 1 or torch.any(upper < 1):
        raise ValueError("Expected positive per-environment stride upper bounds")
    return torch.floor(torch.rand(upper.shape, device=upper.device) * upper).long() + 1


def teledexter_hand_dwell(
    current_tips: torch.Tensor,
    goal_tips: torch.Tensor,
    dwell: torch.Tensor,
    required: torch.Tensor,
    *,
    tolerance_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Test the all-finger threshold for each environment's sampled hold duration."""
    if current_tips.shape != goal_tips.shape or current_tips.shape[-2:] != (5, 3):
        raise ValueError("Expected matching fingertip positions[N,5,3]")
    if dwell.shape != required.shape or dwell.shape != (len(current_tips),) or tolerance_m <= 0:
        raise ValueError("Expected dwell and required[N] with a positive tolerance")
    errors = torch.linalg.vector_norm(current_tips - goal_tips, dim=-1)
    within = (errors < tolerance_m).all(dim=-1)
    next_dwell = torch.where(within, dwell + 1, torch.zeros_like(dwell))
    reached = next_dwell >= required
    return errors, within, reached, next_dwell


def teledexter_hand_reward(
    errors: torch.Tensor,
    reached: torch.Tensor,
    goal_span: torch.Tensor,
    *,
    dense_scale: float = 0.1,
    time_cost: float = 0.1,
) -> torch.Tensor:
    """Reward the completed active goal; span zero marks a cross-trajectory target."""
    if errors.ndim != 2 or errors.shape[1] != 5 or any(item.shape != (len(errors),) for item in (reached, goal_span)):
        raise ValueError("Expected finger errors[N,5] and per-environment reward inputs[N]")
    decay = errors.new_tensor([100.0, 90.0, 90.0, 90.0, 90.0])
    score = 1.5 * 0.5 * torch.exp(-errors * decay).sum(dim=-1)
    step_weight = torch.where(goal_span == 0, 100.0, goal_span.float() + 5.0)
    dense = torch.exp(-50.0 * errors).mean(dim=-1)
    return reached.float() * step_weight * score + dense_scale * dense - time_cost


def update_failure_count(failures: torch.Tensor, within: torch.Tensor, reached: torch.Tensor) -> torch.Tensor:
    """Count out-of-tolerance policy frames until the next subgoal is hit."""
    if failures.shape != within.shape or failures.shape != reached.shape:
        raise ValueError("Expected matching failure, threshold, and reach arrays")
    return torch.where(reached, torch.zeros_like(failures), failures + (~within).long())


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
