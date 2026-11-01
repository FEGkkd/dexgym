"""Pure numerical acceptance and reward contracts for the unified task."""

import torch

from dexteroushands_lab.tasks.direct.bimanual_bottle.reward_terms import cap_turn_physics_valid


MAX_OPENING_STEP_RAD = 0.25


def capture_hold_anchor(anchor: torch.Tensor, latched: torch.Tensor,
                        bottle_pose: torch.Tensor,
                        grasp_established: torch.Tensor,
                        *, max_catch_drift_m: float = 0.02):
    """Capture the bottle frame once at first valid grasp; never chase a loss."""
    if (anchor.ndim != 2 or anchor.shape[-1] != 7 or
            bottle_pose.shape != anchor.shape or
            latched.shape != anchor.shape[:1] or
            grasp_established.shape != latched.shape or max_catch_drift_m <= 0):
        raise ValueError("Anchor state has inconsistent shapes or drift limit")
    in_reach = torch.linalg.vector_norm(
        bottle_pose[:, :3] - anchor[:, :3], dim=-1) <= max_catch_drift_m
    take = (~latched & grasp_established & in_reach
            & torch.isfinite(bottle_pose).all(dim=-1))
    return torch.where(take[:, None], bottle_pose, anchor), latched | take


def valid_turn_step(opening: torch.Tensor, previous_opening: torch.Tensor,
                    axial_travel: torch.Tensor, expected_travel: torch.Tensor,
                    constraint_error: torch.Tensor, stable_bilateral_support: torch.Tensor,
                    *, goal_rad: float, pitch_tolerance_m: float = 0.00005,
                    constraint_tolerance_m: float = 0.001) -> torch.Tensor:
    """Physical 30-degree gate with stability and a finite angle-continuity bound."""
    if previous_opening.shape != opening.shape or stable_bilateral_support.shape != opening.shape:
        raise ValueError("Turn state dimensions differ")
    continuous = (torch.isfinite(previous_opening)
                  & ((opening - previous_opening).abs() <= MAX_OPENING_STEP_RAD))
    return cap_turn_physics_valid(
        opening, axial_travel, expected_travel, constraint_error,
        stable_bilateral_support & continuous, goal_rad=goal_rad,
        pitch_tolerance_m=pitch_tolerance_m,
        constraint_tolerance_m=constraint_tolerance_m)


def bounded_turn_progress(current_opening: torch.Tensor,
                          previous_opening: torch.Tensor,
                          stable_bilateral_support: torch.Tensor,
                          *, goal_rad: float) -> torch.Tensor:
    """Credit only supported real positive progress up to 30 degrees."""
    if goal_rad <= 0 or current_opening.shape != previous_opening.shape or stable_bilateral_support.shape != current_opening.shape:
        raise ValueError("Turn progress inputs differ or goal is nonpositive")
    delta = (current_opening.clamp(max=goal_rad)
             - previous_opening.clamp(max=goal_rad)).clamp(-MAX_OPENING_STEP_RAD,
                                                           MAX_OPENING_STEP_RAD)
    valid = (torch.isfinite(current_opening) & torch.isfinite(previous_opening)
             & ((current_opening - previous_opening).abs() <= MAX_OPENING_STEP_RAD))
    signed = 20.0 * delta
    reward = torch.where(stable_bilateral_support, signed,
                         signed.clamp(max=0))
    return torch.where(valid, reward, torch.full_like(reward, -1.0))
