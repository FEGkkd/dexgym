"""Small bounded reward primitives used by both Phase 1 tasks."""

import torch
import math


def digit_contact_score(forces: torch.Tensor, body_names: tuple[str, ...],
                        digits: tuple[str, ...], scale_n: float = 1.0,
                        aggregation: str = "mean", balanced_weight: float = 0.8) -> torch.Tensor:
    """Bounded correct-object contact; balanced mode favors all requested digits."""
    if forces.ndim != 2 or forces.shape[1] != len(body_names) or scale_n <= 0:
        raise ValueError("Contact forces/body order mismatch")
    parts = []
    for digit in digits:
        selected = [index for index, name in enumerate(body_names) if digit in name]
        if not selected:
            raise ValueError(f"Missing digit in body names: {digit}")
        magnitude = forces[:, selected].sum(dim=-1).clamp_min(0)
        parts.append(1 - torch.exp(-magnitude / scale_n))
    per_digit = torch.stack(parts, dim=-1)
    mean = per_digit.mean(dim=-1)
    if aggregation == "mean":
        return mean
    if aggregation == "balanced":
        if not 0 <= balanced_weight <= 1:
            raise ValueError("balanced_weight must be in [0, 1]")
        geometric = per_digit.prod(dim=-1).clamp_min(0).pow(1.0 / len(digits))
        return (1.0 - balanced_weight) * mean + balanced_weight * geometric
    raise ValueError(f"Unknown contact aggregation: {aggregation}")


def cylinder_surface_proximity(local_sites: torch.Tensor, *, radius_m: float,
                               half_height_m: float, radial_scale_m: float,
                               axial_scale_m: float, axial_inset_m: float = 0.0,
                               digit_count: int = 5) -> torch.Tensor:
    """Bounded fingertip proximity to a cylinder's interior sidewall band."""
    if (local_sites.ndim != 3 or local_sites.shape[-1] != 3 or
            not 1 <= digit_count <= local_sites.shape[1] or
            radius_m <= 0 or half_height_m <= axial_inset_m or
            radial_scale_m <= 0 or axial_scale_m <= 0 or axial_inset_m < 0):
        raise ValueError("Invalid cylinder proximity shape or geometry")
    sites = local_sites[:, :digit_count]
    radial = torch.linalg.vector_norm(sites[..., :2], dim=-1)
    radial_score = torch.exp(-(radial - radius_m).abs() / radial_scale_m)
    axial_excess = torch.relu(sites[..., 2].abs() - (half_height_m - axial_inset_m))
    axial_score = torch.exp(-axial_excess / axial_scale_m)
    return (radial_score * axial_score).mean(dim=-1)


def signed_progress_reward(current_open: torch.Tensor,
                           previous_open: torch.Tensor) -> torch.Tensor:
    """Real cap-vs-bottle angle progress; reverse motion is negative."""
    return 20.0 * (current_open - previous_open).clamp(-0.05, 0.05)


def physically_supported_turn_reward(current_open: torch.Tensor,
                                     previous_open: torch.Tensor,
                                     axial_error_m: torch.Tensor,
                                     turn_support: torch.Tensor,
                                     *, axial_tolerance_m: float = 0.001) -> torch.Tensor:
    """Credit positive cap rotation only with bilateral contact and thread travel.

    Reverse rotation remains penalized even when support is absent. The
    tolerance matches the separately enforced 30-degree success condition.
    """
    if axial_tolerance_m <= 0:
        raise ValueError("Axial tolerance must be positive")
    signed = signed_progress_reward(current_open, previous_open)
    supported = turn_support & (axial_error_m.abs() < axial_tolerance_m)
    return signed.clamp_max(0) + signed.clamp_min(0) * supported.float()


def thread_opening_progress(theta: torch.Tensor, axial: torch.Tensor,
                            theta_at_reset: torch.Tensor, axial_at_reset: torch.Tensor,
                            *, sign: int, pitch: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Opening and local-axis travel measured from the same episode reset."""
    opening = sign * (theta - theta_at_reset)
    travel = axial - axial_at_reset
    expected = pitch * opening / (2 * math.pi)
    return opening, travel, expected


def terminal_failure_penalty(failure: torch.Tensor) -> torch.Tensor:
    """Cost of dropping or numerically breaking the free bottle assembly."""
    return -5.0 * failure.float()


def suspended_bottle(position: torch.Tensor, quaternion_xyzw: torch.Tensor,
                     reset_position: torch.Tensor, reset_quaternion_xyzw: torch.Tensor,
                     env_origin: torch.Tensor) -> torch.Tensor:
    """Reachable suspended workspace, allowing a pregrasp to catch a falling bottle."""
    orientation_dot = (quaternion_xyzw * reset_quaternion_xyzw).sum(dim=-1).abs().clamp(max=1)
    tilt = 2.0 * torch.acos(orientation_dot)
    planar = torch.linalg.vector_norm((position - env_origin)[:, :2], dim=-1)
    reset_planar = torch.linalg.vector_norm((position - reset_position)[:, :2], dim=-1)
    return ((position[:, 2] - env_origin[:, 2] > 0.20)
            & (position[:, 2] - env_origin[:, 2] < 0.80)
            & (planar < 0.30) & (reset_planar < 0.30) & (tilt < 0.35))


def bottle_hold_stability(
    bottle_pose_w: torch.Tensor, bottle_velocity_w: torch.Tensor,
    hold_anchor_pose_w: torch.Tensor,
    cap_spin_velocity_rad_s: torch.Tensor | None = None,
    *, max_drift_m: float = 0.02, max_tilt_rad: float = math.radians(10),
    max_linear_speed_m_s: float = 0.05,
    max_angular_speed_rad_s: float = 0.5,
    max_cap_spin_rad_s: float = 0.2,
) -> torch.Tensor:
    """Bounded motion during a hold window; cap spin is optional for turning."""
    n = bottle_pose_w.shape[0]
    if (bottle_pose_w.shape != (n, 7) or bottle_velocity_w.shape != (n, 6)
            or hold_anchor_pose_w.shape != (n, 7)
            or (cap_spin_velocity_rad_s is not None
                and cap_spin_velocity_rad_s.shape != (n,))
            or min(max_drift_m, max_tilt_rad, max_linear_speed_m_s,
                   max_angular_speed_rad_s, max_cap_spin_rad_s) <= 0):
        raise ValueError("Hold stability state or positive limits are invalid")
    dot = (bottle_pose_w[:, 3:] * hold_anchor_pose_w[:, 3:]).sum(dim=-1).abs().clamp(max=1)
    tilt = 2.0 * torch.acos(dot)
    valid = (torch.linalg.vector_norm(
        bottle_pose_w[:, :3] - hold_anchor_pose_w[:, :3], dim=-1) < max_drift_m)
    valid &= tilt < max_tilt_rad
    valid &= torch.linalg.vector_norm(bottle_velocity_w[:, :3], dim=-1) < max_linear_speed_m_s
    valid &= torch.linalg.vector_norm(bottle_velocity_w[:, 3:], dim=-1) < max_angular_speed_rad_s
    valid &= torch.isfinite(bottle_pose_w).all(dim=-1)
    valid &= torch.isfinite(bottle_velocity_w).all(dim=-1)
    if cap_spin_velocity_rad_s is not None:
        valid &= cap_spin_velocity_rad_s.abs() < max_cap_spin_rad_s
        valid &= torch.isfinite(cap_spin_velocity_rad_s)
    return valid


def grasp_contact_reward(left_score: torch.Tensor, right_score: torch.Tensor) -> torch.Tensor:
    """Reward each correct contact and give a larger bonus to balanced use."""
    return (0.5 * (left_score + right_score)
            + 0.8 * torch.sqrt((left_score * right_score).clamp_min(0)))


def physically_valid_grasp(left_contact: torch.Tensor, right_contact: torch.Tensor,
                           suspended: torch.Tensor, low_speed: torch.Tensor,
                           thread_constraint_error_m: torch.Tensor,
                           tolerance_m: float = 0.001) -> torch.Tensor:
    """Reject a bilateral hold if cap and bottle violate the thread constraint."""
    if tolerance_m <= 0:
        raise ValueError("Thread constraint tolerance must be positive")
    return (left_contact & right_contact & suspended & low_speed
            & torch.isfinite(thread_constraint_error_m)
            & (thread_constraint_error_m.abs() < tolerance_m))


def thread_error_penalty(error_m: torch.Tensor, *, tolerance_m: float = 0.001) -> torch.Tensor:
    """Bounded penalty for hand loading that exceeds the thread travel tolerance."""
    if tolerance_m <= 0:
        raise ValueError("Thread constraint tolerance must be positive")
    magnitude = torch.where(torch.isfinite(error_m), error_m.abs(),
                            torch.full_like(error_m, float("inf")))
    excess = torch.relu(magnitude - tolerance_m)
    return torch.exp(-excess / tolerance_m) - 1.0


def cap_turn_physics_valid(opening_rad: torch.Tensor, axial_travel_m: torch.Tensor,
                           expected_travel_m: torch.Tensor,
                           constraint_error_m: torch.Tensor,
                           bilateral_grasp: torch.Tensor, *, goal_rad: float,
                           pitch_tolerance_m: float = 0.00005,
                           constraint_tolerance_m: float = 0.001) -> torch.Tensor:
    """Check signed thread travel as well as grasp, angle and D6 coupling."""
    if goal_rad <= 0 or pitch_tolerance_m <= 0 or constraint_tolerance_m <= 0:
        raise ValueError("Cap goal and physical tolerances must be positive")
    return (bilateral_grasp & (opening_rad >= goal_rad)
            & (axial_travel_m > 0)
            & ((axial_travel_m - expected_travel_m).abs() < pitch_tolerance_m)
            & (constraint_error_m.abs() < constraint_tolerance_m)
            & torch.isfinite(opening_rad) & torch.isfinite(axial_travel_m)
            & torch.isfinite(expected_travel_m) & torch.isfinite(constraint_error_m))
