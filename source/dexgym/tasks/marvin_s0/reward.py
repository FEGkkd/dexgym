"""Pose scores that require position and orientation to improve together."""

import torch


def balanced_pose_score(
    position_error_m: torch.Tensor,
    orientation_error_rad: torch.Tensor,
    position_scale_m: float,
    orientation_scale_rad: float,
) -> torch.Tensor:
    """Harmonic mean of exponential position and rotation tracking scores."""
    position_score = torch.exp(-position_error_m / position_scale_m)
    orientation_score = torch.exp(-orientation_error_rad / orientation_scale_rad)
    return 2.0 * position_score * orientation_score / (position_score + orientation_score + 1e-8)
