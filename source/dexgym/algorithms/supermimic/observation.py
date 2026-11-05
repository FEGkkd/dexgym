"""Arm-only sensor observation noise shared by real and imagined rollouts."""

import torch


def noisy_proprio_frame(
    frame: torch.Tensor,
    q_scale: torch.Tensor,
    position_rad: float = 0.0,
    velocity_rad_s: float = 0.0,
) -> torch.Tensor:
    """Add uniform q/qd sensor noise in physical units before normalized Actor input."""
    if frame.shape[-1] != 56 or q_scale.shape != (14,):
        raise ValueError("Arm observation frame must be [...,56] with a 14D q scale")
    if position_rad < 0 or velocity_rad_s < 0 or (q_scale <= 0).any():
        raise ValueError("Sensor-noise widths must be nonnegative and q scale positive")
    if position_rad == 0 and velocity_rad_s == 0:
        return frame
    q_noise = (2 * torch.rand_like(frame[..., :14]) - 1) * (position_rad / q_scale)
    qd_noise = (2 * torch.rand_like(frame[..., 14:28]) - 1) * (velocity_rad_s / 2.0)
    return torch.cat((frame[..., :14] + q_noise, frame[..., 14:28] + qd_noise, frame[..., 28:]), dim=-1)
