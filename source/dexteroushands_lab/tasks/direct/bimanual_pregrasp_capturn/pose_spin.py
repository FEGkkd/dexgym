"""Independent pose-based bottle/cap spin diagnostics, with xyzw quaternions."""

import torch

from dexteroushands_lab.assets.batched_wrist_pose import _mul


def relative_pose_yaw(bottle_quat_xyzw: torch.Tensor,
                      cap_quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Signed twist about the bottle-local z axis from actual object poses."""
    if bottle_quat_xyzw.shape != cap_quat_xyzw.shape or bottle_quat_xyzw.shape[-1] != 4:
        raise ValueError("Object quaternion shapes must match and end in 4")
    inverse_bottle = torch.cat((-bottle_quat_xyzw[..., :3],
                                bottle_quat_xyzw[..., 3:]), dim=-1)
    relative = _mul(inverse_bottle, cap_quat_xyzw)
    raw = 2.0 * torch.atan2(relative[..., 2], relative[..., 3])
    return torch.atan2(torch.sin(raw), torch.cos(raw))


def wrapped_delta(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    """One-step signed change, continuous through the ±pi representation seam."""
    if current.shape != previous.shape:
        raise ValueError("Angle shapes must match")
    change = current - previous
    return torch.atan2(torch.sin(change), torch.cos(change))
