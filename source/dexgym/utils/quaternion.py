"""Explicit quaternion boundary conversion: MuJoCo WXYZ, Isaac Lab 3.0 XYZW."""

import torch


def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    if q.shape[-1] != 4:
        raise ValueError("Quaternion last dimension must be 4")
    return q[..., [1, 2, 3, 0]]


def xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    if q.shape[-1] != 4:
        raise ValueError("Quaternion last dimension must be 4")
    return q[..., [3, 0, 1, 2]]
