"""Action-contract helpers for the Wuji arm-hand Manager-Based task."""

from __future__ import annotations

import torch


def split_arm_hand_action(action: torch.Tensor, arm_dof: int, hand_dof: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split one canonical policy action into arm and hand position targets."""

    expected = arm_dof + hand_dof
    if action.ndim != 2 or action.shape[-1] != expected:
        raise ValueError(f"Expected action shape [N, {expected}], got {tuple(action.shape)}")
    return action[..., :arm_dof], action[..., arm_dof:]
