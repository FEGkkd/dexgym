"""Observation-contract helpers for the Wuji arm-hand task."""

from __future__ import annotations

import torch


def build_arm_hand_observation(
    arm_state: torch.Tensor,
    hand_state: torch.Tensor,
    object_state: torch.Tensor,
    goal_state: torch.Tensor,
    previous_action: torch.Tensor,
) -> torch.Tensor:
    """Concatenate named Manager-Based observation blocks in canonical order."""

    blocks = (arm_state, hand_state, object_state, goal_state, previous_action)
    if any(block.ndim != 2 for block in blocks):
        raise ValueError("All arm-hand observation blocks must be rank-2 tensors.")
    if len({block.shape[0] for block in blocks}) != 1:
        raise ValueError("All arm-hand observation blocks must have the same environment count.")
    return torch.cat(blocks, dim=-1)
