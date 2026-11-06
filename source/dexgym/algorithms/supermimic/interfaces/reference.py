"""Optional time-indexed reference contract used by tracking tasks only."""

from typing import Protocol

import torch


class ReferenceProvider(Protocol):
    lengths: torch.Tensor
    fps: torch.Tensor

    def reference_at(self, motion_id: torch.Tensor, frame: torch.Tensor) -> dict[str, torch.Tensor]: ...

    def future_goal_at(self, motion_id: torch.Tensor, frame: torch.Tensor,
                       count: int) -> torch.Tensor: ...

