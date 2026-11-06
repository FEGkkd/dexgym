"""Robot-specific state integration and physical control contract."""

from typing import Protocol

import torch

from dexgym.algorithms.supermimic.core.specs import RobotSpec


class RobotDynamicsAdapter(Protocol):
    spec: RobotSpec

    def integrate(self, state: torch.Tensor, physical_delta: torch.Tensor,
                  action: torch.Tensor, context: torch.Tensor) -> torch.Tensor: ...

    def physical_delta(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor: ...

    def executed_command(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor: ...

    def physical_state(self, env) -> torch.Tensor: ...

