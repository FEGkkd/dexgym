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

    def dynamics_context(self, env) -> torch.Tensor:
        """Return finite [N, context_dim] parameters active during the next real action."""
        ...

    def next_dynamics_context(self, context: torch.Tensor, state: torch.Tensor,
                              action: torch.Tensor, next_state: torch.Tensor,
                              next_task_state) -> torch.Tensor:
        """Required when RobotSpec.context_evolution is adapter_transition."""
        ...

