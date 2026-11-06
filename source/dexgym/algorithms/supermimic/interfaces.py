"""Contracts consumed by the shared trainer; task code supplies concrete behavior."""

from typing import Protocol

import torch


class ReferenceProvider(Protocol):
    length: int

    def reference_at(self, frame: torch.Tensor) -> dict[str, torch.Tensor]: ...

    def reference_features_at(self, frame: torch.Tensor) -> torch.Tensor: ...

    def future_goal_at(self, frame: torch.Tensor, count: int = 10) -> torch.Tensor: ...


class RobotDynamicsAdapter(Protocol):
    state_dim: int
    action_dim: int
    action_schema: str
    action_map: object
    fk: object

    def interpolate_command(self, previous: torch.Tensor, current: torch.Tensor,
                            substep: int) -> torch.Tensor: ...


class TaskMDPAdapter(Protocol):
    reference: ReferenceProvider
    robot: RobotDynamicsAdapter
    termination_mode: str

    def arrival_goal(self, frame: torch.Tensor) -> torch.Tensor: ...

    def imagine(self, actor, world_model, critic, start: dict,
                *, horizon: int) -> dict: ...

    def critic_batch(self, critic, rollout: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


class DynamicsWorldModel(Protocol):
    def __call__(self, state: torch.Tensor, action: torch.Tensor, mapping) -> torch.Tensor: ...

    def autoregressive_loss(self, windows: dict, mapping, **kwargs) -> tuple[torch.Tensor, torch.Tensor]: ...


class PhysicalEvaluator(Protocol):
    def evaluate(self, actor, output_dir) -> dict: ...
