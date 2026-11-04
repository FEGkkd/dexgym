from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ContractError(ValueError):
    """Raised when a task data contract is internally inconsistent."""


@dataclass(frozen=True)
class ObservationField:
    name: str
    size: int
    frame: str | None = None
    description: str = ""

    def __post_init__(self):
        if self.size <= 0:
            raise ContractError(f"Observation field {self.name!r} must have a positive size.")


@dataclass(frozen=True)
class TaskDataContract:
    """Backend-independent action and observation schema."""

    task_id: str
    asset_variant: str
    joint_order: tuple[str, ...]
    action_mode: str
    control_dt: float
    physics_dt: float
    observation_fields: tuple[ObservationField, ...]
    observation_history_length: int = 1
    quaternion_convention: str = "wxyz"
    version: str = "dexteroushands-v1"
    critic_observation_dim: int | None = None

    def __post_init__(self):
        if not self.task_id:
            raise ContractError("task_id must not be empty.")
        if not self.joint_order:
            raise ContractError("joint_order must not be empty.")
        if len(set(self.joint_order)) != len(self.joint_order):
            raise ContractError("joint_order contains duplicate joint names.")
        if self.control_dt <= 0 or self.physics_dt <= 0:
            raise ContractError("control_dt and physics_dt must be positive.")
        if self.control_dt < self.physics_dt:
            raise ContractError("control_dt cannot be smaller than physics_dt.")
        if self.observation_history_length < 1:
            raise ContractError("observation_history_length must be positive.")
        if self.quaternion_convention != "wxyz":
            raise ContractError("Only the explicit wxyz convention is supported in v1.")
        if self.critic_observation_dim is not None and self.critic_observation_dim <= 0:
            raise ContractError("critic_observation_dim must be positive when provided.")

    @property
    def action_dim(self) -> int:
        return len(self.joint_order)

    @property
    def frame_observation_dim(self) -> int:
        return sum(field.size for field in self.observation_fields)

    @property
    def observation_dim(self) -> int:
        return self.frame_observation_dim * self.observation_history_length

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "task_id": self.task_id,
            "asset_variant": self.asset_variant,
            "joint_order": list(self.joint_order),
            "action_mode": self.action_mode,
            "control_dt": self.control_dt,
            "physics_dt": self.physics_dt,
            "quaternion_convention": self.quaternion_convention,
            "observation_history_length": self.observation_history_length,
            "frame_observation_dim": self.frame_observation_dim,
            "observation_fields": [
                {
                    "name": field.name,
                    "size": field.size,
                    "frame": field.frame,
                    "description": field.description,
                }
                for field in self.observation_fields
            ],
            "action_dim": self.action_dim,
            "observation_dim": self.observation_dim,
            "critic_observation_dim": self.critic_observation_dim,
        }
