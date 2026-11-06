"""Task-specific observation, goal, reward and episode semantics."""

from typing import Protocol

import torch

from dexgym.algorithms.supermimic.core.specs import TaskSpec
from dexgym.algorithms.supermimic.core.transition import TaskState


class TaskMDP(Protocol):
    spec: TaskSpec
    termination_mode: str
    success_hold_steps: int

    def advance(self, task_state: TaskState) -> TaskState: ...

    def observation(self, state: torch.Tensor, previous_action: torch.Tensor,
                    task_state: TaskState) -> torch.Tensor: ...

    def goal(self, state: torch.Tensor, task_state: TaskState) -> torch.Tensor: ...

    def critic_features(self, state: torch.Tensor, task_state: TaskState) -> torch.Tensor: ...

    def reward(self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor,
               next_task_state: TaskState, previous_action: torch.Tensor) -> torch.Tensor: ...

    def soft_termination(self, next_state: torch.Tensor,
                         next_task_state: TaskState) -> torch.Tensor: ...

    def termination_features(self, next_state: torch.Tensor,
                             next_task_state: TaskState) -> torch.Tensor: ...

    def valid_imagination_start(self, task_state: TaskState, horizon: int) -> torch.Tensor: ...

    def evaluate_metrics(self, trajectory) -> dict: ...

    def success_mask(self, transition) -> torch.Tensor: ...

    def checkpoint_contract(self) -> dict: ...

