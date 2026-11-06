"""Strongly typed time and task identity for pre-reset physical transitions."""

from dataclasses import dataclass

import torch

from .specs import RobotSpec, TaskSpec


@dataclass
class TaskState:
    task_id: torch.Tensor
    motion_id: torch.Tensor
    frame: torch.Tensor
    data: torch.Tensor

    def index(self, index) -> "TaskState":
        return TaskState(self.task_id[index], self.motion_id[index],
                         self.frame[index], self.data[index])

    def validate(self, task: TaskSpec, batch: int):
        for name in ("task_id", "motion_id", "frame"):
            value = getattr(self, name)
            if value.shape != (batch,) or value.dtype != torch.long:
                raise ValueError(f"TaskState {name} must be a long vector")
        if self.data.shape != (batch, task.task_data_dim) or not self.data.is_floating_point():
            raise ValueError("TaskState data schema differs from TaskSpec")
        if not bool(torch.isfinite(self.data).all()):
            raise ValueError("TaskState data is nonfinite")


@dataclass
class TransitionBatch:
    state: torch.Tensor
    action: torch.Tensor
    previous_action: torch.Tensor
    executed_command: torch.Tensor
    next_state: torch.Tensor
    history: torch.Tensor
    dynamics_context: torch.Tensor
    task_state: TaskState
    next_task_state: TaskState
    terminated: torch.Tensor
    truncated: torch.Tensor
    source_id: torch.Tensor
    failure_reason: torch.Tensor
    reward: torch.Tensor | None = None

    @property
    def done(self) -> torch.Tensor:
        return self.terminated | self.truncated

    def validate(self, robot: RobotSpec, task: TaskSpec, *, num_envs: int):
        shapes = {"state": (num_envs, robot.state_dim),
                  "next_state": (num_envs, robot.state_dim),
                  "action": (num_envs, robot.action_dim),
                  "previous_action": (num_envs, robot.action_dim),
                  "executed_command": (num_envs, robot.command_dim or robot.action_dim),
                  "history": (num_envs, task.history_length, task.observation_dim),
                  "dynamics_context": (num_envs, robot.context_dim)}
        for name, shape in shapes.items():
            value = getattr(self, name)
            if value.shape != shape or not value.is_floating_point():
                raise ValueError(f"Transition {name} schema differs")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Transition {name} contains nonfinite data")
        self.task_state.validate(task, num_envs)
        self.next_task_state.validate(task, num_envs)
        if (not torch.equal(self.next_task_state.frame, self.task_state.frame + 1)
                or not torch.equal(self.next_task_state.task_id, self.task_state.task_id)
                or not torch.equal(self.next_task_state.motion_id, self.task_state.motion_id)):
            raise ValueError("Transition frame or task identity does not advance consistently")
        for name, dtype in (("terminated", torch.bool), ("truncated", torch.bool),
                            ("source_id", torch.int8), ("failure_reason", torch.int32)):
            value = getattr(self, name)
            if value.shape != (num_envs,) or value.dtype != dtype:
                raise ValueError(f"Transition {name} schema differs")
        if self.reward is not None and (self.reward.shape != (num_envs,)
                                        or not bool(torch.isfinite(self.reward).all())):
            raise ValueError("Transition reward schema or finite value differs")


@dataclass
class WindowBatch:
    state: torch.Tensor
    action: torch.Tensor
    previous_action: torch.Tensor
    executed_command: torch.Tensor
    next_state: torch.Tensor
    history: torch.Tensor
    dynamics_context: torch.Tensor
    task_state: TaskState
    next_task_state: TaskState
    terminated: torch.Tensor
    truncated: torch.Tensor
    done: torch.Tensor
    source_id: torch.Tensor
    failure_reason: torch.Tensor
    reward: torch.Tensor
    episode: torch.Tensor
    mask: torch.Tensor
    env_id: torch.Tensor
    absolute_timestamp: torch.Tensor


@dataclass
class StartBatch:
    state: torch.Tensor
    history: torch.Tensor
    dynamics_context: torch.Tensor
    task_state: TaskState
    previous_action: torch.Tensor
    env_id: torch.Tensor
    episode: torch.Tensor
    source_id: torch.Tensor

