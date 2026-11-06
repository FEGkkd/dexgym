"""Schema-driven GPU ring replay of real pre-reset transitions and episode-safe windows."""

from dataclasses import asdict

import torch

from .specs import RobotSpec, TaskSpec
from .transition import StartBatch, TaskState, TransitionBatch, WindowBatch


class SchemaReplay:
    def __init__(self, capacity_steps: int, num_envs: int, robot: RobotSpec,
                 task: TaskSpec, device: str | torch.device):
        if capacity_steps < 1 or num_envs < 1:
            raise ValueError("Replay capacity and environment count must be positive")
        self.capacity, self.num_envs = capacity_steps, num_envs
        self.robot_spec, self.task_spec = robot, task
        self.device = torch.device(device)
        scalar = ()
        self.field_schema = {
            "state": ((robot.state_dim,), torch.float32),
            "action": ((robot.action_dim,), torch.float32),
            "previous_action": ((robot.action_dim,), torch.float32),
            "executed_command": ((robot.command_dim or robot.action_dim,), torch.float32),
            "next_state": ((robot.state_dim,), torch.float32),
            "history": ((task.history_length, task.observation_dim), torch.float32),
            "dynamics_context": ((robot.context_dim,), torch.float32),
            "task_id": (scalar, torch.long), "motion_id": (scalar, torch.long),
            "frame": (scalar, torch.long), "task_data": ((task.task_data_dim,), torch.float32),
            "next_task_id": (scalar, torch.long), "next_motion_id": (scalar, torch.long),
            "next_frame": (scalar, torch.long), "next_task_data": ((task.task_data_dim,), torch.float32),
            "terminated": (scalar, torch.bool), "truncated": (scalar, torch.bool),
            "done": (scalar, torch.bool), "source_id": (scalar, torch.int8),
            "failure_reason": (scalar, torch.int32), "reward": (scalar, torch.float32),
            "episode": (scalar, torch.long),
        }
        self.data = {
            name: torch.zeros((capacity_steps, num_envs, *tail), device=self.device, dtype=dtype)
            for name, (tail, dtype) in self.field_schema.items()
        }
        self.timestamp = torch.full((capacity_steps,), -1, device=self.device, dtype=torch.long)
        self.episode_counter = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.steps = 0

    @property
    def bytes_allocated(self) -> int:
        return sum(value.numel() * value.element_size() for value in
                   (*self.data.values(), self.timestamp, self.episode_counter))

    def __getattr__(self, name):
        # Preserve simple field access without exposing an untyped append API.
        if "data" in self.__dict__ and name in self.data:
            return self.data[name]
        raise AttributeError(name)

    def append(self, batch: TransitionBatch):
        batch.validate(self.robot_spec, self.task_spec, num_envs=self.num_envs)
        slot = self.steps % self.capacity
        fields = {
            "state": batch.state, "action": batch.action,
            "previous_action": batch.previous_action,
            "executed_command": batch.executed_command,
            "next_state": batch.next_state, "history": batch.history,
            "dynamics_context": batch.dynamics_context,
            "task_id": batch.task_state.task_id,
            "motion_id": batch.task_state.motion_id,
            "frame": batch.task_state.frame,
            "task_data": batch.task_state.data,
            "next_task_id": batch.next_task_state.task_id,
            "next_motion_id": batch.next_task_state.motion_id,
            "next_frame": batch.next_task_state.frame,
            "next_task_data": batch.next_task_state.data,
            "terminated": batch.terminated, "truncated": batch.truncated,
            "done": batch.done, "source_id": batch.source_id,
            "failure_reason": batch.failure_reason,
            "reward": (torch.zeros(self.num_envs, device=self.device) if batch.reward is None
                       else batch.reward),
            "episode": self.episode_counter,
        }
        for name, value in fields.items():
            self.data[name][slot].copy_(value.detach().to(self.device))
        self.timestamp[slot] = self.steps
        self.episode_counter += batch.done.to(self.device).long()
        self.steps += 1

    def _active_candidates(self, *, env_range=None, source_id=None, task_id=None,
                           motion_id=None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.steps == 0:
            raise ValueError("Replay is empty")
        low, high = env_range or (0, self.num_envs)
        if not 0 <= low < high <= self.num_envs:
            raise ValueError("Replay environment range is invalid")
        times = torch.arange(max(0, self.steps - self.capacity), self.steps, device=self.device)
        envs = torch.arange(low, high, device=self.device)
        slots = times[:, None] % self.capacity
        selected = torch.ones((len(times), len(envs)), device=self.device, dtype=torch.bool)
        for name, requested in (("source_id", source_id), ("task_id", task_id),
                                ("motion_id", motion_id)):
            if requested is not None:
                selected &= self.data[name][slots, envs[None, :]] == requested
        row, col = selected.nonzero(as_tuple=True)
        if not len(row):
            raise ValueError("Replay has no matching real transitions")
        return times[row], envs[col]

    def window_at(self, times: torch.Tensor, envs: torch.Tensor, horizon: int) -> WindowBatch:
        if times.ndim != 1 or envs.shape != times.shape or horizon < 1 or len(times) < 1:
            raise ValueError("Window start and environment vectors are invalid")
        times, envs = times.to(self.device), envs.to(self.device)
        if bool(((times < max(0, self.steps - self.capacity)) | (times >= self.steps)
                 | (envs < 0) | (envs >= self.num_envs)).any()):
            raise ValueError("Window start was overwritten or is outside Replay")
        absolute = times[:, None] + torch.arange(horizon, device=self.device)
        slots = absolute % self.capacity
        indices = (slots, envs[:, None])
        gathered = {name: value[indices] for name, value in self.data.items()}
        present = ((absolute < self.steps) & (absolute >= max(0, self.steps - self.capacity))
                   & (self.timestamp[slots] == absolute))
        initial_slot = times % self.capacity
        trajectory = (
            (gathered["episode"] == self.episode[initial_slot, envs][:, None])
            & (gathered["task_id"] == self.task_id[initial_slot, envs][:, None])
            & (gathered["motion_id"] == self.motion_id[initial_slot, envs][:, None])
            & (gathered["frame"] == self.frame[initial_slot, envs][:, None]
               + torch.arange(horizon, device=self.device))
        )
        prior_alive = torch.cumprod(torch.cat((
            torch.ones((len(times), 1), device=self.device, dtype=torch.long),
            (~gathered["done"][:, :-1]).long()), dim=1), dim=1).bool()
        mask = present & trajectory & prior_alive
        if not bool(mask[:, 0].all()):
            raise ValueError("Window start was overwritten or invalid")
        for name, value in gathered.items():
            keep = mask.reshape(*mask.shape, *((1,) * (value.ndim - 2)))
            gathered[name] = torch.where(keep, value, torch.zeros_like(value))
        state = TaskState(gathered["task_id"], gathered["motion_id"],
                          gathered["frame"], gathered["task_data"])
        next_task_state = TaskState(gathered["next_task_id"], gathered["next_motion_id"],
                                   gathered["next_frame"], gathered["next_task_data"])
        return WindowBatch(
            state=gathered["state"], action=gathered["action"],
            previous_action=gathered["previous_action"],
            executed_command=gathered["executed_command"],
            next_state=gathered["next_state"], history=gathered["history"],
            dynamics_context=gathered["dynamics_context"], task_state=state,
            next_task_state=next_task_state, terminated=gathered["terminated"],
            truncated=gathered["truncated"], done=gathered["done"],
            source_id=gathered["source_id"], failure_reason=gathered["failure_reason"],
            reward=gathered["reward"], episode=gathered["episode"],
            mask=mask.float(), env_id=envs, absolute_timestamp=absolute)

    def sample_windows(self, batch: int, horizon: int = 8, *, env_range=None,
                       source_id=None, task_id=None, motion_id=None,
                       require_full: bool = False) -> WindowBatch:
        if batch < 1 or horizon < 1:
            raise ValueError("Window batch and horizon must be positive")
        times, envs = self._active_candidates(env_range=env_range, source_id=source_id,
                                              task_id=task_id, motion_id=motion_id)
        if require_full:
            if horizon > self.capacity:
                raise ValueError("Full window exceeds Replay capacity")
            end = times + horizon - 1
            start_slot, end_slot = times % self.capacity, end % self.capacity
            valid = ((end < self.steps) & (self.timestamp[end_slot] == end)
                     & (self.episode[start_slot, envs] == self.episode[end_slot, envs])
                     & (self.task_id[start_slot, envs] == self.task_id[end_slot, envs])
                     & (self.motion_id[start_slot, envs] == self.motion_id[end_slot, envs])
                     & (self.frame[end_slot, envs] == self.frame[start_slot, envs] + horizon - 1))
            if source_id is not None:
                for offset in range(1, horizon):
                    valid &= self.source_id[(times + offset) % self.capacity, envs] == source_id
            times, envs = times[valid], envs[valid]
            if not len(times):
                raise ValueError("Replay has no full episode-safe windows")
        choice = torch.randint(len(times), (batch,), device=self.device)
        return self.window_at(times[choice], envs[choice], horizon)

    def sample_starts(self, batch: int, *, horizon: int, valid_start,
                      env_range=None, source_id=None, task_id=None,
                      motion_id=None) -> StartBatch:
        times, envs = self._active_candidates(env_range=env_range,
                                              source_id=source_id, task_id=task_id,
                                              motion_id=motion_id)
        slot = times % self.capacity
        task_state = TaskState(self.task_id[slot, envs], self.motion_id[slot, envs],
                               self.frame[slot, envs], self.task_data[slot, envs])
        valid = (~self.done[slot, envs]) & valid_start(task_state, horizon)
        if valid.shape != times.shape:
            raise ValueError("Task start-validity mask shape differs")
        times, envs = times[valid], envs[valid]
        if not len(times):
            raise ValueError("Replay has no legal imagination starts")
        chosen = torch.randint(len(times), (batch,), device=self.device)
        slot, env = times[chosen] % self.capacity, envs[chosen]
        return StartBatch(self.state[slot, env], self.history[slot, env],
                          self.dynamics_context[slot, env],
                          TaskState(self.task_id[slot, env], self.motion_id[slot, env],
                                    self.frame[slot, env], self.task_data[slot, env]),
                          self.previous_action[slot, env], env,
                          self.episode[slot, env], self.source_id[slot, env])

    def state_dict(self) -> dict:
        active = self.timestamp.detach().cpu() >= 0
        fields = {}
        for name, value in self.data.items():
            fields[name] = value.detach().cpu().clone()
            fields[name][~active] = 0
        return {"capacity": self.capacity, "num_envs": self.num_envs,
                "robot_spec": asdict(self.robot_spec), "task_spec": asdict(self.task_spec),
                "steps": self.steps, "episode_counter": self.episode_counter.detach().cpu().clone(),
                "timestamp": self.timestamp.detach().cpu().clone(), "fields": fields}

    def load_state_dict(self, record: dict, *, fresh_physics: bool = True):
        if (record["capacity"] != self.capacity or record["num_envs"] != self.num_envs
                or record["robot_spec"] != asdict(self.robot_spec)
                or record["task_spec"] != asdict(self.task_spec)
                or set(record["fields"]) != set(self.data)):
            raise ValueError("Replay schema differs")
        steps = int(record["steps"])
        timestamp = record["timestamp"].to(self.device)
        expected = torch.arange(max(0, steps - self.capacity), steps, device=self.device)
        observed = torch.sort(timestamp[timestamp >= 0]).values
        if timestamp.shape != self.timestamp.shape or steps < 0 or not torch.equal(observed, expected):
            raise ValueError("Replay absolute timestamps differ")
        for name, source in record["fields"].items():
            target = self.data[name]
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(f"Replay {name} schema differs")
            target.copy_(source.to(self.device))
        if record["episode_counter"].shape != self.episode_counter.shape:
            raise ValueError("Replay episode counter schema differs")
        self.timestamp.copy_(timestamp)
        self.episode_counter.copy_(record["episode_counter"].to(self.device)
                                   + int(fresh_physics))
        self.steps = steps
