"""GPU ring replay of pre-reset physical transitions with masked windows."""

import torch


class MotionReplay:
    fields = ("state", "action", "previous_action", "executed_command", "next_state",
              "history", "frame", "next_frame", "motion_id", "source_id",
              "failure_reason", "episode", "done", "terminated", "truncated", "timestamp")

    @staticmethod
    def valid_length_histogram(window: dict) -> dict[int, int]:
        lengths = window["mask"].sum(-1).long()
        values, counts = torch.unique(lengths, return_counts=True)
        return {int(length): int(count) for length, count in zip(values, counts)}

    def __init__(self, capacity_steps: int, num_envs: int, device: str | torch.device):
        if capacity_steps < 1 or num_envs < 1:
            raise ValueError("Replay capacity and environment count must be positive")
        self.capacity = capacity_steps
        self.num_envs = num_envs
        self.device = torch.device(device)
        shape = (capacity_steps, num_envs)
        self.state = torch.empty((*shape, 42), device=self.device)
        self.action = torch.empty((*shape, 14), device=self.device)
        self.previous_action = torch.empty_like(self.action)
        self.executed_command = torch.empty_like(self.action)
        self.next_state = torch.empty_like(self.state)
        self.history = torch.empty((*shape, 10, 56), device=self.device)
        self.frame = torch.empty(shape, device=self.device, dtype=torch.long)
        self.next_frame = torch.empty_like(self.frame)
        self.motion_id = torch.empty_like(self.frame)
        self.source_id = torch.empty(shape, device=self.device, dtype=torch.int8)
        self.failure_reason = torch.empty(shape, device=self.device, dtype=torch.int32)
        self.episode = torch.empty_like(self.frame)
        self.done = torch.empty(shape, device=self.device, dtype=torch.bool)
        self.terminated = torch.empty_like(self.done)
        self.truncated = torch.empty_like(self.done)
        self.timestamp = torch.full((capacity_steps,), -1, device=self.device, dtype=torch.long)
        self.episode_counter = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.steps = 0

    @property
    def bytes_allocated(self):
        fields = (self.state, self.action, self.previous_action, self.executed_command,
                  self.next_state, self.history, self.frame, self.next_frame,
                  self.motion_id, self.source_id, self.failure_reason, self.episode, self.done, self.terminated,
                  self.truncated, self.timestamp, self.episode_counter)
        return sum(field.numel() * field.element_size() for field in fields)

    def state_dict(self):
        """Persist the physical ring and provenance; fresh environments start a new episode."""
        valid_slot = self.timestamp.detach().cpu() >= 0
        fields = {}
        for key in self.fields:
            value = getattr(self, key).detach().cpu().clone()
            if key != "timestamp":
                value[~valid_slot] = 0
            fields[key] = value
        return {"capacity": self.capacity, "num_envs": self.num_envs, "steps": self.steps,
                "episode_counter": self.episode_counter.detach().cpu().clone(),
                "fields": fields}

    def load_state_dict(self, record):
        if (record["capacity"], record["num_envs"]) != (self.capacity, self.num_envs):
            raise ValueError("Replay capacity or environment count differs")
        if set(record["fields"]) != set(self.fields):
            raise ValueError("Replay field schema differs")
        for key in self.fields:
            target, source = getattr(self, key), record["fields"][key]
            if target.shape != source.shape or target.dtype != source.dtype:
                raise ValueError(f"Replay {key} schema differs")
            target.copy_(source.to(self.device))
        self.steps = int(record["steps"])
        expected = torch.arange(max(0, self.steps - self.capacity), self.steps,
                                device=self.device)
        observed = torch.sort(self.timestamp[self.timestamp >= 0]).values
        if self.steps < 0 or (self.timestamp >= self.steps).any() or not torch.equal(observed, expected):
            raise ValueError("Replay absolute timestamps are invalid")
        if record["episode_counter"].shape != self.episode_counter.shape:
            raise ValueError("Replay episode counter schema differs")
        # The Newton environments are recreated on resume. Never stitch the
        # first transition from a fresh environment to a saved trajectory.
        self.episode_counter.copy_(record["episode_counter"].to(self.device) + 1)

    def append(self, state, action, next_state, history, frame, done, *, terminated=None,
               truncated=None, motion_id=None, previous_action=None, executed_command=None,
               source_id=None, failure_reason=None):
        if terminated is None:
            terminated = done
        if truncated is None:
            truncated = done & ~terminated
        if not torch.equal(done, terminated | truncated):
            raise ValueError("done must equal terminated OR truncated")
        if state.shape != (self.num_envs, 42) or action.shape != (self.num_envs, 14):
            raise ValueError("Replay expects per-environment 42D states and 14D actions")
        if next_state.shape != state.shape or frame.shape != (self.num_envs,):
            raise ValueError("Replay next state or reference time has invalid shape")
        if not bool(torch.isfinite(state).all() and torch.isfinite(action).all()
                    and torch.isfinite(next_state).all()):
            raise ValueError("Replay transition has nonfinite physical data")
        slot = self.steps % self.capacity
        self.state[slot] = state.detach()
        self.action[slot] = action.detach()
        self.previous_action[slot] = (torch.zeros_like(action) if previous_action is None
                                      else previous_action).detach()
        self.next_state[slot] = next_state.detach()
        self.executed_command[slot] = (next_state[:, 28:42] if executed_command is None
                                       else executed_command).detach()
        self.history[slot] = history.detach()
        self.frame[slot] = frame.detach()
        self.next_frame[slot] = frame.detach() + 1
        self.motion_id[slot] = (torch.zeros_like(frame) if motion_id is None else motion_id).detach()
        self.source_id[slot] = (torch.zeros_like(frame) if source_id is None else source_id).to(torch.int8)
        self.failure_reason[slot] = (torch.zeros_like(frame) if failure_reason is None
                                     else failure_reason).to(torch.int32)
        self.episode[slot] = self.episode_counter
        self.done[slot] = done.detach()
        self.terminated[slot] = terminated.detach()
        self.truncated[slot] = truncated.detach()
        self.timestamp[slot] = self.steps
        self.episode_counter += done.long()
        self.steps += 1

    def _candidates(self, env_range=None, frame_range=None, source_id=None):
        if self.steps == 0:
            raise ValueError("Replay is empty")
        env_low, env_high = env_range or (0, self.num_envs)
        if not (0 <= env_low < env_high <= self.num_envs):
            raise ValueError("Invalid replay environment range")
        times = torch.arange(max(0, self.steps - self.capacity), self.steps, device=self.device)
        envs = torch.arange(env_low, env_high, device=self.device)
        if frame_range is None and source_id is None:
            return times[:, None].expand(-1, len(envs)).flatten(), envs.repeat(len(times))
        valid = torch.ones((len(times), len(envs)), device=self.device, dtype=torch.bool)
        if frame_range is not None:
            frame = self.frame[times[:, None] % self.capacity, envs[None, :]]
            valid &= (frame >= frame_range[0]) & (frame < frame_range[1])
        if source_id is not None:
            valid &= self.source_id[times[:, None] % self.capacity, envs[None, :]] == source_id
        time_idx, env_idx = valid.nonzero(as_tuple=True)
        if not len(time_idx):
            raise ValueError("Replay lacks requested reference frames")
        return times[time_idx], envs[env_idx]

    def sample_windows(self, batch: int, horizon: int = 8, env_range=None, frame_range=None,
                       *, require_full: bool = False, source_id: int | None = None):
        if batch < 1 or horizon < 1:
            raise ValueError("Batch and horizon must be positive")
        times, envs = self._candidates(env_range, frame_range, source_id)
        if require_full:
            if horizon > self.capacity:
                raise ValueError("A full window exceeds replay capacity")
            start_slot = times % self.capacity
            end_absolute = times + horizon - 1
            end_slot = end_absolute % self.capacity
            full = ((end_absolute < self.steps)
                    & (self.timestamp[end_slot] == end_absolute)
                    & (self.episode[start_slot, envs] == self.episode[end_slot, envs])
                    & (self.motion_id[start_slot, envs] == self.motion_id[end_slot, envs])
                    & (self.frame[end_slot, envs] == self.frame[start_slot, envs] + horizon - 1))
            if source_id is not None:
                for offset in range(1, horizon):
                    full &= self.source_id[(times + offset) % self.capacity, envs] == source_id
            times, envs = times[full], envs[full]
            if not len(times):
                raise ValueError("Replay lacks full episode-safe windows")
        selected = torch.randint(len(times), (batch,), device=self.device)
        start, env = times[selected], envs[selected]
        absolute = start[:, None] + torch.arange(horizon, device=self.device)
        slots = absolute % self.capacity
        env_idx = env[:, None]
        start_slot = start % self.capacity
        episode = self.episode[slots, env_idx]
        frame = self.frame[slots, env_idx]
        motion_id = self.motion_id[slots, env_idx]
        present = ((absolute < self.steps) & (absolute >= max(0, self.steps - self.capacity))
                   & (self.timestamp[slots] == absolute))
        same_trajectory = ((episode == self.episode[start_slot, env][:, None])
                           & (motion_id == self.motion_id[start_slot, env][:, None])
                           & (frame == self.frame[start_slot, env][:, None]
                              + torch.arange(horizon, device=self.device)))
        no_prior_done = torch.cumprod(
            torch.cat((torch.ones((batch, 1), device=self.device, dtype=torch.long),
                       (~self.done[slots, env_idx][:, :-1]).long()), dim=1), dim=1).bool()
        mask = present & same_trajectory & no_prior_done
        assert bool(mask[:, 0].all())
        keys = ("state", "action", "previous_action", "executed_command", "next_state",
                "frame", "next_frame", "motion_id", "source_id", "failure_reason",
                "episode", "done", "terminated", "truncated")
        out = {key: getattr(self, key)[slots, env_idx] for key in keys}
        for key in ("state", "action", "previous_action", "executed_command", "next_state"):
            out[key] = torch.where(mask[..., None], out[key], torch.zeros_like(out[key]))
        for key in ("frame", "next_frame", "motion_id", "source_id", "failure_reason", "episode"):
            out[key] = torch.where(mask, out[key], torch.zeros_like(out[key]))
        for key in ("done", "terminated", "truncated"):
            out[key] = out[key] & mask
        out["mask"] = mask.float()
        out["env_id"] = env
        out["absolute_timestamp"] = absolute
        return out

    def sample_starts(self, batch: int, env_range=None, frame_range=None, *, horizon=None,
                      clip_length=None, source_id=None):
        times, envs = self._candidates(env_range, frame_range, source_id)
        slots = times % self.capacity
        valid = ~self.done[slots, envs]
        if horizon is not None:
            if clip_length is None:
                raise ValueError("A clip length is required for full-horizon imagination")
            length = torch.as_tensor(clip_length, device=self.device)
            if length.ndim:
                length = length[self.motion_id[slots, envs]]
            valid &= self.frame[slots, envs] + horizon < length
        times, envs = times[valid], envs[valid]
        if not len(times):
            raise ValueError("Replay lacks episode-safe imagination starts")
        chosen = torch.randint(len(times), (batch,), device=self.device)
        slot, env = times[chosen] % self.capacity, envs[chosen]
        return {"state": self.state[slot, env], "history": self.history[slot, env],
                "frame": self.frame[slot, env], "motion_id": self.motion_id[slot, env],
                "episode": self.episode[slot, env], "source_id": self.source_id[slot, env], "env_id": env}
