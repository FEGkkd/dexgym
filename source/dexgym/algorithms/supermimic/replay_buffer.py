"""Bounded GPU replay with episode-safe contiguous physical windows."""

import torch


class MotionReplay:
    def __init__(self, capacity_steps: int, num_envs: int, device: str | torch.device):
        if capacity_steps < 40 or num_envs < 1:
            raise ValueError("Replay needs at least 40 time slots and a positive environment count")
        self.capacity = capacity_steps
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.state = torch.empty((capacity_steps, num_envs, 42), device=self.device)
        self.action = torch.empty((capacity_steps, num_envs, 14), device=self.device)
        self.next_state = torch.empty_like(self.state)
        self.history = torch.empty((capacity_steps, num_envs, 10, 56), device=self.device)
        self.frame = torch.empty((capacity_steps, num_envs), device=self.device, dtype=torch.long)
        self.episode = torch.empty_like(self.frame)
        self.done = torch.empty((capacity_steps, num_envs), device=self.device, dtype=torch.bool)
        self.terminated = torch.empty_like(self.done)
        self.timestamp = torch.full((capacity_steps,), -1, device=self.device, dtype=torch.long)
        self.episode_counter = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.steps = 0

    @property
    def bytes_allocated(self):
        fields = (
            self.state,
            self.action,
            self.next_state,
            self.history,
            self.frame,
            self.episode,
            self.done,
            self.terminated,
            self.timestamp,
            self.episode_counter,
        )
        return sum(field.numel() * field.element_size() for field in fields)

    def append(self, state, action, next_state, history, frame, done, *, terminated=None):
        slot = self.steps % self.capacity
        self.state[slot] = state.detach()
        self.action[slot] = action.detach()
        self.next_state[slot] = next_state.detach()
        self.history[slot] = history.detach()
        self.frame[slot] = frame.detach()
        self.episode[slot] = self.episode_counter
        self.done[slot] = done.detach()
        self.terminated[slot] = (done if terminated is None else terminated).detach()
        self.timestamp[slot] = self.steps
        self.episode_counter += done.long()
        self.steps += 1

    def _sample_indices(
        self,
        batch: int,
        horizon: int,
        env_range: tuple[int, int] | None = None,
        frame_range: tuple[int, int] | None = None,
        allow_terminal_at_end: bool = False,
    ):
        low = max(0, self.steps - self.capacity)
        high = self.steps - horizon
        if high < low:
            raise ValueError("Not enough replay steps for requested horizon")
        env_low, env_high = env_range or (0, self.num_envs)
        times = torch.arange(low, high + 1, device=self.device)
        envs = torch.arange(env_low, env_high, device=self.device)
        start = times % self.capacity
        end = (times + horizon - 1) % self.capacity
        # Episode IDs advance immediately after a terminal transition, so equal
        # endpoints prove that no reset occurs inside this contiguous window.
        valid = self.episode[start[:, None], envs[None, :]] == self.episode[end[:, None], envs[None, :]]
        if not allow_terminal_at_end:
            valid &= ~self.done[end[:, None], envs[None, :]]
        if frame_range is not None:
            frame = self.frame[start[:, None], envs[None, :]]
            valid &= (frame >= frame_range[0]) & (frame < frame_range[1])
        candidate_t, candidate_env = valid.nonzero(as_tuple=True)
        if candidate_t.numel() == 0:
            raise ValueError("Replay lacks episode-safe continuous windows")
        selected = torch.randint(candidate_t.numel(), (batch,), device=self.device)
        return times[candidate_t[selected]], envs[candidate_env[selected]]

    def sample_windows(self, batch: int, horizon: int = 8, env_range: tuple[int, int] | None = None):
        t, env = self._sample_indices(batch, horizon, env_range, allow_terminal_at_end=True)
        slots = (t[:, None] + torch.arange(horizon, device=self.device)) % self.capacity
        return {
            "state": self.state[slots, env[:, None]],
            "action": self.action[slots, env[:, None]],
            "next_state": self.next_state[slots, env[:, None]],
            "frame": self.frame[slots, env[:, None]],
            "terminated": self.terminated[slots, env[:, None]],
        }

    def sample_starts(
        self,
        batch: int,
        env_range: tuple[int, int] | None = None,
        frame_range: tuple[int, int] | None = None,
    ):
        t, env = self._sample_indices(batch, 1, env_range, frame_range)
        slots = t % self.capacity
        return {
            "state": self.state[slots, env],
            "history": self.history[slots, env],
            "frame": self.frame[slots, env],
        }
