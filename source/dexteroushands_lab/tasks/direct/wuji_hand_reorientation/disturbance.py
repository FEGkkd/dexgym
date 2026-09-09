"""External object disturbance scheduling for WUJI reorientation."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class DisturbanceScheduler:
    """Schedule per-environment object velocity impulses after warmup."""

    def __init__(self, cfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.next_trigger = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.last_direction = torch.zeros((self.num_envs, 3), device=self.device)
        self.last_velocity = torch.zeros((self.num_envs, 6), device=self.device)
        self.trigger_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _uniform(
        self,
        bounds: tuple[float, float],
        count: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        low, high = map(float, bounds)
        if high < low:
            raise ValueError("disturbance range upper bound must be >= lower bound")
        return low + (high - low) * torch.rand(count, device=self.device, generator=generator)

    def reset(self, env_ids: Sequence[int] | torch.Tensor, now: float | torch.Tensor) -> None:
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        self.last_direction[ids] = 0.0
        self.last_velocity[ids] = 0.0
        self.trigger_count[ids] = 0
        if not self.cfg.enabled:
            self.next_trigger[ids] = float("inf")
            return
        now_tensor = torch.as_tensor(now, device=self.device, dtype=torch.float32)
        now_tensor = now_tensor.expand(ids.numel()) if now_tensor.numel() == 1 else now_tensor
        self.next_trigger[ids] = now_tensor + self._uniform(self.cfg.interval_s, ids.numel())

    def step(
        self,
        now: torch.Tensor,
        progress: torch.Tensor | float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        now = now.to(self.device, dtype=torch.float32)
        progress = torch.as_tensor(progress, device=self.device, dtype=torch.float32)
        progress = progress.expand(self.num_envs) if progress.numel() == 1 else progress
        if not self.cfg.enabled:
            return self.neutral()
        trigger = (now >= float(self.cfg.warmup_time_s)) & (now >= self.next_trigger)
        if trigger.any():
            ids = trigger.nonzero(as_tuple=False).flatten()
            direction = torch.randn((ids.numel(), 3), device=self.device, generator=generator)
            direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
            low, high = map(float, self.cfg.speed_range)
            if high < low:
                raise ValueError("disturbance speed upper bound must be >= lower bound")
            ramp = ((progress[ids] - float(self.cfg.warmup_frac)) / max(float(self.cfg.rampup_frac), 1.0e-6)).clamp(0.0, 1.0)
            max_speed = low + (high - low) * ramp
            speed = low + (max_speed - low) * torch.rand(
                ids.numel(), device=self.device, generator=generator
            )
            velocity = torch.zeros((ids.numel(), 6), device=self.device)
            velocity[:, :3] = direction * speed.unsqueeze(-1)
            self.last_direction[ids] = direction
            self.last_velocity[ids] = velocity
            self.trigger_count[ids] += 1
            self.next_trigger[ids] = now[ids] + self._uniform(
                self.cfg.interval_s, ids.numel(), generator
            )
        return trigger, self.last_direction.clone(), self.last_velocity.clone()

    def neutral(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            torch.zeros_like(self.last_direction),
            torch.zeros_like(self.last_velocity),
        )
