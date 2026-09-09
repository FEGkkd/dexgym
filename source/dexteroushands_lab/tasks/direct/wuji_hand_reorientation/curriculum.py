"""Tensor-only curriculum state for WUJI reorientation."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class CurriculumState:
    """Track success progression and adaptive episode scale per environment."""

    def __init__(self, cfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.scale = torch.ones(self.num_envs, device=self.device)
        self.success_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_target_steps = torch.full(
            (self.num_envs,), float(max(int(cfg.adaptive_target_steps), 1)), device=self.device
        )
        self.total_successes = torch.zeros((), dtype=torch.long, device=self.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.numel():
            self.success_count[ids] = 0

    def update(
        self,
        successes: torch.Tensor,
        episode_steps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        successes = successes.to(self.device, dtype=torch.bool)
        if successes.shape != (self.num_envs,):
            raise ValueError(f"successes must have shape ({self.num_envs},), got {tuple(successes.shape)}")
        if not self.cfg.enabled:
            return self.as_progress()

        self.success_count += successes.to(torch.long)
        self.total_successes += successes.sum()
        threshold = max(int(self.cfg.success_count_threshold), 1)
        reached = self.success_count >= threshold
        if reached.any():
            self.scale[reached] += float(self.cfg.delta_per_loop)
            self.success_count[reached] = 0

        if self.cfg.adaptive_episode_enabled and episode_steps is not None:
            episode_steps = episode_steps.to(self.device, dtype=torch.float32)
            if episode_steps.ndim != 1 or episode_steps.numel() == 0:
                raise ValueError(
                    "episode_steps must be a non-empty one-dimensional tensor"
                )
            target = float(max(int(self.cfg.adaptive_target_steps), 1))
            ratio = episode_steps.clamp_min(1.0).mean() / target
            if ratio > 1.0:
                factor = max(1.0 - float(self.cfg.adaptive_dec_rate), float(self.cfg.adaptive_min_scale))
                self.scale *= factor
            elif ratio < 1.0:
                self.scale *= 1.0 + float(self.cfg.adaptive_inc_rate)
            self.scale.clamp_(min=float(self.cfg.adaptive_min_scale))
            self.episode_target_steps.fill_(target * self.scale.mean().item())
        return self.as_progress()

    def as_progress(self) -> dict[str, torch.Tensor]:
        if self.cfg.enabled:
            scale = self.scale
            target = self.episode_target_steps
        else:
            scale = torch.ones_like(self.scale)
            target = torch.full_like(
                self.episode_target_steps, float(max(int(self.cfg.adaptive_target_steps), 1))
            )
        return {
            "scale": scale.clone(),
            "episode_target_steps": target.clone(),
            "success_count": self.success_count.clone(),
            "total_successes": self.total_successes.expand(self.num_envs).clone(),
        }
