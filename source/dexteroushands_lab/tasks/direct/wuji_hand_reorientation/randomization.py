"""Task-local domain-randomization buffers and capability bookkeeping."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class DomainRandomizationState:
    """Store sampled values that the DirectTask reports after applying."""

    def __init__(self, cfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.ratios = self.neutral(self.num_envs, self.device)
        self.encoder_bias = torch.zeros((self.num_envs, 20), device=self.device)
        self.unsupported_fields: set[str] = set()
        self.applied_fields: set[str] = set()

    @staticmethod
    def neutral(num_envs: int, device: str | torch.device) -> torch.Tensor:
        return torch.ones((int(num_envs), 6), device=device)

    @staticmethod
    def _sample(
        bounds: tuple[float, float],
        shape: tuple[int, ...],
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        low, high = map(float, bounds)
        if high < low:
            raise ValueError("randomization range upper bound must be >= lower bound")
        return low + (high - low) * torch.rand(shape, device=device, generator=generator)

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        defaults: dict[str, torch.Tensor] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        del defaults
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return self.ratios
        if not self.cfg.enabled:
            self.ratios[ids] = 1.0
            self.encoder_bias[ids] = 0.0
            return self.ratios
        cfg = self.cfg
        ranges = (
            cfg.object_mass_scale,
            (1.0 + cfg.object_com_range[0], 1.0 + cfg.object_com_range[1]),
            cfg.object_size_scale,
            cfg.hand_friction_scale,
            cfg.pd_stiffness_scale,
            cfg.joint_damping_scale,
        )
        for index, bounds in enumerate(ranges):
            self.ratios[ids, index] = self._sample(
                bounds, (ids.numel(),), self.device, generator
            )
        self.encoder_bias[ids] = self._sample(
            cfg.encoder_bias_range, (ids.numel(), 20), self.device, generator
        )
        return self.ratios

    def neutralized(self) -> torch.Tensor:
        return self.neutral(self.num_envs, self.device) if not self.cfg.enabled else self.ratios.clone()

    def mark_capability(self, field: str, supported: bool) -> None:
        target = self.applied_fields if supported else self.unsupported_fields
        other = self.unsupported_fields if supported else self.applied_fields
        target.add(field)
        other.discard(field)
