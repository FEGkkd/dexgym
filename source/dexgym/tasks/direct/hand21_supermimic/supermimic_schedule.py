"""Hand-local Dex-Supermimic scheduling, without model-error gates."""

import math
from dataclasses import dataclass
from fractions import Fraction

from dexgym.algorithms.supermimic.core.trainer import SuperMimicTrainer, TrainerConfig


@dataclass(frozen=True)
class HandTrainerConfig(TrainerConfig):
    actor_update_schedule: str = "every_iteration"
    world_warmup_updates: int = 0
    utd: float = 0.4
    warmup_steps: int = 64
    wm_only_iters: int = 500
    lr: float = 2e-4
    lr_decay_iters: int = 250000
    lr_final_ratio: float = 0.03

    def __post_init__(self):
        super().__post_init__()
        if self.actor_update_schedule != "every_iteration" or self.physics_steps_per_iteration != 1:
            raise ValueError("Hand scheduling requires the ungated schedule and one-step collector")
        if not math.isfinite(self.utd) or not 0 < self.utd <= 1:
            raise ValueError("utd must be in (0, 1]")
        if self.warmup_steps < max(self.world_window, self.max_imagination_horizon):
            raise ValueError("warmup_steps must cover WM and imagination horizons")
        if self.wm_only_iters < 0 or self.lr_decay_iters < 1:
            raise ValueError("Invalid warmup or learning-rate schedule")
        if not math.isfinite(self.lr) or self.lr <= 0 or not 0 < self.lr_final_ratio <= 1:
            raise ValueError("Invalid learning rate or floor")

    def collection_steps(self, iteration):
        # Absolute iteration avoids fractional drift and needs no resume accumulator.
        rate = 1 / Fraction(str(self.utd))
        return ((iteration + 1) * rate).__floor__() - (iteration * rate).__floor__()

    def learning_rate(self, iteration):
        fraction = min(1.0, max(0.0, iteration / self.lr_decay_iters))
        return self.lr * (self.lr_final_ratio + (1 - self.lr_final_ratio) * 0.5 * (1 + math.cos(math.pi * fraction)))


class HandSuperMimicTrainer(SuperMimicTrainer):
    def prefill(self):
        """Fresh physics needs new replay; collection never consumes WM-only iterations."""
        for _ in range(self.config.warmup_steps):
            super().collect()
        return self.config.warmup_steps

    def collect(self):
        reports = []
        for _ in range(self.config.collection_steps(self.iteration)):
            reports.extend(super().collect())
        return reports

    def update_actor_critic(self):
        if self.iteration < self.config.wm_only_iters:
            return {"updated": False, "reason": "fixed_wm_only_iterations"}
        return super().update_actor_critic()

    def run_iteration(self):
        lr = self.config.learning_rate(self.iteration)
        for optimizer in self.optimizers.values():
            for group in optimizer.param_groups:
                group["lr"] = lr
        result = super().run_iteration()
        result["lr"] = lr
        result["collection_steps"] = self.config.collection_steps(self.iteration - 1)
        result["training_phase"] = "world_model_only" if self.iteration <= self.config.wm_only_iters else "actor_critic"
        return result
