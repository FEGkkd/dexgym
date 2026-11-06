"""Real-physics collection and independent evaluation contracts."""

from typing import Protocol

from dexgym.algorithms.supermimic.core.transition import TransitionBatch


class PhysicalCollector(Protocol):
    num_envs: int

    def step(self, actor, *, policy: str = "actor") -> TransitionBatch: ...


class PhysicalEvaluator(Protocol):
    def evaluate(self, actor, *, output_dir) -> dict: ...

