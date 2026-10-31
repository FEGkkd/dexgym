"""Numerical health metrics used by the Phase 1 PPO logger."""

import torch

from dexteroushands_lab.tasks.direct.bimanual_bottle.agents.ppo_diagnostics import (
    clip_fraction, explained_variance,
)


def test_clip_fraction_uses_probability_ratio():
    ratio = torch.tensor([1.0, 1.1, 1.3, 0.7])
    assert torch.isclose(clip_fraction(ratio.log(), 0.2), torch.tensor(0.5))


def test_explained_variance_handles_perfect_and_constant_targets():
    target = torch.tensor([[1.0], [2.0], [3.0]])
    assert torch.isclose(explained_variance(target, target), torch.tensor(1.0))
    assert torch.isfinite(explained_variance(torch.ones_like(target), target))
