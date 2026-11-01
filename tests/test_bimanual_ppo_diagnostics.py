"""Numerical health metrics used by the Phase 1 PPO logger."""

import torch
from types import SimpleNamespace

from rsl_rl.base.train import reset_loaded_gaussian_std

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


def test_capturn_transfer_restores_std_one_and_clears_only_std_optimizer_moments():
    std = torch.nn.Parameter(torch.full((52,), 0.65))
    other = torch.nn.Parameter(torch.tensor([0.5]))
    optimizer = torch.optim.Adam([std, other], lr=1e-3)
    (std.sum() + other.sum()).backward()
    optimizer.step()
    assert optimizer.state[std] and optimizer.state[other]
    runner = SimpleNamespace(alg=SimpleNamespace(
        actor=SimpleNamespace(distribution=SimpleNamespace(
            std_type="scalar", std_param=std)), optimizer=optimizer))
    reset_loaded_gaussian_std(runner, 1.0)
    assert torch.allclose(std, torch.ones_like(std))
    assert std not in optimizer.state
    assert optimizer.state[other]
