"""Numerical contract for bounded PPO actions and their log probabilities."""

import pytest
import torch
from torch.distributions import Normal

from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.agents.tanh_distribution import (
    TanhGaussianDistribution,
)


def test_samples_and_deterministic_action_are_strictly_bounded():
    dist = TanhGaussianDistribution(52, init_std=1.0)
    dist.update(torch.tensor([[0.] * 52, [100.] * 52, [-100.] * 52]))
    samples = dist.sample()
    deterministic = dist.deterministic_output(torch.tensor([[100.] * 52]))
    assert torch.isfinite(samples).all()
    assert (samples.abs() < 1).all()
    assert (deterministic.abs() < 1).all()
    assert torch.isfinite(dist.log_prob(samples)).all()
    assert torch.isfinite(dist.entropy).all()


def test_tanh_log_probability_has_exact_jacobian_and_ppo_ratio():
    dist = TanhGaussianDistribution(2, init_std=1.0)
    raw_old = torch.tensor([[0.3, -0.7]], dtype=torch.float64)
    raw_new = torch.tensor([[0.8, -0.2]], dtype=torch.float64)
    latent = torch.tensor([[0.4, -0.5]], dtype=torch.float64)
    action = torch.tanh(latent)
    dist.update(raw_old)
    old_params = dist.params
    old_log = dist.log_prob(action)
    jacobian = torch.log1p(-action.square()).sum(-1)
    manual_old = Normal(old_params[0], old_params[1]).log_prob(latent).sum(-1) - jacobian
    assert torch.allclose(old_log, manual_old, atol=1e-8)
    dist.update(raw_new)
    new_params = dist.params
    new_log = dist.log_prob(action)
    ratio = torch.exp(new_log - old_log)
    latent_ratio = torch.exp((Normal(new_params[0], new_params[1]).log_prob(latent)
                              - Normal(old_params[0], old_params[1]).log_prob(latent)).sum(-1))
    assert torch.allclose(ratio, latent_ratio, atol=1e-8)
    expected_kl = torch.distributions.kl_divergence(
        Normal(old_params[0], old_params[1]), Normal(new_params[0], new_params[1])).sum(-1)
    assert torch.allclose(dist.kl_divergence(old_params, new_params), expected_kl, atol=1e-8)


def test_log_probability_rejects_exact_boundaries_instead_of_faking_finite_ratio():
    dist = TanhGaussianDistribution(1)
    dist.update(torch.zeros(1, 1))
    with pytest.raises(ValueError, match="strictly inside"):
        dist.log_prob(torch.ones(1, 1))


def test_reparameterized_entropy_matches_one_dimensional_quadrature():
    torch.manual_seed(17)
    dist = TanhGaussianDistribution(1, init_std=1.0)
    dist.update(torch.zeros(100_000, 1))
    estimated = dist.entropy.mean()
    z = torch.linspace(-8, 8, 20_001, dtype=torch.float64)
    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    log_jacobian = torch.log1p(-torch.tanh(z).square())
    expected = normal.entropy()[0] + torch.trapz(normal.log_prob(z).exp() * log_jacobian, z)
    assert abs(float((estimated - expected).detach())) < 0.015
