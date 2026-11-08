"""Contract for a bounded Gaussian action head used by PPO."""

from dataclasses import asdict
from importlib import import_module

import torch
from dexgym.tasks.ppo_cfg import RightHandCleanHybrid125BetaPPOCfg, RightHandCleanHybrid125TanhGaussianPPOCfg
from rsl_rl.models import MLPModel
from tensordict import TensorDict


def _distribution():
    cls = import_module("dexgym.algorithms.action_distributions").TanhGaussianDistribution
    return cls(output_dim=2, init_std=0.3)


def test_tanh_gaussian_samples_and_deterministic_actions_stay_in_bounds():
    distribution = _distribution()
    mean = torch.tensor([[10.0, -10.0], [0.2, -0.4]])
    distribution.update(mean)
    sample = distribution.sample()
    deterministic = distribution.deterministic_output(mean)
    assert sample.shape == mean.shape
    assert torch.all(sample.abs() <= 1.0)
    assert torch.all(deterministic.abs() <= 1.0)
    assert torch.allclose(deterministic, torch.tanh(mean))
    assert torch.isfinite(distribution.log_prob(sample)).all()


def test_tanh_gaussian_log_prob_and_kl_match_change_of_variables():
    distribution = _distribution()
    mean = torch.tensor([[0.1, -0.3]], requires_grad=True)
    distribution.update(mean)
    latent = torch.tensor([[0.2, -0.5]])
    action = torch.tanh(latent)
    normal = torch.distributions.Normal(mean, distribution.std)
    expected = (normal.log_prob(latent) - torch.log1p(-action.square())).sum(dim=-1)
    actual = distribution.log_prob(action)
    assert torch.allclose(actual, expected, atol=1e-5)
    old_params = distribution.params
    new_params = (mean + 0.2, distribution.std * 1.1)
    expected_kl = torch.distributions.kl_divergence(
        torch.distributions.Normal(*old_params), torch.distributions.Normal(*new_params)
    ).sum(dim=-1)
    assert torch.allclose(distribution.kl_divergence(old_params, new_params), expected_kl)
    (-actual.mean()).backward()
    assert torch.isfinite(mean.grad).all()


def test_rsl_rl_actor_builds_and_samples_both_bounded_pilot_distributions():
    obs = TensorDict({"policy": torch.randn(4, 125)}, batch_size=[4])
    for cfg_class in (RightHandCleanHybrid125TanhGaussianPPOCfg, RightHandCleanHybrid125BetaPPOCfg):
        cfg = cfg_class()
        actor = MLPModel(
            obs,
            {"actor": ["policy"]},
            "actor",
            20,
            hidden_dims=[16],
            activation="elu",
            distribution_cfg=asdict(cfg.actor.distribution_cfg),
        )
        action = actor(obs, stochastic_output=True)
        assert action.shape == (4, 20)
        assert torch.isfinite(action).all()
        assert (action.abs() <= 1.0).all()
        assert torch.isfinite(actor.get_output_log_prob(action)).all()
        assert torch.isfinite(actor.output_entropy).all()
        if cfg_class is RightHandCleanHybrid125BetaPPOCfg:
            assert torch.allclose(actor.output_std.mean(), torch.tensor(0.3), atol=0.01)
