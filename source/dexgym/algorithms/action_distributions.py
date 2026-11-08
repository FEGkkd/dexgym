"""Bounded PPO action distribution for the right-hand diagnostic."""

import math

import torch
from rsl_rl.modules.distribution import BetaDistribution, GaussianDistribution


class TanhGaussianDistribution(GaussianDistribution):
    """Gaussian latent action with a tanh map and its change-of-variables density."""

    def sample(self) -> torch.Tensor:
        return torch.tanh(super().sample())

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return torch.tanh(mlp_output)

    def as_deterministic_output_module(self) -> torch.nn.Module:
        return torch.nn.Tanh()

    @property
    def mean(self) -> torch.Tensor:
        """Return the bounded deterministic action; the transformed expectation is not closed-form."""
        return torch.tanh(self._distribution.mean)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Use latent Gaussian parameters for KL, which tanh preserves."""
        return self._distribution.mean, self._distribution.stddev

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        bounded = outputs.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        latent = torch.atanh(bounded)
        return (self._distribution.log_prob(latent) - torch.log1p(-bounded.square())).sum(dim=-1)

    @property
    def entropy(self) -> torch.Tensor:
        """Monte Carlo entropy estimate in bounded action space."""
        latent = self._distribution.rsample()
        return -self.log_prob(torch.tanh(latent))


class MatchedBetaDistribution(BetaDistribution):
    """Native RSL-RL Beta with a symmetric initial action spread matching Gaussian 0.3."""

    def __init__(
        self,
        output_dim: int,
        init_std: float = 0.3,
        action_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__(output_dim, action_range=action_range)
        half_range = (action_range[1] - action_range[0]) / 2.0
        if init_std <= 0.0 or init_std >= half_range / math.sqrt(3.0):
            raise ValueError("Initial Beta std must be positive and support alpha=beta>1")
        self.init_std = init_std
        self._initial_concentration = ((half_range / init_std) ** 2 - 1.0) / 2.0

    def init_mlp_weights(self, mlp: torch.nn.Module) -> None:
        concentration_offset = self._initial_concentration - 1.0
        raw_concentration = concentration_offset + math.log(-math.expm1(-concentration_offset))
        torch.nn.init.zeros_(mlp[-2].weight)
        torch.nn.init.constant_(mlp[-2].bias, raw_concentration)
