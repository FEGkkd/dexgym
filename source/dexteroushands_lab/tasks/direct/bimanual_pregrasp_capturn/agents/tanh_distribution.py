"""Bounded Gaussian policy with exact interior density for PPO ratios."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from rsl_rl.modules.distribution import GaussianDistribution


def _latent_mean(raw: torch.Tensor) -> torch.Tensor:
    # A finite latent mean makes float32 tanh saturation extremely unlikely
    # under the std=1.0 controlled comparison; it remains differentiable.
    return 2.0 * torch.tanh(raw / 2.0)


def _log_tanh_jacobian(latent: torch.Tensor) -> torch.Tensor:
    # log(1 - tanh(z)^2), in a stable form used for the exact transformed PDF.
    return 2.0 * (math.log(2.0) - latent - F.softplus(-2.0 * latent))


class _BoundedDeterministicOutput(nn.Module):
    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.tanh(_latent_mean(raw))


class TanhGaussianDistribution(GaussianDistribution):
    """Tanh transform of a learned diagonal Gaussian.

    The latent mean is smoothly limited to [-2, 2] before sampling. Actions
    and PPO ratios use the transformed density. KL is exact because tanh is
    bijective on the open action interval. Entropy uses a reparameterized
    Monte Carlo estimate of the transformed entropy.
    """

    def update(self, mlp_output: torch.Tensor) -> None:
        super().update(_latent_mean(mlp_output))

    def sample(self) -> torch.Tensor:
        action = torch.tanh(self._distribution.sample())
        if not bool((action.abs() < 1).all()):
            raise FloatingPointError("Tanh action rounded to its boundary; PPO ratio is undefined")
        return action

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return torch.tanh(_latent_mean(mlp_output))

    def as_deterministic_output_module(self) -> nn.Module:
        return _BoundedDeterministicOutput()

    @property
    def latent_mean(self) -> torch.Tensor:
        return self._distribution.mean

    @property
    def mean(self) -> torch.Tensor:
        # RSL's diagnostics use this action-space deterministic output.
        return torch.tanh(self.latent_mean)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        # KL must operate on the latent Gaussian, not the transformed output.
        return self.latent_mean, self._distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        latent = self._distribution.rsample()
        return (self._distribution.entropy()
                + _log_tanh_jacobian(latent)).sum(dim=-1)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        if not bool((torch.isfinite(outputs) & (outputs.abs() < 1)).all()):
            raise ValueError("Tanh PPO actions must be finite and strictly inside (-1, 1)")
        latent = torch.atanh(outputs)
        return (self._distribution.log_prob(latent)
                - _log_tanh_jacobian(latent)).sum(dim=-1)
