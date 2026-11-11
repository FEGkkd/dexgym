# Ported from CumulusAlpha/Dex-Supermimic @ 2becb738; dimensions injected by task.
"""Compact dynamics and an independent real-termination/failure head."""

import torch
from torch import nn

from .common import EmpiricalNorm, RunningNorm, mlp


class WorldModel(nn.Module):
    def __init__(self, dynamics_dim, contact_dim, codec, action_dim, error_floor, hidden=(2048, 2048, 2048),
                 amp=True, *, control_error_features=False):
        super().__init__()
        self.codec = codec
        self.delta_dim = len(error_floor)
        self.contact_dim = contact_dim
        self.amp = amp
        self.control_error_features = control_error_features
        # Dynamics context contains physical state, startup simulator parameters
        # and the measured/sampled transition disturbance. Task command state is
        # deliberately absent: it is recursively derived after each WM step.
        inp = dynamics_dim + action_dim
        self.norm = RunningNorm(inp)
        self.net = mlp(
            inp,
            list(hidden),
            self.delta_dim + contact_dim,
            act=nn.SiLU,
            layer_norm=True,
        )
        self.delta_norm = RunningNorm(self.delta_dim, eps=1e-4)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[self.delta_dim :] = -3.0
        self.register_buffer("error_floor", torch.as_tensor(error_floor, dtype=torch.float32))

    def inputs(self, dynamics, u):
        if self.control_error_features:
            u = u - dynamics[..., :u.shape[-1]]
        return torch.cat((dynamics, u), -1)

    def forward(self, x, u, dynamics):
        features = self.norm(self.inputs(dynamics, u))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and x.is_cuda):
            y = self.net(features)
        y = y.float()
        d1, c = y.split((self.delta_dim, self.contact_dim), -1)
        scale = self.delta_norm.std()
        return (
            self.codec.integrate(x, d1 * scale + self.delta_norm.mean),
            c,
        )


class FailureHead(nn.Module):
    """Predict physical termination from task-defined state and context features."""

    def __init__(self, context_dim, state_dim, *, control_error_features=False):
        super().__init__()
        dim = state_dim + context_dim
        self.norm = RunningNorm(dim)
        self.net = mlp(dim, [256, 128], 1)
        self.control_error_features = control_error_features

    def features(self, x, context):
        if self.control_error_features:
            context = context - x[..., :context.shape[-1]]
        return torch.cat((x, context), -1)

    def forward(self, features):
        return self.net(self.norm(features)).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.norm = EmpiricalNorm(obs_dim)
        self.net = mlp(obs_dim, [512, 256, 128], action_dim)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs):
        return torch.tanh(self.net(self.norm(obs)))


class Critic(nn.Module):
    def __init__(self, dim, amp=True, empirical_norm=False):
        super().__init__()
        self.amp = amp
        self.norm = EmpiricalNorm(dim) if empirical_norm else RunningNorm(dim)
        self.net = mlp(dim, [512, 512, 256, 128], 1)

    def forward(self, x):
        if isinstance(self.norm, EmpiricalNorm):
            return self.net(self.norm(x)).squeeze(-1)
        features = self.norm(x)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and x.is_cuda):
            return self.net(features).float().squeeze(-1)
