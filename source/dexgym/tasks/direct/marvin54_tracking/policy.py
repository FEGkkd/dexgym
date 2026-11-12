"""Motion encoder and full-body decoder for Marvin reference tracking."""

import torch
from torch import nn

from dexgym.algorithms.supermimic.dex_reference.common import EmpiricalNorm, mlp


class MotionTrackingActor(nn.Module):
    def __init__(self, history_dim, goal_dim, action_dim, latent_dim=32, direct_goal_dim=0):
        super().__init__()
        widths = [2048, 1024, 512, 256]
        self.history_dim = history_dim
        self.direct_goal_dim = direct_goal_dim
        self.norm = EmpiricalNorm(history_dim + goal_dim)
        self.encoder = mlp(goal_dim, widths, latent_dim)
        self.decoder = mlp(history_dim + latent_dim + direct_goal_dim, widths, action_dim)
        nn.init.orthogonal_(self.decoder[-1].weight, gain=0.01)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, observation):
        observation = self.norm(observation)
        history = observation[..., : self.history_dim]
        goal = observation[..., self.history_dim :]
        latent = self.encoder(goal).tanh()
        return self.decoder(torch.cat((history, latent, goal[..., :self.direct_goal_dim]), dim=-1)).tanh()
