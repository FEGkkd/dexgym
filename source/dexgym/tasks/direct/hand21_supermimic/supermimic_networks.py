"""Dex-style direct Actor/Critic for causal single-hand goals.

Actor returns pre-tanh actions: the existing hand adapter applies tanh exactly
once in both physics and imagination. Current goals replace object commands;
no future reference frames are consumed.
"""

import torch
from torch import nn

from dexgym.algorithms.supermimic.core.models import Critic, GoalActor
from dexgym.algorithms.supermimic.core.normalization import RunningNorm


def elu_mlp(widths):
    layers = []
    for i, (before, after) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(before, after))
        if i < len(widths) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def norm(width, config):
    return (
        RunningNorm(width, min_std=config.normalizer_min_std, max_count=config.normalizer_max_count)
        if config.normalize
        else nn.Identity()
    )


class HandDirectActor(GoalActor):
    def __init__(self, robot, task, config):
        nn.Module.__init__(self)
        self.robot_spec, self.task_spec, self.network_spec = robot, task, config
        self.goal_latent_tanh = False
        self.observation_norm = norm(task.observation_dim, config)
        self.goal_norm = norm(task.goal_dim, config)
        width = task.observation_dim * task.history_length + task.goal_dim * task.goal_horizon
        self.decoder = elu_mlp((width, *config.actor_hidden, robot.action_dim))
        nn.init.orthogonal_(self.decoder[-1].weight, gain=0.01)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, history, goal):
        self._validate(history, goal)
        return self.decoder(
            torch.cat((self.observation_norm(history).flatten(-2), self.goal_norm(goal).flatten(-2)), -1)
        )


class HandCritic(Critic):
    def __init__(self, task, config):
        nn.Module.__init__(self)
        self.task_spec, self.network_spec = task, config
        self.input_norm = norm(task.critic_dim, config)
        self.network = elu_mlp((task.critic_dim, *config.critic_hidden, 1))
