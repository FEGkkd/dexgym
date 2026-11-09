"""Dex-style direct Actor/Critic for causal single-hand goals.

Legacy v3 returns pre-tanh actions; v4 returns tanh-bounded actions. Each
version uses its matching adapter in physics and imagination. Both consume
current goals without future reference frames.
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


class HandDexActor(HandDirectActor):
    """v4: stacked empirical normalization, then ELU MLP and tanh."""

    checkpoint_output_activation = "tanh_action_v1"
    normalize_before_collection = True

    def __init__(self, robot, task, config):
        super().__init__(robot, task, config)
        from .supermimic_normalization import EmpiricalNorm

        del self.observation_norm, self.goal_norm
        width = task.observation_dim * task.history_length + task.goal_dim * task.goal_horizon
        self.input_norm = EmpiricalNorm(width) if config.normalize else nn.Identity()

    def inputs(self, history, goal):
        self._validate(history, goal)
        return torch.cat((history.flatten(-2), goal.flatten(-2)), -1)

    @torch.no_grad()
    def update_norms(self, history, goal):
        if hasattr(self.input_norm, "update"):
            features = self.inputs(history, goal)
            features = features[torch.isfinite(features).all(-1)]
            if len(features):
                self.input_norm.update(features)

    def forward(self, history, goal):
        return self.decoder(self.input_norm(self.inputs(history, goal))).tanh()


class HandDexCritic(HandCritic):
    def __init__(self, task, config):
        super().__init__(task, config)
        from .supermimic_normalization import RunningNorm

        self.input_norm = RunningNorm(task.critic_dim) if config.normalize else nn.Identity()

    @torch.no_grad()
    def update_norms(self, features):
        if hasattr(self.input_norm, "update"):
            self.input_norm.update(features)
