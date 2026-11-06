"""Config-driven World Model, Actor, Critic and task termination networks."""

from dataclasses import dataclass

import torch
from torch import nn

from .normalization import RunningNorm
from .dynamics_contract import (first_dynamics_difference,
                                world_model_dynamics_signature)
from .specs import NetworkSpec, RobotSpec, TaskSpec


def mlp(widths: tuple[int, ...], *, output_zero: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (before, after) in enumerate(zip(widths[:-1], widths[1:])):
        linear = nn.Linear(before, after)
        if output_zero and index == len(widths) - 2:
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)
        layers.append(linear)
        if index != len(widths) - 2:
            layers.extend((nn.LayerNorm(after), nn.SiLU()))
    return nn.Sequential(*layers)


def _norm(dimension: int, config: NetworkSpec) -> nn.Module:
    return (RunningNorm(dimension, min_std=config.normalizer_min_std,
                        max_count=config.normalizer_max_count)
            if config.normalize else nn.Identity())


class GoalActor(nn.Module):
    def __init__(self, robot: RobotSpec, task: TaskSpec, config: NetworkSpec):
        super().__init__()
        self.robot_spec, self.task_spec = robot, task
        self.network_spec = config
        self.goal_latent_tanh = config.goal_latent_tanh
        self.observation_norm = _norm(task.observation_dim, config)
        self.goal_norm = _norm(task.goal_dim, config)
        self.goal_encoder = mlp((task.goal_dim * task.goal_horizon,
                                 *config.goal_hidden, config.latent_dim))
        self.decoder = mlp((task.observation_dim * task.history_length + config.latent_dim,
                            *config.actor_hidden, robot.action_dim), output_zero=True)

    @torch.no_grad()
    def update_norms(self, history: torch.Tensor, goal: torch.Tensor):
        self._validate(history, goal)
        if isinstance(self.observation_norm, RunningNorm):
            # Each real physical observation enters the statistics once. Old
            # history frames and sampled Replay windows are not reweighted.
            self.observation_norm.update(history[:, -1])
            self.goal_norm.update(goal)

    def _validate(self, history: torch.Tensor, goal: torch.Tensor):
        task = self.task_spec
        if history.ndim != 3 or history.shape[1:] != (task.history_length, task.observation_dim):
            raise ValueError("Actor history shape differs from TaskSpec")
        if goal.ndim != 3 or goal.shape != (len(history), task.goal_horizon, task.goal_dim):
            raise ValueError("Actor goal shape differs from TaskSpec")

    def forward(self, history: torch.Tensor, goal: torch.Tensor):
        self._validate(history, goal)
        latent = self.goal_encoder(self.goal_norm(goal).flatten(-2))
        if self.goal_latent_tanh:
            latent = latent.tanh()
        return self.decoder(torch.cat((self.observation_norm(history).flatten(-2), latent), dim=-1))


class Critic(nn.Module):
    def __init__(self, task: TaskSpec, config: NetworkSpec):
        super().__init__()
        self.task_spec = task
        self.network_spec = config
        self.input_norm = _norm(task.critic_dim, config)
        self.network = mlp((task.critic_dim, *config.critic_hidden, 1))

    @torch.no_grad()
    def update_norms(self, features: torch.Tensor):
        if isinstance(self.input_norm, RunningNorm):
            self.input_norm.update(features)

    def forward(self, features: torch.Tensor):
        if features.shape[-1] != self.task_spec.critic_dim:
            raise ValueError("Critic feature dimension differs from TaskSpec")
        return self.network(self.input_norm(features)).squeeze(-1)


class StateDeltaWorldModel(nn.Module):
    def __init__(self, robot: RobotSpec, config: NetworkSpec):
        super().__init__()
        self.robot_spec = robot
        self.network_spec = config
        self.input_norm = _norm(robot.state_dim + robot.action_dim + robot.context_dim, config)
        self.delta_norm = _norm(robot.delta_dim, config)
        self.network = mlp((robot.state_dim + robot.action_dim + robot.context_dim,
                            *config.world_hidden, robot.delta_dim), output_zero=True)

    def predict_delta(self, state: torch.Tensor, action: torch.Tensor,
                      context: torch.Tensor) -> torch.Tensor:
        spec = self.robot_spec
        if (state.ndim != 2 or state.shape[1] != spec.state_dim
                or action.shape != (len(state), spec.action_dim)
                or context.shape != (len(state), spec.context_dim)):
            raise ValueError("World Model state/action/context schema differs")
        features = torch.cat((state, action, context), dim=-1)
        delta = self.network(self.input_norm(features))
        if isinstance(self.delta_norm, RunningNorm):
            delta = delta * self.delta_norm.std() + self.delta_norm.mean
        return delta

    def forward(self, state: torch.Tensor, action: torch.Tensor,
                context: torch.Tensor, robot_adapter) -> torch.Tensor:
        return robot_adapter.integrate(state, self.predict_delta(state, action, context),
                                       action, context)

    @torch.no_grad()
    def update_norms(self, state: torch.Tensor, action: torch.Tensor,
                     context: torch.Tensor, next_state: torch.Tensor, robot_adapter):
        if isinstance(self.input_norm, RunningNorm):
            self.input_norm.update(torch.cat((state, action, context), dim=-1))
            self.delta_norm.update(robot_adapter.physical_delta(state, next_state))

    def autoregressive_loss(self, window, robot_adapter, **kwargs):
        from .world_model import masked_autoregressive_loss

        return masked_autoregressive_loss(self, window, robot_adapter, **kwargs)


class TaskTerminationHead(nn.Module):
    def __init__(self, robot: RobotSpec, task: TaskSpec, config: NetworkSpec):
        super().__init__()
        self.robot_spec, self.task_spec = robot, task
        self.network_spec = config
        width = task.termination_feature_dim or task.goal_dim
        self.feature_dim = width
        self.network = mlp((robot.state_dim + width, *config.termination_hidden, 1))
        nn.init.constant_(self.network[-1].bias, -4.0)

    def logits(self, state: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != self.robot_spec.state_dim or features.shape != (*state.shape[:-1], self.feature_dim):
            raise ValueError("Task termination feature schema differs")
        return self.network(torch.cat((state, features), dim=-1)).squeeze(-1)

    def forward(self, state: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(state, features))


@dataclass
class NetworkModules:
    world_model: StateDeltaWorldModel
    actor: GoalActor
    critic: Critic
    termination: TaskTerminationHead


class NetworkFactory:
    @staticmethod
    def build(robot: RobotSpec, task: TaskSpec, config: NetworkSpec, *,
              robot_adapter=None,
              shared_world_model: StateDeltaWorldModel | None = None) -> NetworkModules:
        if shared_world_model is not None and shared_world_model.robot_spec != robot:
            raise ValueError("Shared World Model robot/action/context schema differs")
        if robot_adapter is not None and robot_adapter.spec != robot:
            raise ValueError("Robot Adapter and RobotSpec differ")
        signature = (world_model_dynamics_signature(robot_adapter, config)
                     if robot_adapter is not None else None)
        if shared_world_model is not None:
            old = getattr(shared_world_model, "dynamics_signature", None)
            if old is None or signature is None:
                raise ValueError("Shared World Model requires explicit source and target physics adapters")
            if old[1] != signature[1]:
                raise ValueError("Shared World Model dynamics mismatch: "
                                 + str(first_dynamics_difference(old[0], signature[0])))
        world_model = shared_world_model if shared_world_model is not None else StateDeltaWorldModel(robot, config)
        if signature is not None and shared_world_model is None:
            world_model.dynamics_signature = signature
        return NetworkModules(world_model, GoalActor(robot, task, config),
                              Critic(task, config), TaskTerminationHead(robot, task, config))
