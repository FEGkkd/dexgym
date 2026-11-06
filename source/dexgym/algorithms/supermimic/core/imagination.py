"""Task-agnostic short-horizon differentiable model imagination."""

from dataclasses import dataclass

import torch

from .returns import discounted_imagined_return, lambda_returns, survival_weights
from .transition import StartBatch


@dataclass
class ImaginedRollout:
    actor_loss: torch.Tensor
    rewards: torch.Tensor
    terminations: torch.Tensor
    survival: torch.Tensor
    values: torch.Tensor
    critic_features: torch.Tensor
    terminal_features: torch.Tensor
    states: torch.Tensor
    terminal_state: torch.Tensor


class ImaginationEngine:
    def __init__(self, *, gamma: float = 0.99, lam: float = 0.95,
                 value_upper_bound: float | None = None):
        if not 0 <= gamma < 1 or not 0 <= lam <= 1:
            raise ValueError("Imagination gamma or lambda is invalid")
        if value_upper_bound is not None and value_upper_bound <= 0:
            raise ValueError("Value bound must follow a positive reward upper bound")
        self.gamma, self.lam, self.value_upper_bound = gamma, lam, value_upper_bound

    def rollout(self, start: StartBatch, actor, world_model, critic, robot, task, *,
                horizon: int = 32, bootstrap_weight: float = 1.0,
                termination_head=None, observation_noise=None) -> ImaginedRollout:
        if horizon < 1 or not bool(task.valid_imagination_start(start.task_state, horizon).all()):
            raise ValueError("Imagination would cross the task episode boundary")
        state, history, task_state = start.state, start.history, start.task_state
        previous_action = start.previous_action
        context = start.dynamics_context
        if (context.shape != (len(state), robot.spec.context_dim)
                or not bool(torch.isfinite(context).all())):
            raise ValueError("Initial dynamics context differs from RobotSpec")
        evolve_context = robot.spec.context_evolution == "adapter_transition"
        if evolve_context and not callable(getattr(robot, "next_dynamics_context", None)):
            raise ValueError("Time-varying dynamics context requires next_dynamics_context")
        rewards, probabilities, values, features, states = [], [], [], [], []
        for _ in range(horizon):
            goal = task.goal(state, task_state)
            critic_input = task.critic_features(state, task_state)
            if critic_input.shape != (len(state), task.spec.critic_dim):
                raise ValueError("Task Critic features differ from TaskSpec")
            features.append(critic_input.detach())
            values.append(critic(critic_input))
            states.append(state.detach())
            action = actor(history, goal)
            next_state = world_model(state, action, context, robot)
            next_task_state = task.advance(task_state)
            reward = task.reward(state, next_state, action, next_task_state, previous_action)
            if task.termination_mode == "learned":
                if termination_head is None:
                    raise ValueError("Learned termination needs a task-shaped head")
                term_features = task.termination_features(next_state, next_task_state)
                probability = termination_head(next_state, term_features)
            else:
                probability = task.soft_termination(next_state, next_task_state)
            if (reward.shape != (len(state),) or probability.shape != (len(state),)
                    or not bool(torch.isfinite(reward).all() and torch.isfinite(probability).all())
                    or not bool(((probability >= 0) & (probability <= 1)).all())):
                raise ValueError("Task reward or soft termination is invalid")
            rewards.append(reward)
            probabilities.append(probability)
            next_observation = task.observation(next_state, action, next_task_state)
            if next_observation.shape != (len(state), task.spec.observation_dim):
                raise ValueError("Task observation differs from TaskSpec")
            if observation_noise is not None:
                next_observation = observation_noise(next_observation, mode="train")
            history = torch.cat((history[:, 1:], next_observation[:, None]), dim=1)
            if evolve_context:
                context = robot.next_dynamics_context(context, state, action, next_state,
                                                      next_task_state)
                if (context.shape != (len(state), robot.spec.context_dim)
                        or not bool(torch.isfinite(context).all())):
                    raise ValueError("Next dynamics context differs from RobotSpec")
            state, task_state, previous_action = next_state, next_task_state, action
        terminal_features = task.critic_features(state, task_state)
        values.append(critic(terminal_features))
        reward_tensor = torch.stack(rewards)
        termination_tensor = torch.stack(probabilities)
        value_tensor = torch.stack(values)
        bounded_terminal = (value_tensor[-1].clamp(max=self.value_upper_bound)
                            if self.value_upper_bound is not None else value_tensor[-1])
        return ImaginedRollout(
            actor_loss=-discounted_imagined_return(
                reward_tensor, termination_tensor,
                bootstrap_weight * bounded_terminal, gamma=self.gamma).mean(),
            rewards=reward_tensor, terminations=termination_tensor,
            survival=survival_weights(termination_tensor), values=value_tensor,
            critic_features=torch.stack(features),
            terminal_features=terminal_features.detach(), states=torch.stack(states),
            terminal_state=state.detach())


def critic_td_lambda_batch(rollout: ImaginedRollout, *, gamma=0.99, lam=0.95,
                           value_upper_bound: float | None = None):
    with torch.no_grad():
        values = rollout.values.detach()
        if value_upper_bound is not None:
            values = values.clamp(max=value_upper_bound)
        targets = lambda_returns(rollout.rewards.detach(), values,
                                 rollout.terminations.detach(), gamma=gamma, lam=lam)
    return rollout.critic_features.reshape(-1, rollout.critic_features.shape[-1]).detach(), targets.flatten().detach()
