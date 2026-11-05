"""Thirty-two-step BPTT through a frozen world model, with TD(lambda) critic targets."""

import torch
from dexgym.motion.motion_manager import MotionManager

from .action import ArmActionMap
from .differentiable_fk import DifferentiableMarvinFK
from .networks import DeltaWorldModel, GoalActor, PrivilegedCritic
from .observation import noisy_proprio_frame
from .rewards import RewardWeights, tracking_reward


def survival_weights(termination_probabilities: torch.Tensor) -> torch.Tensor:
    """Return c[0:H+1,B] with c[t+1] = c[t] * (1 - d[t+1])."""
    if termination_probabilities.ndim != 2 or len(termination_probabilities) == 0:
        raise ValueError("Termination probabilities must have shape [H,B] with H > 0")
    first = torch.ones_like(termination_probabilities[:1])
    return torch.cumprod(torch.cat((first, 1 - termination_probabilities), dim=0), dim=0)


def discounted_imagined_return(rewards, termination_probabilities, terminal_value, *, gamma=0.99):
    """Paper Eq. 2: reward on arrival, then soften the survival of later steps."""
    if rewards.shape != termination_probabilities.shape or terminal_value.shape != rewards.shape[1:]:
        raise ValueError("Reward, termination, and terminal value shapes must agree")
    horizon = len(rewards)
    survival = survival_weights(termination_probabilities)
    discounts = gamma ** torch.arange(horizon, device=rewards.device, dtype=rewards.dtype)
    return (discounts[:, None] * survival[:-1] * rewards).sum(0) + gamma**horizon * survival[-1] * terminal_value


def lambda_returns(
    rewards: torch.Tensor, values: torch.Tensor, termination_probabilities=None, *, gamma=0.99, lam=0.95
):
    """Rewards [H,B], values [H+1,B]; bootstrap after the imagined horizon."""
    if values.shape != (len(rewards) + 1, rewards.shape[1]):
        raise ValueError("Critic values must have shape [H+1,B]")
    if termination_probabilities is None:
        termination_probabilities = torch.zeros_like(rewards)
    if termination_probabilities.shape != rewards.shape:
        raise ValueError("Termination probabilities must have shape [H,B]")
    result = []
    running = values[-1]
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * (1 - termination_probabilities[t]) * ((1 - lam) * values[t + 1] + lam * running)
        result.append(running)
    return torch.stack(result[::-1])


def imagined_rollout(
    actor: GoalActor,
    world_model: DeltaWorldModel,
    critic: PrivilegedCritic,
    fk: DifferentiableMarvinFK,
    mapping: ArmActionMap,
    motion: MotionManager,
    start: dict[str, torch.Tensor],
    *,
    horizon: int = 32,
    weights: RewardWeights = RewardWeights(),
    gamma: float = 0.99,
    prior_actor: GoalActor | None = None,
    prior_weight: float = 0.0,
    observation_position_noise_rad: float = 0.0,
    observation_velocity_noise_rad_s: float = 0.0,
):
    state = start["state"]
    history = start["history"]
    frame = start["frame"]
    rewards, terminations, states, frames, values, prior_costs = [], [], [], [], [], []
    for step in range(horizon):
        current = (frame + step).clamp(max=motion.length - 1)
        goal = motion.goal_at(current)
        values.append(critic(state, motion.data["q_arm"][current], goal[:, 0]))
        states.append(state.detach())
        frames.append(current)
        action = actor(history, goal)
        if prior_actor is not None:
            with torch.no_grad():
                prior_action = prior_actor(history.detach(), goal)
            prior_costs.append((action - prior_action).square().mean(-1))
        next_state = world_model(state, action, mapping)
        raw_next = frame + step + 1
        next_frame = raw_next.clamp(max=motion.length - 1)
        target = {
            key: motion.data[key][next_frame]
            for key in ("link_pos", "link_rotmat", "link_linear_vel", "link_angular_vel")
        }
        beyond = raw_next >= motion.length
        for key in ("link_linear_vel", "link_angular_vel"):
            target[key] = torch.where(beyond[:, None, None, None], 0.0, target[key])
        reward, _ = tracking_reward(
            next_state[:, :14],
            state[:, :14],
            next_state[:, 28:42],
            state[:, 28:42],
            target,
            fk,
            mapping.lower,
            mapping.upper,
            weights,
            bounded_action=action.tanh(),
            previous_bounded_action=history[:, -1, 28:42],
        )
        rewards.append(reward)
        next_goal = motion.goal_at(next_frame)
        terminations.append(world_model.termination_probability(next_state, next_goal[:, 0]))
        observation = torch.cat(
            (
                (next_state[:, :14] - mapping.origin) / mapping.scale,
                next_state[:, 14:28] / 2.0,
                action.tanh(),
                (next_state[:, 28:42] - mapping.origin) / mapping.scale,
            ),
            dim=-1,
        )
        observation = noisy_proprio_frame(
            observation,
            mapping.scale,
            position_rad=observation_position_noise_rad,
            velocity_rad_s=observation_velocity_noise_rad_s,
        )
        history = torch.cat((history[:, 1:], observation[:, None]), dim=1)
        state = next_state
    terminal_frame = (frame + horizon).clamp(max=motion.length - 1)
    terminal_goal = motion.goal_at(terminal_frame)
    values.append(critic(state, motion.data["q_arm"][terminal_frame], terminal_goal[:, 0]))
    reward_tensor = torch.stack(rewards)
    termination_tensor = torch.stack(terminations)
    value_tensor = torch.stack(values)
    survival = survival_weights(termination_tensor)
    objective = discounted_imagined_return(reward_tensor, termination_tensor, value_tensor[-1], gamma=gamma)
    if prior_costs:
        discounts = gamma ** torch.arange(horizon, device=state.device, dtype=state.dtype)
        objective = objective - prior_weight * (discounts[:, None] * survival[:-1] * torch.stack(prior_costs)).sum(0)
    return {
        "actor_loss": -objective.mean(),
        "imagined_rewards": reward_tensor,
        "termination_probabilities": termination_tensor,
        "survival_weights": survival,
        "critic_values": value_tensor,
        "states_detached": torch.stack(states),
        "frames": torch.stack(frames),
        "terminal_state_detached": state.detach(),
        "terminal_frame": terminal_frame,
    }


def critic_td_lambda_loss(critic: PrivilegedCritic, motion: MotionManager, rollout: dict, *, gamma=0.99, lam=0.95):
    states = rollout["states_detached"]
    frames = rollout["frames"]
    flat_state = states.reshape(-1, 42)
    flat_frame = frames.reshape(-1)
    goal = motion.goal_at(flat_frame)[:, 0]
    prediction = critic(flat_state, motion.data["q_arm"][flat_frame], goal).reshape(frames.shape)
    with torch.no_grad():
        terminal_frame = rollout["terminal_frame"]
        terminal_goal = motion.goal_at(terminal_frame)[:, 0]
        terminal = critic(rollout["terminal_state_detached"], motion.data["q_arm"][terminal_frame], terminal_goal)
        values = torch.cat((prediction.detach(), terminal[None]), dim=0)
        targets = lambda_returns(
            rollout["imagined_rewards"].detach(),
            values,
            rollout["termination_probabilities"].detach(),
            gamma=gamma,
            lam=lam,
        )
    return (prediction - targets).square().mean()
