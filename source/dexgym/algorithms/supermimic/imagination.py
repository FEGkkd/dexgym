"""Thirty-two-step BPTT through a frozen world model, with TD(lambda) critic targets."""

import torch
from dexgym.motion.motion_manager import MotionManager

from .action import ArmActionMap
from .differentiable_fk import DifferentiableMarvinFK
from .networks import DeltaWorldModel, GoalActor, PrivilegedCritic
from .rewards import RewardWeights, tracking_reward


def lambda_returns(rewards: torch.Tensor, values: torch.Tensor, *, gamma=0.99, lam=0.95):
    """Rewards [H,B], values [H+1,B]; bootstrap after the imagined horizon."""
    result = torch.zeros_like(rewards)
    running = values[-1]
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * ((1 - lam) * values[t + 1] + lam * running)
        result[t] = running
    return result


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
):
    state = start["state"]
    history = start["history"]
    frame = start["frame"]
    rewards, states, frames, values, prior_costs = [], [], [], [], []
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
        )
        rewards.append(reward)
        observation = torch.cat(
            (
                (next_state[:, :14] - mapping.origin) / mapping.scale,
                next_state[:, 14:28] / 2.0,
                action.tanh(),
                (next_state[:, 28:42] - mapping.origin) / mapping.scale,
            ),
            dim=-1,
        )
        history = torch.cat((history[:, 1:], observation[:, None]), dim=1)
        state = next_state
    terminal_frame = (frame + horizon).clamp(max=motion.length - 1)
    terminal_goal = motion.goal_at(terminal_frame)
    values.append(critic(state, motion.data["q_arm"][terminal_frame], terminal_goal[:, 0]))
    reward_tensor = torch.stack(rewards)
    value_tensor = torch.stack(values)
    discounts = gamma ** torch.arange(horizon, device=state.device, dtype=state.dtype)
    objective = (discounts[:, None] * reward_tensor).sum(0) + gamma**horizon * value_tensor[-1]
    if prior_costs:
        objective = objective - prior_weight * (discounts[:, None] * torch.stack(prior_costs)).sum(0)
    return {
        "actor_loss": -objective.mean(),
        "imagined_rewards": reward_tensor,
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
        targets = lambda_returns(rollout["imagined_rewards"].detach(), values, gamma=gamma, lam=lam)
    return (prediction - targets).square().mean()
