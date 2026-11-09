"""One shared discrete TELEDEXTER goal transition for physics and imagination."""

from dataclasses import dataclass, fields

import torch

from .logic import curriculum_stride_upper, sample_goal_span, teledexter_hand_dwell, update_failure_count


@dataclass
class HandGoals:
    motion_ids: torch.Tensor
    goal_frame: torch.Tensor
    goal_span: torch.Tensor
    dwell: torch.Tensor
    dwell_required: torch.Tensor
    fail_frames: torch.Tensor
    fail_budget: torch.Tensor
    curriculum_steps: torch.Tensor
    goal_age: torch.Tensor
    goals_reached: torch.Tensor
    completed: torch.Tensor

    @classmethod
    def from_env(cls, env):
        return cls(**{f.name: getattr(env, f.name).clone() for f in fields(cls)})

    def apply(self, env):
        for f in fields(self):
            setattr(env, f.name, getattr(self, f.name))

    def pack(self):
        return torch.stack([getattr(self, f.name).float() for f in fields(self)], -1)

    @classmethod
    def unpack(cls, data):
        if data.shape[-1] != len(fields(cls)):
            raise ValueError("Hand goal state must contain eleven fields")
        values = {}
        for i, f in enumerate(fields(cls)):
            value = data[:, i]
            values[f.name] = (
                value.bool() if f.name == "completed" else (value if f.name == "fail_budget" else value.long())
            )
        return cls(**values)


@dataclass
class GoalAdvance:
    goals: HandGoals
    errors: torch.Tensor
    within: torch.Tensor
    reached: torch.Tensor
    crossed: torch.Tensor
    exceeded_budget: torch.Tensor
    stride_upper: torch.Tensor


def advance_hand_goals(current_tips, old: HandGoals, bank, cfg) -> GoalAdvance:
    """Score the old goal and sample its successor only after a successful hold.

    No tensors in ``old`` are mutated. Reference switches do not reset a physical
    episode. Random draws intentionally retain the original PPO task's ordering.
    """
    goal_tips = bank.fingertips(old.motion_ids, old.goal_frame)
    errors, within, reached, next_dwell = teledexter_hand_dwell(
        current_tips, goal_tips, old.dwell, old.dwell_required, tolerance_m=cfg.goal_tolerance_m
    )
    upper = curriculum_stride_upper(
        old.curriculum_steps,
        minimum=cfg.curriculum_stride_min,
        maximum=cfg.curriculum_stride_max,
        anneal_steps=cfg.curriculum_anneal_steps,
        sigma_min=cfg.curriculum_sigma_min,
    )
    count, device = len(current_tips), current_tips.device
    lengths = bank.lengths[old.motion_ids]
    at_end = old.goal_frame >= lengths - 1
    wants_switch = reached & ((torch.rand(count, device=device) < cfg.cross_trajectory_probability) | at_end)
    crossed = torch.zeros_like(reached)
    motion, frame, span, budget = (
        old.motion_ids.clone(),
        old.goal_frame.clone(),
        old.goal_span.clone(),
        old.fail_budget.clone(),
    )
    if wants_switch.any():
        candidates, available = bank.sample_other_at_frame(old.motion_ids[wants_switch], old.goal_frame[wants_switch])
        motion[wants_switch], crossed[wants_switch] = candidates, available
    span = torch.where(crossed, torch.zeros_like(span), span)
    budget = torch.where(crossed, cfg.cross_trajectory_failure_budget, budget)
    forward = reached & ~crossed & ~at_end
    if forward.any():
        frame[forward] = torch.minimum(old.goal_frame[forward] + sample_goal_span(upper[forward]), lengths[forward] - 1)
        span[forward] = frame[forward] - old.goal_frame[forward]
        budget[forward] = cfg.failure_tolerance_scale * span[forward]
    failures = update_failure_count(old.fail_frames, within, reached)
    sampled_dwell = torch.randint(cfg.dwell_min_steps, cfg.dwell_max_steps + 1, (count,), device=device)
    goals = HandGoals(
        motion,
        frame,
        span,
        torch.where(reached, 0, next_dwell),
        torch.where(reached, sampled_dwell, old.dwell_required),
        failures,
        budget,
        old.curriculum_steps + 1,
        torch.where(reached, 0, old.goal_age + 1),
        old.goals_reached + reached.long(),
        old.completed | (reached & at_end & ~crossed),
    )
    return GoalAdvance(goals, errors, within, reached, crossed, failures > old.fail_budget, upper)


def unsafe_hand_state(q, qdot, limits):
    return (
        ~torch.isfinite(q).all(-1)
        | ~torch.isfinite(qdot).all(-1)
        | (q < limits[:, 0] - 0.05).any(-1)
        | (q > limits[:, 1] + 0.05).any(-1)
    )


def paper_hand_reward(errors, reached, goal_span, current_landmarks, goal_landmarks, *, dense_scale=0.1, time_cost=0.1):
    """TELEDEXTER hand terms: fingertip sparse + fingertip/knuckle dense terms.

    Landmarks are [batch, tips/lvl1/lvl2, five fingers, xyz], wrist-relative.
    The original simplified PPO dense reward remains available in logic.py.
    """
    decay = errors.new_tensor([100.0, 90.0, 90.0, 90.0, 90.0])
    score = 0.75 * torch.exp(-errors * decay).sum(-1)
    weight = torch.where(goal_span == 0, 100.0, goal_span.float() + 5.0)
    intermediate = torch.linalg.vector_norm(current_landmarks[:, 1:] - goal_landmarks[:, 1:], dim=-1)
    tips = (torch.exp(-errors * decay) * errors.new_tensor([1.0, 0.8, 0.8, 0.8, 0.8])).sum(-1)
    dense = (
        tips
        + 0.6 * torch.exp(-50.0 * intermediate[:, 0].mean(-1))
        + 0.4 * torch.exp(-40.0 * intermediate[:, 1].mean(-1))
    )
    return reached.float() * weight * score + dense_scale * dense - time_cost
