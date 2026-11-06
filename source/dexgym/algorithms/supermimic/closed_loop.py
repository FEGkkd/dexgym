"""Distinct action-conditioned and independently closed-loop model predictions."""

import torch

from .observation import noisy_proprio_frame
from .transition import ReferenceTimeIndex


@torch.no_grad()
def model_actor_rollout(actor, world_model, mapping, reference, state, history, frame, *,
                        horizon: int, physical_actions: torch.Tensor | None = None):
    if state.shape[-1] != 42 or history.shape != (len(state), 10, 56) or frame.shape != (len(state),):
        raise ValueError("Model validation requires aligned 42D state, history and frame")
    if horizon < 1 or not bool(ReferenceTimeIndex(
            frame, torch.full_like(frame, reference.length)).full_horizon_valid(horizon).all()):
        raise ValueError("Model validation would cross the motion boundary")
    if physical_actions is not None and physical_actions.shape != (horizon, len(state), 14):
        raise ValueError("Physical action sequence must align with horizon and batch")
    predicted, predicted_history = state.clone(), history.clone()
    states, actions = [], []
    for offset in range(horizon):
        goal = reference.future_goal_at(frame + offset)
        action = (actor(predicted_history, goal) if physical_actions is None else
                  physical_actions[offset])
        predicted = world_model(predicted, action, mapping)
        observation = torch.cat((
            (predicted[:, :14] - mapping.origin) / mapping.scale,
            predicted[:, 14:28] / 2.0,
            action.tanh(),
            (predicted[:, 28:42] - mapping.origin) / mapping.scale,
        ), dim=-1)
        predicted_history = torch.cat((predicted_history[:, 1:],
                                       noisy_proprio_frame(observation, mapping.scale)[:, None]), dim=1)
        states.append(predicted)
        actions.append(action)
    return {"kind": "policy_closed_loop" if physical_actions is None else
            "action_conditioned_open_loop",
            "states": torch.stack(states), "actions": torch.stack(actions),
            "final_history": predicted_history}
