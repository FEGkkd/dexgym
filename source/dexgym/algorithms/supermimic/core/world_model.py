"""Eight-step masked autoregressive world-model fitting over true physics transitions."""

import torch
from torch.nn import functional as F

from .transition import WindowBatch


def masked_autoregressive_loss(model, window: WindowBatch, robot_adapter, *,
                               termination_head=None, task=None,
                               termination_pos_weight: float = 1.0):
    if window.action.ndim != 3 or window.action.shape[1] != 8:
        raise ValueError("World Model requires eight physical transition slots")
    mask = window.mask.bool()
    if mask.shape != window.action.shape[:2] or not bool(mask[:, 0].all()):
        raise ValueError("World Model mask must include each first real transition")
    if termination_head is not None and task is None:
        raise ValueError("Learned termination requires a Task MDP")
    if termination_pos_weight <= 0:
        raise ValueError("Termination positive weight must be positive")
    prediction = window.state[:, 0]
    squared_error = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    bce_sum = torch.zeros_like(squared_error)
    step_errors = []
    for step in range(8):
        proposal = model(prediction, window.action[:, step],
                         window.dynamics_context[:, step], robot_adapter)
        active = mask[:, step]
        # Invalid slots may contain a reset from another episode. They never
        # alter the prediction or the loss for this trajectory.
        prediction = torch.where(active[:, None], proposal, prediction)
        physical_error = robot_adapter.physical_delta(window.next_state[:, step], prediction)
        normalized_error = physical_error / model.delta_norm.std()
        squared_error = squared_error + normalized_error[active].square().sum()
        step_errors.append(torch.where(active, physical_error.square().mean(-1).sqrt(),
                                       torch.zeros_like(active, dtype=prediction.dtype)))
        if termination_head is not None and bool(active.any()):
            arrival = window.next_task_state.index((slice(None), step))
            features = task.termination_features(prediction[active], arrival.index(active))
            logits = termination_head.logits(prediction[active], features)
            labels = window.terminated[:, step][active].float()
            bce_sum = bce_sum + F.binary_cross_entropy_with_logits(
                logits, labels, reduction="sum",
                pos_weight=torch.as_tensor(termination_pos_weight, device=logits.device))
    valid_elements = mask.sum() * model.robot_spec.delta_dim
    loss = squared_error / valid_elements
    if termination_head is not None:
        loss = loss + bce_sum / mask.sum()
    return loss, torch.stack(step_errors, dim=1)
