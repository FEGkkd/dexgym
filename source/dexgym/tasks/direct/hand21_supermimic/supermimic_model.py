"""Existing delta world model plus a real-label model of sparse goal rewards."""

import torch
from torch.nn import functional as F

from dexgym.algorithms.supermimic.core.models import StateDeltaWorldModel, mlp

from .subgoals import HandGoals


class HandWorldModel(StateDeltaWorldModel):
    # Stable checkpoint identity from before the task-directory split.
    checkpoint_class_name = "dexgym.tasks.direct.right_hand_tracking.supermimic_model.HandWorldModel"

    def __init__(self, robot, config, task):
        super().__init__(robot, config)
        self.reach_head = mlp((7, 64, 64, 1))
        torch.nn.init.constant_(self.reach_head[-1].bias, -4.0)
        # Task is not a torch Module; avoid a second parameter owner.
        self.task_adapter = task
        task.reward_head = self.reach_head
        self.last_event_metrics = {}

    def autoregressive_loss(self, window, robot_adapter, **kwargs):
        loss, errors = super().autoregressive_loss(window, robot_adapter, **kwargs)
        active = window.mask.bool()
        arrival = window.next_state[active]
        old = window.task_state.index(active)
        next_goals = HandGoals.unpack(window.next_task_state.data[active])
        old_goals = HandGoals.unpack(old.data)
        reached = (next_goals.goals_reached > old_goals.goals_reached).float()
        features = self.task_adapter.reach_features(arrival, old).detach()
        logits = self.reach_head(features).squeeze(-1)
        event_loss = F.binary_cross_entropy_with_logits(logits, reached)
        self.last_event_metrics = {
            "reach_bce": float(event_loss.detach()),
            "reach_label_rate": float(reached.mean()),
            "reach_prediction_rate": float(logits.detach().sigmoid().mean()),
        }
        return loss + event_loss, errors
