"""Hand v4: executed-command dynamics and Dex physical-error scaling.

The reach reward head, its real labels and mean BCE remain in HandWorldModel.
Termination BCE retains its task-specific training objective and optimizer.
"""

import torch
from torch import nn
from torch.nn import functional as F

from dexgym.algorithms.supermimic.core.models import TaskTerminationHead

from .supermimic_model import HandWorldModel
from .supermimic_normalization import RunningNorm


class HandDexWorldModel(HandWorldModel):
    checkpoint_class_name = "dexgym.tasks.direct.hand21_supermimic.supermimic_dex_model.HandDexWorldModel"

    def __init__(self, robot, config, task):
        super().__init__(robot, config, task)
        self.input_norm = (
            RunningNorm(robot.state_dim + robot.action_dim + robot.context_dim) if config.normalize else nn.Identity()
        )
        self.delta_norm = RunningNorm(robot.delta_dim, eps=1e-4) if config.normalize else nn.Identity()
        self.register_buffer("error_floor", torch.tensor([0.003] * 20 + [0.05] * 20))

    def predict_delta(self, state, command, context):
        if state.shape != (len(state), 60) or command.shape != (len(state), 20) or context.shape != (len(state), 0):
            raise ValueError("Hand WM state/executed-command/context schema differs")
        delta = self.network(self.input_norm(torch.cat((state, context, command), -1)))
        if isinstance(self.delta_norm, RunningNorm):
            delta = delta * self.delta_norm.std() + self.delta_norm.mean
        return delta

    def step_command(self, state, command, context):
        delta = self.predict_delta(state, command, context)
        return torch.cat((state[:, :40] + delta, command), -1)

    def forward(self, state, action, context, robot_adapter):
        # This conversion stays differentiable during Actor BPTT.
        return self.step_command(state, robot_adapter.executed_command(state, action), context)

    @torch.no_grad()
    def update_norms(self, state, command, context, next_state, robot_adapter):
        if isinstance(self.input_norm, RunningNorm):
            self.input_norm.update(torch.cat((state, context, command), -1))
            self.delta_norm.update(robot_adapter.physical_delta(state, next_state))

    def dynamics_loss(self, window, robot_adapter, *, termination_head=None, task=None, termination_pos_weight=1.0):
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
        state_sum = prediction.new_zeros(())
        bce_sum = prediction.new_zeros(())
        errors = []
        scale = (
            torch.maximum(self.delta_norm.std(), self.error_floor)
            if isinstance(self.delta_norm, RunningNorm)
            else self.error_floor
        )
        for step in range(8):
            # Use captured physical targets. Recomputing from predicted q_cmd
            # would train on different inputs than the transitions in replay.
            proposal = self.step_command(prediction, window.executed_command[:, step], window.dynamics_context[:, step])
            active = mask[:, step]
            prediction = torch.where(active[:, None], proposal, prediction)
            physical_error = robot_adapter.physical_delta(window.next_state[:, step], prediction)
            state_sum = state_sum + (physical_error[active] / scale).square().mean(-1).sum()
            errors.append(torch.where(active, physical_error.square().mean(-1).sqrt(), 0.0))
            if termination_head is not None and bool(active.any()):
                arrival = window.next_task_state.index((slice(None), step))
                features = task.termination_features(prediction[active], arrival.index(active))
                logits = termination_head.logits(prediction[active], features)
                bce_sum = bce_sum + F.binary_cross_entropy_with_logits(
                    logits,
                    window.terminated[:, step][active].float(),
                    reduction="sum",
                    pos_weight=logits.new_tensor(termination_pos_weight),
                )
        # Dex sums time and averages trajectories, not active time slots.
        loss = state_sum / mask.shape[0]
        if termination_head is not None:
            loss = loss + bce_sum / mask.sum()
        return loss, torch.stack(errors, 1)


class HandDexTerminationHead(TaskTerminationHead):
    """Normalize real arrival features; preserve the existing failure objective."""

    def __init__(self, robot, task, config):
        super().__init__(robot, task, config)
        self.input_norm = RunningNorm(robot.state_dim + self.feature_dim) if config.normalize else nn.Identity()

    @torch.no_grad()
    def update_norms(self, state, features):
        if hasattr(self.input_norm, "update"):
            self.input_norm.update(torch.cat((state, features), -1))

    def logits(self, state, features):
        if state.shape[-1] != self.robot_spec.state_dim or features.shape != (*state.shape[:-1], self.feature_dim):
            raise ValueError("Task termination feature schema differs")
        return self.network(self.input_norm(torch.cat((state, features), -1))).squeeze(-1)
