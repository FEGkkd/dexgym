"""Causal 50 Hz fixed-base target input and stateful PyTorch policy inference.

This is a simulation/recorded-stream interface; it does not command hardware.
"""

import torch

from dexgym.tasks.marvin_supermimic.se3_reach_hold import rotation_6d_from_matrix

from .teleop_trajectory import MarvinTeleopWorkspaceFollow


class LiveDualTCPTarget:
    names = ("teleop_live",)
    hashes = {"teleop_live": "causal_fixed_base_50hz_v1"}

    def __init__(self, device, *, period_s=0.02, timestamp_tolerance_s=0.003):
        self.device = torch.device(device)
        self.lengths = torch.tensor([2**31 - 1], dtype=torch.long, device=self.device)
        self.period_s = period_s
        self.timestamp_tolerance_s = timestamp_tolerance_s
        self.frame = -1
        self.last_timestamp = None
        self.current_pose = None

    def reset(self):
        self.frame, self.last_timestamp, self.current_pose = -1, None, None

    def push(self, timestamp_s: float, position: torch.Tensor, rotation: torch.Tensor,
             *, frame_id="marvin_fixed_base"):
        if frame_id != "marvin_fixed_base":
            raise ValueError("TCP target must be expressed in Marvin fixed-base coordinates")
        if (position.shape != (2, 3) or rotation.shape != (2, 3, 3)
                or position.device != self.device or rotation.device != self.device
                or not bool(torch.isfinite(position).all())
                or not bool(torch.isfinite(rotation).all())):
            raise ValueError("Dual TCP target must be finite [2,3] and [2,3,3] on policy device")
        identity = torch.eye(3, device=self.device, dtype=rotation.dtype)
        if (float((rotation.transpose(-1, -2) @ rotation - identity).abs().max()) > 1e-3
                or bool(((torch.linalg.det(rotation) - 1).abs() > 1e-3).any())):
            raise ValueError("TCP target orientation must be a proper SO(3) rotation")
        timestamp = float(timestamp_s)
        if not torch.isfinite(torch.tensor(timestamp)):
            raise ValueError("Target timestamp must be finite")
        if (self.last_timestamp is not None
                and abs(timestamp - self.last_timestamp - self.period_s)
                    > self.timestamp_tolerance_s):
            raise ValueError("TCP targets must arrive in monotonic 50 Hz order")
        self.frame += 1
        self.last_timestamp = timestamp
        self.current_pose = torch.cat((position,
            rotation_6d_from_matrix(rotation)), dim=-1).reshape(1, 18).detach().clone()

    def pose_at(self, motion_id, frame):
        if (self.current_pose is None or motion_id.shape != frame.shape
                or bool((motion_id != 0).any())
                or bool((frame != self.frame).any())):
            raise ValueError("Only the current received TCP target is available")
        return self.current_pose.expand(len(frame), -1)


class StatefulTeleopPolicy:
    """Turns one 50 Hz target and 42D arm state into a bounded arm command."""

    def __init__(self, actor, robot, *, device):
        self.actor, self.robot = actor, robot
        self.reference = LiveDualTCPTarget(device)
        self.task = MarvinTeleopWorkspaceFollow(self.reference, robot)
        self.actor.eval()
        self.reset()

    def reset(self):
        self.reference.reset()
        self.history = None
        self.previous_action = torch.zeros(1, self.robot.spec.action_dim,
                                           device=self.reference.device)

    @torch.no_grad()
    def step(self, physical_state, timestamp_s, target_position, target_rotation,
             *, frame_id="marvin_fixed_base"):
        if (physical_state.shape != (1, self.robot.spec.state_dim)
                or physical_state.device != self.reference.device
                or not bool(torch.isfinite(physical_state).all())):
            raise ValueError("Policy input needs one finite physical arm-only state")
        self.reference.push(timestamp_s, target_position, target_rotation, frame_id=frame_id)
        task_state = self.task.task_state(
            torch.zeros(1, dtype=torch.long, device=self.reference.device),
            torch.tensor([self.reference.frame], device=self.reference.device))
        observation = self.task.observation(physical_state, self.previous_action, task_state)
        if self.history is None:
            self.history = observation[:, None].expand(-1, self.task.spec.history_length, -1).clone()
        else:
            self.history = torch.cat((self.history[:, 1:], observation[:, None]), dim=1)
        action = self.actor(self.history, self.task.goal(physical_state, task_state))
        if action.shape != (1, self.robot.spec.action_dim) or not bool(torch.isfinite(action).all()):
            raise RuntimeError("Actor produced an invalid arm action")
        command = self.robot.executed_command(physical_state, action)
        self.previous_action = action.detach().clone()
        return action, command
