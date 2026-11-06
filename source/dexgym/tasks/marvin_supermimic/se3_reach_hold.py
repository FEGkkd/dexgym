"""IK-free dual-TCP SE(3) reach-and-hold MDP on the existing Marvin dynamics."""

from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F

from dexgym.algorithms.supermimic.core.specs import TaskSpec
from dexgym.algorithms.supermimic.core.transition import TaskState
from dexgym.algorithms.supermimic.rewards import so3_geodesic, so3_log
from dexgym.algorithms.supermimic.termination import (
    TerminationThresholds, analytic_soft_termination)
from .contact_audit import unsafe_arm_states


def rotation_6d_from_matrix(rotation: torch.Tensor) -> torch.Tensor:
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("Rotation matrix must end in 3×3")
    return torch.cat((rotation[..., :, 0], rotation[..., :, 1]), dim=-1)


def rotation_6d_to_matrix(continuous: torch.Tensor) -> torch.Tensor:
    if continuous.shape[-1] != 6 or not bool(torch.isfinite(continuous).all()):
        raise ValueError("Rotation 6D must be finite and have six components")
    first = continuous[..., :3]
    second = continuous[..., 3:]
    orthogonal = (second - (first * second).sum(-1, keepdim=True) * first
                  / first.square().sum(-1, keepdim=True).clamp_min(1e-8))
    if bool((first.norm(dim=-1) < 0.5).any() or (orthogonal.norm(dim=-1) < 0.5).any()):
        raise ValueError("Rotation 6D axes are degenerate")
    axis_x = F.normalize(first, dim=-1)
    axis_y = F.normalize(second - (axis_x * second).sum(-1, keepdim=True) * axis_x,
                         dim=-1)
    return torch.stack((axis_x, axis_y, torch.cross(axis_x, axis_y, dim=-1)), dim=-1)


def pack_dual_tcp_pose(position: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    if position.shape[-2:] != (2, 3) or rotation.shape[-3:] != (2, 3, 3):
        raise ValueError("Dual TCP pose must have two positions and rotations")
    return torch.cat((position, rotation_6d_from_matrix(rotation)), dim=-1).flatten(-2)


def unpack_dual_tcp_pose(pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pose.shape[-1] != 18 or not bool(torch.isfinite(pose).all()):
        raise ValueError("Dual TCP goal must have 18 finite values")
    per_arm = pose.reshape(*pose.shape[:-1], 2, 9)
    return per_arm[..., :3], rotation_6d_to_matrix(per_arm[..., 3:])


@dataclass(frozen=True)
class SE3RewardConfig:
    position_weight: float = 1.5
    rotation_weight: float = 1.2
    progress_weight: float = 0.45
    low_speed_weight: float = 0.25
    hold_weight: float = 0.8
    action_weight: float = 0.015
    action_rate_weight: float = 0.02
    command_jump_weight: float = 0.02
    joint_soft_limit_weight: float = 0.1
    velocity_weight: float = 0.01
    position_scale_m: float = 0.04
    rotation_scale_rad: float = 0.20
    step_scale: float = 0.02


@dataclass(frozen=True)
class SE3SuccessConfig:
    position_m: float = 0.005
    rotation_rad: float = math.radians(2.0)
    max_joint_speed_rad_s: float = 0.15
    arrival_deadline_steps: int = 125
    hold_steps: int = 25
    episode_steps: int = 200


class MarvinBimanualSE3ReachHold:
    """Task state stores only two target TCP poses; witness q is discarded at reset."""

    termination_mode = "analytic"

    def __init__(self, robot, *, reward_config=SE3RewardConfig(),
                 success_config=SE3SuccessConfig(),
                 safety_thresholds=TerminationThresholds()):
        self.robot = robot
        self.reward_config, self.success_config = reward_config, success_config
        self.safety_thresholds = safety_thresholds
        self.episode_length = success_config.episode_steps
        self.success_hold_steps = success_config.hold_steps
        self.spec = TaskSpec(
            "marvin_bimanual_se3_reach_hold", 68, 10, 18, 1, 18, 60,
            "dual_tcp_base_se3_rotation6d_1x18_v1",
            "arm_q_qd_qcmd_previous_action_tcp_error_log_68_v1",
            "physical_state42_target_pose18_v1", 18)

    def checkpoint_contract(self):
        return {"task_adapter_version": "marvin_bimanual_se3_reach_hold_v1",
                "reward": asdict(self.reward_config),
                "success": asdict(self.success_config),
                "safety_thresholds": asdict(self.safety_thresholds),
                "episode_semantics": "fixed_goal_200_steps_no_success_reset_v1",
                "goal_frame": "fixed_robot_base_v1",
                "witness_joint_angle_usage": "sampler_only_not_transition_v1"}

    @staticmethod
    def task_state_from_goal(goal_pose: torch.Tensor, *, frame=None) -> TaskState:
        unpack_dual_tcp_pose(goal_pose)
        count = len(goal_pose)
        device = goal_pose.device
        return TaskState(torch.full((count,), 2, device=device, dtype=torch.long),
                         torch.full((count,), -1, device=device, dtype=torch.long),
                         torch.zeros(count, device=device, dtype=torch.long) if frame is None else frame,
                         goal_pose)

    @staticmethod
    def physical_task_state(native):
        return native.task_state()

    @staticmethod
    def advance(task_state):
        return TaskState(task_state.task_id, task_state.motion_id,
                         task_state.frame + 1, task_state.data)

    def goal(self, state, task_state):
        return task_state.data[:, None]

    def termination_features(self, state, task_state):
        return task_state.data

    def _errors(self, state, goal_pose):
        position, rotation = self.robot.fk(state[..., :14])
        tcp_position, tcp_rotation = position[..., :, 2, :], rotation[..., :, 2, :, :]
        goal_position, goal_rotation = unpack_dual_tcp_pose(goal_pose)
        position_error = torch.linalg.vector_norm(tcp_position - goal_position, dim=-1)
        rotation_error = so3_geodesic(tcp_rotation, goal_rotation)
        rotation_vector = so3_log(goal_rotation @ tcp_rotation.transpose(-1, -2))
        return position_error, rotation_error, rotation_vector

    def observation(self, state, previous_action, task_state):
        mapping = self.robot.action_map
        _, _, rotation_vector = self._errors(state, task_state.data)
        position = self.robot.fk(state[..., :14])[0]
        goal_position, _ = unpack_dual_tcp_pose(task_state.data)
        position_vector = goal_position - position[..., :, 2, :]
        return torch.cat(((state[..., :14] - mapping.origin) / mapping.scale,
                          state[..., 14:28] / 2.0,
                          (state[..., 28:42] - mapping.origin) / mapping.scale,
                          previous_action.tanh(),
                          position_vector.flatten(-2),
                          rotation_vector.flatten(-2)), dim=-1)

    def critic_features(self, state, task_state):
        return torch.cat((state, task_state.data), dim=-1)

    def reward(self, state, next_state, action, next_task_state, previous_action=None):
        weights = self.reward_config
        goal_pose = next_task_state.data
        old_position, old_rotation, _ = self._errors(state, goal_pose)
        position, rotation, _ = self._errors(next_state, goal_pose)
        velocity = next_state[..., 14:28].abs()
        bounded_action = action.tanh()
        previous_bounded = (torch.zeros_like(bounded_action) if previous_action is None
                            else previous_action.tanh())
        accuracy = (weights.position_weight *
                    torch.exp(-(position / weights.position_scale_m).square()).mean(-1)
                    + weights.rotation_weight *
                    torch.exp(-(rotation / weights.rotation_scale_rad).square()).mean(-1))
        progress = torch.tanh(((old_position - position).mean(-1) / 0.02
                               + (old_rotation - rotation).mean(-1) / 0.15))
        near = (torch.sigmoid((0.02 - position.max(-1).values) / 0.005)
                * torch.sigmoid((0.08 - rotation.max(-1).values) / 0.02))
        slow = torch.exp(-velocity.mean(-1) / self.success_config.max_joint_speed_rad_s)
        hold = (torch.sigmoid((self.success_config.position_m - position.max(-1).values) / 0.002)
                * torch.sigmoid((self.success_config.rotation_rad - rotation.max(-1).values) / 0.01)
                * torch.sigmoid((self.success_config.max_joint_speed_rad_s - velocity.max(-1).values) / 0.03))
        command_jump = (next_state[..., 28:42] - state[..., 28:42]).square().mean(-1) / 0.01**2
        margin = 0.05
        soft_limit = (F.relu(self.robot.action_map.lower + margin - next_state[..., :14]).square()
                      + F.relu(next_state[..., :14] - self.robot.action_map.upper + margin).square()).mean(-1)
        penalty = (weights.action_weight * bounded_action.square().mean(-1)
                   + weights.action_rate_weight * (bounded_action - previous_bounded).square().mean(-1)
                   + weights.command_jump_weight * command_jump
                   + weights.joint_soft_limit_weight * soft_limit
                   + weights.velocity_weight * velocity.square().mean(-1))
        return weights.step_scale * (accuracy + weights.progress_weight * progress
                                     + weights.low_speed_weight * near * slow
                                     + weights.hold_weight * hold - penalty)

    def reward_upper_bound(self):
        weights = self.reward_config
        return weights.step_scale * (weights.position_weight + weights.rotation_weight
                                     + weights.progress_weight + weights.low_speed_weight
                                     + weights.hold_weight)

    def soft_termination(self, next_state, next_task_state):
        return analytic_soft_termination(
            next_state[..., :14], next_state[..., 14:28],
            self.robot.action_map.lower, self.robot.action_map.upper,
            next_state.new_zeros((len(next_state), 2)), self.safety_thresholds)

    def hard_termination(self, next_state, next_task_state, contacts):
        return unsafe_arm_states(
            next_state[..., :14], next_state[..., 14:28],
            self.robot.action_map.lower, self.robot.action_map.upper,
            next_state.new_zeros((len(next_state), 2)), contacts,
            thresholds=self.safety_thresholds)

    def valid_imagination_start(self, task_state, horizon):
        return task_state.frame + horizon <= self.episode_length

    def success_mask(self, transition):
        position, rotation, _ = self._errors(transition.next_state,
                                             transition.next_task_state.data)
        config = self.success_config
        return ((position <= config.position_m).all(-1)
                & (rotation <= config.rotation_rad).all(-1)
                & (transition.next_state[..., 14:28].abs() <= config.max_joint_speed_rad_s).all(-1)
                & ~transition.terminated)

    @staticmethod
    def truncation_reason(transition):
        return "time_limit"

    def _episode_statistics(self, trajectory):
        states = torch.cat([item.next_state for item in trajectory])
        goals = torch.cat([item.next_task_state.data for item in trajectory])
        position, rotation, _ = self._errors(states, goals)
        config = self.success_config
        position_arrived = (position <= config.position_m).all(-1)
        se3_arrived = position_arrived & (rotation <= config.rotation_rad).all(-1)
        valid = se3_arrived & (states[:, 14:28].abs() <= config.max_joint_speed_rad_s).all(-1)

        def first_true(mask):
            found = torch.nonzero(mask, as_tuple=False)
            return int(found[0, 0]) + 1 if len(found) else None

        streak = longest = 0
        first_hold = None
        for step, is_valid in enumerate(valid.tolist(), 1):
            streak = streak + 1 if is_valid else 0
            longest = max(longest, streak)
            if streak >= config.hold_steps and first_hold is None:
                first_hold = step
        return position, rotation, first_true(position_arrived), first_true(se3_arrived), first_hold, longest

    def episode_success(self, trajectory, *, complete: bool):
        _, _, _, first_se3, _, longest = self._episode_statistics(trajectory)
        return bool(complete and len(trajectory) == self.success_config.episode_steps
                    and int(trajectory[-1].next_task_state.frame[0]) == self.success_config.episode_steps
                    and trajectory[-1].truncated[0]
                    and not any(bool(item.terminated[0]) for item in trajectory)
                    and first_se3 is not None
                    and first_se3 <= self.success_config.arrival_deadline_steps
                    and longest >= self.success_config.hold_steps)

    def evaluate_metrics(self, trajectory):
        position, rotation, first_position, first_se3, first_hold, longest = (
            self._episode_statistics(trajectory))
        states = torch.cat([item.next_state for item in trajectory])
        actions = torch.cat([item.action.tanh() for item in trajectory])
        commands = states[:, 28:42]
        action_rate = (actions[1:] - actions[:-1]).abs() if len(actions) > 1 else actions.new_zeros((1, 14))
        command_jump = (commands[1:] - commands[:-1]).abs() if len(commands) > 1 else commands.new_zeros((1, 14))
        dt = self.robot.spec.control_dt
        return {
            "first_position_arrival_step": first_position,
            "first_complete_se3_arrival_step": first_se3,
            "first_valid_hold_step": first_hold,
            "first_position_arrival_time_s": None if first_position is None else first_position * dt,
            "first_complete_se3_arrival_time_s": None if first_se3 is None else first_se3 * dt,
            "first_valid_hold_time_s": None if first_hold is None else first_hold * dt,
            "longest_hold_steps": longest,
            "tcp_position_mean_m": position.mean(0).tolist(),
            "tcp_position_p95_m": torch.quantile(position, 0.95, dim=0).tolist(),
            "tcp_position_max_m": position.max(0).values.tolist(),
            "tcp_position_final_m": position[-1].tolist(),
            "tcp_so3_mean_rad": rotation.mean(0).tolist(),
            "tcp_so3_p95_rad": torch.quantile(rotation, 0.95, dim=0).tolist(),
            "tcp_so3_max_rad": rotation.max(0).values.tolist(),
            "tcp_so3_final_rad": rotation[-1].tolist(),
            "joint_velocity_abs_max_rad_s": float(states[:, 14:28].abs().max()),
            "action_rate_abs_max": float(action_rate.max()),
            "command_jump_abs_max_rad": float(command_jump.max()),
            "contact_event_steps": sum(bool(int(item.failure_reason[0]) & 8) for item in trajectory),
            "joint_violation_steps": sum(bool(int(item.failure_reason[0]) & 2) for item in trajectory),
        }
