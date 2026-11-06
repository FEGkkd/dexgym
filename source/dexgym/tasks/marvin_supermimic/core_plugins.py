"""Marvin robot and two task plugins for the dimension-driven SuperMimic Core.

This module owns all Marvin dimensions, FK, reward and reference semantics. The
algorithm Core imports neither this module nor any Wave asset.
"""

import hashlib

import torch

from dexgym.algorithms.supermimic.core.specs import RobotSpec, TaskSpec
from dexgym.algorithms.supermimic.core.transition import TaskState
from dexgym.algorithms.supermimic.rewards import REWARD_PROFILES, reward_upper_bound, tracking_reward
from dexgym.algorithms.supermimic.rewards import so3_geodesic
from dexgym.algorithms.supermimic.termination import (
    TerminationThresholds, analytic_soft_termination)
from dexgym.tasks.marvin_supermimic.contact_audit import unsafe_arm_states
from dexgym.tasks.marvin_supermimic.robot_adapter import MarvinArmDynamicsAdapter


class MarvinCoreRobot:
    """42D arm state; 28D learned q/qd delta; exact analytic servo command."""

    def __init__(self, reference_for_joint_order, device, *, action_schema="robot_fixed_v3"):
        self.native = MarvinArmDynamicsAdapter(reference_for_joint_order, device,
                                               action_schema=action_schema)
        self.action_map, self.fk = self.native.action_map, self.native.fk
        self.spec = RobotSpec(
            "marvin_arm_only", 42, 14, 28, 0, action_schema,
            "q14_qd14_qcmd14_v1", control_dt=0.02, physics_dt=0.002,
            asset_sha256=hashlib.sha256(self.native.mjcf.read_bytes()).hexdigest(),
            joint_order=tuple(reference_for_joint_order.joint_names_arm))

    def integrate(self, state, physical_delta, action, context):
        if (state.shape[-1] != 42 or physical_delta.shape != (*state.shape[:-1], 28)
                or action.shape != (*state.shape[:-1], 14)
                or context.shape != (*state.shape[:-1], 0)):
            raise ValueError("Marvin state/action/delta/context contract differs")
        command = self.action_map(action, state[..., 28:42])
        return torch.cat((state[..., :28] + physical_delta, command), dim=-1)

    @staticmethod
    def physical_delta(state, next_state):
        return next_state[..., :28] - state[..., :28]

    def executed_command(self, state, action):
        return self.action_map(action, state[..., 28:42])

    def physical_state(self, env):
        return self.native.physical_state(env)


class MarvinMotionTracking:
    """Six-link M4.2 tracking math expressed through the generic task protocol."""

    termination_mode = "analytic"

    def __init__(self, reference, robot: MarvinCoreRobot, *,
                 reward_profile="marvin_tcp_priority", termination_mode="analytic",
                 thresholds=TerminationThresholds()):
        if reward_profile not in REWARD_PROFILES or termination_mode not in ("analytic", "learned"):
            raise ValueError("Unknown tracking reward profile or termination mode")
        self.reference, self.robot = reference, robot
        self.reward_profile, self.termination_mode, self.thresholds = (
            reward_profile, termination_mode, thresholds)
        self.spec = TaskSpec("marvin_motion_tracking", 56, 10, 90, 10, 0, 146,
                             "six_link_future_10x90_v1", "arm_proprio_history_10x56_v1",
                             "state42_reference_q14_next_link90_v1", 90)

    def physical_task_state(self, native):
        motion_index = self.reference.names.index(native.motion.motion_id)
        return TaskState(torch.zeros(native.num_envs, device=native.device, dtype=torch.long),
                         torch.full((native.num_envs,), motion_index, device=native.device, dtype=torch.long),
                         native.motion.frame.clone(),
                         torch.empty((native.num_envs, 0), device=native.device))

    def physical_bootstrap_action(self, native):
        arrival = (native.motion.frame + 1).clamp(max=native.motion.length - 1)
        target = native.motion.data["q_arm"][arrival]
        mapping = self.robot.action_map
        normalized = ((target - mapping.origin) / mapping.scale).clamp(-0.98, 0.98)
        return torch.atanh(normalized)

    def advance(self, task_state):
        return TaskState(task_state.task_id, task_state.motion_id,
                         task_state.frame + 1, task_state.data)

    def goal(self, state, task_state):
        return self.reference.future_goal_at(task_state.motion_id, task_state.frame, 10)

    def termination_features(self, state, task_state):
        return self.reference.reference_features_at(task_state.motion_id, task_state.frame)

    def observation(self, state, previous_action, task_state):
        mapping = self.robot.action_map
        return torch.cat(((state[..., :14] - mapping.origin) / mapping.scale,
                          state[..., 14:28] / 2, previous_action.tanh(),
                          (state[..., 28:42] - mapping.origin) / mapping.scale), dim=-1)

    def critic_features(self, state, task_state):
        reference = self.reference.reference_at(task_state.motion_id, task_state.frame)
        goal = self.goal(state, task_state)
        return torch.cat((state, reference["q_arm"], goal[:, 0]), dim=-1)

    def reward(self, state, next_state, action, next_task_state, previous_action=None):
        ref = self.reference.reference_at(next_task_state.motion_id, next_task_state.frame)
        if previous_action is None:
            previous_action = torch.zeros_like(action)
        reward, _ = tracking_reward(
            next_state[..., :14], state[..., :14], next_state[..., 28:42], state[..., 28:42],
            {key: ref[key] for key in ("link_pos", "link_rotmat",
                                        "link_linear_vel", "link_angular_vel")},
            self.robot.fk, self.robot.action_map.lower, self.robot.action_map.upper,
            weights=REWARD_PROFILES[self.reward_profile], dt=self.robot.spec.control_dt,
            bounded_action=action.tanh(), previous_bounded_action=previous_action.tanh())
        return reward

    def reward_upper_bound(self):
        return reward_upper_bound(REWARD_PROFILES[self.reward_profile], dt=self.robot.spec.control_dt)

    def soft_termination(self, next_state, next_task_state):
        target = self.reference.reference_at(next_task_state.motion_id, next_task_state.frame)
        pos = self.robot.fk(next_state[:, :14])[0][:, :, 2]
        tcp_error = torch.linalg.vector_norm(pos - target["link_pos"][:, :, 2], dim=-1)
        return analytic_soft_termination(
            next_state[:, :14], next_state[:, 14:28], self.robot.action_map.lower,
            self.robot.action_map.upper, tcp_error, self.thresholds)

    def hard_termination(self, next_state, next_task_state, contacts):
        target = self.reference.reference_at(next_task_state.motion_id, next_task_state.frame)
        pos = self.robot.fk(next_state[:, :14])[0][:, :, 2]
        tcp_error = torch.linalg.vector_norm(pos - target["link_pos"][:, :, 2], dim=-1)
        return unsafe_arm_states(next_state[:, :14], next_state[:, 14:28],
                                 self.robot.action_map.lower, self.robot.action_map.upper,
                                 tcp_error, contacts, thresholds=self.thresholds)

    def valid_imagination_start(self, task_state, horizon):
        return task_state.frame + horizon < self.reference.lengths[task_state.motion_id]

    def evaluate_metrics(self, trajectory):
        positions, rotations = [], []
        for transition in trajectory:
            next_state = transition.next_state
            ref = self.reference.reference_at(transition.next_task_state.motion_id,
                                              transition.next_task_state.frame)
            pos, rot = self.robot.fk(next_state[:, :14])
            positions.append(torch.linalg.vector_norm(
                pos[:, :, 2] - ref["link_pos"][:, :, 2], dim=-1))
            rotations.append(so3_geodesic(
                rot[:, :, 2], ref["link_rotmat"][:, :, 2]))
        position = torch.stack(positions).flatten(0, 1)
        rotation = torch.stack(rotations).flatten(0, 1)
        return {"tcp_position_mean_m": position.mean(0).tolist(),
                "tcp_position_p95_m": torch.quantile(position, 0.95, dim=0).tolist(),
                "tcp_position_max_m": position.max(0).values.tolist(),
                "tcp_so3_mean_rad": rotation.mean(0).tolist(),
                "tcp_so3_p95_rad": torch.quantile(rotation, 0.95, dim=0).tolist(),
                "motion_completion_fraction": float((~torch.stack(
                    [item.terminated for item in trajectory]).any(0)).float().mean())}


class MarvinDualTCPReach:
    """Static Cartesian dual-TCP goals, with no six-link or motion-goal encoding."""

    termination_mode = "analytic"

    def __init__(self, robot: MarvinCoreRobot, *, episode_length=100,
                 thresholds=TerminationThresholds()):
        self.robot, self.episode_length, self.thresholds = robot, episode_length, thresholds
        self.spec = TaskSpec("marvin_dual_tcp_reach", 48, 4, 6, 1, 6, 48,
                             "dual_tcp_position_1x6_v1", "arm_state_tcp_error_48_v1",
                             "state42_target_tcp6_v1", 6)

    @staticmethod
    def physical_task_state(native):
        return native.task_state()

    def task_state_from_reset(self, state, *, frame=None):
        current = self.robot.fk(state[:, :14])[0][:, :, 2].reshape(len(state), 6)
        offset = current.new_tensor([0.025, 0.015, 0, -0.025, -0.015, 0])
        target = current.detach() + offset
        return TaskState(torch.ones(len(state), device=state.device, dtype=torch.long),
                         torch.full((len(state),), -1, device=state.device, dtype=torch.long),
                         (torch.zeros(len(state), device=state.device, dtype=torch.long)
                          if frame is None else frame), target)

    @staticmethod
    def advance(task_state):
        return TaskState(task_state.task_id, task_state.motion_id,
                         task_state.frame + 1, task_state.data)

    def goal(self, state, task_state):
        return task_state.data[:, None, :]

    def termination_features(self, state, task_state):
        return task_state.data

    def observation(self, state, previous_action, task_state):
        mapping = self.robot.action_map
        tcp = self.robot.fk(state[:, :14])[0][:, :, 2].reshape(len(state), 6)
        return torch.cat(((state[:, :14] - mapping.origin) / mapping.scale,
                          state[:, 14:28] / 2,
                          (state[:, 28:42] - mapping.origin) / mapping.scale,
                          task_state.data - tcp), dim=-1)

    def critic_features(self, state, task_state):
        return torch.cat((state, task_state.data), dim=-1)

    def reward(self, state, next_state, action, next_task_state, previous_action=None):
        tcp = self.robot.fk(next_state[:, :14])[0][:, :, 2].reshape(len(state), 2, 3)
        target = next_task_state.data.reshape(len(state), 2, 3)
        distance_sq = (tcp - target).square().sum(-1)
        command_rate = (next_state[:, 28:42] - state[:, 28:42]).square().mean(-1)
        return 0.02 * (torch.exp(-40 * distance_sq).mean(-1) - 0.2 * command_rate)

    @staticmethod
    def reward_upper_bound():
        return 0.02

    def soft_termination(self, next_state, next_task_state):
        return analytic_soft_termination(
            next_state[:, :14], next_state[:, 14:28], self.robot.action_map.lower,
            self.robot.action_map.upper, next_state.new_zeros((len(next_state), 2)), self.thresholds)

    def hard_termination(self, next_state, next_task_state, contacts):
        return unsafe_arm_states(
            next_state[:, :14], next_state[:, 14:28], self.robot.action_map.lower,
            self.robot.action_map.upper, next_state.new_zeros((len(next_state), 2)),
            contacts, thresholds=self.thresholds)

    def valid_imagination_start(self, task_state, horizon):
        return task_state.frame + horizon <= self.episode_length

    def evaluate_metrics(self, trajectory):
        states = torch.cat([item.next_state for item in trajectory])
        tcp = self.robot.fk(states[:, :14])[0][:, :, 2].reshape(-1, 2, 3)
        target = torch.cat([item.next_task_state.data for item in trajectory]).reshape(-1, 2, 3)
        distances = torch.linalg.vector_norm(tcp - target, dim=-1)
        return {"tcp_distance_mean_m": float(distances.mean()),
                "tcp_distance_max_m": float(distances.max()),
                "success_rate": float((distances < 0.015).all(-1).float().mean())}
