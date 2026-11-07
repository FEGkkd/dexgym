"""Time-indexed dual-TCP pose following; joint witnesses are reset-only data."""

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from dexgym.algorithms.supermimic.core.specs import TaskSpec
from dexgym.algorithms.supermimic.core.transition import TaskState
from dexgym.algorithms.supermimic.rewards import so3_geodesic, so3_log
from dexgym.algorithms.supermimic.termination import TerminationThresholds, analytic_soft_termination
from dexgym.motion.library import MotionLibrary
from dexgym.motion.motion_manager import quaternion_xyzw_to_6d

from .contact_audit import unsafe_arm_states
from .se3_reach_hold import unpack_dual_tcp_pose


class SE3TCPReferenceProvider:
    """Immutable 50 Hz TCP targets; q reference is accessible only to reset physics."""

    def __init__(self, registry: dict[str, str | Path], device: str | torch.device,
                 *, include_reverse: bool = False):
        library = MotionLibrary(registry, device)
        self.device = torch.device(device)
        self.joint_names_arm = library.joint_names_arm
        self.names = tuple(name for original in library.names
                           for name in ((original, original + "_reverse") if include_reverse
                                        else (original,)))
        self.hashes = {}
        poses, resets, lengths = [], [], []
        for name, clip in zip(library.names, library.clips, strict=True):
            pose = torch.cat((clip.data["tcp_pos"],
                              quaternion_xyzw_to_6d(clip.data["tcp_quat_xyzw"])),
                             dim=-1).flatten(-2)
            if pose.shape != (clip.length, 18) or not bool(torch.isfinite(pose).all()):
                raise ValueError("Reference must contain finite dual TCP SE(3) poses")
            variants = ((name, pose, clip.data["q_arm"], "forward_v1"),)
            if include_reverse:
                variants += ((name + "_reverse", pose.flip(0),
                              clip.data["q_arm"].flip(0), "reverse_v1"),)
            for variant_name, variant_pose, variant_q, transform in variants:
                poses.append(variant_pose)
                resets.append(variant_q)
                lengths.append(clip.length)
                source = library.hashes[name] + ":" + transform
                self.hashes[variant_name] = hashlib.sha256(source.encode()).hexdigest()
        self.lengths = torch.tensor(lengths, device=self.device, dtype=torch.long)
        longest = max(lengths)
        self._poses = torch.stack([F.pad(pose, (0, 0, 0, longest - len(pose)),
                                          mode="constant") for pose in poses])
        self._reset_q = torch.stack([F.pad(q, (0, 0, 0, longest - len(q)),
                                            mode="constant") for q in resets])

    def _validate(self, motion_id: torch.Tensor, frame: torch.Tensor):
        if (motion_id.shape != frame.shape or motion_id.ndim != 1
                or motion_id.dtype != torch.long or frame.dtype != torch.long
                or bool(((motion_id < 0) | (motion_id >= len(self.names))).any())):
            raise ValueError("Reference motion ID and frame vectors differ")
        if bool(((frame < 0) | (frame >= self.lengths[motion_id])).any()):
            raise ValueError("Reference frame is outside its motion")

    def pose_at(self, motion_id: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        self._validate(motion_id, frame)
        return self._poses[motion_id, frame]

    def future_goal_at(self, motion_id: torch.Tensor, frame: torch.Tensor,
                       count: int) -> torch.Tensor:
        self._validate(motion_id, frame)
        if count < 1:
            raise ValueError("Future goal count must be positive")
        offsets = torch.arange(1, count + 1, device=self.device)
        indices = torch.minimum(frame[:, None] + offsets,
                                self.lengths[motion_id, None] - 1)
        return self._poses[motion_id[:, None], indices]

    def reset_q_at(self, motion_id: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        """Only the physical environment uses this witness to make safe resets."""
        self._validate(motion_id, frame)
        return self._reset_q[motion_id, frame]


@dataclass(frozen=True)
class ContinuousSE3RewardConfig:
    position_weight: float = 1.5
    rotation_weight: float = 1.2
    progress_weight: float = 0.35
    action_weight: float = 0.01
    action_rate_weight: float = 0.02
    command_jump_weight: float = 0.01
    velocity_weight: float = 0.005
    soft_joint_limit_weight: float = 0.1
    position_scale_m: float = 0.03
    rotation_scale_rad: float = 0.15
    step_scale: float = 0.02


@dataclass(frozen=True)
class ContinuousSE3SuccessConfig:
    mean_position_m: float = 0.005
    p95_position_m: float = 0.010
    mean_rotation_rad: float = math.radians(2.0)
    p95_rotation_rad: float = math.radians(4.0)


class MarvinContinuousSE3Follow:
    """The reward uses only physical q and TCP targets, never reference q."""

    termination_mode = "analytic"
    success_hold_steps = 1

    def __init__(self, reference: SE3TCPReferenceProvider, robot, *,
                 reward_config=ContinuousSE3RewardConfig(),
                 success_config=ContinuousSE3SuccessConfig(),
                 safety_thresholds=TerminationThresholds()):
        self.reference, self.robot = reference, robot
        self.reward_config, self.success_config = reward_config, success_config
        self.safety_thresholds = safety_thresholds
        self.spec = TaskSpec(
            "marvin_bimanual_continuous_se3_follow", 68, 10, 18, 10, 0, 222,
            "future_dual_tcp_se3_rotation6d_10x18_v1",
            "arm_q_qd_qcmd_previous_action_current_tcp_error_log_68_v1",
            "physical_state42_future_target_pose180_v1", 18)

    def checkpoint_contract(self):
        return {"task_adapter_version": "marvin_continuous_se3_follow_v1",
                "reward": asdict(self.reward_config),
                "success": asdict(self.success_config),
                "safety_thresholds": asdict(self.safety_thresholds),
                "reference_hashes": self.reference.hashes,
                "episode_semantics": "frame_t_action_arrival_t_plus_1_clip_end_v1",
                "goal_frame": "fixed_robot_base_v1",
                "witness_joint_angle_usage": "physics_reset_only_v1"}

    def task_state(self, motion_id: torch.Tensor, frame: torch.Tensor) -> TaskState:
        if (motion_id.shape != frame.shape or motion_id.ndim != 1
                or motion_id.dtype != torch.long or frame.dtype != torch.long
                or bool(((motion_id < 0) | (motion_id >= len(self.reference.names))).any())
                or bool(((frame < 0) | (frame >= self.reference.lengths[motion_id])).any())):
            raise ValueError("Task reference identity or frame is invalid")
        return TaskState(torch.full_like(frame, 3), motion_id, frame,
                         torch.empty((len(frame), 0), device=frame.device))

    @staticmethod
    def physical_task_state(native):
        return native.task_state()

    @staticmethod
    def advance(task_state):
        return TaskState(task_state.task_id, task_state.motion_id,
                         task_state.frame + 1, task_state.data)

    @staticmethod
    def _require_arm_state(state):
        if state.ndim != 2 or state.shape[1] != 42:
            raise ValueError("Continuous SE(3) Task MDP requires the 42D arm-only physical state")

    def goal(self, state, task_state):
        return self.reference.future_goal_at(task_state.motion_id, task_state.frame, 10)

    def termination_features(self, state, task_state):
        return self.reference.pose_at(task_state.motion_id, task_state.frame)

    def _errors(self, state, task_state):
        self._require_arm_state(state)
        target = self.reference.pose_at(task_state.motion_id, task_state.frame)
        desired_pos, desired_rot = unpack_dual_tcp_pose(target)
        pos, rot = self.robot.fk(state[..., :14])
        actual_pos, actual_rot = pos[..., :, 2, :], rot[..., :, 2, :, :]
        position_error = torch.linalg.vector_norm(actual_pos - desired_pos, dim=-1)
        rotation_error = so3_geodesic(actual_rot, desired_rot)
        rotation_vector = so3_log(desired_rot @ actual_rot.transpose(-1, -2))
        return position_error, rotation_error, desired_pos - actual_pos, rotation_vector

    def observation(self, state, previous_action, task_state):
        mapping = self.robot.action_map
        _, _, pos_error, rotation_error = self._errors(state, task_state)
        return torch.cat(((state[..., :14] - mapping.origin) / mapping.scale,
                          state[..., 14:28] / 2,
                          (state[..., 28:42] - mapping.origin) / mapping.scale,
                          previous_action.tanh(), pos_error.flatten(-2),
                          rotation_error.flatten(-2)), dim=-1)

    def critic_features(self, state, task_state):
        self._require_arm_state(state)
        return torch.cat((state, self.goal(state, task_state).flatten(1)), dim=-1)

    def reward(self, state, next_state, action, next_task_state, previous_action=None):
        weights = self.reward_config
        old_position, old_rotation, _, _ = self._errors(state, next_task_state)
        position, rotation, _, _ = self._errors(next_state, next_task_state)
        bounded_action = action.tanh()
        bounded_previous = (torch.zeros_like(bounded_action) if previous_action is None
                            else previous_action.tanh())
        accuracy = (
            weights.position_weight * torch.exp(-(position / weights.position_scale_m).square()).mean(-1)
            + weights.rotation_weight * torch.exp(-(rotation / weights.rotation_scale_rad).square()).mean(-1))
        progress = torch.tanh(((old_position - position).mean(-1) / 0.02
                               + (old_rotation - rotation).mean(-1) / 0.15))
        command_jump = ((next_state[..., 28:42] - state[..., 28:42]) / 0.01).square().mean(-1)
        margin = 0.05
        soft_limit = (F.relu(self.robot.action_map.lower + margin - next_state[..., :14]).square()
                      + F.relu(next_state[..., :14] - self.robot.action_map.upper + margin).square()).mean(-1)
        penalty = (weights.action_weight * bounded_action.square().mean(-1)
                   + weights.action_rate_weight * (bounded_action - bounded_previous).square().mean(-1)
                   + weights.command_jump_weight * command_jump
                   + weights.velocity_weight * next_state[..., 14:28].square().mean(-1)
                   + weights.soft_joint_limit_weight * soft_limit)
        return weights.step_scale * (accuracy + weights.progress_weight * progress - penalty)

    def reward_upper_bound(self):
        weights = self.reward_config
        return weights.step_scale * (weights.position_weight + weights.rotation_weight
                                     + weights.progress_weight)

    def soft_termination(self, next_state, next_task_state):
        self._require_arm_state(next_state)
        return analytic_soft_termination(
            next_state[..., :14], next_state[..., 14:28],
            self.robot.action_map.lower, self.robot.action_map.upper,
            next_state.new_zeros((len(next_state), 2)), self.safety_thresholds)

    def hard_termination(self, next_state, next_task_state, contacts):
        self._require_arm_state(next_state)
        return unsafe_arm_states(
            next_state[..., :14], next_state[..., 14:28],
            self.robot.action_map.lower, self.robot.action_map.upper,
            next_state.new_zeros((len(next_state), 2)), contacts,
            thresholds=self.safety_thresholds)

    def valid_imagination_start(self, task_state, horizon):
        return task_state.frame + horizon <= self.reference.lengths[task_state.motion_id] - 1

    def success_mask(self, transition):
        position, rotation, _, _ = self._errors(transition.next_state,
                                                transition.next_task_state)
        return ((position <= self.success_config.mean_position_m).all(-1)
                & (rotation <= self.success_config.mean_rotation_rad).all(-1)
                & ~transition.terminated)

    @staticmethod
    def truncation_reason(transition):
        return "motion_end"

    def _trajectory_errors(self, trajectory):
        states = torch.cat([step.next_state for step in trajectory])
        ids = torch.cat([step.next_task_state.motion_id for step in trajectory])
        frames = torch.cat([step.next_task_state.frame for step in trajectory])
        return self._errors(states, self.task_state(ids, frames))[:2]

    def episode_success(self, trajectory, *, complete: bool):
        if not complete or not bool(trajectory[-1].truncated[0]):
            return False
        if any(bool(step.terminated[0]) for step in trajectory):
            return False
        final = trajectory[-1].next_task_state
        if not bool((final.frame == self.reference.lengths[final.motion_id] - 1).all()):
            return False
        pos, rot = self._trajectory_errors(trajectory)
        cfg = self.success_config
        return bool(((pos.mean(0) <= cfg.mean_position_m)
                     & (torch.quantile(pos, 0.95, dim=0) <= cfg.p95_position_m)
                     & (rot.mean(0) <= cfg.mean_rotation_rad)
                     & (torch.quantile(rot, 0.95, dim=0) <= cfg.p95_rotation_rad)).all())

    def evaluate_metrics(self, trajectory):
        pos, rot = self._trajectory_errors(trajectory)
        return {"tcp_position_mean_m": pos.mean(0).tolist(),
                "tcp_position_p95_m": torch.quantile(pos, 0.95, dim=0).tolist(),
                "tcp_position_max_m": pos.max(0).values.tolist(),
                "tcp_so3_mean_rad": rot.mean(0).tolist(),
                "tcp_so3_p95_rad": torch.quantile(rot, 0.95, dim=0).tolist(),
                "tcp_so3_max_rad": rot.max(0).values.tolist(),
                "reference_end_frame": int(self.reference.lengths[trajectory[-1].next_task_state.motion_id[0]] - 1),
                "observed_end_frame": int(trajectory[-1].next_task_state.frame[0])}
