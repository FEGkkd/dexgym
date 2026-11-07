"""Causal, fixed-base 50 Hz dual TCP pose tracking for streamed targets."""

from dataclasses import asdict

import torch

from dexgym.algorithms.supermimic.core.specs import TaskSpec
from dexgym.algorithms.supermimic.core.transition import TaskState

from .se3_trajectory import (ContinuousSE3RewardConfig, ContinuousSE3SuccessConfig,
                             MarvinContinuousSE3Follow)


class MarvinTeleopWorkspaceFollow(MarvinContinuousSE3Follow):
    """The target at t is held while action a_t executes; no future target is read.

    The FK joint witness in a generated reference is available to physical
    reset only. Neither this MDP nor its Actor, Critic or Reward reads it.
    """

    inference_reference_agnostic = True

    def __init__(self, reference, robot, *,
                 reward_config=ContinuousSE3RewardConfig(),
                 success_config=ContinuousSE3SuccessConfig(),
                 safety_thresholds=None):
        kwargs = {"reward_config": reward_config, "success_config": success_config}
        if safety_thresholds is not None:
            kwargs["safety_thresholds"] = safety_thresholds
        super().__init__(reference, robot, **kwargs)
        self.spec = TaskSpec(
            "marvin_bimanual_teleop_workspace_se3_follow", 68, 10, 18, 1,
            0, 60, "current_dual_tcp_se3_rotation6d_1x18_v1",
            "arm_q_qd_qcmd_previous_action_current_tcp_error_log_68_v1",
            "physical_state42_current_target_pose18_v1", 18)

    def checkpoint_contract(self):
        return {
            "task_adapter_version": "marvin_teleop_workspace_follow_v1",
            "reward": asdict(self.reward_config),
            "success": asdict(self.success_config),
            "safety_thresholds": asdict(self.safety_thresholds),
            "reference_hashes": self.reference.hashes,
            "episode_semantics": "target_t_held_during_action_t_arrival_t_plus_1_v1",
            "target_stream": "fixed_base_se3_50hz_current_only_v1",
            "witness_joint_angle_usage": "physics_reset_only_v1",
        }

    def task_state(self, motion_id: torch.Tensor, frame: torch.Tensor) -> TaskState:
        state = super().task_state(motion_id, frame)
        return TaskState(torch.full_like(frame, 4), state.motion_id, state.frame, state.data)

    def goal(self, state, task_state):
        return self.reference.pose_at(task_state.motion_id, task_state.frame)[:, None, :]

    def critic_features(self, state, task_state):
        self._require_arm_state(state)
        return torch.cat((state, self.goal(state, task_state).flatten(1)), dim=-1)

    @staticmethod
    def _commanded_target_state(next_task_state):
        if bool((next_task_state.frame < 1).any()):
            raise ValueError("An arrival must follow a real commanded target frame")
        return TaskState(next_task_state.task_id, next_task_state.motion_id,
                         next_task_state.frame - 1, next_task_state.data)

    def reward(self, state, next_state, action, next_task_state, previous_action=None):
        return super().reward(state, next_state, action,
                              self._commanded_target_state(next_task_state), previous_action)

    def success_mask(self, transition):
        position, rotation, _, _ = self._errors(
            transition.next_state,
            self._commanded_target_state(transition.next_task_state))
        return ((position <= self.success_config.mean_position_m).all(-1)
                & (rotation <= self.success_config.mean_rotation_rad).all(-1)
                & ~transition.terminated)

    def _trajectory_errors(self, trajectory):
        states = torch.cat([step.next_state for step in trajectory])
        ids = torch.cat([step.next_task_state.motion_id for step in trajectory])
        frames = torch.cat([step.next_task_state.frame for step in trajectory]) - 1
        return self._errors(states, self.task_state(ids, frames))[:2]

    def evaluate_metrics(self, trajectory):
        result = super().evaluate_metrics(trajectory)
        motion_id = int(trajectory[0].task_state.motion_id[0])
        result["reference_motion_id"] = motion_id
        if hasattr(self.reference, "difficulty"):
            result["reference_difficulty"] = int(self.reference.difficulty[motion_id])
            result["reference_family"] = self.reference.family[motion_id]
            result["reference_split"] = ("train", "validation", "test")[
                int(self.reference.split[motion_id])]
        return result
