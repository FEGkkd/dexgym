"""Static far-goal SE(3) arrival and hold on the arm-only Marvin dynamics."""

from dataclasses import asdict, dataclass, replace

import torch

from dexgym.algorithms.supermimic.core.specs import TaskSpec
from dexgym.algorithms.supermimic.core.transition import TaskState

from .se3_reach_hold import MarvinBimanualSE3ReachHold, SE3RewardConfig, SE3SuccessConfig


@dataclass(frozen=True)
class WorkspaceReachRewardConfig(SE3RewardConfig):
    # These scales make a 0.4 m / 1 rad goal an ordinary training example.
    # The linear distance term below keeps a gradient beyond either scale.
    position_scale_m: float = 0.4
    rotation_scale_rad: float = 1.2


class MarvinWorkspaceSE3ReachHold(MarvinBimanualSE3ReachHold):
    """Use a hidden-witness FK goal bank as a true fixed-goal MDP.

    The bank's joint witness is used by the Newton reset, never by the MDP.
    TaskState contains only current goal SE(3), motion ID and elapsed time.
    """

    inference_reference_agnostic = True

    def __init__(self, reference, robot, *,
                 reward_config=WorkspaceReachRewardConfig(),
                 success_config=None, safety_thresholds=None,
                 success_semantics="hold_start_deadline_v2",
                 temporal_context=None):
        if success_semantics not in ("pre_deadline_arrival_v1", "hold_start_deadline_v2"):
            raise ValueError("Unknown workspace reach success semantics")
        if temporal_context not in (None, "normalized_time_remaining_v3"):
            raise ValueError("Unknown workspace reach temporal context")
        if temporal_context is not None and success_semantics != "hold_start_deadline_v2":
            raise ValueError("Time-aware task requires the corrected hold deadline")
        if getattr(reference, "version", "marvin_teleop_workspace_bank_v3") not in (
                "marvin_teleop_workspace_bank_v3", "marvin_teleop_workspace_bank_v4",
                "marvin_teleop_workspace_bank_v5"):
            raise ValueError("Workspace reach-and-hold requires a static far-goal bank")
        lengths = reference.lengths
        if lengths.ndim != 1 or not bool((lengths == lengths[0]).all()) or int(lengths[0]) < 33:
            raise ValueError("Far-goal episodes need equal lengths and a 32-step imagination window")
        steps = int(lengths[0]) - 1
        if success_config is None:
            success_config = replace(SE3SuccessConfig(), episode_steps=steps,
                                     arrival_deadline_steps=round(steps * 2 / 3))
        if success_config.episode_steps != steps:
            raise ValueError("Reach-and-hold episode length differs from the target bank")
        kwargs = {"reward_config": reward_config, "success_config": success_config,
                  "hold_speed_mode": "policy_step_displacement_v2"}
        if safety_thresholds is not None:
            kwargs["safety_thresholds"] = safety_thresholds
        super().__init__(robot, **kwargs)
        self.reference = reference
        self.success_semantics = success_semantics
        self.temporal_context = temporal_context
        if temporal_context is None:
            self.spec = TaskSpec(
                "marvin_bimanual_workspace_se3_reach_hold", 68, 10, 18, 1, 18, 60,
                "static_dual_tcp_base_se3_rotation6d_1x18_v1",
                "arm_q_qd_qcmd_previous_action_tcp_error_log_68_v1",
                "physical_state42_target_pose18_v1", 18)
        else:
            self.spec = TaskSpec(
                "marvin_bimanual_workspace_se3_reach_hold", 69, 10, 18, 1, 18, 61,
                "static_dual_tcp_base_se3_rotation6d_1x18_v1",
                "arm_q_qd_qcmd_previous_action_tcp_error_log_remaining_time_69_v3",
                "physical_state42_target_pose18_remaining_time_61_v3", 18)

    def checkpoint_contract(self):
        contract = {"task_adapter_version": "marvin_workspace_se3_reach_hold_v1",
                "reward": asdict(self.reward_config),
                "reward_shape": "linear_dual_tcp_distance_plus_se3_progress_and_hold_v1",
                "success": asdict(self.success_config),
                "safety_thresholds": asdict(self.safety_thresholds),
                "reference_hashes": self.reference.hashes,
                "episode_semantics": "static_goal_arrival_deadline_and_25_step_hold_v1",
                "target_stream": "static_dual_tcp_50hz_fixed_base_v1",
                "goal_frame": "fixed_robot_base_v1",
                "witness_joint_angle_usage": "physics_reset_only_v1"}
        if self.success_semantics == "hold_start_deadline_v2":
            contract["task_adapter_version"] = "marvin_workspace_se3_reach_hold_v2"
            contract["episode_semantics"] = "static_goal_hold_starts_before_deadline_v2"
            contract["success_deadline_contract"] = self.success_semantics
        if self.temporal_context is not None:
            contract["task_adapter_version"] = "marvin_workspace_se3_reach_hold_v3"
            contract["episode_semantics"] = "finite_horizon_time_aware_terminal_timeout_v3"
            contract["temporal_context"] = self.temporal_context
            contract["timeout_soft_termination"] = "exact_one_at_episode_end_v3"
        return contract

    def _remaining_time(self, task_state):
        return ((self.episode_length - task_state.frame).clamp(min=0)
                .to(dtype=task_state.data.dtype) / self.episode_length)[:, None]

    def observation(self, state, previous_action, task_state):
        clean = super().observation(state, previous_action, task_state)
        return (torch.cat((clean, self._remaining_time(task_state)), dim=-1)
                if self.temporal_context is not None else clean)

    def critic_features(self, state, task_state):
        features = super().critic_features(state, task_state)
        return (torch.cat((features, self._remaining_time(task_state)), dim=-1)
                if self.temporal_context is not None else features)

    def soft_termination(self, next_state, next_task_state):
        safety = super().soft_termination(next_state, next_task_state)
        if self.temporal_context is None:
            return safety
        return torch.where(next_task_state.frame >= self.episode_length,
                           torch.ones_like(safety), safety)

    def episode_success(self, trajectory, *, complete: bool):
        if not super().episode_success(trajectory, complete=complete):
            return False
        if self.success_semantics == "pre_deadline_arrival_v1":
            return True
        _, _, _, _, first_hold, _ = self._episode_statistics(trajectory)
        return (first_hold is not None and
                first_hold - self.success_config.hold_steps + 1
                <= self.success_config.arrival_deadline_steps)

    def task_state(self, motion_id: torch.Tensor, frame: torch.Tensor) -> TaskState:
        if motion_id.shape != frame.shape or motion_id.ndim != 1:
            raise ValueError("Far-goal motion IDs and frame vectors differ")
        # Validate both motion ID and frame against the bank; the target itself
        # is fixed for the entire episode and is stored in typed TaskState.data.
        self.reference.pose_at(motion_id, frame)
        goal = self.reference.pose_at(motion_id, torch.zeros_like(frame))
        return TaskState(torch.full_like(frame, 5), motion_id, frame, goal)

    def valid_imagination_start(self, task_state, horizon):
        return task_state.frame + horizon <= self.reference.lengths[task_state.motion_id] - 1

    def reward(self, state, next_state, action, next_task_state, previous_action=None):
        """Add a nonvanishing far-distance gradient to the proven hold reward."""
        weights = self.reward_config
        position, rotation, _ = self._errors(next_state, next_task_state.data)
        nearby_accuracy = (
            weights.position_weight * torch.exp(-(position / weights.position_scale_m).square()).mean(-1)
            + weights.rotation_weight * torch.exp(-(rotation / weights.rotation_scale_rad).square()).mean(-1))
        broad_accuracy = (weights.position_weight * (1 - position.mean(-1) / weights.position_scale_m)
                          + weights.rotation_weight * (1 - rotation.mean(-1) / weights.rotation_scale_rad))
        return (super().reward(state, next_state, action, next_task_state, previous_action)
                + weights.step_scale * (broad_accuracy - nearby_accuracy))
