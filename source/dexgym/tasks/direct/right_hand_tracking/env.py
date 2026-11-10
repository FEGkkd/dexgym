"""TeleDexter-inspired consecutive fingertip tracking with the right hand only.

Source: https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/task_workflows.html
"""

import json
from collections.abc import Sequence

import torch
from isaaclab.envs import DirectRLEnv

from dexgym.assets.marvin_wuji.robot_cfg import PACKAGED_KEYFRAMES
from dexgym.data.motion_bank import MotionBank, advance_goal_frames
from dexgym.data.wuji_retarget import TIP_SITE_NAMES

from .env_cfg import RightHandTrackingDirectCfg
from .subgoals import HandGoals, advance_hand_goals, unsafe_hand_state
from .logic import (
    build_right_hand_observation,
    curriculum_stride_upper,
    fingertips_in_wrist_frame,
    residual_joint_target,
    right_hand_joint_indices,
    sample_goal_span,
    teledexter_hand_reward,
    tracking_step,
)

TIP_DISTAL_BODY_NAMES = (
    "r_thumb_distal",
    "r_index_finger_distal",
    "r_middle_finger_distal",
    "r_ring_finger_distal",
    "r_pinky_distal",
)
# Tip site positions in their distal-link frames, from the pinned right.xml.
TIP_LOCAL_Z = (-0.02978, -0.02475, -0.02475, -0.02475, -0.02475)


class RightHandTrackingDirectEnv(DirectRLEnv):
    cfg: RightHandTrackingDirectCfg

    def __init__(self, cfg: RightHandTrackingDirectCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        if self.robot.num_joints not in (20, 54):
            raise ValueError("Expected the 20-DoF Hand 2.1 or legacy 54-DoF articulation")
        standalone_hand = self.robot.num_joints == 20
        self.right_ids = right_hand_joint_indices(self.robot.joint_names)
        wrist_ids, _ = self.robot.find_bodies("r_wrist")
        if len(wrist_ids) != 1:
            raise ValueError("Expected exactly one right-wrist body")
        self.wrist_body_id = wrist_ids[0]
        self.tip_body_ids = []
        for name in TIP_SITE_NAMES if standalone_hand else TIP_DISTAL_BODY_NAMES:
            ids, _ = self.robot.find_bodies(name)
            if len(ids) != 1:
                raise ValueError(f"Expected exactly one fingertip body {name}")
            self.tip_body_ids.append(ids[0])
        offsets = (0.0,) * 5 if standalone_hand else TIP_LOCAL_Z
        self.tip_offsets = torch.tensor([[0.0, 0.0, z] for z in offsets], dtype=torch.float32, device=self.device)
        if standalone_hand:
            self.home_q = self.robot.data.default_joint_pos.torch[0].clone()
        else:
            keyframes = json.loads(PACKAGED_KEYFRAMES.read_text())
            home_by_name = dict(zip(keyframes["joint_order"], keyframes["keyframes"]["home"], strict=True))
            self.home_q = torch.tensor([home_by_name[name] for name in self.robot.joint_names], device=self.device)
        self.full_target = self.home_q[None].repeat(self.num_envs, 1)
        self.target_q = self.full_target[:, self.right_ids].clone()
        self.joint_limits = self.robot.data.joint_pos_limits.torch[0, self.right_ids].clone()
        self.bank = MotionBank(
            cfg.library_path, split=cfg.split, device=self.device, min_frames=cfg.subgoal_stride + 2
        )
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_tips = torch.zeros((self.num_envs, 5, 3), device=self.device)
        self.dwell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.previous_action = torch.zeros((self.num_envs, 20), device=self.device)
        self.action_delta = torch.zeros_like(self.previous_action)
        self.goals_reached = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._step_reward = torch.zeros(self.num_envs, device=self.device)
        self.curriculum_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_span = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.dwell_required = torch.full((self.num_envs,), cfg.goal_dwell_steps, dtype=torch.long, device=self.device)
        self.fail_frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.fail_budget = torch.full((self.num_envs,), float(cfg.max_goal_steps), device=self.device)
        self.last_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_crossed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _current_tips(self) -> torch.Tensor:
        return fingertips_in_wrist_frame(
            self.robot.data.body_link_pose_w.torch, self.wrist_body_id, self.tip_body_ids, self.tip_offsets
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        bounded = actions.clamp(-1.0, 1.0)
        self.action_delta = bounded - self.previous_action
        self.previous_action = bounded
        self.target_q = residual_joint_target(
            self.target_q,
            bounded,
            self.joint_limits,
            scale=self.cfg.action_delta_rad,
            deadzone=self.cfg.action_deadzone,
        )
        self.full_target[:, self.right_ids] = self.target_q

    def _apply_action(self) -> None:
        # Legacy full-body commands hold both arms and the left hand at home;
        # the standalone hand command contains only the 20 tracked joints.
        # Source: https://isaac-sim.github.io/IsaacLab/v3.0.0/source/api/lab/isaaclab.assets.html
        self.robot.actuators.target_command.set_position_index(value=self.full_target, full_data=True)

    def _get_observations(self) -> dict:
        q = self.robot.data.joint_pos.torch[:, self.right_ids]
        qdot = self.robot.data.joint_vel.torch[:, self.right_ids]
        tips = self._current_tips()
        policy = build_right_hand_observation(
            q,
            qdot,
            tips,
            self.goal_tips,
            self.previous_action,
            command_q=self.target_q if self.cfg.include_command_error else None,
        )
        return {"policy": policy}

    def _get_rewards(self) -> torch.Tensor:
        return self._step_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.teledexter_style:
            return self._get_dones_teledexter()
        # DirectRLEnv calls this hook before _get_rewards. Update the current
        # subgoal here so completion and stalled-goal termination occur on the
        # same transition that earns its reward.
        reward, reached, self.dwell, mean_error = tracking_step(
            self._current_tips(),
            self.goal_tips,
            self.dwell,
            self.action_delta,
            tolerance_m=self.cfg.goal_tolerance_m,
            dwell_steps=self.cfg.goal_dwell_steps,
            sparse_bonus=self.cfg.reward_sparse_bonus,
            dense_scale=self.cfg.reward_dense_scale,
            action_rate_penalty=self.cfg.reward_action_rate_penalty,
            time_cost=self.cfg.reward_time_cost,
        )
        self.last_reached = reached
        self.last_crossed = torch.zeros_like(reached)
        self.goal_age += 1
        self.goals_reached += reached.long()
        next_frame, just_completed = advance_goal_frames(
            self.goal_frame, self.bank.lengths[self.motion_ids], reached, stride=self.cfg.subgoal_stride
        )
        self.completed |= just_completed
        self.goal_frame = next_frame
        self.goal_tips = self.bank.fingertips(self.motion_ids, next_frame)
        self.dwell = torch.where(reached, torch.zeros_like(self.dwell), self.dwell)
        self.goal_age = torch.where(reached, torch.zeros_like(self.goal_age), self.goal_age)
        self.extras["log"] = {
            "mean_tip_error_m": mean_error.mean(),
            "goal_reach_fraction": reached.float().mean(),
            "mean_goals_reached": self.goals_reached.float().mean(),
        }
        self._step_reward = reward
        q = self.robot.data.joint_pos.torch[:, self.right_ids]
        qdot = self.robot.data.joint_vel.torch[:, self.right_ids]
        unsafe = (
            ~torch.isfinite(q).all(dim=-1)
            | ~torch.isfinite(qdot).all(dim=-1)
            | (q < self.joint_limits[:, 0] - 0.05).any(dim=-1)
            | (q > self.joint_limits[:, 1] + 0.05).any(dim=-1)
        )
        failed = unsafe | (self.goal_age >= self.cfg.max_goal_steps) | self.completed
        timeout = self.episode_length_buf >= self.max_episode_length
        return failed, timeout

    def _get_dones_teledexter(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared exact transition; reward and terminal snapshots precede reset."""
        cfg = self.cfg
        old = HandGoals.from_env(self)
        result = advance_hand_goals(self._current_tips(), old, self.bank, cfg)
        q = self.robot.data.joint_pos.torch[:, self.right_ids]
        qdot = self.robot.data.joint_vel.torch[:, self.right_ids]
        self._step_reward = teledexter_hand_reward(
            result.errors, result.reached, old.goal_span,
            dense_scale=cfg.reward_dense_scale, time_cost=cfg.reward_time_cost)
        result.goals.apply(self)
        self.goal_tips = self.bank.fingertips(self.motion_ids, self.goal_frame)
        self.last_reached, self.last_crossed = result.reached, result.crossed
        unsafe = unsafe_hand_state(q, qdot, self.joint_limits)
        failed = unsafe | result.exceeded_budget | self.completed
        timeout = self.episode_length_buf >= self.max_episode_length
        if getattr(cfg, "capture_model_transitions", False):
            # These are deliberately NOT cleared by _reset_idx: DirectRLEnv
            # auto-resets before returning env.step, but Replay needs arrival.
            self.last_hand_state = torch.cat((q, qdot, self.target_q), -1).detach().clone()
            self.last_hand_goal_data = result.goals.pack().detach().clone()
            self.last_hand_control_frame = self.episode_length_buf.detach().clone().long()
            self.last_hand_reached = result.reached.clone()
            self.last_hand_errors = result.errors.detach().clone()
            self.last_hand_failure_reason = (unsafe.int() + 2 * result.exceeded_budget.int()
                                             + 4 * self.completed.int()).to(torch.int32)
        self.extras["log"] = {
            "mean_tip_error_m": result.errors.mean(),
            "goal_reach_fraction": result.reached.float().mean(),
            "mean_goals_reached": self.goals_reached.float().mean(),
            "cross_trajectory_fraction": result.crossed.float().mean(),
            "mean_goal_span_frames": self.goal_span.float().mean(),
            "curriculum_stride_upper": result.stride_upper.float().mean(),
            "mean_fail_frames": self.fail_frames.float().mean(),
        }
        return failed, timeout

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        count = len(env_ids)
        if self.cfg.teledexter_style:
            motion_ids, start = self.bank.sample_reference_starts(
                count, first_fraction=self.cfg.reference_start_fraction
            )
        else:
            motion_ids, start = self.bank.sample_starts(count, stride=self.cfg.subgoal_stride)
        q = self.home_q[None].repeat(count, 1)
        q[:, self.right_ids] = self.bank.joints(motion_ids, start)
        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=self.robot.data.default_root_vel.torch[env_ids].clone(), env_ids=env_ids
        )
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
        self.robot.actuators.target_command.set_position_index(value=q, env_ids=env_ids)
        self.full_target[env_ids] = q
        self.target_q[env_ids] = q[:, self.right_ids]
        self.motion_ids[env_ids] = motion_ids
        if self.cfg.teledexter_style:
            upper = curriculum_stride_upper(
                self.curriculum_steps[env_ids],
                minimum=self.cfg.curriculum_stride_min,
                maximum=self.cfg.curriculum_stride_max,
                anneal_steps=self.cfg.curriculum_anneal_steps,
                sigma_min=self.cfg.curriculum_sigma_min,
            )
            span = sample_goal_span(upper)
            goal = torch.minimum(start + span, self.bank.lengths[motion_ids] - 1)
            self.goal_span[env_ids] = goal - start
            self.goal_frame[env_ids] = goal
            self.fail_budget[env_ids] = self.cfg.failure_tolerance_scale * self.goal_span[env_ids]
            self.fail_frames[env_ids] = 0
            self.dwell_required[env_ids] = torch.randint(
                self.cfg.dwell_min_steps, self.cfg.dwell_max_steps + 1, (count,), device=self.device
            )
        else:
            self.goal_frame[env_ids] = start + self.cfg.subgoal_stride
        self.goal_tips[env_ids] = self.bank.fingertips(motion_ids, self.goal_frame[env_ids])
        self.dwell[env_ids] = 0
        self.goal_age[env_ids] = 0
        self.completed[env_ids] = False
        self.previous_action[env_ids] = 0.0
        self.action_delta[env_ids] = 0.0
        self.goals_reached[env_ids] = 0
        self.last_reached[env_ids] = False
        self.last_crossed[env_ids] = False
