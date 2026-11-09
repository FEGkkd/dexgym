"""Standalone Hand21 PPO physical environment."""

from collections.abc import Sequence

import torch
from isaaclab.envs import DirectRLEnv

from dexgym.data.motion_bank import MotionBank
from dexgym.data.wuji_retarget import TIP_SITE_NAMES

from .env_cfg import WujiHand21TeleDexterCfg
from .logic import (
    build_right_hand_observation,
    curriculum_stride_upper,
    fingertips_in_wrist_frame,
    residual_joint_target,
    right_hand_joint_indices,
    sample_goal_span,
    teledexter_hand_reward,
)
from .subgoals import HandGoals, advance_hand_goals, unsafe_hand_state


class Hand21PPOEnv(DirectRLEnv):
    cfg: WujiHand21TeleDexterCfg

    def __init__(self, cfg: WujiHand21TeleDexterCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        if self.robot.num_joints != 20:
            raise ValueError("Expected the standalone 20-DoF Hand 2.1")
        self.right_ids = right_hand_joint_indices(self.robot.joint_names)
        wrist_ids, _ = self.robot.find_bodies("r_wrist")
        if len(wrist_ids) != 1:
            raise ValueError("Expected exactly one right-wrist body")
        self.wrist_body_id = wrist_ids[0]
        self.tip_body_ids = []
        for name in TIP_SITE_NAMES:
            ids, _ = self.robot.find_bodies(name)
            if len(ids) != 1:
                raise ValueError(f"Expected exactly one fingertip body {name}")
            self.tip_body_ids.append(ids[0])
        offsets = (0.0,) * 5
        self.tip_offsets = torch.tensor([[0.0, 0.0, z] for z in offsets], dtype=torch.float32, device=self.device)
        self.home_q = self.robot.data.default_joint_pos.torch[0].clone()
        self.full_target = self.home_q[None].repeat(self.num_envs, 1)
        self.target_q = self.full_target[:, self.right_ids].clone()
        self.joint_limits = self.robot.data.joint_pos_limits.torch[0, self.right_ids].clone()
        self.bank = MotionBank(cfg.library_path, split=cfg.split, device=self.device, min_frames=cfg.subgoal_stride + 2)
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
        return self._get_dones_teledexter()

    def _get_dones_teledexter(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared exact transition; reward and terminal snapshots precede reset."""
        cfg = self.cfg
        old = HandGoals.from_env(self)
        result = advance_hand_goals(self._current_tips(), old, self.bank, cfg)
        q = self.robot.data.joint_pos.torch[:, self.right_ids]
        qdot = self.robot.data.joint_vel.torch[:, self.right_ids]
        self._step_reward = teledexter_hand_reward(
            result.errors,
            result.reached,
            old.goal_span,
            dense_scale=cfg.reward_dense_scale,
            time_cost=cfg.reward_time_cost,
        )
        result.goals.apply(self)
        self.goal_tips = self.bank.fingertips(self.motion_ids, self.goal_frame)
        self.last_reached, self.last_crossed = (result.reached, result.crossed)
        unsafe = unsafe_hand_state(q, qdot, self.joint_limits)
        failed = unsafe | result.exceeded_budget | self.completed
        timeout = self.episode_length_buf >= self.max_episode_length
        self.extras["log"] = {
            "mean_tip_error_m": result.errors.mean(),
            "goal_reach_fraction": result.reached.float().mean(),
            "mean_goals_reached": self.goals_reached.float().mean(),
            "cross_trajectory_fraction": result.crossed.float().mean(),
            "mean_goal_span_frames": self.goal_span.float().mean(),
            "curriculum_stride_upper": result.stride_upper.float().mean(),
            "mean_fail_frames": self.fail_frames.float().mean(),
        }
        return (failed, timeout)

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        count = len(env_ids)
        motion_ids, start = self.bank.sample_reference_starts(count, first_fraction=self.cfg.reference_start_fraction)
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
        self.goal_tips[env_ids] = self.bank.fingertips(motion_ids, self.goal_frame[env_ids])
        self.dwell[env_ids] = 0
        self.goal_age[env_ids] = 0
        self.completed[env_ids] = False
        self.previous_action[env_ids] = 0.0
        self.action_delta[env_ids] = 0.0
        self.goals_reached[env_ids] = 0
        self.last_reached[env_ids] = False
        self.last_crossed[env_ids] = False
