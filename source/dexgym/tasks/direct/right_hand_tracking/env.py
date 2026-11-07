"""TeleDexter-inspired consecutive fingertip tracking with the right hand only.

Source: https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/task_workflows.html
"""

import json
from collections.abc import Sequence

import torch
from isaaclab.envs import DirectRLEnv

from dexgym.assets.marvin_wuji.robot_cfg import PACKAGED_KEYFRAMES
from dexgym.data.motion_bank import MotionBank, advance_goal_frames

from .env_cfg import RightHandTrackingDirectCfg
from .logic import fingertips_in_wrist_frame, right_hand_joint_indices, tracking_step

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
        if self.robot.num_joints != 54:
            raise ValueError("Expected the pinned 54-DoF Marvin/Wuji articulation")
        self.right_ids = right_hand_joint_indices(self.robot.joint_names)
        wrist_ids, _ = self.robot.find_bodies("r_wrist")
        if len(wrist_ids) != 1:
            raise ValueError("Expected exactly one right-wrist body")
        self.wrist_body_id = wrist_ids[0]
        self.tip_body_ids = []
        for name in TIP_DISTAL_BODY_NAMES:
            ids, _ = self.robot.find_bodies(name)
            if len(ids) != 1:
                raise ValueError(f"Expected exactly one distal body {name}")
            self.tip_body_ids.append(ids[0])
        self.tip_offsets = torch.tensor([[0.0, 0.0, z] for z in TIP_LOCAL_Z], dtype=torch.float32, device=self.device)
        keyframes = json.loads(PACKAGED_KEYFRAMES.read_text())
        home_by_name = dict(zip(keyframes["joint_order"], keyframes["keyframes"]["home"], strict=True))
        self.home_q = torch.tensor([home_by_name[name] for name in self.robot.joint_names], device=self.device)
        self.full_target = self.home_q[None].repeat(self.num_envs, 1)
        self.target_q = self.full_target[:, self.right_ids].clone()
        self.joint_limits = self.robot.data.joint_pos_limits.torch[0, self.right_ids].clone()
        self.bank = MotionBank(cfg.library_path, split=cfg.split, device=self.device)
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

    def _current_tips(self) -> torch.Tensor:
        return fingertips_in_wrist_frame(
            self.robot.data.body_link_pose_w.torch, self.wrist_body_id, self.tip_body_ids, self.tip_offsets
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        bounded = actions.clamp(-1.0, 1.0)
        self.action_delta = bounded - self.previous_action
        self.previous_action = bounded
        self.target_q = torch.clamp(
            self.target_q + bounded * self.cfg.action_delta_rad,
            min=self.joint_limits[:, 0],
            max=self.joint_limits[:, 1],
        )
        self.full_target[:, self.right_ids] = self.target_q

    def _apply_action(self) -> None:
        # The full command holds both arms and the left hand at home.
        # Source: https://isaac-sim.github.io/IsaacLab/v3.0.0/source/api/lab/isaaclab.assets.html
        self.robot.actuators.target_command.set_position_index(value=self.full_target, full_data=True)

    def _get_observations(self) -> dict:
        q = self.robot.data.joint_pos.torch[:, self.right_ids]
        qdot = self.robot.data.joint_vel.torch[:, self.right_ids]
        tips = self._current_tips()
        policy = torch.cat(
            (
                q,
                qdot,
                tips.flatten(1),
                self.goal_tips.flatten(1),
                (self.goal_tips - tips).flatten(1),
                self.previous_action,
            ),
            dim=-1,
        )
        return {"policy": policy}

    def _get_rewards(self) -> torch.Tensor:
        return self._step_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
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
        )
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

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        count = len(env_ids)
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
        self.goal_frame[env_ids] = start + self.cfg.subgoal_stride
        self.goal_tips[env_ids] = self.bank.fingertips(motion_ids, self.goal_frame[env_ids])
        self.dwell[env_ids] = 0
        self.goal_age[env_ids] = 0
        self.completed[env_ids] = False
        self.previous_action[env_ids] = 0.0
        self.action_delta[env_ids] = 0.0
        self.goals_reached[env_ids] = 0
