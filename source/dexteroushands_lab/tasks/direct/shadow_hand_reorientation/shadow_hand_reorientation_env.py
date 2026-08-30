from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_mul, sample_uniform

from .shadow_hand_reorientation_env_cfg import ShadowHandReorientationDirectEnvCfg


class ShadowHandReorientationDirectEnv(DirectRLEnv):
    """Native DirectTask port with explicit object, goal, reward, and reset state."""

    cfg: ShadowHandReorientationDirectEnvCfg

    def __init__(self, cfg: ShadowHandReorientationDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Preserve the legacy DexterousHands action order as the backend contract.
        self.actuated_joint_ids = [self.robot.joint_names.index(name) for name in cfg.actuated_joint_names]
        self.joint_lower_limits = self.robot.data.joint_limits.torch[:, self.actuated_joint_ids, 0]
        self.joint_upper_limits = self.robot.data.joint_limits.torch[:, self.actuated_joint_ids, 1]
        self.num_actuated_joints = len(self.actuated_joint_ids)
        if self.num_actuated_joints != self.cfg.action_space:
            raise RuntimeError("The configured action dimension does not match the actuated joint mapping.")

        self.actions = torch.zeros((self.num_envs, self.num_actuated_joints), device=self.device)
        self.joint_targets = torch.zeros_like(self.actions)
        self.goal_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.tensor(cfg.object_start_pos, device=self.device).repeat(self.num_envs, 1)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.object = RigidObject(self.cfg.object)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object

        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.to(self.device).clone().clamp(-1.0, 1.0)

    def _apply_action(self):
        default_pos = self.robot.data.default_joint_pos.torch[:, self.actuated_joint_ids]
        self.joint_targets = default_pos + self.cfg.action_scale * self.actions
        self.joint_targets = torch.clamp(self.joint_targets, self.joint_lower_limits, self.joint_upper_limits)
        self.robot.set_joint_position_target_index(target=self.joint_targets, joint_ids=self.actuated_joint_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        joint_pos = self.robot.data.joint_pos.torch[:, self.actuated_joint_ids]
        joint_vel = self.robot.data.joint_vel.torch[:, self.actuated_joint_ids]
        joint_pos_normalized = 2.0 * (joint_pos - self.joint_lower_limits)
        joint_pos_normalized = joint_pos_normalized / (self.joint_upper_limits - self.joint_lower_limits) - 1.0

        object_pose = self.object.data.root_pose_w.torch
        object_vel = self.object.data.root_vel_w.torch
        object_pos = object_pose[:, :3] - self.scene.env_origins
        object_rot = object_pose[:, 3:7]
        quat_error = quat_mul(object_rot, quat_conjugate(self.goal_rot))

        obs = torch.cat(
            (
                joint_pos_normalized,
                self.cfg.velocity_scale * joint_vel,
                object_pos,
                object_rot,
                object_vel[:, :3],
                self.cfg.velocity_scale * object_vel[:, 3:6],
                self.goal_pos,
                self.goal_rot,
                quat_error,
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _get_rewards(self) -> torch.Tensor:
        object_pose = self.object.data.root_pose_w.torch
        object_pos = object_pose[:, :3] - self.scene.env_origins
        object_rot = object_pose[:, 3:7]
        quat_error = quat_mul(object_rot, quat_conjugate(self.goal_rot))
        orientation_error = 1.0 - torch.abs(quat_error[:, 0])
        position_error = torch.linalg.norm(object_pos - self.goal_pos, dim=-1)
        reward = (
            self.cfg.orientation_reward_scale * torch.exp(-4.0 * orientation_error)
            + self.cfg.position_reward_scale * torch.exp(-4.0 * position_error)
            - self.cfg.action_penalty_scale * torch.mean(self.actions.square(), dim=-1)
        )
        self.extras.setdefault("log", {})["Metrics/orientation_error"] = orientation_error.mean().item()
        self.extras["log"]["Metrics/position_error"] = position_error.mean().item()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        object_height = self.object.data.root_pose_w.torch[:, 2]
        terminated = object_height < self.cfg.object_fall_height
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        default_pos = self.robot.data.default_joint_pos.torch[env_ids][:, self.actuated_joint_ids]
        default_vel = self.robot.data.default_joint_vel.torch[env_ids][:, self.actuated_joint_ids]
        joint_pos = default_pos + sample_uniform(-0.10, 0.10, default_pos.shape, self.device)
        joint_pos = torch.clamp(joint_pos, self.joint_lower_limits[env_ids], self.joint_upper_limits[env_ids])
        joint_vel = torch.zeros_like(default_vel)
        self.robot.set_joint_position_target_index(target=joint_pos, joint_ids=self.actuated_joint_ids, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(
            position=joint_pos, joint_ids=self.actuated_joint_ids, env_ids=env_ids
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=joint_vel, joint_ids=self.actuated_joint_ids, env_ids=env_ids
        )

        object_pose = self.object.data.default_root_pose.torch[env_ids].clone()
        object_vel = self.object.data.default_root_vel.torch[env_ids].clone()
        object_pose[:, :3] = torch.tensor(self.cfg.object_start_pos, device=self.device)
        object_pose[:, :3] += sample_uniform(-0.02, 0.02, object_pose[:, :3].shape, self.device)
        object_pose[:, :3] += self.scene.env_origins[env_ids]
        object_pose[:, 3:7] = self._sample_goal_rotation(len(env_ids))
        object_vel.zero_()
        self.object.write_root_pose_to_sim_index(root_pose=object_pose, env_ids=env_ids)
        self.object.write_root_velocity_to_sim_index(root_velocity=object_vel, env_ids=env_ids)

        self.goal_pos[env_ids] = object_pose[:, :3] - self.scene.env_origins[env_ids]
        self.goal_rot[env_ids] = self._sample_goal_rotation(len(env_ids))
        self.actions[env_ids] = 0.0
        self.joint_targets[env_ids] = joint_pos

    def _sample_goal_rotation(self, count: int) -> torch.Tensor:
        angle = sample_uniform(-torch.pi, torch.pi, (count,), self.device)
        half = 0.5 * angle
        quat = torch.zeros((count, 4), device=self.device)
        quat[:, 0] = torch.cos(half)
        quat[:, 3] = torch.sin(half)
        return quat
