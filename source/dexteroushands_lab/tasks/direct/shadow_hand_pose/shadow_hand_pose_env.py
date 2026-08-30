from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .shadow_hand_pose_env_cfg import DirectShadowHandPoseEnvCfg


class DirectShadowHandPoseEnv(DirectRLEnv):
    """Explicit DirectRLEnv lifecycle for a Shadow Hand pose-tracking task.

    This is an architecture baseline, not the final manipulation objective.
    The same methods are the required migration surface for the original
    DexterousHands tasks.
    """

    cfg: DirectShadowHandPoseEnvCfg

    def __init__(self, cfg: DirectShadowHandPoseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.num_joints = self.robot.num_joints
        if self.num_joints != self.cfg.action_space:
            raise RuntimeError(
                f"Shadow Hand asset exposes {self.num_joints} joints, "
                f"but the task declares {self.cfg.action_space} actions."
            )

        self.joint_lower_limits = self.robot.data.joint_limits.torch[..., 0]
        self.joint_upper_limits = self.robot.data.joint_limits.torch[..., 1]
        self.target_joint_pos = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.joint_targets = torch.zeros_like(self.actions)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.to(self.device).clone().clamp(-1.0, 1.0)

    def _apply_action(self):
        self.joint_targets = self.robot.data.default_joint_pos.torch + self.cfg.action_scale * self.actions
        self.joint_targets = torch.clamp(self.joint_targets, self.joint_lower_limits, self.joint_upper_limits)
        self.robot.set_joint_position_target_index(target=self.joint_targets)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        joint_pos = self.robot.data.joint_pos.torch
        joint_vel = self.robot.data.joint_vel.torch
        joint_pos_normalized = 2.0 * (joint_pos - self.joint_lower_limits)
        joint_pos_normalized = joint_pos_normalized / (self.joint_upper_limits - self.joint_lower_limits) - 1.0
        target_error = self.target_joint_pos - joint_pos

        obs = torch.cat(
            (
                joint_pos_normalized,
                self.cfg.velocity_scale * joint_vel,
                target_error,
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _get_rewards(self) -> torch.Tensor:
        pose_error = torch.mean(torch.abs(self.target_joint_pos - self.robot.data.joint_pos.torch), dim=-1)
        pose_reward = torch.exp(-5.0 * pose_error)
        action_penalty = 0.01 * torch.mean(self.actions.square(), dim=-1)
        self.extras.setdefault("log", {})["Metrics/pose_error"] = pose_error.mean().item()
        return pose_reward - action_penalty

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        default_pos = self.robot.data.default_joint_pos.torch[env_ids]
        default_vel = self.robot.data.default_joint_vel.torch[env_ids]
        joint_pos = default_pos + sample_uniform(-0.05, 0.05, default_pos.shape, self.device)
        joint_pos = torch.clamp(joint_pos, self.joint_lower_limits[env_ids], self.joint_upper_limits[env_ids])
        joint_vel = torch.zeros_like(default_vel)

        self.robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        lower = self.joint_lower_limits[env_ids]
        upper = self.joint_upper_limits[env_ids]
        self.target_joint_pos[env_ids] = lower + torch.rand_like(lower) * (upper - lower)
        self.actions[env_ids] = 0.0
        self.joint_targets[env_ids] = joint_pos
