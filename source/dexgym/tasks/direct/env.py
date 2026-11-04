"""DirectRLEnv cart joint target tracking on Newton."""

from collections.abc import Sequence

import torch
from isaaclab.envs import DirectRLEnv

from .env_cfg import JointTrackingDirectCfg


class JointTrackingDirectEnv(DirectRLEnv):
    cfg: JointTrackingDirectCfg

    def __init__(self, cfg: JointTrackingDirectCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        cart_ids, _ = self.robot.find_joints("slider_to_cart")
        pole_ids, _ = self.robot.find_joints("cart_to_pole")
        self.cart_ids = torch.tensor(cart_ids, device=self.device)
        self.pole_ids = torch.tensor(pole_ids, device=self.device)
        self.goal = torch.zeros((self.num_envs, 1), device=self.device)
        self.target = torch.zeros_like(self.goal)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.target = (1.5 * actions.clamp(-1, 1)).clamp(-1.5, 1.5)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target_index(target=self.target, joint_ids=self.cart_ids)

    def _get_observations(self) -> dict:
        pos = self.robot.data.joint_pos.torch[:, self.cart_ids]
        vel = self.robot.data.joint_vel.torch[:, self.cart_ids]
        return {"policy": torch.cat((pos, vel, self.goal, self.goal - pos), dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        pos = self.robot.data.joint_pos.torch[:, self.cart_ids]
        vel = self.robot.data.joint_vel.torch[:, self.cart_ids]
        return (1.0 - (self.goal - pos).square() - 0.01 * vel.square()).sum(dim=-1) * self.step_dt

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.robot.data.joint_pos.torch[:, self.cart_ids]
        failed = (pos.abs() > 2.8).any(dim=-1) | ~torch.isfinite(pos).all(dim=-1)
        timeout = self.episode_length_buf >= self.max_episode_length
        return failed, timeout

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        count = len(env_ids)
        self.goal[env_ids] = torch.empty((count, 1), device=self.device).uniform_(-1.0, 1.0)
        pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        vel = self.robot.data.default_joint_vel.torch[env_ids].clone()
        pos[:, self.cart_ids] = torch.empty((count, 1), device=self.device).uniform_(-0.5, 0.5)
        pos[:, self.pole_ids] = torch.empty((count, 1), device=self.device).uniform_(-0.1, 0.1)
        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=self.robot.data.default_root_vel.torch[env_ids].clone(), env_ids=env_ids
        )
        self.robot.write_joint_position_to_sim_index(position=pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=vel, env_ids=env_ids)
        self.target[env_ids] = pos[:, self.cart_ids]
