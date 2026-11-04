"""Goal-conditioned 14-action control of the complete 54-DoF Marvin/Wuji robot.

Isaac Lab 3.0 DirectRLEnv lifecycle:
https://isaac-sim.github.io/IsaacLab/v3.0.0/source/tutorials/03_envs/create_direct_rl_env.html
"""

import hashlib
import json
import math
from collections.abc import Sequence

import mujoco
import torch
from isaaclab.envs import DirectRLEnv

from dexgym.assets.marvin_wuji.controller import ARM_JOINT_NAMES, apply_newton_arm_tracking_profile
from dexgym.assets.marvin_wuji.robot_cfg import JOINT_ORDER, PACKAGED_KEYFRAMES, SOURCE_SHA256, resolve_source_mjcf

from .env_cfg import MarvinBimanualS0Cfg
from .kinematics import MarvinArmFK, orientation_error
from .reward import balanced_pose_score
from .safety import JointCommandLimiter


class MarvinBimanualS0Env(DirectRLEnv):
    cfg: MarvinBimanualS0Cfg

    def __init__(self, cfg: MarvinBimanualS0Cfg, render_mode: str | None = None, **kwargs):
        if not 0.0 <= cfg.bimanual_fraction <= 1.0:
            raise ValueError("bimanual_fraction must be in [0, 1]")
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        if self.robot.num_joints != 54 or tuple(self.robot.joint_names) != JOINT_ORDER:
            raise ValueError("S0 requires the verified complete 54-DoF articulation and named joint order")
        source = resolve_source_mjcf()
        if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise ValueError("The external Marvin MJCF differs from the pinned FK reference")
        model = mujoco.MjModel.from_xml_path(str(source))
        self.fk = MarvinArmFK(model, device=self.device)
        self.arm_ids = torch.tensor(
            [self.robot.joint_names.index(name) for name in ARM_JOINT_NAMES], device=self.device
        )
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES]
        self.arm_lower = torch.tensor(model.jnt_range[joint_ids, 0].copy(), dtype=torch.float32, device=self.device)
        self.arm_upper = torch.tensor(model.jnt_range[joint_ids, 1].copy(), dtype=torch.float32, device=self.device)
        self.arm_lower += cfg.joint_limit_margin_rad
        self.arm_upper -= cfg.joint_limit_margin_rad
        if torch.any(self.arm_lower >= self.arm_upper):
            raise ValueError("The S0 safety margin exceeds a joint's available range")
        keyframes = json.loads(PACKAGED_KEYFRAMES.read_text())
        if keyframes["source_sha256"] != SOURCE_SHA256 or tuple(keyframes["joint_order"]) != JOINT_ORDER:
            raise ValueError("Packaged Home does not match the pinned source model")
        self.home = torch.tensor(keyframes["keyframes"]["home"], dtype=torch.float32, device=self.device)
        self.home_arm = self.home[self.arm_ids]
        self.tcp_ids = torch.tensor(
            [self.robot.body_names.index(f"TCP_Link_{side}") for side in ("L", "R")], device=self.device
        )
        self.limiter = JointCommandLimiter(
            dt=self.step_dt,
            velocity_limit=cfg.command_velocity_limit_rad_s,
            acceleration_limit=cfg.command_acceleration_limit_rad_s2,
            tracking_lag_limit=cfg.command_tracking_lag_limit_rad,
        )
        self.joint_target = self.home.repeat(self.num_envs, 1)
        self.command_velocity = torch.zeros((self.num_envs, 14), device=self.device)
        self.previous_action = torch.zeros_like(self.command_velocity)
        self.active_mask = torch.zeros((self.num_envs, 2), device=self.device)
        self.mode_one_hot = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_pose = torch.zeros((self.num_envs, 2, 7), device=self.device)
        self.previous_metric = torch.zeros((self.num_envs, 2), device=self.device)
        self.success_steps = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        self.invalid_action = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.last_reward_terms: dict[str, torch.Tensor] = {}
        self.controller_profile = apply_newton_arm_tracking_profile(
            self.robot, damping_scale_source=cfg.controller_damping_scale
        )

    def set_eval_cases(
        self, start_arm_q: torch.Tensor, target_tcp_pose_xyzw: torch.Tensor, active_mask: torch.Tensor
    ) -> None:
        """Inject fixed offline cases; only TCP poses enter the actor observation."""
        start = start_arm_q.to(device=self.device, dtype=torch.float32)
        target = target_tcp_pose_xyzw.to(device=self.device, dtype=torch.float32)
        mask = active_mask.to(device=self.device, dtype=torch.float32)
        if start.shape != (self.num_envs, 14) or target.shape != (self.num_envs, 2, 7):
            raise ValueError("Evaluation starts or TCP goals have the wrong shape")
        if mask.shape != (self.num_envs, 2) or not torch.all((mask == 0.0) | (mask == 1.0)):
            raise ValueError("Evaluation active mask must be binary [num_envs, 2]")
        if torch.any(mask.sum(-1) < 1) or not torch.isfinite(start).all() or not torch.isfinite(target).all():
            raise ValueError("Evaluation cases must be active and finite")
        if torch.any(start < self.arm_lower) or torch.any(start > self.arm_upper):
            raise ValueError("Evaluation start violates guarded joint limits")
        if torch.any(torch.abs(torch.linalg.vector_norm(target[..., 3:], dim=-1) - 1.0) > 1e-3):
            raise ValueError("Evaluation TCP quaternion must be normalized XYZW")
        full = self.home.repeat(self.num_envs, 1)
        full[:, self.arm_ids] = start
        self.robot.write_joint_position_to_sim_index(position=full)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(full))
        self.joint_target[:] = full
        self.goal_pose[:] = target
        self.active_mask[:] = mask
        modes = torch.where(mask.sum(-1) == 2, 2, mask[:, 1].long())
        self.mode_one_hot[:] = torch.nn.functional.one_hot(modes, num_classes=3).float()
        self.command_velocity.zero_()
        self.previous_action.zero_()
        self.success_steps.zero_()
        self.invalid_action.zero_()
        self.episode_length_buf.zero_()
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(0.0)
        p, r = self._pose_errors()
        self.previous_metric[:] = torch.linalg.vector_norm(p, dim=-1) / 0.06 + torch.linalg.vector_norm(r, dim=-1) / 0.4

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.invalid_action = ~torch.isfinite(actions).all(dim=-1)
        self.previous_action = torch.nan_to_num(actions).clamp(-1.0, 1.0) * self.active_mask.repeat_interleave(7, dim=1)
        actual = self.robot.data.joint_pos.torch[:, self.arm_ids]
        previous = self.joint_target[:, self.arm_ids]
        target, velocity = self.limiter.apply(
            self.previous_action,
            actual,
            previous,
            self.command_velocity,
            self.arm_lower,
            self.arm_upper,
        )
        self.joint_target[:, self.arm_ids] = target
        self.command_velocity = velocity

    def _apply_action(self) -> None:
        self.robot.actuators.target_command.set_position_index(value=self.joint_target, full_data=True)

    def _pose_errors(self) -> tuple[torch.Tensor, torch.Tensor]:
        tcp = self.robot.data.body_pose_w.torch[:, self.tcp_ids]
        position_error = self.goal_pose[..., :3] + self.scene.env_origins[:, None, :] - tcp[..., :3]
        rotation_error = orientation_error(self.goal_pose[..., 3:], tcp[..., 3:])
        return position_error, rotation_error

    def _get_observations(self) -> dict:
        q = self.robot.data.joint_pos.torch[:, self.arm_ids]
        qd = self.robot.data.joint_vel.torch[:, self.arm_ids]
        pos_error, rot_error = self._pose_errors()
        midpoint = (self.arm_lower + self.arm_upper) * 0.5
        half_range = (self.arm_upper - self.arm_lower) * 0.5
        obs = torch.cat(
            (
                (q - midpoint) / half_range,
                qd / self.cfg.command_velocity_limit_rad_s,
                pos_error.reshape(self.num_envs, 6),
                rot_error.reshape(self.num_envs, 6),
                self.previous_action,
                (self.joint_target[:, self.arm_ids] - midpoint) / half_range,
                self.active_mask,
                self.mode_one_hot,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.robot.data.joint_pos.torch[:, self.arm_ids]
        qd = self.robot.data.joint_vel.torch[:, self.arm_ids]
        pos_error, rot_error = self._pose_errors()
        pos_norm = torch.linalg.vector_norm(pos_error, dim=-1)
        rot_norm = torch.linalg.vector_norm(rot_error, dim=-1)
        reached = (
            ((pos_norm < self.cfg.position_success_m) & (rot_norm < self.cfg.orientation_success_rad))
            | (self.active_mask < 0.5)
        ).all(dim=-1)
        self.success_steps = torch.where(reached, self.success_steps + 1, 0)
        nonfinite = ~torch.isfinite(q).all(dim=-1) | ~torch.isfinite(qd).all(dim=-1)
        limit = (
            (q < self.arm_lower - self.cfg.joint_limit_margin_rad)
            | (q > self.arm_upper + self.cfg.joint_limit_margin_rad)
        ).any(dim=-1)
        failed = nonfinite | limit | self.invalid_action
        timeout = self.episode_length_buf >= self.max_episode_length
        return failed, timeout

    def _get_rewards(self) -> torch.Tensor:
        position_error, rotation_error = self._pose_errors()
        pos = torch.linalg.vector_norm(position_error, dim=-1)
        rot = torch.linalg.vector_norm(rotation_error, dim=-1)
        mask = self.active_mask
        active_count = mask.sum(dim=-1).clamp_min(1.0)
        pos_coarse = (torch.exp(-pos / 0.06) * mask).sum(-1) / active_count
        rot_coarse = (torch.exp(-rot / 0.4) * mask).sum(-1) / active_count
        pos_fine = (torch.exp(-pos / 0.008) * mask).sum(-1) / active_count
        rot_fine = (torch.exp(-rot / 0.06) * mask).sum(-1) / active_count
        se3_coarse = (balanced_pose_score(pos, rot, 0.06, 0.4) * mask).sum(-1) / active_count
        se3_fine = (balanced_pose_score(pos, rot, 0.008, 0.06) * mask).sum(-1) / active_count
        metric = pos / 0.06 + rot / 0.4
        progress = ((self.previous_metric - metric).clamp(-1.0, 1.0) * mask).sum(-1) / active_count
        self.previous_metric = metric.detach()
        hold = (self.success_steps >= math.ceil(self.cfg.success_hold_s / self.step_dt)).float()
        smooth = self.command_velocity.square().mean(-1)
        qd = self.robot.data.joint_vel.torch[:, self.arm_ids]
        velocity = qd.square().mean(-1)
        q = self.robot.data.joint_pos.torch[:, self.arm_ids]
        lower_margin = q - self.arm_lower
        upper_margin = self.arm_upper - q
        near_limit = torch.relu(0.1 - torch.minimum(lower_margin, upper_margin)).mean(-1)
        terms = {
            "position_coarse": 0.5 * pos_coarse,
            "orientation_coarse": 0.5 * rot_coarse,
            "se3_coarse": self.cfg.se3_coarse_weight * se3_coarse,
            "position_fine": 0.25 * pos_fine,
            "orientation_fine": 0.25 * rot_fine,
            "se3_fine": self.cfg.se3_fine_weight * se3_fine,
            "progress": 0.4 * progress,
            "success_hold": 2.0 * hold,
            "command_smoothness": -0.02 * smooth,
            "joint_velocity": -0.002 * velocity,
            "joint_limit": -0.5 * near_limit,
        }
        self.last_reward_terms = {name: value.detach() for name, value in terms.items()}
        if self.common_step_counter % 100 == 0:
            self.extras["log"] = {f"Reward/{name}": value.mean().detach() for name, value in terms.items()}
        return sum(terms.values())

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        count = len(env_ids)
        if count == 0:
            return
        q = self.home.repeat(count, 1)
        start = self.home_arm + (2.0 * torch.rand((count, 14), device=self.device) - 1.0) * self.cfg.initial_radius_rad
        start = torch.maximum(torch.minimum(start, self.arm_upper), self.arm_lower)
        mode = torch.randint(0, 2, (count,), device=self.device)
        if self.cfg.bimanual_fraction:
            mode = torch.where(torch.rand((count,), device=self.device) < self.cfg.bimanual_fraction, 2, mode)
        mask = torch.stack((mode != 1, mode != 0), dim=-1).float()
        goal = start + (2.0 * torch.rand_like(start) - 1.0) * self.cfg.goal_radius_rad * mask.repeat_interleave(7, 1)
        goal = torch.maximum(torch.minimum(goal, self.arm_upper), self.arm_lower)
        q[:, self.arm_ids] = start
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
        self.goal_pose[env_ids] = self.fk(goal)
        self.joint_target[env_ids] = q
        self.command_velocity[env_ids] = 0.0
        self.previous_action[env_ids] = 0.0
        self.active_mask[env_ids] = mask
        self.mode_one_hot[env_ids] = torch.nn.functional.one_hot(mode, num_classes=3).float()
        self.success_steps[env_ids] = 0
        self.invalid_action[env_ids] = False
        start_pose = self.fk(start)
        p = torch.linalg.vector_norm(self.goal_pose[env_ids, :, :3] - start_pose[:, :, :3], dim=-1)
        r = torch.linalg.vector_norm(orientation_error(self.goal_pose[env_ids, :, 3:], start_pose[:, :, 3:]), dim=-1)
        self.previous_metric[env_ids] = p / 0.06 + r / 0.4
