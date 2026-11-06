"""Arm-only bimanual motion tracking with real Newton MJWarp stepping."""

from collections.abc import Sequence

import numpy as np
import torch

from dexgym.algorithms.supermimic.observation import noisy_proprio_frame
from dexgym.algorithms.supermimic.rewards import REWARD_PROFILES, tracking_reward
from dexgym.assets.supermimic import PD_WAVE_STATES, WAVE_REFERENCE
from dexgym.motion.motion_manager import MotionManager

from .base_env import MarvinArmNewtonBaseEnv
from .contact_audit import unsafe_arm_states
from .env_cfg import MarvinWaveDirectCfg

REFERENCE = str(WAVE_REFERENCE)
PD_STATES = PD_WAVE_STATES


class MarvinWaveDirectEnv(MarvinArmNewtonBaseEnv):
    cfg: MarvinWaveDirectCfg

    def __init__(self, cfg: MarvinWaveDirectCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.motion = MotionManager(cfg.reference_path, self.num_envs, self.device)
        start = self.motion.data["q_arm"][0]
        self._setup_arm_physics(self.motion, start)
        if cfg.initialize_hands_from_pd_replay:
            with np.load(PD_STATES, allow_pickle=False) as replay:
                if replay["q"].shape != (self.motion.length, 1, 54) or replay["qd"].shape != (
                    self.motion.length,
                    1,
                    54,
                ):
                    raise ValueError("Pinned Newton PD state shape mismatch")
                self.pd_hand_q = torch.tensor(replay["q"][:, 0].copy(), device=self.device)[:, self.model_to_newton]
                self.pd_hand_qd = torch.tensor(replay["qd"][:, 0].copy(), device=self.device)[:, self.model_to_newton]
        self.history = torch.zeros((self.num_envs, 10, 56), device=self.device)
        self.prev_link_pos = self.motion.data["link_pos"][0][None].repeat(self.num_envs, 1, 1, 1)
        if cfg.reward_profile not in REWARD_PROFILES:
            raise ValueError(f"Unknown reward profile: {cfg.reward_profile}")
        self.reward_weights = REWARD_PROFILES[cfg.reward_profile]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
        self.motion.advance()

    def _get_observations(self) -> dict:
        q, qd = self._arm_state()
        frame = torch.cat(
            (
                (q - self.action_map.origin) / self.action_map.scale,
                qd / 2.0,
                self.last_action.tanh(),
                (self.command - self.action_map.origin) / self.action_map.scale,
            ),
            dim=-1,
        )
        frame = noisy_proprio_frame(
            frame,
            self.action_map.scale,
            position_rad=self.cfg.observation_position_noise_rad,
            velocity_rad_s=self.cfg.observation_velocity_noise_rad_s,
        )
        self.history = torch.cat((self.history[:, 1:], frame[:, None]), dim=1)
        goal = self.motion.goal()
        return {"policy": torch.cat((self.history.flatten(1), goal.flatten(1)), dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        q, _ = self._arm_state()
        self.last_physical_q = self.robot.data.joint_pos.torch.detach().clone()
        self.last_physical_qd = self.robot.data.joint_vel.torch.detach().clone()
        self.last_physical_torque = self.robot.data.applied_torque.torch.detach().clone()
        target = {
            key: self.motion.current(key) for key in ("link_pos", "link_rotmat", "link_linear_vel", "link_angular_vel")
        }
        reward, self.last_tracking = tracking_reward(
            q,
            self.previous_q,
            self.command,
            self.previous_command,
            target,
            self.fk,
            self.action_map.lower,
            self.action_map.upper,
            weights=self.reward_weights,
            dt=self.step_dt,
            bounded_action=self.last_action.tanh(),
            previous_bounded_action=self.previous_action.tanh(),
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._record_contacts()
        q, qd = self._arm_state()
        self.last_physical_arm_state = torch.cat((q, qd, self.command), dim=-1).detach().clone()
        tcp_pos = self.fk(q)[0][:, :, 2]
        reference_tcp = self.motion.current("link_pos")[:, :, 2]
        tcp_error = torch.linalg.vector_norm(tcp_pos - reference_tcp, dim=-1)
        self.last_policy_contact_counts = self._contact_accum.clone()
        self.last_policy_contact_physics_steps = int(self._contact_accum.sum())
        from dexgym.algorithms.supermimic.termination import TerminationThresholds
        thresholds = TerminationThresholds()
        self.last_failure_reason = (
            ((tcp_error > thresholds.tcp_error_m).any(-1).int() * 1)
            | (((q < self.action_map.lower - thresholds.joint_limit_margin_rad)
                | (q > self.action_map.upper + thresholds.joint_limit_margin_rad)).any(-1).int() * 2)
            | ((qd.abs() > thresholds.speed_rad_s).any(-1).int() * 4)
            | (self.last_policy_contact_counts.gt(0).int() * 8)
            | ((~torch.isfinite(q).all(-1) | ~torch.isfinite(qd).all(-1)).int() * 16)
        )
        bad = unsafe_arm_states(
            q, qd, self.action_map.lower, self.action_map.upper, tcp_error, self.last_policy_contact_counts,
            thresholds=thresholds,
        )
        timeout = (self.episode_length_buf >= self.max_episode_length) | (self.motion.frame >= self.motion.length - 1)
        return bad, timeout

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        count = len(env_ids)
        fixed_start = getattr(self.cfg, "fixed_reference_start_frame", None)
        if fixed_start is not None and not 0 <= fixed_start < self.motion.length - 1:
            raise ValueError("Evaluation reference start frame is outside the motion")
        frames = (torch.full((count,), fixed_start, device=self.device, dtype=torch.long)
                  if fixed_start is not None else
                  torch.randint(0, self.motion.length - 1, (count,), device=self.device)
                  if self.cfg.random_reference_start else
                  torch.zeros(count, device=self.device, dtype=torch.long))
        if self.cfg.random_reference_start and (self.cfg.active_reset_fraction or self.cfg.early_reset_fraction):
            selector = torch.rand(count, device=self.device)
            active = selector < self.cfg.active_reset_fraction
            early = (selector >= self.cfg.active_reset_fraction) & (
                selector < self.cfg.active_reset_fraction + self.cfg.early_reset_fraction
            )
            active_start, active_end = self.cfg.active_reset_frame_range
            early_start, early_end = self.cfg.early_reset_frame_range
            frames = torch.where(active, torch.randint(active_start, active_end, (count,), device=self.device), frames)
            frames = torch.where(early, torch.randint(early_start, early_end, (count,), device=self.device), frames)
        self.motion.reset(env_ids, frames)
        q = self.motion.current("q_full")[env_ids][:, self.model_to_newton].clone()
        qd = torch.zeros_like(q)
        if self.cfg.initialize_hands_from_pd_replay:
            q[:, self.hand_ids] = self.pd_hand_q[frames][:, self.hand_ids]
            qd[:, self.hand_ids] = self.pd_hand_qd[frames][:, self.hand_ids]
        arm_q = self.motion.current("q_arm")[env_ids]
        arm_qd = self.motion.current("qd_arm")[env_ids]
        perturb = (torch.rand_like(arm_q) * 2 - 1) * self.cfg.reset_joint_perturb_rad
        arm_q = torch.clamp(arm_q + perturb, self.action_map.lower, self.action_map.upper)
        self._write_robot_reset(env_ids, q, qd, arm_q, arm_qd)
        first_observation = torch.cat(
            (
                (arm_q - self.action_map.origin) / self.action_map.scale,
                arm_qd / 2.0,
                torch.zeros_like(arm_q),
                (arm_q - self.action_map.origin) / self.action_map.scale,
            ),
            dim=-1,
        )
        first_observation = noisy_proprio_frame(
            first_observation,
            self.action_map.scale,
            position_rad=self.cfg.observation_position_noise_rad,
            velocity_rad_s=self.cfg.observation_velocity_noise_rad_s,
        )
        self.history[env_ids] = first_observation[:, None, :]
