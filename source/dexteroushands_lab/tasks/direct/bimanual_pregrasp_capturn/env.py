"""One joint actor learns pregrasp, bottle support, and contact-driven cap turn."""

import torch

from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env import BimanualBottleDirectEnv
from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.action_math import policy_wrist_speed

from .contract import MAX_OPENING_STEP_RAD
from .reward_terms import SUCCESS_BONUS, accumulate_reward_terms, unified_reward_terms


class PregraspCapTurn30DirectEnv(BimanualBottleDirectEnv):
    """Reuses the physical scene and owns the unified task's reward and fault gate."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        if (cfg.thread_backend != "mimic" or cfg.sim.physics.solver_type != 0
                or cfg.stable_reset_probability != 0):
            raise ValueError("Phase 1B requires pregrasp-only PGS Mimic")
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._angle_fault = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "_angle_fault"):
            self._angle_fault[env_ids] = False

    def _pre_physics_step(self, actions):
        self._policy_wrist_start_pos = [w.current_pos.clone() for w in self.wrists]
        self._policy_wrist_start_quat = [w.current_quat.clone() for w in self.wrists]
        super()._pre_physics_step(actions)

    def _update_metrics(self):
        super()._update_metrics()
        if not hasattr(self, "_angle_fault"):
            self._angle_fault = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        jump = ((self._metrics["opening"] - self._previous_open).abs()
                > MAX_OPENING_STEP_RAD)
        self._angle_fault |= jump | ~torch.isfinite(self._metrics["opening"])
        self._metrics["cap_ok"] &= ~self._angle_fault
        self._success_steps = torch.where(
            self._angle_fault, torch.zeros_like(self._success_steps), self._success_steps)

    def _get_dones(self):
        terminated, timeout = super()._get_dones()
        terminated |= self._angle_fault
        episode = self.extras["phase1_episode"]
        episode["angle_fault"] = self._angle_fault.clone()
        episode["failure"] |= self._angle_fault
        episode["success"] &= ~self._angle_fault
        return terminated, timeout

    def _get_rewards(self):
        m = self._metrics
        proximity = torch.stack((m["left_proximity"], m["right_proximity"]), dim=-1)
        previous = torch.where(self._proximity_fresh[:, None],
                               self._previous_proximity, proximity)
        pitch_error = m["axial_travel"] - m["axial_goal"]
        terms = unified_reward_terms(
            m["left_contact_score"], m["right_contact_score"],
            proximity[:, 0], proximity[:, 1], previous[:, 0], previous[:, 1],
            m["opening"], self._previous_open, m["turn_support"],
            m["bottle_stable"], pitch_error, self._last_action,
            dt=self.step_dt, goal_rad=self.cfg.cap_goal_rad,
            pitch_tolerance_m=self.cfg.cap_pitch_tolerance_m,
            profile=self.cfg.unified_reward_profile)
        terms["success"] = SUCCESS_BONUS * (self._success_steps == 5).float()
        terms["failure"] = -5.0 * self.extras["phase1_episode"]["failure"].float()
        reward = sum(terms.values())
        left_linear_speed, left_angular_speed = policy_wrist_speed(
            self._policy_wrist_start_pos[0], self.wrists[0].current_pos,
            self._policy_wrist_start_quat[0], self.wrists[0].current_quat,
            self.step_dt)
        right_linear_speed, right_angular_speed = policy_wrist_speed(
            self._policy_wrist_start_pos[1], self.wrists[1].current_pos,
            self._policy_wrist_start_quat[1], self.wrists[1].current_quat,
            self.step_dt)
        self._previous_proximity = proximity
        self._proximity_fresh[:] = True
        self._previous_open = m["opening"].clone()
        self.extras["metrics"] = {
            "left_contact_rate": m["left_contact"].float().mean(),
            "right_contact_rate": m["right_contact"].float().mean(),
            "bilateral_support_rate": m["turn_support"].float().mean(),
            "bottle_stable_rate": m["bottle_stable"].float().mean(),
            "bottle_height_mean_m": self.bottle.data.root_pose_w.torch[:, 2].mean(),
            "bottle_linear_speed_mean_m_s": torch.linalg.vector_norm(
                self.bottle.data.root_vel_w.torch[:, :3], dim=-1).mean(),
            "bottle_angular_speed_mean_rad_s": torch.linalg.vector_norm(
                self.bottle.data.root_vel_w.torch[:, 3:], dim=-1).mean(),
            "cap_linear_speed_mean_m_s": torch.linalg.vector_norm(
                self.cap.data.root_vel_w.torch[:, :3], dim=-1).mean(),
            "actual_left_wrist_linear_speed_m_s": left_linear_speed.mean(),
            "actual_right_wrist_linear_speed_m_s": right_linear_speed.mean(),
            "actual_left_wrist_angular_speed_rad_s": left_angular_speed.mean(),
            "actual_right_wrist_angular_speed_rad_s": right_angular_speed.mean(),
            "opening_mean_rad": m["opening"].mean(),
            "opening_max_rad": m["opening"].max(),
            "pitch_error_abs_mean_m": pitch_error.abs().mean(),
            "constraint_error_abs_mean_m": self._latest_thread.constraint_error.abs().mean(),
            "angle_fault_rate": self._angle_fault.float().mean(),
            "raw_action_element_outside_fraction": (self._last_raw_action.abs() > 1).float().mean(),
            "clipped_action_abs_mean": self._last_action.abs().mean(),
            "left_wrist_action_abs_mean": self._last_raw_action[:, :6].abs().mean(),
            "left_finger_action_abs_mean": self._last_raw_action[:, 6:26].abs().mean(),
            "right_wrist_action_abs_mean": self._last_raw_action[:, 26:32].abs().mean(),
            "right_finger_action_abs_mean": self._last_raw_action[:, 32:].abs().mean(),
            "joint_target_clip_fraction": self._joint_clip_count.float().sum() /
                (2 * (self.episode_length_buf.float() + 1).sum()),
            "wrist_workspace_clip_fraction": (
                self.wrists[0].workspace_clip_count.float().sum() +
                self.wrists[1].workspace_clip_count.float().sum()) /
                (2 * (self.episode_length_buf.float() + 1).sum()),
            "invalid_action_fraction": self._invalid_action_count.float().sum() /
                (self.episode_length_buf.float() + 1).sum(),
        }
        self.extras["log"] = {f"phase1b/{key}": value for key, value in self.extras["metrics"].items()}
        accumulate_reward_terms(self._reward_sums, terms)
        for name, value in terms.items():
            self.extras["log"][f"reward/{name}_mean"] = value.mean()
            self.extras["log"][f"reward/{name}_sum_mean"] = self._reward_sums[name].mean()
        return reward
