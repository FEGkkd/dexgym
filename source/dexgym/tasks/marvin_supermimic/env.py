"""Arm-only bimanual motion tracking with real Newton MJWarp stepping."""

from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp
from isaaclab.envs import DirectRLEnv

from dexgym.algorithms.supermimic.observation import noisy_proprio_frame
from dexgym.algorithms.supermimic.rewards import REWARD_PROFILES, tracking_reward
from dexgym.assets.marvin_wuji.controller import apply_newton_arm_tracking_profile
from dexgym.motion.motion_manager import MotionManager

from .contact_audit import contact_counts, unsafe_arm_states
from .env_cfg import MarvinWaveDirectCfg
from .robot_adapter import MarvinArmDynamicsAdapter

REFERENCE = "/home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/cmu_141_16_wave_bimanual.npz"
MJCF = "/home/ubuntu/dexgym/source/dexgym/assets/marvin_wuji/source_model/dual_arm_marvin.xml"
PD_STATES = Path(
    "/home/ubuntu/dexgym_data/marvin_motion_library_v1/bimanual/physics_replays/newton_1env/newton_pd_states.npz"
)


class MarvinWaveDirectEnv(DirectRLEnv):
    cfg: MarvinWaveDirectCfg

    def __init__(self, cfg: MarvinWaveDirectCfg, render_mode: str | None = None, **kwargs):
        if cfg.decimation != MarvinArmDynamicsAdapter.physics_substeps or cfg.sim.dt != MarvinArmDynamicsAdapter.physics_dt:
            raise ValueError("Marvin action contract requires ten 2 ms Newton substeps per control step")
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        if self.robot.num_joints != 54:
            raise ValueError("Newton robot must retain all 54 joints")
        self.motion = MotionManager(cfg.reference_path, self.num_envs, self.device)
        self.arm_ids = torch.tensor(
            [self.robot.joint_names.index(name) for name in self.motion.joint_names_arm],
            device=self.device,
            dtype=torch.long,
        )
        model = mujoco.MjModel.from_xml_path(MJCF)
        self.dynamics_adapter = MarvinArmDynamicsAdapter(
            self.motion, self.device, action_schema=cfg.action_schema, mjcf=MJCF
        )
        self.action_map = self.dynamics_adapter.action_map
        start = self.motion.data["q_arm"][0]
        model_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(54))
        if set(model_names) != set(self.robot.joint_names):
            raise ValueError("Newton/MuJoCo 54DoF joint names differ")
        self.model_to_newton = torch.tensor(
            [model_names.index(name) for name in self.robot.joint_names],
            device=self.device,
            dtype=torch.long,
        )
        self.home_full = torch.tensor(
            model.key_qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")].copy(),
            device=self.device,
            dtype=torch.float32,
        )[self.model_to_newton]
        self.hand_ids = torch.tensor(
            [j for j, name in enumerate(self.robot.joint_names) if name not in self.motion.joint_names_arm],
            device=self.device,
            dtype=torch.long,
        )
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
        self.full_command = self.robot.data.default_joint_pos.torch.clone()
        self.command = start[None].repeat(self.num_envs, 1)
        self.previous_command = self.command.clone()
        self.previous_q = self.command.clone()
        self.last_action = torch.zeros((self.num_envs, 14), device=self.device)
        self.previous_action = self.last_action.clone()
        self.history = torch.zeros((self.num_envs, 10, 56), device=self.device)
        self.prev_link_pos = self.motion.data["link_pos"][0][None].repeat(self.num_envs, 1, 1, 1)
        self.profile = apply_newton_arm_tracking_profile(self.robot, damping_scale_source=0.5)
        if cfg.reward_profile not in REWARD_PROFILES:
            raise ValueError(f"Unknown reward profile: {cfg.reward_profile}")
        self.reward_weights = REWARD_PROFILES[cfg.reward_profile]
        self.fk = self.dynamics_adapter.fk
        self._substep = 0
        if cfg.audit_contacts_each_physics_step and self._physics_handles_decimation and cfg.decimation > 1:
            raise RuntimeError("Newton internal decimation prevents physics-step contact auditing")
        self._contact_accum = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.last_policy_contact_counts = self._contact_accum.clone()
        self.last_policy_contact_physics_steps = 0

    def _record_contacts(self) -> None:
        data = self.sim.physics_manager._solver.mjw_data
        active_count = int(data.nacon.numpy()[0])
        if active_count:
            world_ids = wp.to_torch(data.contact.worldid)
            self._contact_accum += contact_counts(world_ids, active_count, self.num_envs)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 14) or not torch.isfinite(actions).all():
            raise ValueError("Invalid 14D policy action")
        self.previous_q = self.robot.data.joint_pos.torch[:, self.arm_ids].clone()
        self.previous_command = self.command.clone()
        self.previous_action = self.last_action.clone()
        self.command = self.action_map(actions, self.command)
        self.last_action = actions
        self.motion.advance()
        self._substep = 0
        self._contact_accum.zero_()

    def _apply_action(self) -> None:
        if self.cfg.audit_contacts_each_physics_step and self._substep:
            self._record_contacts()
        self.full_command[:] = self.home_full
        self.full_command[:, self.arm_ids] = self.dynamics_adapter.interpolate_command(
            self.previous_command, self.command, self._substep
        )
        self.robot.actuators.target_command.set_position_index(value=self.full_command, full_data=True)
        self._substep += 1

    def _arm_state(self):
        return self.robot.data.joint_pos.torch[:, self.arm_ids], self.robot.data.joint_vel.torch[:, self.arm_ids]

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
        bad = unsafe_arm_states(
            q, qd, self.action_map.lower, self.action_map.upper, tcp_error, self.last_policy_contact_counts
        )
        timeout = (self.episode_length_buf >= self.max_episode_length) | (self.motion.frame >= self.motion.length - 1)
        return bad, timeout

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        count = len(env_ids)
        frames = (
            torch.randint(0, self.motion.length - 1, (count,), device=self.device)
            if self.cfg.random_reference_start
            else torch.zeros(count, device=self.device, dtype=torch.long)
        )
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
        self.command[env_ids] = arm_q
        self.previous_command[env_ids] = arm_q
        self.previous_q[env_ids] = arm_q
        q[:, self.arm_ids] = arm_q
        qd[:, self.arm_ids] = arm_qd
        root = self.robot.data.default_root_pose.torch[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=root, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=self.robot.data.default_root_vel.torch[env_ids].clone(), env_ids=env_ids
        )
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=env_ids)
        self.last_action[env_ids] = 0
        self.previous_action[env_ids] = 0
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
