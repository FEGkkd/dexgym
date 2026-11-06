"""Shared 54DoF Marvin Newton physics and 14D position servo, without a task MDP."""

from collections.abc import Sequence

import mujoco
import torch
import warp as wp
from isaaclab.envs import DirectRLEnv

from dexgym.assets.marvin_wuji.controller import ARM_JOINT_NAMES, apply_newton_arm_tracking_profile

from .contact_audit import contact_counts
from .robot_adapter import DEFAULT_MJCF, MarvinArmDynamicsAdapter


class MarvinArmNewtonBaseEnv(DirectRLEnv):
    """Physics layer shared by tracking and reach; subclasses own reset goals and MDP."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        if (cfg.decimation != MarvinArmDynamicsAdapter.physics_substeps
                or cfg.sim.dt != MarvinArmDynamicsAdapter.physics_dt):
            raise ValueError("Marvin action contract requires ten 2 ms Newton substeps per control step")
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_arm_physics(self, joint_source, initial_command):
        self.robot = self.scene["robot"]
        if self.robot.num_joints != 54 or tuple(joint_source.joint_names_arm) != ARM_JOINT_NAMES:
            raise ValueError("Newton robot requires 54 joints and the canonical 14 arm joint order")
        self.arm_ids = torch.tensor(
            [self.robot.joint_names.index(name) for name in ARM_JOINT_NAMES],
            device=self.device, dtype=torch.long)
        model = mujoco.MjModel.from_xml_path(str(DEFAULT_MJCF))
        self.dynamics_adapter = MarvinArmDynamicsAdapter(
            joint_source, self.device, action_schema=self.cfg.action_schema, mjcf=DEFAULT_MJCF)
        self.action_map = self.dynamics_adapter.action_map
        model_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(54))
        if set(model_names) != set(self.robot.joint_names):
            raise ValueError("Newton/MuJoCo 54DoF joint names differ")
        self.model_to_newton = torch.tensor(
            [model_names.index(name) for name in self.robot.joint_names],
            device=self.device, dtype=torch.long)
        self.home_full = torch.tensor(
            model.key_qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")].copy(),
            device=self.device, dtype=torch.float32)[self.model_to_newton]
        self.hand_ids = torch.tensor(
            [j for j, name in enumerate(self.robot.joint_names) if name not in ARM_JOINT_NAMES],
            device=self.device, dtype=torch.long)
        self.full_command = self.robot.data.default_joint_pos.torch.clone()
        self.command = initial_command[None].repeat(self.num_envs, 1)
        self.previous_command = self.command.clone()
        self.previous_q = self.command.clone()
        self.last_action = torch.zeros((self.num_envs, 14), device=self.device)
        self.previous_action = self.last_action.clone()
        self.profile = apply_newton_arm_tracking_profile(self.robot, damping_scale_source=0.5)
        self.fk = self.dynamics_adapter.fk
        self._substep = 0
        if self.cfg.audit_contacts_each_physics_step and self._physics_handles_decimation and self.cfg.decimation > 1:
            raise RuntimeError("Newton internal decimation prevents physics-step contact auditing")
        self._contact_accum = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.last_policy_contact_counts = self._contact_accum.clone()
        self.last_policy_contact_physics_steps = 0
        self.last_failure_reason = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

    def _record_contacts(self):
        data = self.sim.physics_manager._solver.mjw_data
        active_count = int(data.nacon.numpy()[0])
        if active_count:
            world_ids = wp.to_torch(data.contact.worldid)
            self._contact_accum += contact_counts(world_ids, active_count, self.num_envs)

    def _pre_physics_step(self, actions):
        if actions.shape != (self.num_envs, 14) or not bool(torch.isfinite(actions).all()):
            raise ValueError("Invalid 14D policy action")
        self.previous_q = self.robot.data.joint_pos.torch[:, self.arm_ids].clone()
        self.previous_command = self.command.clone()
        self.previous_action = self.last_action.clone()
        self.command = self.action_map(actions, self.command)
        # Reset writes last_action[env_ids] after the physics step. Keep the
        # caller's action tensor independent so Replay retains the action
        # actually executed on the terminal transition.
        self.last_action = actions.clone()
        self._substep = 0
        self._contact_accum.zero_()

    def _apply_action(self):
        if self.cfg.audit_contacts_each_physics_step and self._substep:
            self._record_contacts()
        self.full_command[:] = self.home_full
        self.full_command[:, self.arm_ids] = self.dynamics_adapter.interpolate_command(
            self.previous_command, self.command, self._substep)
        self.robot.actuators.target_command.set_position_index(value=self.full_command, full_data=True)
        self._substep += 1

    def _arm_state(self):
        return self.robot.data.joint_pos.torch[:, self.arm_ids], self.robot.data.joint_vel.torch[:, self.arm_ids]

    def _write_robot_reset(self, env_ids: Sequence[int] | torch.Tensor, q, qd, arm_q, arm_qd):
        self.command[env_ids] = arm_q
        self.previous_command[env_ids] = arm_q
        self.previous_q[env_ids] = arm_q
        q[:, self.arm_ids] = arm_q
        qd[:, self.arm_ids] = arm_qd
        root = self.robot.data.default_root_pose.torch[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=root, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=self.robot.data.default_root_vel.torch[env_ids].clone(), env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=env_ids)
        self.last_action[env_ids] = 0
        self.previous_action[env_ids] = 0

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
