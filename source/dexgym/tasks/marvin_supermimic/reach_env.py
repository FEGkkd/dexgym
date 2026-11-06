"""Independent dual-TCP Reach MDP on the shared Marvin arm Newton physics."""

from types import SimpleNamespace

import numpy as np
import torch
from isaaclab.utils import configclass

from dexgym.assets.marvin_wuji.controller import ARM_JOINT_NAMES

from .base_env import MarvinArmNewtonBaseEnv
from .core_plugins import MarvinDualTCPReach, MarvinCoreRobot
from .env_cfg import MarvinArmNewtonBaseCfg


PROCEDURAL_RESET = (
    "/home/ubuntu/dexgym_data/marvin_supermimic_m42/second_task/"
    "procedural_bimanual_reach_v1.npz")


@configclass
class MarvinReachDirectCfg(MarvinArmNewtonBaseCfg):
    reset_state_path = PROCEDURAL_RESET
    action_schema = "robot_fixed_v3"
    observation_space = 4 * 48 + 6
    episode_length_s = 2.0


class MarvinReachDirectEnv(MarvinArmNewtonBaseEnv):
    cfg: MarvinReachDirectCfg

    def __init__(self, cfg: MarvinReachDirectCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        joint_source = SimpleNamespace(joint_names_arm=ARM_JOINT_NAMES)
        with np.load(cfg.reset_state_path, allow_pickle=False) as source:
            if tuple(source["joint_names_arm"]) != ARM_JOINT_NAMES or source["q_full"].shape[1] != 54:
                raise ValueError("Reach safe reset pose has an incompatible Marvin joint schema")
            initial_arm = torch.as_tensor(source["q_arm"][0].copy(), device=self.device)
            initial_velocity = torch.as_tensor(source["qd_arm"][0].copy(), device=self.device)
            initial_full = torch.as_tensor(source["q_full"][0].copy(), device=self.device)
        self._setup_arm_physics(joint_source, initial_arm)
        self.reset_arm_q = initial_arm
        self.reset_arm_qd = initial_velocity
        self.reset_full_q = initial_full[self.model_to_newton]
        self.core_robot = MarvinCoreRobot(joint_source, self.device, action_schema=cfg.action_schema)
        self.reach_task = MarvinDualTCPReach(self.core_robot, episode_length=100)
        self.reach_target = torch.zeros((self.num_envs, 6), device=self.device)
        self.reach_frame = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reach_history = torch.zeros((self.num_envs, 4, 48), device=self.device)

    def task_state(self):
        from dexgym.algorithms.supermimic.core.transition import TaskState

        return TaskState(torch.ones(self.num_envs, device=self.device, dtype=torch.long),
                         torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long),
                         self.reach_frame.clone(), self.reach_target.clone())

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self.reach_frame.add_(1)

    def _get_observations(self):
        state = self.core_robot.physical_state(self)
        frame = self.reach_task.observation(state, self.last_action, self.task_state())
        self.reach_history = torch.cat((self.reach_history[:, 1:], frame[:, None]), dim=1)
        return {"policy": torch.cat((self.reach_history.flatten(1), self.reach_target), dim=-1)}

    def _get_rewards(self):
        state = self.core_robot.physical_state(self)
        previous_state = torch.cat((self.previous_q, state[:, 14:28], self.previous_command), dim=-1)
        self.last_physical_q = self.robot.data.joint_pos.torch.detach().clone()
        self.last_physical_qd = self.robot.data.joint_vel.torch.detach().clone()
        self.last_physical_torque = self.robot.data.applied_torque.torch.detach().clone()
        return self.reach_task.reward(previous_state, state, self.last_action,
                                      self.task_state(), self.previous_action)

    def _get_dones(self):
        self._record_contacts()
        state = self.core_robot.physical_state(self)
        self.last_physical_arm_state = state.detach().clone()
        self.last_policy_contact_counts = self._contact_accum.clone()
        self.last_policy_contact_physics_steps = int(self._contact_accum.sum())
        task_state = self.task_state()
        bad = self.reach_task.hard_termination(state, task_state, self.last_policy_contact_counts)
        self.last_failure_reason = bad.int()
        timeout = (self.episode_length_buf >= self.max_episode_length) | (
            self.reach_frame >= self.reach_task.episode_length)
        return bad, timeout

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "reach_task"):
            return
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        count = len(env_ids)
        q = self.reset_full_q[None].repeat(count, 1)
        qd = torch.zeros_like(q)
        arm_q = self.reset_arm_q[None].repeat(count, 1)
        arm_qd = self.reset_arm_qd[None].repeat(count, 1)
        self._write_robot_reset(env_ids, q, qd, arm_q, arm_qd)
        state = self.core_robot.physical_state(self)
        target = self.reach_task.task_state_from_reset(state).data
        self.reach_target[env_ids] = target[env_ids]
        self.reach_frame[env_ids] = 0
        first = self.reach_task.observation(state, self.last_action, self.task_state())
        self.reach_history[env_ids] = first[env_ids, None].expand(-1, 4, -1)
