"""Newton dual-TCP Reach physics using the existing validated 54DoF servo."""

import torch
from isaaclab.utils import configclass

from .core_plugins import MarvinDualTCPReach, MarvinCoreRobot
from .env import MarvinWaveDirectEnv
from .env_cfg import MarvinWaveDirectCfg


PROCEDURAL_RESET = (
    "/home/ubuntu/dexgym_data/marvin_supermimic_m42/second_task/"
    "procedural_bimanual_reach_v1.npz")


@configclass
class MarvinReachDirectCfg(MarvinWaveDirectCfg):
    reference_path = PROCEDURAL_RESET  # Used only to initialize a safe full-body pose.
    action_schema = "robot_fixed_v3"
    initialize_hands_from_pd_replay = False
    random_reference_start = False
    reset_joint_perturb_rad = 0.0
    observation_space = 4 * 48 + 6
    episode_length_s = 2.0


class MarvinReachDirectEnv(MarvinWaveDirectEnv):
    cfg: MarvinReachDirectCfg

    def __init__(self, cfg: MarvinReachDirectCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.core_robot = MarvinCoreRobot(self.motion, self.device, action_schema=cfg.action_schema)
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
        state = self.core_robot.physical_state(self)
        target = self.reach_task.task_state_from_reset(state).data
        self.reach_target[env_ids] = target[env_ids]
        self.reach_frame[env_ids] = 0
        first = self.reach_task.observation(state, self.last_action, self.task_state())
        self.reach_history[env_ids] = first[env_ids, None].expand(-1, 4, -1)
