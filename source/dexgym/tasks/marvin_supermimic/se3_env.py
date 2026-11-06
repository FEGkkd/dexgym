"""IK-free dual-arm SE(3) reach and hold on the shared Marvin Newton servo."""

from types import SimpleNamespace

import torch
from isaaclab.utils import configclass

from dexgym.assets.marvin_wuji.controller import ARM_JOINT_NAMES

from .base_env import MarvinArmNewtonBaseEnv
from .core_plugins import MarvinCoreRobot
from .env_cfg import MarvinArmNewtonBaseCfg
from .se3_goal_sampler import SE3GoalSampler
from .se3_reach_hold import MarvinBimanualSE3ReachHold


@configclass
class MarvinSE3ReachHoldCfg(MarvinArmNewtonBaseCfg):
    action_schema = "arm_joint_delta_v1"
    observation_space = 10 * 68 + 18
    episode_length_s = 4.0
    goal_train_seed = 5101
    goal_eval_seed = 9501
    curriculum_level = "near"
    hold_speed_mode = "instantaneous_qd_v1"
    evaluation_mode = False
    collection_contract_version = None


class MarvinSE3ReachHoldEnv(MarvinArmNewtonBaseEnv):
    """Task-specific goal/reset/MDP; all servo and 54DoF physics live in BaseEnv."""

    cfg: MarvinSE3ReachHoldCfg

    def __init__(self, cfg: MarvinSE3ReachHoldCfg, render_mode=None, **kwargs):
        if cfg.action_schema != "arm_joint_delta_v1" or cfg.episode_length_s != 4.0:
            raise ValueError("SE(3) reach and hold requires its versioned delta and 200-step episode")
        super().__init__(cfg, render_mode, **kwargs)
        source = SimpleNamespace(joint_names_arm=ARM_JOINT_NAMES)
        self.core_robot = MarvinCoreRobot(source, self.device, action_schema=cfg.action_schema)
        self._setup_arm_physics(source, self.core_robot.action_map.origin)
        self.se3_task = MarvinBimanualSE3ReachHold(
            self.core_robot, hold_speed_mode=cfg.hold_speed_mode)
        self.goal_sampler = SE3GoalSampler(
            self.core_robot, train_seed=cfg.goal_train_seed,
            evaluation_seed=cfg.goal_eval_seed)
        self.goal_pose = torch.zeros((self.num_envs, 18), device=self.device)
        self.goal_frame = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.se3_history = torch.zeros((self.num_envs, 10, 68), device=self.device)
        self.previous_qd = torch.zeros((self.num_envs, 14), device=self.device)
        self.reset_position_difficulty_m = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_rotation_difficulty_rad = torch.zeros((self.num_envs, 2), device=self.device)

    def task_state(self):
        return self.se3_task.task_state_from_goal(self.goal_pose.clone(),
                                                   frame=self.goal_frame.clone())

    def _pre_physics_step(self, actions):
        self.previous_qd = self.robot.data.joint_vel.torch[:, self.arm_ids].clone()
        super()._pre_physics_step(actions)
        self.goal_frame.add_(1)

    def _get_observations(self):
        state = self.core_robot.physical_state(self)
        frame = self.se3_task.observation(state, self.last_action, self.task_state())
        self.se3_history = torch.cat((self.se3_history[:, 1:], frame[:, None]), dim=1)
        return {"policy": torch.cat((self.se3_history.flatten(1), self.goal_pose), dim=-1)}

    def _get_rewards(self):
        state = self.core_robot.physical_state(self)
        previous_state = torch.cat((self.previous_q, self.previous_qd,
                                    self.previous_command), dim=-1)
        self.last_physical_q = self.robot.data.joint_pos.torch.detach().clone()
        self.last_physical_qd = self.robot.data.joint_vel.torch.detach().clone()
        self.last_physical_torque = self.robot.data.applied_torque.torch.detach().clone()
        return self.se3_task.reward(previous_state, state, self.last_action,
                                    self.task_state(), self.previous_action)

    def _get_dones(self):
        self._record_contacts()
        state = self.core_robot.physical_state(self)
        self.last_physical_arm_state = state.detach().clone()
        self.last_policy_contact_counts = self._contact_accum.clone()
        self.last_policy_contact_physics_steps = int(self._contact_accum.sum())
        thresholds = self.se3_task.safety_thresholds
        q, qd = state[:, :14], state[:, 14:28]
        self.last_failure_reason = (
            (((q < self.action_map.lower - thresholds.joint_limit_margin_rad)
               | (q > self.action_map.upper + thresholds.joint_limit_margin_rad)).any(-1).int() * 2)
            | ((qd.abs() > thresholds.speed_rad_s).any(-1).int() * 4)
            | (self.last_policy_contact_counts.gt(0).int() * 8)
            | ((~torch.isfinite(q).all(-1) | ~torch.isfinite(qd).all(-1)).int() * 16))
        bad = self.se3_task.hard_termination(state, self.task_state(),
                                             self.last_policy_contact_counts)
        timeout = (self.episode_length_buf >= self.max_episode_length) | (
            self.goal_frame >= self.se3_task.episode_length)
        return bad, timeout

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # DirectRLEnv invokes virtual reset during its own construction.
        if not hasattr(self, "goal_sampler"):
            return
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        count = len(env_ids)
        samples = (self.goal_sampler.sample_evaluation(count, self.cfg.curriculum_level)
                   if self.cfg.evaluation_mode else
                   self.goal_sampler.sample(count, self.cfg.curriculum_level))
        full_q = self.home_full[None].repeat(count, 1)
        full_qd = torch.zeros_like(full_q)
        arm_q = samples.start_q.to(self.device)
        self._write_robot_reset(env_ids, full_q, full_qd, arm_q,
                                torch.zeros_like(arm_q))
        self.goal_pose[env_ids] = samples.goal_pose.to(self.device)
        self.goal_frame[env_ids] = 0
        self.reset_position_difficulty_m[env_ids] = samples.position_difficulty_m.to(self.device)
        self.reset_rotation_difficulty_rad[env_ids] = samples.rotation_difficulty_rad.to(self.device)
        state = self.core_robot.physical_state(self)
        first = self.se3_task.observation(state, self.last_action, self.task_state())
        self.se3_history[env_ids] = first[env_ids, None].expand(-1, 10, -1)
