"""Newton environment for continuous dual-TCP SE(3) pose following."""

from types import SimpleNamespace

import torch
from isaaclab.utils import configclass

from dexgym.assets.marvin_wuji.controller import ARM_JOINT_NAMES
from dexgym.assets.supermimic import PROCEDURAL_REFERENCE

from .base_env import MarvinArmNewtonBaseEnv
from .core_plugins import MarvinCoreRobot
from .env_cfg import MarvinArmNewtonBaseCfg
from .se3_trajectory import MarvinContinuousSE3Follow, SE3TCPReferenceProvider


@configclass
class MarvinContinuousSE3FollowCfg(MarvinArmNewtonBaseCfg):
    action_schema = "arm_joint_delta_v1"
    observation_space = 10 * 68 + 10 * 18
    episode_length_s = 4.0
    reference_path = str(PROCEDURAL_REFERENCE)
    include_reverse = True
    random_reference_start = True
    max_reference_start_frame = 100
    reset_joint_perturb_rad = 0.005
    evaluation_mode = False
    evaluation_motion_id = None
    audit_contacts_each_physics_step = True


class MarvinContinuousSE3FollowEnv(MarvinArmNewtonBaseEnv):
    """The task owns time, TCP targets, reward and reset; BaseEnv owns physics."""

    cfg: MarvinContinuousSE3FollowCfg

    def __init__(self, cfg: MarvinContinuousSE3FollowCfg, render_mode=None, **kwargs):
        if cfg.action_schema != "arm_joint_delta_v1" or cfg.episode_length_s != 4.0:
            raise ValueError("Continuous SE(3) following needs delta actions and 200 control steps")
        super().__init__(cfg, render_mode, **kwargs)
        source = SimpleNamespace(joint_names_arm=ARM_JOINT_NAMES)
        self.core_robot = MarvinCoreRobot(source, self.device, action_schema=cfg.action_schema)
        self._setup_arm_physics(source, self.core_robot.action_map.origin)
        self.reference = SE3TCPReferenceProvider(
            {"procedural_bimanual_reach_v1": cfg.reference_path}, self.device,
            include_reverse=cfg.include_reverse)
        self.follow_task = MarvinContinuousSE3Follow(self.reference, self.core_robot)
        if (cfg.max_reference_start_frame < 0
                or cfg.max_reference_start_frame + 32 >= int(self.reference.lengths.min())):
            raise ValueError("Reference reset start must leave a 32-step imagination window")
        if cfg.evaluation_motion_id is not None and not 0 <= cfg.evaluation_motion_id < len(self.reference.names):
            raise ValueError("Evaluation motion ID is outside the reference library")
        self.reference_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reference_frame = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.follow_history = torch.zeros((self.num_envs, 10, 68), device=self.device)
        self.previous_qd = torch.zeros((self.num_envs, 14), device=self.device)

    def task_state(self):
        return self.follow_task.task_state(self.reference_id.clone(), self.reference_frame.clone())

    def _pre_physics_step(self, actions):
        self.previous_qd = self.robot.data.joint_vel.torch[:, self.arm_ids].clone()
        super()._pre_physics_step(actions)
        self.reference_frame.add_(1)

    def _get_observations(self):
        state = self.core_robot.physical_state(self)
        task_state = self.task_state()
        observation = self.follow_task.observation(state, self.last_action, task_state)
        self.follow_history = torch.cat((self.follow_history[:, 1:], observation[:, None]), dim=1)
        goal = self.follow_task.goal(state, task_state)
        return {"policy": torch.cat((self.follow_history.flatten(1), goal.flatten(1)), dim=-1)}

    def _get_rewards(self):
        state = self.core_robot.physical_state(self)
        previous_state = torch.cat((self.previous_q, self.previous_qd,
                                    self.previous_command), dim=-1)
        self.last_physical_q = self.robot.data.joint_pos.torch.detach().clone()
        self.last_physical_qd = self.robot.data.joint_vel.torch.detach().clone()
        self.last_physical_torque = self.robot.data.applied_torque.torch.detach().clone()
        return self.follow_task.reward(previous_state, state, self.last_action,
                                       self.task_state(), self.previous_action)

    def _get_dones(self):
        self._record_contacts()
        state = self.core_robot.physical_state(self)
        self.last_physical_arm_state = state.detach().clone()
        self.last_policy_contact_counts = self._contact_accum.clone()
        self.last_policy_contact_physics_steps = int(self._contact_accum.sum())
        thresholds = self.follow_task.safety_thresholds
        q, qd = state[:, :14], state[:, 14:28]
        self.last_failure_reason = (
            (((q < self.action_map.lower - thresholds.joint_limit_margin_rad)
               | (q > self.action_map.upper + thresholds.joint_limit_margin_rad)).any(-1).int() * 2)
            | ((qd.abs() > thresholds.speed_rad_s).any(-1).int() * 4)
            | (self.last_policy_contact_counts.gt(0).int() * 8)
            | ((~torch.isfinite(q).all(-1) | ~torch.isfinite(qd).all(-1)).int() * 16))
        bad = self.follow_task.hard_termination(state, self.task_state(),
                                                 self.last_policy_contact_counts)
        timeout = ((self.episode_length_buf >= self.max_episode_length)
                   | (self.reference_frame >= self.reference.lengths[self.reference_id] - 1))
        return bad, timeout

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # DirectRLEnv calls virtual reset while its own constructor is running.
        if not hasattr(self, "reference"):
            return
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        count = len(env_ids)
        if self.cfg.evaluation_mode:
            motion_id = (torch.arange(count, device=self.device) % len(self.reference.names)
                         if self.cfg.evaluation_motion_id is None else
                         torch.full((count,), self.cfg.evaluation_motion_id,
                                    device=self.device, dtype=torch.long))
            frame = torch.zeros(count, device=self.device, dtype=torch.long)
            perturb = self.cfg.reset_joint_perturb_rad
        else:
            motion_id = torch.randint(len(self.reference.names), (count,), device=self.device)
            frame = (torch.randint(self.cfg.max_reference_start_frame + 1,
                                   (count,), device=self.device)
                     if self.cfg.random_reference_start else
                     torch.zeros(count, device=self.device, dtype=torch.long))
            perturb = self.cfg.reset_joint_perturb_rad
        arm_q = self.reference.reset_q_at(motion_id, frame).clone()
        if perturb:
            arm_q += (torch.rand_like(arm_q) * 2 - 1) * perturb
            arm_q = arm_q.clamp(self.action_map.lower + 0.01,
                                self.action_map.upper - 0.01)
        full_q = self.home_full[None].repeat(count, 1)
        full_qd = torch.zeros_like(full_q)
        self._write_robot_reset(env_ids, full_q, full_qd, arm_q, torch.zeros_like(arm_q))
        self.reference_id[env_ids] = motion_id
        self.reference_frame[env_ids] = frame
        state = self.core_robot.physical_state(self)
        first = self.follow_task.observation(state, self.last_action, self.task_state())
        self.follow_history[env_ids] = first[env_ids, None].expand(-1, 10, -1)
