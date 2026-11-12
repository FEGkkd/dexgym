"""Shared S0 task transition contract for GPU collection and native evaluation."""

from types import SimpleNamespace

import torch

from ..marvin54_tracking.task import JOINTS
from .physics import COLLISION, JOINT_LIMIT, NONFINITE, OVERFLOW, SUBSTEPS, TRACKING
from .task import HandCentricTask, TRACKING_LIMIT_M


class HandCentricEnv:
    def __init__(self, backend, library, *, reset_noise=True):
        self.backend = backend
        self.num_envs, self.device = backend.num_envs, backend.device
        self.task = HandCentricTask(self, backend.model, library)
        self.reset_noise = reset_noise
        self.history = torch.zeros(
            self.num_envs, self.task.history_steps, self.task.history_frame_dim, device=self.device
        )
        self._state = backend.state()
        self.memory = {
            "frame": torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
            "action": torch.zeros(self.num_envs, JOINTS, device=self.device),
            "q": self._state[:, :JOINTS],
            "tip": torch.zeros(self.num_envs, 30, device=self.device),
        }
        self.transition = None
        self.common_step_counter = 0
        self.disturbance = SimpleNamespace(adaptive_scale=0.0, success_scale=0.0)

    def state(self):
        return self._state

    def observation(self):
        return {"policy": self.task.observation_from_history(self.history, self.memory)}

    def reset(self, env_ids=None, frame=None, *, q=None, qd=None):
        task, lib = self.task, self.task.library
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        sample = frame is None
        frame = lib.sample(len(env_ids)) if sample else frame
        if q is None:
            q = lib.qpos[frame]
        if qd is None:
            qd = lib.qvel[frame]
        if sample and self.reset_noise:
            strength = min(1.0, self.common_step_counter / 6000)
            q = (q + 0.45 * strength * task.action_step * torch.randn_like(q)).clamp(task.lo, task.hi)
            qd = qd + 0.12 * strength * torch.randn_like(qd)
        self.backend.reset(env_ids, q, qd)
        self._state = self._state.index_copy(0, env_ids, torch.cat((q, qd), -1))
        self.memory = {key: value.clone() for key, value in self.memory.items()}
        self.memory["frame"][env_ids] = frame
        self.memory["action"][env_ids] = 0
        self.memory["q"][env_ids] = q
        self.memory["tip"][env_ids] = task.geometry.hand_pose(q)["tip"].flatten(1)
        obs = task.policy_observation(self._state, self.memory, None)
        self.history[env_ids] = obs[env_ids, None]
        return self.observation(), {}

    def step(self, action, *, target_override=None):
        task, before = self.task, self.memory
        _, target = task.control_target(action, before, None)
        if target_override is not None:
            target = target_override.clamp(task.lo, task.hi)
            action = ((target - before["q"]) / task.action_step).clamp(-1, 1)
        x, torque, depth, first, flags = self.backend.interval(target, task)
        self._state = x
        safety = flags != 0
        invalid = (flags & (NONFINITE | OVERFLOW)) != 0
        full_step = ((first == 0) | (first == SUBSTEPS)) & ~invalid
        clean_x = x.nan_to_num(nan=0., posinf=0., neginf=0.)
        pose = task.geometry.hand_pose(clean_x[:, :JOINTS])
        tracking = task.tracking_error(pose, before) > TRACKING_LIMIT_M
        flags = flags | TRACKING * (tracking & ~safety).int()
        contact = task.contact_labels(depth)
        reward, tip, _ = task.reward(clean_x, before, action, target, contact, torque, pose=pose)
        reward = torch.where(invalid, 0., reward)
        self.memory = task.advance_memory(clean_x, before, action, target, tip, None, soft=False)
        obs = task.policy_observation(clean_x, self.memory, None)
        self.history = torch.cat((self.history[:, 1:], obs[:, None]), 1)
        frame = self.memory["frame"]
        complete = (frame == task.library.end[frame]) & (flags == 0)
        terminated = (flags != 0) | complete
        events = dict(failure_bits=flags, safety_failure=safety, safety_substep=first, reference_complete=complete,
                      full_step=full_step)
        self.transition = dict(
            x1=x, contact=contact, torque=torque, drop=(flags & (COLLISION | JOINT_LIMIT | OVERFLOW)) != 0,
            nan=invalid, c=target, u=target, depth=depth, **events
        )
        self.common_step_counter += 1
        return self.observation(), reward, terminated, torch.zeros_like(terminated), events
