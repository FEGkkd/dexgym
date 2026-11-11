"""Training-only initialization and regularization from physical teacher rollouts."""

import json

import numpy as np
import torch

from .model import ARM_JOINTS
from .task import JOINTS


class PhysicalTeacher:
    def __init__(self, task, file, model_hash):
        lib = task.library
        if len(lib.start) != 1 or task.action_mode != "home" or not task.first_event or not task.goal_dim:
            raise ValueError("The physical teacher requires single-clip first-event absolute-action tracking")
        self.task = task
        with np.load(file, allow_pickle=False) as data:
            expected = dict(model_sha256=model_hash, motion_sha256=lib.sha256[0], joint_names=lib.names)
            if json.loads(data["metadata"].item()) != expected:
                raise ValueError("Teacher provenance differs from the model or clip")
            self.q = task.tensor(data["qpos"])
            self.qd = task.tensor(data["qvel"])
            targets = task.tensor(data["targets"])
        if (self.q.shape != lib.qpos.shape or self.qd.shape != lib.qvel.shape
                or targets.shape != (lib.frames - 1, JOINTS)
                or not all(torch.isfinite(value).all() for value in (self.q, self.qd, targets))):
            raise ValueError("Invalid teacher trajectories")
        self.targets = torch.cat((targets, targets[-1:]))
        self.history_action = torch.cat((torch.zeros_like(targets[:1]), self.encode(targets)))
        self.history_offsets = torch.arange(-task.history_steps + 1, 1, device=task.device)
        self.sigma = task.tensor([0.018 if name in ARM_JOINTS else 0.030 for name in lib.names])

    def encode(self, targets):
        task = self.task
        offset = targets - task.home
        travel = torch.where(offset >= 0, task.upper_travel, task.lower_travel)
        return (offset / torch.where(travel > 0, travel, 1.0)).clamp(-1, 1)

    def target(self, state, phase):
        q, qd = state.split(JOINTS, dim=-1)
        correction = (0.5 * (self.q[phase] - q) + 0.008 * (self.qd[phase] - qd)).clamp(-0.075, 0.075)
        return (self.targets[phase] + correction).clamp(self.task.lo, self.task.hi)

    def penalty(self, state, memory, executed):
        error = (executed - self.target(state.detach(), memory["frame"])) / self.sigma
        return error.square().mean(-1)

    def batch(self, n):
        task = self.task
        lib = task.library
        phase = torch.randint(lib.frames - 1, (n,), device=task.device)
        age = torch.randint(3 * task.history_steps, (n, 1), device=task.device).clamp_max(task.history_steps)
        step = torch.maximum(self.history_offsets[None], -age)
        ids = (phase[:, None] + step).clamp_min(0)
        # Mix exact reference reset histories and physically executed histories.
        blend = torch.rand(n, 1, 1, device=task.device)
        blend = torch.where(age[:, :, None] == 0, 1.0, blend * 0.35)
        q = torch.lerp(self.q[ids], lib.qpos[ids], blend)
        qd = torch.lerp(self.qd[ids], lib.qvel[ids], blend)
        noise = torch.rand(n, 1, 1, device=task.device)
        q = (q + noise * 0.035 * torch.randn(n, 1, JOINTS, device=task.device)).clamp(task.lo, task.hi)
        qd = qd + noise * 0.25 * torch.randn(n, 1, JOINTS, device=task.device)
        prev = self.history_action[ids]
        prev = torch.where((step == -age)[:, :, None], 0.0, prev)
        prev = (prev + 0.008 * noise * torch.randn_like(prev)).clamp(-1, 1)
        state = torch.cat((q[:, -1], qd[:, -1]), dim=-1)
        memory = {"frame": phase, "action": prev[:, -1]}
        history = torch.cat((q - task.home, qd, prev), dim=-1)
        return task.observation_from_history(history, memory), state, memory

    def initialize(self, actor, steps, batch_size=768):
        with torch.no_grad():
            for _ in range(12):
                actor.norm.update(self.batch(batch_size)[0])
        optimizer = torch.optim.AdamW(actor.parameters(), lr=2e-4, weight_decay=0)
        losses = []
        for step in range(steps):
            obs, state, memory = self.batch(batch_size)
            action = actor(obs)
            _, target = self.task.control_target(action, memory, None)
            loss = self.penalty(state, memory, target).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if (step + 1) in (steps // 2, 4 * steps // 5):
                optimizer.param_groups[0]["lr"] *= 0.35
            if (step + 1) % 250 == 0 or step + 1 == steps:
                losses.append(dict(step=step + 1, target_loss=loss.item()))
                print("ACTOR_INITIALIZATION", json.dumps(losses[-1]), flush=True)
        return losses

    def anchor(self, actor, batch_size=256):
        obs, state, memory = self.batch(batch_size)
        _, target = self.task.control_target(actor(obs), memory, None)
        return self.penalty(state, memory, target).mean()
