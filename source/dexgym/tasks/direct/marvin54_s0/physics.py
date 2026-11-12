"""Shared transition contract for Isaac Lab Newton and native MuJoCo."""

from contextlib import contextmanager

import mujoco
import numpy as np
import torch

from ..marvin54_tracking.model import drive_parameters
from ..marvin54_tracking.task import DT, HARD_CONTACT_DEPTH_M, JOINT_OVERFLOW_RAD, JOINTS


PHYSICS_DT = 0.002
SUBSTEPS = round(DT / PHYSICS_DT)
AUDIT_DEPTH_M = 0.0005
COLLISION, JOINT_LIMIT, NONFINITE, TRACKING, OVERFLOW = 1, 2, 4, 8, 16
FAILURES = {COLLISION: "collision", JOINT_LIMIT: "joint_limit", NONFINITE: "nonfinite",
            TRACKING: "tracking", OVERFLOW: "buffer_overflow"}


class Physics:
    def __init__(self, model, names, num_envs, device):
        self.model, self.num_envs, self.device = model, num_envs, device
        if not np.isclose(model.opt.timestep, PHYSICS_DT):
            raise ValueError("The 50 Hz action contract requires 2 ms physics steps")
        if (model.nq, model.nv, model.nu) != (JOINTS, JOINTS, JOINTS):
            raise ValueError("Expected the fixed-base 54 DoF model")
        self.actuator_ids = [v["actuator_id"] for v in drive_parameters(model, names).values()]

    @staticmethod
    def flags(task, state, depth):
        invalid = ~torch.isfinite(state).all(-1) | ~torch.isfinite(depth)
        limit = task.position_violation(state) > JOINT_OVERFLOW_RAD
        return (COLLISION * (depth > HARD_CONTACT_DEPTH_M).int()
                + JOINT_LIMIT * limit.int() + NONFINITE * invalid.int())


class CpuPhysics(Physics):
    def __init__(self, model, names, num_envs):
        super().__init__(model, names, num_envs, "cpu")
        self.data = [mujoco.MjData(model) for _ in range(num_envs)]

    def state(self):
        return torch.as_tensor(np.stack([np.r_[d.qpos, d.qvel] for d in self.data]), dtype=torch.float32)

    def reset(self, ids, q, qd):
        for i, pos, vel in zip(ids.tolist(), q.numpy(), qd.numpy(), strict=True):
            d = self.data[i]
            mujoco.mj_resetData(self.model, d)
            d.qpos[:], d.qvel[:] = pos, vel
            d.ctrl[self.actuator_ids] = pos
            mujoco.mj_forward(self.model, d)

    def interval(self, target, task):
        m = self.model
        peak = torch.zeros(self.num_envs)
        deep_substeps = torch.zeros(self.num_envs, dtype=torch.int32)
        substeps = torch.zeros(self.num_envs, dtype=torch.int32)
        flags = torch.zeros_like(substeps)
        torque = []
        for i, d in enumerate(self.data):
            d.ctrl[self.actuator_ids] = target[i].numpy()
            for j in range(1, SUBSTEPS + 1):
                previous_time = d.time
                mujoco.mj_step(m, d)
                mujoco.mj_fwdPosition(m, d)
                depth = max(0.0, -float(np.min(d.contact.dist, initial=0.0)))
                peak[i] = max(peak[i], depth)
                deep_substeps[i] += depth > AUDIT_DEPTH_M
                state = torch.as_tensor(np.r_[d.qpos, d.qvel], dtype=torch.float32)[None]
                event = int(self.flags(task, state, peak[i:i + 1]))
                if d.time <= previous_time:
                    event |= NONFINITE
                if event:
                    flags[i], substeps[i] = event, j
                    break
            torque.append(d.qfrc_actuator.copy())
        return self.state(), torch.as_tensor(np.stack(torque), dtype=torch.float32), peak, substeps, flags, deep_substeps


BACKEND_NAMES = {
    "cpu": "native MuJoCo",
    "newton": "Isaac Lab 3 / Newton MJWarp",
    "warp": "MuJoCo Warp",
}


@contextmanager
def open_physics(backend, model, names, num_envs, *, graph=True):
    if backend == "cpu":
        yield CpuPhysics(model, names, num_envs)
    elif backend == "newton":
        from isaaclab.app import launch_simulation

        from ..marvin54_tracking.native import simulation_cfg
        from .newton_physics import NewtonPhysics

        cfg = simulation_cfg()
        with launch_simulation(cfg):
            yield NewtonPhysics(cfg, model, names, num_envs)
    elif backend == "warp":
        from .warp_physics import WarpPhysics

        yield WarpPhysics(model, names, num_envs, graph=graph)
    else:
        raise ValueError(f"Unknown simulator backend {backend}")
