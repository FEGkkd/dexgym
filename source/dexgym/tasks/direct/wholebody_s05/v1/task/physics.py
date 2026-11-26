"""Native MuJoCo servo and exact physical traces for S0-5 acceptance."""

from dataclasses import dataclass
from hashlib import sha256
import json
from os.path import relpath
from typing import NamedTuple

import mujoco
import numpy as np
import torch

from ...p02.model import RobotModel, asset_files
from ....marvin54_tracking.model import drive_parameters
from . import JOINTS, PHYSICS_DT, SUBSTEPS
from .safety import Stop


@dataclass(frozen=True)
class ServoProfile:
    arm_tau_seconds: float = .016
    gravity_multiplier: float = 1.

    def __post_init__(self):
        if not np.isfinite(self.arm_tau_seconds) or self.arm_tau_seconds <= 0:
            raise ValueError("Arm servo requires a finite positive target time constant")
        if not np.isfinite(self.gravity_multiplier) or self.gravity_multiplier < 0:
            raise ValueError("Gravity compensation requires a nonnegative multiplier")

    @property
    def alpha(self):
        return np.r_[np.full(14, -np.expm1(-PHYSICS_DT / self.arm_tau_seconds)), np.ones(40)]

    def applied_interval(self, start, command, count=SUBSTEPS):
        """Deterministic target history, usable by imagination without safety authority."""
        decay = 1 - start.new_tensor(self.alpha)
        powers = torch.arange(1, count + 1, device=start.device, dtype=start.dtype)
        return command[:, None] + (start - command)[:, None] * decay[None, None].pow(powers[None, :, None])


class Interval(NamedTuple):
    state: torch.Tensor
    servo: torch.Tensor
    applied: torch.Tensor
    ctrl: torch.Tensor
    torque: torch.Tensor
    contact: torch.Tensor
    depth: torch.Tensor
    minimum_gap: torch.Tensor
    substeps: torch.Tensor
    event: torch.Tensor


def signature(robot, profile):
    values = drive_parameters(robot.model, robot.names)
    files = asset_files(robot.path)
    root = robot.path.parent
    relative_files = {
        str(relpath(file, root)).replace("\\", "/"): digest
        for file, digest in files.items()
    }
    identity = {
        "assets": relative_files, "joints": list(robot.names), "filters_s": [profile.arm_tau_seconds] * 14 + [0.] * 40,
        "gravity_multiplier": profile.gravity_multiplier, "opt_gravity": robot.model.opt.gravity.tolist(),
        "qpos_ids": robot.q_indices.tolist(), "qvel_ids": robot.v_indices.tolist(),
        "drive": values, "physics_dt": PHYSICS_DT,
        "control_substeps": SUBSTEPS, "mujoco": mujoco.__version__,
    }
    identity["sha256"] = sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return identity


def context(robot, profile, device):
    values = tuple(drive_parameters(robot.model, robot.names).values())
    flattened = ([profile.gravity_multiplier] + [profile.arm_tau_seconds] * 14 + [0.] * 40
                 + [v["kp"] for v in values] + [v["kd"] for v in values] + [v["effort"] for v in values])
    return torch.tensor(flattened, dtype=torch.float32, device=device)


def contact_labels(robot):
    """Assign contact geometry ownership by named independent finger hinges."""
    m = robot.model
    # 0/1 are left/right, -1 is an unrelated scene or table body.
    side = np.full(m.ngeom, -1, dtype=np.int32)
    finger = np.full(m.ngeom, -1, dtype=np.int32)
    first_joints = [[robot.joints[14 + h * 20 + f * 4].id for f in range(5)] for h in range(2)]
    fingers = {joint: h * 5 + f for h, group in enumerate(first_joints) for f, joint in enumerate(group)}
    arm_by_side = {joint.id: i // 7 for i, joint in enumerate(robot.joints[:14])}
    palms = {m.body(f"{s}_wrist").id: h for h, s in enumerate("lr")}
    for g in range(m.ngeom):
        body = int(m.geom_bodyid[g])
        while body:
            joint = int(m.body_jntadr[body])
            if joint in fingers and finger[g] < 0:
                finger[g] = fingers[joint]
            if body in palms:
                side[g] = palms[body]
            if joint in arm_by_side:
                side[g] = arm_by_side[joint]
                break
            body = int(m.body_parentid[body])
    return side, finger


def _contact_row(data, side, finger):
    row = np.zeros(13, dtype=np.float32)
    depth = 0.
    for hit in data.contact:
        if hit.dist > 0:
            continue
        a, b = int(hit.geom[0]), int(hit.geom[1])
        depth = max(depth, -float(hit.dist))
        for f in (finger[a], finger[b]):
            if f >= 0:
                row[f] = 1.
        if side[a] < 0 or side[b] < 0 or side[a] == side[b]:
            row[10] = 1.
        elif finger[a] >= 0 and finger[b] >= 0:
            row[12] = 1.
        else:
            row[11] = 1.
    return row, depth


class CpuServo:
    """Double precision MuJoCo reference with independently applied command guard."""

    device = "cpu"

    def __init__(self, robot: RobotModel, num_envs: int, profile: ServoProfile):
        # The actuator position target is already bounded by the command guard.
        # Arm gravity-compensation offset is added after that and before the signed force cap.
        self.model = mujoco.MjModel.from_xml_path(str(robot.path))
        self.robot = robot
        self.num_envs = num_envs
        self.profile = profile
        if num_envs < 1 or abs(self.model.opt.timestep - PHYSICS_DT) > 1e-10:
            raise ValueError("Native physics requires a positive batch at 500 Hz")
        if not np.allclose(self.model.opt.gravity, [0., 0., -9.81], atol=1e-9):
            raise ValueError("Only the signed normal-gravity profile is accepted")
        drive = drive_parameters(self.model, robot.names)
        self.actuator_ids = np.asarray([row["actuator_id"] for row in drive.values()])
        self.effort = np.asarray([row["effort"] for row in drive.values()])
        self.arm_joints = np.asarray([joint.id for joint in robot.joints[:14]])
        self.arm_bodies = self.model.jnt_bodyid[self.arm_joints]
        self.mass_over_kp = (profile.gravity_multiplier * self.model.body_subtreemass[self.arm_bodies]
                             / self.model.actuator_gainprm[self.actuator_ids[:14], 0])
        self.arm_ctrl_low, self.arm_ctrl_high = self.model.actuator_ctrlrange[self.actuator_ids[:14]].T.copy()
        self.model.actuator_ctrllimited[self.actuator_ids[:14]] = False
        self.data = tuple(mujoco.MjData(self.model) for _ in range(num_envs))
        self.filtered = np.repeat(robot.home[None], num_envs, axis=0)
        self.geom_side, self.geom_finger = contact_labels(robot)
        self.reset(torch.arange(num_envs), torch.as_tensor(self.filtered), torch.zeros((num_envs, JOINTS)))

    @staticmethod
    def _float(array):
        return torch.as_tensor(np.asarray(array), dtype=torch.float32)

    def state(self):
        return self._float([
            np.r_[d.qpos[self.robot.q_indices], d.qvel[self.robot.v_indices]] for d in self.data
        ])

    def servo(self):
        return self._float(self.filtered.copy())

    def reset(self, ids, q, qd):
        if q.shape != qd.shape or q.shape != (ids.numel(), JOINTS):
            raise ValueError("Expected B x 54 reset position and velocity")
        if not torch.isfinite(q).all() or not torch.isfinite(qd).all():
            raise ValueError("Reset state must be finite")
        for i, position, velocity in zip(ids.tolist(), q.detach().numpy(), qd.detach().numpy(), strict=True):
            d = self.data[i]
            mujoco.mj_resetData(self.model, d)
            d.qpos[self.robot.q_indices] = position
            d.qvel[self.robot.v_indices] = velocity
            d.ctrl[self.actuator_ids] = position
            self.filtered[i] = position
            mujoco.mj_forward(self.model, d)

    def interval(self, command, allowed, guard, *, count=SUBSTEPS):
        if not 1 <= count <= SUBSTEPS:
            raise ValueError("A physical interval is at most ten MuJoCo substeps")
        m = self.model
        targets = np.zeros((self.num_envs, count, JOINTS), dtype=np.float32)
        controls = np.zeros_like(targets)
        torques = np.zeros_like(targets)
        contacts = np.zeros((self.num_envs, 13), dtype=np.float32)
        completed = np.zeros(self.num_envs, dtype=np.int32)
        events = np.zeros_like(completed)
        peaks = np.zeros(self.num_envs, dtype=np.float32)
        minimum = np.full(self.num_envs, .005, dtype=np.float32)
        alpha = self.profile.alpha
        for i, (d, target, run) in enumerate(zip(self.data, command.detach().numpy(), allowed.tolist(), strict=True)):
            if not run:
                continue
            for j in range(count):
                before = d.qpos[self.robot.q_indices].copy()
                time = d.time
                self.filtered[i] += alpha * (target - self.filtered[i])
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                lever = d.subtree_com[self.arm_bodies] - d.xanchor[self.arm_joints]
                gravity = -self.mass_over_kp * np.sum(d.xaxis[self.arm_joints] * np.cross(lever, m.opt.gravity), axis=-1)
                d.ctrl[self.actuator_ids] = self.filtered[i]
                arm_reference = np.clip(self.filtered[i, :14], self.arm_ctrl_low, self.arm_ctrl_high)
                d.ctrl[self.actuator_ids[:14]] = arm_reference + gravity
                targets[i, j] = self.filtered[i]
                controls[i, j] = d.ctrl[self.actuator_ids]
                mujoco.mj_step(m, d)
                torques[i, j] = d.qfrc_actuator[self.robot.v_indices]
                mujoco.mj_fwdPosition(m, d)
                labels, depth = _contact_row(d, self.geom_side, self.geom_finger)
                contacts[i] = np.maximum(contacts[i], labels)
                peaks[i] = max(peaks[i], depth)
                minimum[i] = min(minimum[i], guard.sweep.minimum_gap(d))
                completed[i] += 1
                actual = d.qpos[self.robot.q_indices]
                velocity = d.qvel[self.robot.v_indices]
                if not (np.isfinite(actual).all() and np.isfinite(velocity).all() and np.isfinite(torques[i, j]).all()):
                    events[i] |= Stop.BAD_FEEDBACK
                    break
                measured = self._float(actual[None])
                if (depth > 0 or not guard.positions_valid(measured)[0]
                        or not guard.velocities_valid(self._float(velocity[None]))[0]
                        or not guard.sweep.permits(before, actual)):
                    events[i] |= Stop.PHYSICAL_CONSTRAINT
                if np.any(np.abs(torques[i, j]) > self.effort + 1e-4):
                    events[i] |= Stop.PHYSICAL_CONSTRAINT
                if abs(d.time - time - PHYSICS_DT) > 1e-9:
                    events[i] |= Stop.TIMING
                if events[i]:
                    break
        return Interval(self.state(), self.servo(), self._float(targets), self._float(controls), self._float(torques),
                        self._float(contacts), self._float(peaks), self._float(minimum),
                        torch.from_numpy(completed), torch.from_numpy(events))
