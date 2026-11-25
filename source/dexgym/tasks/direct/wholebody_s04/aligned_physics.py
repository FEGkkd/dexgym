"""Batched implementation of the measured normal-gravity CPU servo profile."""

import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw

from dexgym.tasks.direct.marvin54_s0.warp_physics import WarpPhysics


@wp.kernel
def apply_bilateral_servo(
    target: wp.array2d(dtype=wp.float32),
    filtered: wp.array2d(dtype=wp.float32),
    anchors: wp.array2d(dtype=wp.vec3),
    axes: wp.array2d(dtype=wp.vec3),
    center: wp.array2d(dtype=wp.vec3),
    joint: wp.array(dtype=wp.int32),
    body: wp.array(dtype=wp.int32),
    drive: wp.array(dtype=wp.int32),
    mass_over_kp: wp.array(dtype=wp.float32),
    limits: wp.array(dtype=wp.vec2),
    gravity: wp.vec3,
    alpha: wp.float32,
    ctrl: wp.array2d(dtype=wp.float32),
):
    env, i = wp.tid()
    reference = filtered[env, i] + alpha * (target[env, i] - filtered[env, i])
    filtered[env, i] = reference
    lever = center[env, body[i]] - anchors[env, joint[i]]
    offset = -mass_over_kp[i] * wp.dot(axes[env, joint[i]], wp.cross(lever, gravity))
    bounded_reference = wp.clamp(reference, limits[i][0], limits[i][1])
    # The original actuator force cap clamps the sum of feedback and feedforward.
    ctrl[env, drive[i]] = bounded_reference + offset


class AlignedPhysics(WarpPhysics):
    """The policy issues the original qref + residual, then the drive filters it at 500 Hz."""

    def __init__(self, model, names, num_envs, profile, *, graph=True):
        if (profile["delay_ms"] != 0 or profile["velocity_feedforward"] != 0
                or profile["damping_scale"] != 1 or profile["target_filter_ms"] <= 0):
            raise ValueError("Expected the validated filtered drive profile")
        super().__init__(model, names, num_envs, graph=False, count_touching=True)
        arms = tuple(f"Joint{i}_{s}" for s in ("L", "R") for i in range(1, 8))
        joint = np.array([model.joint(name).id for name in arms], dtype=np.int32)
        drive = np.array([model.actuator(f"{name}_position").id for name in arms], dtype=np.int32)
        body = model.jnt_bodyid[joint]
        if not (np.all(model.actuator_forcelimited[drive]) and np.all(model.actuator_ctrllimited[drive])):
            raise ValueError("The arm drive requires its original effort and command limits")
        scale = profile["gravity_scale"] * model.body_subtreemass[body] / model.actuator_gainprm[drive, 0]
        control_limit = model.actuator_ctrllimited.copy()
        control_limit[drive] = False
        self.arm = [names.index(name) for name in arms]
        if not np.array_equal(self.arm, model.jnt_qposadr[joint]):
            raise ValueError("Motor and policy joint orders differ")
        with self.simulation_stream():
            self.sim.actuator_ctrllimited = wp.array(control_limit.astype(bool), dtype=wp.bool)
            target = wp.zeros((num_envs, len(arms)), dtype=wp.float32)
            filtered = wp.zeros_like(target)
            self.arm_target = wp.to_torch(target)
            self.filtered_target = wp.to_torch(filtered)
            d = self.data
            self.servo_args = [
                target, filtered, d.xanchor, d.xaxis, d.subtree_com,
                wp.array(joint, dtype=wp.int32), wp.array(body, dtype=wp.int32), wp.array(drive, dtype=wp.int32),
                wp.array(scale, dtype=wp.float32), wp.array(model.actuator_ctrlrange[drive], dtype=wp.vec2),
                wp.vec3(*map(float, model.opt.gravity)),
                float(-np.expm1(-1000 * model.opt.timestep / profile["target_filter_ms"])), d.ctrl,
            ]
            self.substep()
            if graph:
                with wp.ScopedCapture(stream=self.stream) as captured:
                    self.substep()
                self.graph = captured.graph

    def substep(self):
        mjw.kinematics(self.sim, self.data)
        mjw.com_pos(self.sim, self.data)
        wp.launch(apply_bilateral_servo, (self.num_envs, len(self.arm)), inputs=self.servo_args)
        super().substep()

    def reset(self, ids, q, qd):
        super().reset(ids, q, qd)
        with self.simulation_stream():
            self.arm_target[ids] = q[:, self.arm]
            self.filtered_target[ids] = q[:, self.arm]

    def interval(self, target, task):
        with self.simulation_stream():
            self.arm_target.copy_(target[:, self.arm])
        return super().interval(target, task)
