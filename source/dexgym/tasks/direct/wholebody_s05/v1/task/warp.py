"""MuJoCo Warp dynamics with the same guarded servo and canonical trace as native."""

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from ....marvin54_s0.warp_physics import WarpPhysics
from ....marvin54_tracking.model import drive_parameters
from ....wholebody_s04.aligned_physics import apply_bilateral_servo
from . import JOINTS, PHYSICS_DT, SUBSTEPS
from .physics import Interval, contact_labels
from .safety import Stop


@wp.kernel
def _label_contacts(
    count: wp.array(dtype=wp.int32), world: wp.array(dtype=wp.int32),
    distances: wp.array(dtype=wp.float32), geometries: wp.array(dtype=wp.vec2i),
    side: wp.array(dtype=wp.int32), finger: wp.array(dtype=wp.int32),
    marks: wp.array2d(dtype=wp.int32),
):
    i = wp.tid()
    if i < count[0] and distances[i] <= 0.0:
        env = world[i]
        g = geometries[i]
        a, b = g[0], g[1]
        if env >= 0 and env < marks.shape[0] and a >= 0 and b >= 0:
            fa, fb = finger[a], finger[b]
            if fa >= 0:
                wp.atomic_max(marks, env, fa, 1)
            if fb >= 0:
                wp.atomic_max(marks, env, fb, 1)
            sa, sb = side[a], side[b]
            if sa < 0 or sb < 0 or sa == sb:
                wp.atomic_max(marks, env, 10, 1)
            elif fa >= 0 and fb >= 0:
                wp.atomic_max(marks, env, 12, 1)
            else:
                wp.atomic_max(marks, env, 11, 1)


class WarpServo(WarpPhysics):
    """Actual parallel Warp physics, initially certified by the native swept check.

    The native check is an explicit bottleneck in this qualification backend.
    Scaling it requires a replacement that independently preserves its envelope.
    """

    def __init__(self, robot, num_envs, profile, *, device="cuda:0"):
        model = robot.model
        if not np.allclose(model.opt.gravity, [0., 0., -9.81], atol=1e-9):
            raise ValueError("Only the signed normal-gravity profile is accepted")
        super().__init__(model, robot.names, num_envs, device=device, graph=False, count_touching=False)
        self.robot = robot
        self.profile = profile
        self.pos_id = torch.as_tensor(robot.q_indices, device=device)
        self.vel_id = torch.as_tensor(robot.v_indices, device=device)
        drive = drive_parameters(model, robot.names)
        self.effort = torch.tensor([row["effort"] for row in drive.values()], device=device)
        ids = np.asarray(self.actuator_ids[:14], dtype=np.int32)
        joint = np.asarray([joint.id for joint in robot.joints[:14]], dtype=np.int32)
        body = model.jnt_bodyid[joint]
        mass_over_kp = profile.gravity_multiplier * model.body_subtreemass[body] / model.actuator_gainprm[ids, 0]
        limits = model.actuator_ctrllimited.copy()
        limits[ids] = False
        labels = contact_labels(robot)
        with self.simulation_stream():
            self.sim.actuator_ctrllimited = wp.array(limits.astype(bool), dtype=wp.bool)
            target = wp.zeros((num_envs, 14), dtype=wp.float32)
            filtered = wp.zeros_like(target)
            self.arm_target = wp.to_torch(target)
            self.filtered_arm = wp.to_torch(filtered)
            self.filtered_hand = torch.zeros((num_envs, 40), device=device)
            self.contact_marks = wp.zeros((num_envs, 13), dtype=wp.int32)
            self.mark_tensor = wp.to_torch(self.contact_marks)
            d = self.data
            self.contact_args = [d.nacon, d.contact.worldid, d.contact.dist, d.contact.geom,
                                 *(wp.array(a, dtype=wp.int32) for a in labels), self.contact_marks]
            self.servo_args = [target, filtered, d.xanchor, d.xaxis, d.subtree_com,
                               wp.array(joint, dtype=wp.int32), wp.array(body, dtype=wp.int32),
                               wp.array(ids, dtype=wp.int32), wp.array(mass_over_kp, dtype=wp.float32),
                               wp.array(model.actuator_ctrlrange[ids], dtype=wp.vec2),
                               wp.vec3(*model.opt.gravity), float(profile.alpha[0]), d.ctrl]
        home = torch.tensor(robot.home, dtype=torch.float32, device=device)[None].repeat(num_envs, 1)
        self.reset(torch.arange(num_envs, device=device), home, torch.zeros_like(home))

    def state(self):
        return torch.cat((self.qpos[:, self.pos_id], self.qvel[:, self.vel_id]), dim=-1)

    def servo(self):
        return torch.cat((self.filtered_arm, self.filtered_hand), dim=-1)

    def reset(self, ids, q, qd):
        qp = torch.empty_like(q)
        qv = torch.empty_like(qd)
        qp[:, self.pos_id] = q
        qv[:, self.vel_id] = qd
        # WarpPhysics.reset expects q arrays in raw model order; ctrl is canonical below.
        with self.simulation_stream():
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[ids] = True
            mjw.reset_data(self.sim, self.data, reset=wp.from_torch(mask, dtype=wp.bool))
            self.qpos[ids], self.qvel[ids] = qp, qv
            self.ctrl[ids[:, None], self.actuator_ids] = q
            self.arm_target[ids] = q[:, :14]
            self.filtered_arm[ids] = q[:, :14]
            self.filtered_hand[ids] = q[:, 14:]

    def interval(self, command, allowed, guard, *, count=SUBSTEPS):
        if not 1 <= count <= SUBSTEPS:
            raise ValueError("A physical interval is at most ten MuJoCo substeps")
        shape = (self.num_envs, count, JOINTS)
        targets = command.new_zeros(shape)
        controls = command.new_zeros(shape)
        torques = command.new_zeros(shape)
        contacts = command.new_zeros((self.num_envs, 13))
        peaks = command.new_zeros(self.num_envs)
        gap = command.new_full((self.num_envs,), .005)
        completed = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        events = torch.zeros_like(completed)
        active = allowed.clone()
        state = self.state()
        for j in range(count):
            if not active.any():
                break
            initial_servo = self.servo()
            with self.simulation_stream() as caller:
                self.arm_target.copy_(torch.where(active[:, None], command[:, :14], initial_servo[:, :14]))
                self.filtered_hand.copy_(torch.where(active[:, None], command[:, 14:], initial_servo[:, 14:]))
                self.ctrl[:, self.actuator_ids[14:]] = self.filtered_hand
                d = self.data
                mjw.kinematics(self.sim, d)
                mjw.com_pos(self.sim, d)
                wp.launch(apply_bilateral_servo, (self.num_envs, 14), inputs=self.servo_args)
                self.substep()
                self.contact_marks.zero_()
                wp.launch(_label_contacts, d.contact.dist.shape[0], inputs=self.contact_args)
                post = self.state()
                step_servo = self.servo()
                ctrl = self.ctrl[:, self.actuator_ids]
                torque = self.force[:, self.vel_id]
                depth = self.depth_torch.clone()
                overflow = (self.overflow != 0).clone()
                if overflow.ndim != 1:
                    overflow = overflow.flatten(1).any(-1)
                marks = self.mark_tensor.float()
                # A stopped replica must never leak a later inactive substep into the trace.
                self.qpos[:, self.pos_id] = torch.where(active[:, None], post[:, :JOINTS], state[:, :JOINTS])
                self.qvel[:, self.vel_id] = torch.where(active[:, None], post[:, JOINTS:], state[:, JOINTS:])
                for value in (post, step_servo, ctrl, torque, depth, overflow, marks):
                    value.record_stream(caller)
            actual = post[:, :JOINTS]
            finite = torch.isfinite(post).all(-1) & torch.isfinite(torque).all(-1)
            valid = guard.positions_valid(actual) & guard.velocities_valid(post[:, JOINTS:])
            check = active & valid & (depth == 0.) & ~overflow
            cleared = guard.permitted_paths(state[:, :JOINTS], actual, check)
            # The measured signed clearance remains useful at actual contacts
            # and stops, independently of whether the motion remains permitted.
            subgap = guard.minimum_gaps(actual, active & torch.isfinite(actual).all(-1))
            safe_force = (torque.abs() <= self.effort + 1e-4).all(-1)
            event = ((~cleared | ~safe_force).int() * Stop.PHYSICAL_CONSTRAINT
                     | (~finite).int() * Stop.BAD_FEEDBACK | overflow.int() * Stop.OVERFLOW)
            completed += active.int()
            events = torch.where(active, event, events)
            step_mask = active[:, None]
            state = torch.where(step_mask, post, state)
            targets[:, j] = torch.where(step_mask, step_servo, 0.)
            controls[:, j] = torch.where(step_mask, ctrl, 0.)
            torques[:, j] = torch.where(step_mask, torque, 0.)
            contacts = torch.where(step_mask, torch.maximum(contacts, marks), contacts)
            peaks = torch.where(active, torch.maximum(peaks, depth), peaks)
            gap = torch.where(active, torch.minimum(gap, subgap), gap)
            active = active & (event == 0)
        return Interval(state, self.servo(), targets, controls, torques, contacts, peaks, gap, completed, events)
