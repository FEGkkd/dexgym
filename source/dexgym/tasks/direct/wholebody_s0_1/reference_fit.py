"""Native FK, damped bimanual retargeting, and bounded servo checks."""

import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.data.egopro_handcentric import skew

from ..wholebody_s0.robot import FINGER_SITES, FIRMWARE_FINGERS
from ..wholebody_s0.sync import runs
from . import DOF, DT
from .reference_timing import RATE_TOLERANCE, local_clock, position_rates
from .targets import actuator_ids, command_bounds, joint_names
from .wrist_calibration import WRIST_BODIES

CONTACT_STOP = .003
REFERENCE_CLEARANCE = .0015


def rotation_error(target, measured):
    shape = target.shape[:-2]
    rotation = target @ measured.swapaxes(-1, -2)
    return Rotation.from_matrix(rotation.reshape(-1, 3, 3)).as_rotvec().reshape(*shape, 3)


class ReferenceRobot:
    def __init__(self, path, home, rates=None):
        self.model = m = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(m)
        self.scratch_jac = np.zeros((3, DOF))
        self.names = joint_names(m)
        self.actuators = actuator_ids(m, self.names)
        self.arm_ids = [np.array([self.names.index(f"Joint{i}_{side}") for i in range(1, 8)]) for side in "LR"]
        self.arm = np.array([name.startswith("Joint") for name in self.names])
        self.lo, self.hi = command_bounds(m, self.names)
        self.home = np.asarray(home, dtype=float)
        self.wrists = np.array([m.body(name).id for name in WRIST_BODIES])
        self.shoulders = np.array([m.body(f"Link2_{side}").id for side in "LR"])
        self.elbows = np.array([m.body(f"Link4_{side}").id for side in "LR"])
        self.tips = np.array([[m.site(f"{side}_{finger}").id for finger in FINGER_SITES] for side in "lr"])
        self.anchors = np.array([[[int(m.actuator(f"{side}_{name}{j}").trnid[0]) for j in (2, 3)]
                                  for name in FIRMWARE_FINGERS] for side in "lr"])
        self.slew = position_rates(self.names, rates) * DT
        # IK searches the posture offline. Its numerical trust region is independent of the command envelope.
        self.look_ahead = position_rates(self.names) * DT * 4
        self._identity = np.eye(DOF)
        self.native_margin = m.geom_margin.copy(), m.pair_margin.copy()

    def proximity(self, enabled):
        geom, pair = self.native_margin
        extra = REFERENCE_CLEARANCE + .001 if enabled else 0.
        self.model.geom_margin[:] = geom + extra
        self.model.pair_margin[:] = pair + extra

    def forward(self, q, *, collision=False, jacobian=False):
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        if jacobian:
            mujoco.mj_comPos(self.model, self.data)
        if collision:
            mujoco.mj_collision(self.model, self.data)

    def body_pose(self, ids):
        return self.data.xpos[ids], self.data.xmat[ids].reshape(*np.shape(ids), 3, 3)

    def measured(self):
        wrist, rotation = self.body_pose(self.wrists)
        tips = self.data.site_xpos[self.tips]
        world = np.concatenate((self.data.xanchor[self.anchors], tips[..., None, :]), axis=-2)
        hand = np.einsum("sji,sfkj->sfki", rotation, world - wrist[:, None, None])
        return wrist, rotation, hand

    def depth(self):
        return max(0., -float(self.data.contact.dist.min(initial=0.)))

    def trajectory(self, q):
        wrist, rotation, hand, depth = [], [], [], []
        for angles in q:
            self.forward(angles, collision=True)
            p, r, h = self.measured()
            wrist.append(p)
            rotation.append(r)
            hand.append(h)
            depth.append(self.depth())
        return np.stack(wrist), np.stack(rotation), np.stack(hand), np.asarray(depth)

    def fit_reference_rates(self, q, goal, fit, admissible, thresholds):
        """Reuse the target fit with a speed-constrained joint search on the source clock."""
        wrist, rotation, hand, _ = goal
        measured_wrist, measured_rot, measured_hand, depth = fit
        result = q.copy()
        position_budget = np.minimum(thresholds["wrist_m"],
                                     np.linalg.norm(measured_wrist - wrist, axis=-1).max(-1) + .006)
        angle_budget = np.minimum(np.deg2rad(thresholds["wrist_deg"]),
                                  np.linalg.norm(rotation_error(rotation, measured_rot), axis=-1).max(-1)
                                  + np.deg2rad(3.))
        shape_error = np.linalg.norm(measured_hand - hand, axis=-1)
        shape_budget = np.minimum(thresholds["hand_mean_m"], shape_error.mean((1, 2, 3)) + .0015)
        tip_budget = shape_error[..., -1] + .002
        posture_budget = np.where(self.arm, .18, .20)
        step = np.minimum(self.slew * (1. - 1e-6), self.look_ahead)

        frames = list(zip(*goal, strict=True))
        self.proximity(True)
        for a, b in runs(admissible):
            samples = q[a:b]
            if not (np.abs(np.diff(samples, axis=0)) > self.slew * (1. + RATE_TOLERANCE)).any():
                continue
            best_cost = self.retime_cost(samples)
            for order in (range(a + 1, b), range(b - 2, a - 1, -1)):
                candidate = samples.copy()
                for k in order:
                    last = candidate[k - a - order.step]
                    if (np.abs(q[k] - last) <= step).all():
                        continue
                    clipped = np.clip(q[k], last - step, last + step)
                    if (np.abs(clipped - q[k]) > posture_budget).any():
                        continue
                    refined = self.solve_reference(*frames[k], last, 2, seed=clipped, step_limit=step)
                    if (np.abs(refined - q[k]) > posture_budget).any():
                        continue
                    self.forward(refined, collision=True)
                    if self.depth() > max(.0004, depth[k]):
                        continue
                    p, r, h = self.measured()
                    if np.linalg.norm(p - wrist[k], axis=-1).max() > position_budget[k]:
                        continue
                    if np.linalg.norm(rotation_error(rotation[k], r), axis=-1).max() > angle_budget[k]:
                        continue
                    finger_error = np.linalg.norm(h - hand[k], axis=-1)
                    if finger_error.mean() <= shape_budget[k] and (finger_error[..., -1] <= tip_budget[k]).all():
                        candidate[k - a] = refined
                candidate = self.merge_rate_fit(samples, candidate, posture_budget)
                candidate_cost = self.retime_cost(candidate)
                if candidate_cost < best_cost:
                    result[a:b] = candidate
                    best_cost = candidate_cost
        self.proximity(False)
        return result

    def retime_cost(self, samples):
        ratio = (np.abs(np.diff(samples, axis=0)) / self.slew).max(-1)
        _, timing = local_clock(ratio, slow=True)
        return timing["added_intervals"], timing["retimed_original_intervals"]

    def merge_rate_fit(self, original, fitted, max_step):
        """Keep locally useful refits when an unconstrained fallback breaks continuity."""
        changed = (np.abs(fitted - original) > 1e-8).any(-1)
        merged = original.copy()
        for a, b in runs(changed):
            start, end = max(0, a - 2), min(len(original), b + 2)
            # Bound the joins to the same offline search region already used by the original fit.
            previous = merged[start:end].copy()
            merged[a:b] = fitted[a:b]
            steps = np.abs(np.diff(merged[start:end], axis=0))
            old_steps = np.abs(np.diff(previous, axis=0))
            if (steps > np.maximum(old_steps, max_step) + 1e-7).any() or (
                    self.retime_cost(merged[start:end]) >= self.retime_cost(previous)):
                merged[a:b] = original[a:b]
        return merged

    def body_jacobian(self, body):
        p, r = np.zeros((2, 3, DOF))
        mujoco.mj_jacBody(self.model, self.data, p, r, int(body))
        return p, r

    def solve_legacy_arm(self, q, wrist_shift, iterations=4):
        """Preserve inherited wrist rotation while applying a small world displacement."""
        self.forward(q)
        desired, rotation = self.body_pose(self.wrists)
        desired += wrist_shift
        result = q.copy()
        for _ in range(iterations):
            self.forward(result, jacobian=True)
            position, orientation = self.body_pose(self.wrists)
            errors = rotation_error(rotation, orientation)
            for side, joints in enumerate(self.arm_ids):
                p, r = self.body_jacobian(self.wrists[side])
                j = np.concatenate((p[:, joints], .16 * r[:, joints]))
                residual = np.r_[desired[side] - position[side], .16 * errors[side]]
                step = j.T @ np.linalg.solve(j @ j.T + .0003 * np.eye(6), residual)
                result[joints] = np.clip(result[joints] + step, self.lo[joints], self.hi[joints])
        return result

    def solve_reference(self, wrist, rotation, hand, elbow, initial, iterations, *, seed=None, step_limit=None):
        """Fit physical wrists, wrist-local landmarks and a weak human elbow prior."""
        m, d = self.model, self.data
        step_limit = self.look_ahead if step_limit is None else step_limit
        lower = np.maximum(self.lo + .002, initial - step_limit) if iterations < 10 else self.lo + .002
        upper = np.minimum(self.hi - .002, initial + step_limit) if iterations < 10 else self.hi - .002
        q = np.clip(initial if seed is None else seed, lower, upper)
        for _ in range(iterations):
            self.forward(q, collision=True, jacobian=True)
            position, orientation, measured_hand = self.measured()
            residuals, jacobians = [], []

            def add(error, jac, weight):
                residuals.append((weight * error).reshape(-1))
                jacobians.append((weight * jac).reshape(-1, DOF))

            angle = rotation_error(rotation, orientation)
            world_error = np.einsum("sij,sfkj->sfki", orientation, hand - measured_hand)
            world_points = np.concatenate((d.xanchor[self.anchors], d.site_xpos[self.tips][..., None, :]), axis=-2)
            for side, body in enumerate(self.wrists):
                jpos, jrot = self.body_jacobian(body)
                add(wrist[side] - position[side], jpos, 58.)
                add(angle[side], jrot, 7.0)
                add(elbow[side] - d.xpos[self.elbows[side]], self.body_jacobian(self.elbows[side])[0], 6.0)
                for finger in range(5):
                    for point in range(3):
                        jp = self.scratch_jac
                        xyz = world_points[side, finger, point]
                        if point == 2:
                            mujoco.mj_jacSite(m, d, jp, None, int(self.tips[side, finger]))
                        else:
                            joint = self.anchors[side, finger, point]
                            mujoco.mj_jac(m, d, jp, None, xyz, int(m.jnt_bodyid[joint]))
                        jlocal = jp - jpos + skew(xyz - position[side]) @ jrot
                        add(world_error[side, finger, point], jlocal, 21. if point == 2 else 12.)

            closest = {}
            for i, contact in enumerate(d.contact):
                if contact.dist < REFERENCE_CLEARANCE:
                    key = tuple(contact.geom)
                    if key not in closest or contact.dist < d.contact[closest[key]].dist:
                        closest[key] = i
            for i in closest.values():
                contact = d.contact[i]
                j = np.zeros((2, 3, DOF))
                for k, geom in enumerate(contact.geom):
                    mujoco.mj_jac(m, d, j[k], None, contact.pos, int(m.geom_bodyid[geom]))
                add(np.asarray(REFERENCE_CLEARANCE - contact.dist), contact.frame[:3] @ (j[1] - j[0]), 300.)

            add(initial - q, self._identity, .55)
            J, error = np.concatenate(jacobians), np.concatenate(residuals)
            rhs = J.T @ error
            free = ~(((q <= lower + 1e-8) & (rhs < 0)) | ((q >= upper - 1e-8) & (rhs > 0)))
            if not free.any():
                break
            jf = J[:, free]
            delta = np.zeros(DOF)
            delta[free] = np.linalg.solve(jf.T @ jf + .30 * self._identity[:free.sum(), :free.sum()], rhs[free])
            if np.abs(delta).max() < 1e-5:
                break
            delta = np.clip(delta, -.18, .18)
            depth = self.depth()
            for step in (1., .5, .25):
                next_q = np.clip(q + step * delta, lower, upper)
                self.forward(next_q, collision=True)
                if self.depth() <= max(.0008, depth - .0001):
                    q = next_q
                    break
            else:
                break
        return q

    def replay(self, q, qd, ref_wrist, ref_hand, *, cap_speed=True):
        m, d = self.model, self.data
        count = round(DT / m.opt.timestep)
        if not math.isclose(count * m.opt.timestep, DT):
            raise ValueError("Must replay at the training control rate")
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[self.actuators] = q[0], qd[0], q[0]
        mujoco.mj_forward(m, d)
        command = q[0].copy()
        errors_wrist, errors_hand, peak = [], [], self.depth()
        valid = np.ones(len(q), dtype=bool)
        event = None
        for frame in range(1, len(q)):
            target = q[frame]
            command = command + np.clip(target - command, -self.slew, self.slew) if cap_speed else target
            d.ctrl[self.actuators] = command
            for step in range(1, count + 1):
                mujoco.mj_step(m, d)
                # mj_step leaves contact data at the beginning of the substep.
                mujoco.mj_kinematics(m, d)
                mujoco.mj_collision(m, d)
                peak = max(peak, self.depth())
                expected = ((frame - 1) * count + step) * m.opt.timestep
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or abs(d.time - expected) > 1e-8:
                    event = "invalid_physics"
                elif peak > CONTACT_STOP:
                    event = "contact"
                elif np.any(d.qpos < m.jnt_range[:, 0] - .05) or np.any(d.qpos > m.jnt_range[:, 1] + .05):
                    event = "joint_limit"
                if event:
                    valid[frame:] = False
                    break
            if event:
                break
            wrist, _, hand = self.measured()
            errors_wrist.extend(np.linalg.norm(wrist - ref_wrist[frame], axis=-1))
            errors_hand.extend(np.linalg.norm(hand[..., -1, :] - ref_hand[frame, ..., -1, :], axis=-1).flatten())
        p95 = lambda values: float(np.percentile(values, 95) * 1000.) if len(values) else None
        return {"complete": event is None, "event": event, "valid_frames": int(valid.sum()),
                "peak_depth_mm": 1000. * peak, "wrist_execution_p95_mm": p95(errors_wrist),
                "tip_execution_p95_mm": p95(errors_hand)}
