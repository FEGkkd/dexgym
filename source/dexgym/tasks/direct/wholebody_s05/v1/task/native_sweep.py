"""Independent continuous MuJoCo path certification with no Torch dependency."""

import mujoco
import numpy as np

from . import JOINTS


class NativeSweep:
    """Prove clearance on straight canonical-q intervals with MuJoCo geometry.

    In a pair's common moving frame, distance can change by at most the sum
    of the two objects' maximum point displacements. Radius bounds contain
    each finite collider, and summed link lengths bound the revolute lever
    arm across configurations. Ambiguous pairs are subdivided; expiry rejects.
    """

    def __init__(self, robot, *, clearance_m=None, max_depth=7, max_queries=63):
        self.robot = robot
        self.model = robot.model
        self.data = mujoco.MjData(self.model)
        self.clearance = robot.clearance_m if clearance_m is None else clearance_m
        self.max_depth = max_depth
        self.max_queries = max_queries
        if max_depth < 0 or max_queries < 1 or self.clearance <= 0:
            raise ValueError("Expected finite positive clearance and a bounded search")
        m = self.model
        self.a, self.b = robot.geom_a, robot.geom_b
        if self.a.size == 0:
            raise ValueError("No collision catalogue in simulation asset")
        self.has_plane = robot.has_plane
        self.radii = robot.radius
        canonical = {joint.id: axis for axis, joint in enumerate(robot.joints)}
        weights = np.zeros((m.ngeom, JOINTS))
        for g in np.unique(np.r_[self.a, self.b]):
            body = int(m.geom_bodyid[g])
            moving_extent = np.linalg.norm(m.geom_pos[g]) + m.geom_rbound[g]
            while body:
                address = int(m.body_jntadr[body])
                joint_count = int(m.body_jntnum[body])
                if joint_count > 1:
                    raise ValueError("Swept bound requires one hinge per body")
                joint_shift = 0.
                if joint_count:
                    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                        raise ValueError("Swept bound requires stationary planes")
                    joint_shift = np.linalg.norm(m.jnt_pos[address])
                    weights[g, canonical[address]] = moving_extent + joint_shift
                moving_extent += np.linalg.norm(m.body_pos[body]) + 2 * joint_shift
                body = int(m.body_parentid[body])
        first, second = weights[self.a], weights[self.b]
        self.weights = np.where((first > 0) & (second > 0), 0., first + second)
        self.mask = np.arange(len(self.a))

    def _probe(self, q, subset, required):
        m, d = self.model, self.data
        d.qpos[self.robot.q_indices] = q
        mujoco.mj_kinematics(m, d)
        a, b = self.a[subset], self.b[subset]
        broad_distance = np.linalg.norm(d.geom_xpos[a] - d.geom_xpos[b], axis=-1) - self.radii[subset]
        possible = np.flatnonzero((broad_distance <= required) | self.has_plane[subset])
        unresolved = []
        for i in possible:
            # mj_geomDistance returns the supplied cap when no nearer point exists.
            cap = float(required[i] + 1e-8)
            distance = mujoco.mj_geomDistance(m, d, int(a[i]), int(b[i]), cap, None)
            if not np.isfinite(distance) or distance < self.clearance:
                return None
            if distance <= required[i]:
                unresolved.append(i)
        return np.asarray(unresolved, dtype=int)

    def minimum_gap(self, data, cap=.005):
        a, b, m = self.a, self.b, self.model
        broad = np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b], axis=-1) - self.radii
        minimum = cap
        for i in np.flatnonzero((broad <= cap) | self.has_plane):
            result = mujoco.mj_geomDistance(m, data, int(a[i]), int(b[i]), float(cap), None)
            minimum = min(minimum, result)
        return minimum

    def pair_distances(self, q, cap=.02):
        """Exact narrow-phase distances below cap, conservative cap above it."""
        m, data = self.model, self.data
        data.qpos[self.robot.q_indices] = q
        mujoco.mj_kinematics(m, data)
        broad = np.linalg.norm(data.geom_xpos[self.a] - data.geom_xpos[self.b], axis=-1) - self.radii
        result = np.full(len(self.a), cap)
        for i in np.flatnonzero((broad <= cap) | self.has_plane):
            result[i] = mujoco.mj_geomDistance(m, data, int(self.a[i]), int(self.b[i]), float(cap), None)
        return np.nan_to_num(result, nan=-np.inf, neginf=-np.inf, posinf=-np.inf)

    def pose_gap(self, q):
        self.data.qpos[self.robot.q_indices] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.minimum_gap(self.data)

    def permits(self, start, end):
        start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            return False
        center = .5 * (start + end)
        delta = .5 * (end - start)
        stack = [(center, delta, self.mask, 0)]
        queries = 0
        while stack:
            if queries >= self.max_queries:
                return False
            center, offset, subset, depth = stack.pop()
            queries += 1
            bound = np.sum(self.weights[subset] * np.abs(offset), axis=-1)
            unresolved = self._probe(center, subset, self.clearance + bound)
            if unresolved is None:
                return False
            if unresolved.size:
                if depth >= self.max_depth:
                    return False
                offset = offset * .5
                subset = subset[unresolved]
                stack.append((center + offset, offset, subset, depth + 1))
                stack.append((center - offset, offset, subset, depth + 1))
        return True
