"""Independent functional-geometry and epigraph regressions for P02 refinement."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.p02.intent import CHAINS, robot_target, semantic_order
from dexgym.tasks.direct.wholebody_s05.p02.minimax import solve_minimax, tip_constraints
from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel, independent_frames
from dexgym.tasks.direct.wholebody_s05.p02.oracle import so3
from workspace import apply, apply_pinch, m0, palm_geometry, weight


def edge_matrix():
    edges = np.zeros((4, 5))
    edges[:, 0] = -1.0
    edges[:, 1:] = np.eye(4)
    return edges


class FunctionalPinchTest(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(310)
        self.points = random.normal(0.03, 0.025, (4, 2, 5, 4, 3))
        self.desired = random.normal(0.0, 0.015, (4, 2, 4, 3))

    def test_binary_pinch_is_the_exact_minimum_displacement_projection(self):
        active = ((0,), (1,), (0, 2), ())
        weights = np.zeros((4, 2, 4))
        expected = self.points.copy()
        edges = edge_matrix()
        for frame, entries in enumerate(active):
            for side in range(2):
                indices = list(entries if side == 0 else tuple(3 - j for j in entries))
                weights[frame, side, indices] = 1.0
                if not indices:
                    continue
                c = edges[indices]
                old_tips = self.points[frame, side, :, -1]
                residual = self.desired[frame, side, indices] - c @ old_tips
                delta = c.T @ np.linalg.solve(c @ c.T, residual)
                expected[frame, side] += delta[:, None] * np.linspace(0.0, 1.0, 4)[None, :, None]
        actual = apply_pinch(self.points, self.desired, weights)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-16)
        np.testing.assert_array_equal(actual[:, :, :, 0], self.points[:, :, :, 0])
        np.testing.assert_array_equal(actual[-1], self.points[-1])

    def test_fractional_pinch_agrees_with_independent_penalized_least_squares(self):
        weights = np.broadcast_to([0.2, 0.0, 0.73, 0.45], (4, 2, 4))
        c = edge_matrix()
        penalty = np.diag(weights[0, 0] / (1.0 - weights[0, 0]))
        actual = apply_pinch(self.points, self.desired, weights)
        for frame in range(4):
            for side in range(2):
                old = self.points[frame, side, :, -1]
                reference = np.linalg.solve(np.eye(5) + c.T @ penalty @ c,
                                            old + c.T @ penalty @ self.desired[frame, side])
                np.testing.assert_allclose(actual[frame, side, :, -1], reference, rtol=0, atol=2e-16)

    def test_enter_and_exit_threshold_are_continuous_with_continuous_slope(self):
        np.testing.assert_allclose(
            weight(np.array([0.015, 0.016, 0.018, 0.020, 0.022, 0.024, 0.025])),
            [1.0, 1.0, 0.84375, 0.5, 0.15625, 0.0, 0.0],
            rtol=0, atol=2e-15,
        )
        points = np.zeros((1, 2, 5, 4, 3))
        points[0, 0, 1, -1, 0] = 0.07

        def result(distance, smooth):
            desired = np.zeros((1, 2, 4, 3))
            desired[0, 0, 0, 0] = distance
            coefficients = np.zeros((1, 2, 4))
            coefficients[0, 0, 0] = weight(distance) if smooth else float(distance <= 0.020)
            return apply_pinch(points, desired, coefficients)[0, 0, [0, 1], -1, 0]

        h = 1e-8
        hard_jump = np.linalg.norm(result(0.020 + h, False) - result(0.020 - h, False))
        self.assertGreater(hard_jump, 0.03)
        for crossing in (0.016, 0.020, 0.024):
            below, at, above = (result(crossing + offset, True) for offset in (-h, 0.0, h))
            self.assertLess(np.linalg.norm(above - below), 5e-7)
            np.testing.assert_allclose((at - below) / h, (above - at) / h, rtol=0, atol=4e-5)


class RegistrationTest(unittest.TestCase):
    def setUp(self):
        self.local = np.random.default_rng(553).normal(0.01, 0.03, (2, 2, 5, 4, 3))
        self.p = np.array([[[0.2, -0.17, 0.0], [0.41, 0.27, 0.4]],
                           [[0.15, -0.08, 0.12], [0.58, 0.43, 0.37]]])
        self.rotations = Rotation.from_rotvec([[0.13, 0.24, -0.09], [-0.29, 0.11, 0.08],
                                              [0.36, -0.02, 0.01], [0.18, 0.48, -0.21]])
        self.wrist = np.concatenate((self.p, self.rotations.as_quat().reshape(2, 2, 4)), axis=-1)
        self.cal = {
            'world_rotation': Rotation.from_euler('z', 0.38).as_matrix().tolist(),
            'world_translation_m': [0.11, -0.05, 0.086],
            'tracker_to_tcp_orientation_mount': Rotation.from_rotvec(
                [[0.27, 0.04, 0.03], [-0.13, 0.23, 0.08]]).as_matrix().tolist(),
            'wrist_to_palm_rotation': Rotation.from_rotvec(
                [[0.07, -0.04, 0.14], [0.17, -0.13, 0.08]]).as_matrix().tolist(),
            'wrist_to_palm_translation': [[0.02, 0.03, -0.01], [0.06, 0.01, -0.02]],
        }

    def test_m0_rigid_registration_preserves_relative_geometry_and_named_frames(self):
        cfg = m0(self.cal)
        w, r, xyz = apply(cfg, self.wrist, self.local, self.cal)
        world = np.asarray(self.cal['world_rotation'])
        t = self.cal['world_translation_m']
        tracker = self.rotations.as_matrix().reshape(2, 2, 3, 3)
        mount = np.asarray(self.cal['tracker_to_tcp_orientation_mount'])
        palm_r = np.asarray(self.cal['wrist_to_palm_rotation'])
        palm_t = np.asarray(self.cal['wrist_to_palm_translation'])
        for frame in range(2):
            for side in range(2):
                expected_w = world @ self.p[frame, side] + t
                expected_r = world @ tracker[frame, side] @ mount[side]
                point = self.local[frame, side, 3, 1]
                expected_p = expected_w + expected_r @ (palm_t[side] + palm_r[side] @ point)
                np.testing.assert_allclose(w[frame, side], expected_w, rtol=0, atol=2e-15)
                np.testing.assert_allclose(r[frame, side], expected_r, rtol=0, atol=2e-15)
                np.testing.assert_allclose(xyz[frame, side, 3, 1], expected_p, rtol=0, atol=2e-15)
        np.testing.assert_allclose((w[:, 1] - w[:, 0]) @ world, self.p[:, 1] - self.p[:, 0],
                                   rtol=0, atol=2e-15)
        baseline_mount = tracker @ mount
        np.testing.assert_allclose(r[:, 0].swapaxes(-1, -2) @ r[:, 1],
                                   baseline_mount[:, 0].swapaxes(-1, -2) @ baseline_mount[:, 1],
                                   rtol=0, atol=2e-15)

    def test_m2_scales_wrist_spacing_only(self):
        original = m0(self.cal)
        scaled = original | {'scale': 0.81, 'pivot': [0.11, -0.32, 0.10]}
        w, r, xyz = apply(original, self.wrist, self.local, self.cal)
        w2, r2, xyz2 = apply(scaled, self.wrist, self.local, self.cal)
        shift = ((self.p - scaled['pivot']) * (scaled['scale'] - 1.0)) @ np.asarray(original['rotation']).T
        np.testing.assert_allclose(w2 - w, shift, rtol=0, atol=2e-15)
        np.testing.assert_allclose(w2[:, 1] - w2[:, 0], 0.81 * (w[:, 1] - w[:, 0]),
                                   rtol=0, atol=2e-15)
        np.testing.assert_array_equal(r, r2)
        np.testing.assert_allclose(xyz2 - xyz, np.broadcast_to(shift[:, :, None, None], xyz.shape),
                                   rtol=0, atol=2e-15)
        tips_before, tips_after = xyz[:, :, :, -1], xyz2[:, :, :, -1]
        np.testing.assert_allclose(tips_after[:, :, 1:] - tips_after[:, :, :1],
                                   tips_before[:, :, 1:] - tips_before[:, :, :1],
                                   rtol=0, atol=2e-15)

    def test_mirrored_node_identity_and_right_handed_frames(self):
        knuckles = np.array([[0.026, 0.058, -0.010], [0.073, 0.036, 0.004],
                             [0.078, 0.013, 0.007], [0.068, -0.010, 0.003],
                             [0.063, -0.030, -0.001]])
        direction = np.array([[0.008, 0.014, -0.011], [0.019, 0.004, 0.001],
                              [0.022, 0.000, -0.006], [0.016, -0.002, 0.001],
                              [0.012, -0.004, -0.003]])
        left = knuckles[:, None] + direction[:, None] * np.arange(4)[None, :, None]
        mirror = np.array([1., -1., 1.])
        local = np.stack((left, left * mirror))
        robot = SimpleNamespace(geometry=lambda _: SimpleNamespace(local=local), home=np.zeros(54))
        palm = np.zeros((2, 2, 25, 7))
        palm[..., -1] = 1.0
        orders = []
        expected_relative = np.empty((2, 2, 4, 3))
        scales = np.array([[0.82, 1.14], [1.07, 0.91]])
        random = np.random.default_rng(26)
        for side in range(2):
            source_xyz = [np.zeros(3)]
            nodes = [dict(node_id=899, parent_node_id=899, joint_type='Invalid', chain_type='Hand')]
            root_id = nodes[0]['node_id']
            for finger, chain in enumerate(CHAINS):
                points = local[side, finger]
                path = list(zip(('MCP', 'PIP', 'DIP', 'TIP'), points))
                if finger:
                    path.insert(2, ('IP', 0.5 * (points[1] + points[2])))
                parent = root_id
                for joint, location in path:
                    idx = 900 + len(nodes)
                    nodes.append(dict(node_id=idx, parent_node_id=parent,
                                      joint_type=joint, chain_type=chain))
                    source_xyz.append(location)
                    parent = idx
            permutation = random.permutation(25)
            for channel, original in enumerate(permutation):
                nodes[original]['original_order'] = channel
            orders.append(semantic_order({'joints': [nodes[i] for i in reversed(permutation)]}))
            for frame in range(2):
                rot = Rotation.from_rotvec([0.17 * side, -0.08 * frame, 0.21]).as_matrix()
                cloud = (np.array(source_xyz)[permutation] * scales[frame, side]) @ rot.T
                palm[frame, side, :, :3] = cloud + [0.11, -0.02 * side, 0.1]
                tips = local[side, :, -1]
                expected_relative[frame, side] = (tips[1:] - tips[:1]) @ rot.T * scales[frame, side]
        shape, raw, transforms, usable = palm_geometry(palm, orders, robot)
        np.testing.assert_array_equal(usable, [True, True])
        np.testing.assert_allclose(shape, np.broadcast_to(local, shape.shape), rtol=0, atol=3e-15)
        np.testing.assert_allclose(raw, expected_relative, rtol=0, atol=3e-15)
        np.testing.assert_allclose(np.linalg.det(transforms), np.ones((2, 2)), rtol=0, atol=2e-15)


class EpigraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path('docs/s0-5/evidence/P02/plan.json').read_bytes())
        cls.robot = RobotModel(cls.config['model'])

    def test_named_tip_epigraph_and_jacobian_match_independent_fk(self):
        robot = self.robot
        random = np.random.default_rng(192)
        q = robot.home + random.normal(0., 0.029, 54)
        target = robot_target(robot, robot.geometry(robot.home), self.config['cross_hand_pairs'])
        target.world[:, :, -1] += [0.005, -0.006, 0.009]
        point = np.r_[q, 3.1]
        names = [robot.model.site(i).name for i in robot.tip_ids]
        targets = target.world[:, :, -1].reshape(10, 3)

        def independent(y):
            frames = independent_frames(robot, y[:-1])
            tips = np.stack([frames['site', name][0] for name in names])
            return y[-1] - np.linalg.norm(tips - targets, axis=-1) / 0.01

        value, jac = tip_constraints(robot, target, point)
        np.testing.assert_allclose(value, independent(point), rtol=0, atol=2e-12)
        directions = np.vstack((np.eye(55)[[0, 8, 14, 34, 53, 54]], random.normal(size=55)))
        for d in directions:
            d /= np.linalg.norm(d)
            h = 1e-6
            derivative = (independent(point + h * d) - independent(point - h * d)) / (2.0 * h)
            np.testing.assert_allclose(jac @ d, derivative, rtol=1e-7, atol=2e-7)

    def test_exact_legal_witness_is_retained_by_minimax(self):
        robot = self.robot
        home = robot.home.copy()
        self.assertTrue(robot.legality(home)['feasible'])
        target = robot_target(robot, robot.geometry(home), self.config['cross_hand_pairs'])
        displaced = home.copy()
        displaced[0] += 0.025
        result = solve_minimax(robot, target, self.config, [('PERTURBED', displaced),
                                                          ('SIGNED_HOME', home)], iterations=12)
        self.assertTrue(result['legality']['feasible'])
        self.assertGreater(result['attempts'][0]['initial_epigraph_m'], 1e-3)
        self.assertLess(result['epigraph_m'], 1e-10)
        self.assertLess(max(np.ravel(result['metrics']['tip_m'])), 1e-10)

    def test_so3_log_jacobian_near_pi_matches_world_angular_velocity(self):
        goal = Rotation.from_rotvec([-0.29, 0.16, 0.31]).as_matrix()
        axis = np.array([0.3, -0.7, 0.2])
        axis /= np.linalg.norm(axis)
        actual = goal @ Rotation.from_rotvec((np.pi - 0.018) * axis).as_matrix()
        d = np.array([-0.4, 0.2, 0.9])
        h = 1e-6

        def independent(dt):
            perturbation = Rotation.from_rotvec(dt * d).as_matrix()
            return Rotation.from_matrix(goal.T @ perturbation @ actual).as_rotvec()

        value, jac = so3(goal, actual, np.eye(3))
        np.testing.assert_allclose(value, (np.pi - 0.018) * axis, rtol=0, atol=3e-14)
        np.testing.assert_allclose(jac @ d, (independent(h) - independent(-h)) / (2 * h),
                                   rtol=1e-8, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
