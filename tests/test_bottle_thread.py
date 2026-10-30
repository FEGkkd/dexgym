import math
import unittest

from dexteroushands_lab.assets.bottle import BottleConfig, ThreadCoupler, sample_helix


IDENTITY = (0.0, 0.0, 0.0, 1.0)


def yaw(angle):
    return (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))


class BottleThreadTests(unittest.TestCase):
    def test_geometry_has_pitch_handedness_and_no_high_density_mesh(self):
        cfg = BottleConfig()
        helix = sample_helix(cfg, segments_per_turn=8)
        self.assertEqual(len(helix), cfg.thread_turns * 8 + 1)
        self.assertAlmostEqual(helix[-1][2] - helix[0][2], cfg.thread_pitch * cfg.thread_turns)
        left = sample_helix(BottleConfig(thread_handedness="left"), segments_per_turn=8)
        self.assertAlmostEqual(left[2][2] - left[0][2], -(helix[2][2] - helix[0][2]))

    def test_unwrapped_angle_and_pitch_coupling(self):
        cfg = BottleConfig()
        coupler = ThreadCoupler(cfg)
        z0 = cfg.cap_initial_z
        coupler.observe((0, 0, 0), IDENTITY, (0, 0, z0), IDENTITY)
        for angle in (2.5, 3.1, -3.1, -2.5):
            state = coupler.observe((0, 0, 0), IDENTITY, (0, 0, z0), yaw(angle))
        self.assertGreater(state.theta_unwrapped, math.pi)
        self.assertAlmostEqual(
            state.desired_axial - z0,
            cfg.thread_pitch * state.theta_unwrapped / (2 * math.pi), places=8,
        )

    def test_tilted_bottle_uses_local_thread_axis(self):
        cfg = BottleConfig()
        coupler = ThreadCoupler(cfg)
        tilt = (0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))
        state = coupler.observe((0, 0, 0), tilt, (cfg.cap_initial_z, 0, 0), tilt)
        self.assertAlmostEqual(state.axial, cfg.cap_initial_z, places=7)
        self.assertAlmostEqual(state.lateral_error, 0.0, places=7)

    def test_equal_opposite_reaction_and_bounded_wrench(self):
        cfg = BottleConfig()
        coupler = ThreadCoupler(cfg)
        z0 = cfg.cap_initial_z
        coupler.observe((0, 0, 0), IDENTITY, (0, 0, z0), IDENTITY)
        result = coupler.wrenches(
            (0, 0, 0), IDENTITY, (0, 0, 0), (0, 0, 0),
            (0.004, 0, z0 + 0.002), yaw(0.4), (0, 0, 0), (0, 0, 0.1),
        )
        self.assertTrue(all(math.isfinite(x) for x in result.cap_force + result.cap_torque))
        self.assertLessEqual(math.dist(result.cap_force, (0, 0, 0)), cfg.max_constraint_force + 1e-8)
        self.assertAlmostEqual(sum(result.cap_force[i] + result.bottle_force[i] for i in range(3)), 0.0)

    def test_breakaway_stiction_is_capped_without_plastic_drift(self):
        cfg = BottleConfig()
        coupler = ThreadCoupler(cfg)
        z0 = cfg.cap_initial_z
        args = ((0, 0, 0), IDENTITY, (0, 0, 0), (0, 0, 0))
        cap = lambda angle: ((0, 0, z0), yaw(angle), (0, 0, 0), (0, 0, 0))
        coupler.wrenches(*args, *cap(0))
        tiny = coupler.wrenches(*args, *cap(0.001))
        self.assertLess(abs(tiny.breakaway_resistance_torque), cfg.breakaway_torque)
        yielded = coupler.wrenches(*args, *cap(0.2))
        self.assertAlmostEqual(abs(yielded.breakaway_resistance_torque), cfg.breakaway_torque)
        returned = coupler.wrenches(*args, *cap(0.001))
        self.assertAlmostEqual(returned.breakaway_resistance_torque,
                               tiny.breakaway_resistance_torque)

    def test_running_regime_releases_stiction_and_resticks(self):
        cfg = BottleConfig()
        coupler = ThreadCoupler(cfg)
        z0 = cfg.cap_initial_z
        bottle = ((0, 0, 0), IDENTITY, (0, 0, 0), (0, 0, 0))
        coupler.wrenches(*bottle, (0, 0, z0), IDENTITY, (0, 0, 0), (0, 0, 0))
        slipping = coupler.wrenches(*bottle, (0, 0, z0), yaw(0.2),
                                     (0, 0, 0), (0, 0, 1))
        self.assertAlmostEqual(slipping.breakaway_resistance_torque, 0.0)
        self.assertLess(slipping.resistance_torque, 0.0)
        stopped = coupler.wrenches(*bottle, (0, 0, z0), yaw(0.2),
                                    (0, 0, 0), (0, 0, 0))
        self.assertAlmostEqual(stopped.breakaway_resistance_torque, 0.0)
        resticking = coupler.wrenches(*bottle, (0, 0, z0), yaw(0.21),
                                       (0, 0, 0), (0, 0, 0))
        self.assertLess(resticking.breakaway_resistance_torque, 0.0)

    def test_common_rigid_rotation_has_no_spurious_joint_damping(self):
        cfg = BottleConfig()
        coupler = ThreadCoupler(cfg)
        z0 = cfg.cap_initial_z
        spin = (0, 1, 0)
        result = coupler.wrenches(
            (0, 0, 0), IDENTITY, (0, 0, 0), spin,
            (0, 0, z0), IDENTITY, (z0, 0, 0), spin,
        )
        self.assertLess(math.dist(result.cap_force, (0, 0, 0)), 1e-9)

    def test_finite_thread_travel_has_a_physical_end_stop(self):
        cfg = BottleConfig()
        coupler = ThreadCoupler(cfg)
        z0 = cfg.cap_initial_z
        bottle = ((0, 0, 0), IDENTITY, (0, 0, 0), (0, 0, 0))
        result = None
        for index in range(0, 100):
            angle = index * 0.2
            result = coupler.wrenches(*bottle, (0, 0, z0), yaw(angle),
                                      (0, 0, 0), (0, 0, 0))
        self.assertGreater(result.state.theta_unwrapped, cfg.thread_turns * 2 * math.pi)
        self.assertLess(result.end_stop_resistance_torque, 0.0)
        self.assertLessEqual(result.applied_coupling_torque_nm,
                             cfg.max_constraint_torque + 1e-9)

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            BottleConfig(cap_outer_radius=0.01)
        with self.assertRaises(ValueError):
            BottleConfig(thread_pitch=0)

    def test_initial_cap_shell_clears_bottle_and_neck_top(self):
        cfg = BottleConfig()
        cap_bottom = cfg.cap_initial_z - cfg.cap_height / 2
        cap_top = cfg.cap_initial_z + cfg.cap_height / 2
        self.assertGreater(cap_bottom, cfg.bottle_height / 2)
        self.assertGreater(cap_top, cfg.bottle_height / 2 + cfg.neck_height)


if __name__ == "__main__":
    unittest.main()
