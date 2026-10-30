import math
import unittest

from dexteroushands_lab.assets.wrist_6dof import Wrist6DoFController, WristLimits


class WristControllerTests(unittest.TestCase):
    def test_translation_and_rotation_commands_are_bounded(self):
        ctrl = Wrist6DoFController((0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0), WristLimits())
        ctrl.command((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        self.assertLessEqual(ctrl.target_pos[0], ctrl.limits.max_command_translation + 1e-9)
        self.assertLessEqual(ctrl.target_rot_angle, ctrl.limits.max_command_rotation + 1e-9)
        force, torque = ctrl.wrench(
            (0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        )
        self.assertGreater(force[0], 0.0)
        self.assertGreater(torque[2], 0.0)
        self.assertLessEqual(math.dist(force, (0, 0, 0)), ctrl.limits.max_force + 1e-9)
        self.assertLessEqual(math.dist(torque, (0, 0, 0)), ctrl.limits.max_torque + 1e-9)

    def test_velocity_damping_opposes_motion(self):
        ctrl = Wrist6DoFController((0, 0, 0.5), (0, 0, 0, 1), WristLimits())
        force, torque = ctrl.wrench(
            (0, 0, 0.5), (0, 0, 0, 1), (1, 0, 0), (0, 1, 0),
        )
        self.assertLess(force[0], 0)
        self.assertLess(torque[1], 0)

    def test_world_workspace_limits_and_quaternion_sign(self):
        ctrl = Wrist6DoFController((0, 0, 0.5), (0, 0, 0, 1), WristLimits())
        for _ in range(100):
            ctrl.command((1, 1, -1), (0, 0, 0))
        self.assertGreaterEqual(ctrl.target_pos[2], ctrl.limits.min_height)
        self.assertLessEqual(math.dist(ctrl.target_pos, (0, 0, 0.5)), ctrl.limits.max_target_radius + 1e-9)
        _, torque = ctrl.wrench(ctrl.target_pos, tuple(-q for q in ctrl.target_quat), (0, 0, 0), (0, 0, 0))
        self.assertLess(math.dist(torque, (0, 0, 0)), 1e-6)


if __name__ == "__main__":
    unittest.main()
