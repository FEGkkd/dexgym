"""The offline proposal seam consumes only current inputs and explicit sim state."""

from types import SimpleNamespace
import unittest

import numpy as np

from dexgym.tasks.direct.wholebody_s05.v1.rmb.offline_nominal import (
    OfflineNominalStatus, OfflineTeleopNominal,
)


class RecordingKernel:
    def __init__(self):
        self.fault = False
        self.calls = []

    def home_reference(self, home):
        self.home = home.copy()
        return np.zeros(2)

    def fk(self, side, q):
        return np.eye(4)

    def propose(self, target, measured, previous, branch, dt_s):
        self.calls.append((target.copy(), measured.copy(), previous.copy(), branch.copy(), dt_s))
        return tuple(SimpleNamespace(pair_accepted=not self.fault, pair_hard_fault=self.fault, accepted=True,
                                     ik_target_accepted=True, analytic_braking=False, analytic_exact_stop_used=False,
                                     rate_limited=True, workspace_projected=False,
                                     joint_command=previous[side * 7:(side + 1) * 7] + .001,
                                     arm_angle_command_rad=branch[side] + .0001)
                     for side in range(2))


class RecordingSdk:
    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls.clear()

    def step(self, side, points):
        self.calls.append((side, points.copy()))
        return np.full(20, .10 + points[0, 2])


class OfflineNominalTest(unittest.TestCase):
    def setUp(self):
        self.joints = np.zeros(54)
        self.frames = np.repeat(np.eye(4)[None], 7, axis=0)
        self.frames[0, 0, 3] = .3
        self.frames[1, 0, 3] = .1
        self.frames[4, 1, 3] = -.1
        model = SimpleNamespace(bounds=np.tile([-1., 1.], (54, 1)))
        self.kernel, self.sdk = RecordingKernel(), RecordingSdk()
        self.nominal = OfflineTeleopNominal(model, self.kernel, self.sdk, self.frames)
        self.branch = self.nominal.reset(self.joints)
        self.wrists = self.frames[1:3].copy()
        self.hand = np.zeros((2, 21, 7))
        self.order = (np.arange(21),) * 2

    def test_current_measurement_and_transform_order(self):
        predicted = np.linalg.inv(self.frames[1:3]) @ self.frames[0] @ self.wrists @ self.frames[3:5] @ self.frames[5:7]
        measured = np.full(54, .01)
        previous = self.joints.copy()
        proposed = self.nominal.propose(self.wrists, measured, previous, self.branch)
        frames, actual, prior, seed, dt_s = self.kernel.calls[-1]
        np.testing.assert_allclose(frames, predicted)
        np.testing.assert_array_equal(actual, measured[:14])
        np.testing.assert_array_equal(prior, previous[:14])
        np.testing.assert_array_equal(seed, self.branch)
        np.testing.assert_array_equal(previous, self.joints)
        self.assertAlmostEqual(dt_s, .004)
        np.testing.assert_array_equal(proposed.q_requested[:14], .001)
        np.testing.assert_array_equal(self.branch, 0.)
        np.testing.assert_array_equal(proposed.q_branch, .0001)

    def test_repeat_hold_mask_and_new_hand_source(self):
        self.nominal.refresh(self.wrists, self.hand, self.order, ("original-L", "original-R"), tick_ns=0)
        self.hand[0, 0, 2] = .10
        self.nominal.refresh(self.wrists, self.hand, self.order, ("original-L", "original-R"), tick_ns=20_000_000)
        self.assertEqual(len(self.sdk.calls), 2)
        self.nominal.refresh(self.wrists, self.hand, self.order, ("renewed-L", None), tick_ns=40_000_000)
        self.assertEqual(len(self.sdk.calls), 3)
        masks = np.zeros(40, dtype=bool)
        masks[0] = True
        out = self.nominal.propose(self.wrists, self.joints, self.joints, self.branch, hand_axes=masks)
        np.testing.assert_allclose(out.raw_hand_q[0], .20)
        np.testing.assert_allclose(out.filtered_hand_q[0], .13)
        np.testing.assert_array_equal(out.raw_hand_valid, [True, False])
        np.testing.assert_array_equal(out.q_requested[14], .02)
        np.testing.assert_array_equal(out.q_requested[15:], 0.)
        self.assertEqual(out.status[2], OfflineNominalStatus.LIMITED_ACTIVE)
        self.assertEqual(out.status[3], OfflineNominalStatus.HELD_BY_START_STAGE)

    def test_hold_fault_and_monotonic_cutoff(self):
        self.nominal.refresh(self.wrists, self.hand, self.order, ("left", "right"), tick_ns=20)
        with self.assertRaises(ValueError):
            self.nominal.refresh(self.wrists, self.hand, self.order, ("future-left", "future-right"), tick_ns=19)
        out = self.nominal.propose(None, self.joints, self.joints, self.branch)
        self.assertFalse(self.kernel.calls)
        np.testing.assert_array_equal(out.status, OfflineNominalStatus.INPUT_MISSING)
        self.kernel.fault = True
        out = self.nominal.propose(self.wrists, self.joints, self.joints, self.branch)
        self.assertTrue(out.hard_fault)
        np.testing.assert_array_equal(out.q_requested, 0.)
        np.testing.assert_array_equal(out.q_branch, 0.)
        np.testing.assert_array_equal(out.status, OfflineNominalStatus.NATIVE_PAIR_FAILED)


if __name__ == "__main__":
    unittest.main()
