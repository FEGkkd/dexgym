import unittest
from types import SimpleNamespace

import numpy as np

from dexgym.tasks.direct.wholebody_s05.v1.rmb.nominal import SDKFailure
from dexgym.tasks.direct.wholebody_s05.v1.rmb.teleop_candidate import (
    ARM_DT_S,
    ARM_SUBSTEP_NS,
    CandidateStatus,
    TeleopCandidate,
)


def frozen_target(displacement):
    return {"wrist_xyz": np.array([[displacement, 0, 0], [displacement, 0, 0]]),
            "wrist_xyzw": np.array([[0., 0., 0., 1.], [0., 0., 0., 1.]])}


class SignedModel:
    home = np.zeros(54)
    bounds = np.tile([-1., 1.], (54, 1))

    def legality(self, q):
        return {"feasible": bool(np.max(np.abs(q)) <= 1.)}


class Sweep:
    def __init__(self):
        self.refuse_motion = False
        self.paths = []

    def permits(self, previous, proposed):
        self.paths.append((previous.copy(), proposed.copy()))
        return not self.refuse_motion or np.array_equal(previous, proposed)


class Hand:
    def __init__(self):
        self.steps = []
        self.resets = 0
        self.failing_side = None

    def reset(self):
        self.resets += 1

    def step(self, side, points):
        self.steps.append((side, points.copy()))
        if side == self.failing_side:
            raise SDKFailure("fixture retarget failure")
        return np.full(20, .5, dtype=np.float32)


class Kernel:
    def __init__(self):
        self.frames = []
        self.stationary = False
        self.failing_side = None
        self.resets = 0

    def home_reference(self, home):
        self.resets += 1
        np.testing.assert_array_equal(home, np.zeros(14))
        return np.array([.01, -.01])

    def fk(self, side, q):
        return np.eye(4)

    def propose(self, goals, measured, previous, branch, *, dt_s):
        self.frames.append((goals.copy(), previous.copy(), branch.copy(), dt_s))
        np.testing.assert_array_equal(measured, previous)
        rows = []
        for side in range(2):
            valid = side != self.failing_side
            command = previous[side * 7:(side + 1) * 7].copy()
            if not self.stationary:
                command[0] += .001
            rows.append(SimpleNamespace(
                accepted=valid, ik_target_accepted=valid, analytic_braking=not valid,
                joint_command=command, arm_angle_command_rad=branch[side] + .002,
                workspace_projected=True, rate_limited=True, backoff_scale=.5,
                step_target=goals[side].copy(),
            ))
        return rows


class TeleopCandidateTests(unittest.TestCase):
    def setUp(self):
        self.kernel, self.hand, self.sweep = Kernel(), Hand(), Sweep()
        wrist_frames = [(np.eye(4), np.eye(4))] * 2
        self.candidate = TeleopCandidate(SignedModel(), self.kernel, self.hand, self.sweep, wrist_frames)
        self.common = np.zeros((2, 21, 7))
        self.order = [np.arange(21)] * 2

    def tick(self, offset, number, source=(20, 40)):
        return self.candidate.step(frozen_target(offset), self.common, self.order, source,
                                   tick_ns=number * 20_000_000)

    def test_five_current_input_steps_and_exact_wuji_dedup(self):
        first = self.tick(.5, 0)
        self.assertEqual(len(first.substeps), 5)
        self.assertEqual([row.issue_tick_ns for row in first.substeps], list(range(0, 20_000_000, ARM_SUBSTEP_NS)))
        self.assertFalse(first.q_valid)
        self.assertTrue(np.all(first.status == CandidateStatus.RECOVERING))
        self.assertTrue(first.substeps[0].native_arms[0].workspace_projected)
        np.testing.assert_allclose(first.q_nominal[[0, 7, 14, 34]], [.005, .005, .1, .1])
        for index, (goals, previous, branch, dt_s) in enumerate(self.kernel.frames):
            np.testing.assert_array_equal(goals[:, 0, 3], [.5, .5])
            np.testing.assert_allclose(previous[[0, 7]], index * .001)
            np.testing.assert_allclose(branch, [.01 + index * .002, -.01 + index * .002])
            self.assertEqual(dt_s, ARM_DT_S)
        second = self.tick(.7, 1)
        self.assertTrue(second.q_valid)
        self.assertFalse(second.qd_valid)
        self.assertEqual(len(self.hand.steps), 2)
        third = self.tick(.8, 2, source=(21, 40))
        self.assertTrue(third.qd_valid)
        self.assertEqual(len(self.hand.steps), 3)
        self.assertEqual(self.hand.steps[-1][0], 0)
        np.testing.assert_allclose(third.qd_nominal[[0, 7]], [.25, .25])
        np.testing.assert_array_equal(first.native_tcp_target[:, 0, 3], [.5, .5])
        np.testing.assert_array_equal(first.substeps[0].q_issued[[0, 7]], [.001, .001])

    def test_native_refusal_does_not_advance_held_arm_or_branch(self):
        self.kernel.failing_side = 1
        result = self.tick(.5, 0)
        self.assertFalse(result.q_valid)
        self.assertEqual(result.status[1], CandidateStatus.NATIVE_ARM_HOLD)
        np.testing.assert_allclose(result.q_nominal[[0, 7]], [.005, 0])
        self.assertEqual(result.substeps[-1].arm_branches_rad[1], -.01)

    def test_swept_refusal_does_not_commit_any_pair_or_hand_goal(self):
        self.sweep.refuse_motion = True
        result = self.tick(.5, 0)
        self.assertTrue(np.all(result.status == CandidateStatus.WHOLE_BODY_PATH_HOLD))
        np.testing.assert_array_equal(result.q_nominal, np.zeros(54))
        for substep in result.substeps:
            self.assertFalse(substep.whole_body_path_accepted)
            np.testing.assert_array_equal(substep.arm_branches_rad, [.01, -.01])
            self.assertTrue(substep.hand_candidate_limit.all())
            self.assertFalse(substep.hand_limit_applied.any())

    def test_stationary_unmet_step_cannot_count_as_accepted_tracking(self):
        self.kernel.stationary = True
        for index in range(2):
            result = self.tick(.5, index)
            self.assertTrue(np.all(result.status[:2] == CandidateStatus.NATIVE_ARM_HOLD))
        self.candidate.cut()
        self.tick(0, 3)
        settled = self.tick(0, 4)
        self.assertTrue(settled.q_valid)
        self.assertEqual(self.kernel.resets, 2)

    def test_failed_hand_retains_failure_and_never_reuses_same_input(self):
        self.hand.failing_side = 0
        self.tick(.5, 0)
        repeated = self.tick(.6, 1)
        self.assertEqual(len(self.hand.steps), 2)
        self.assertEqual(repeated.status[2], CandidateStatus.WUJI_SDK_HOLD)
        self.assertEqual(repeated.hand_failure_reason[0], "fixture retarget failure")
        self.assertFalse(repeated.raw_hand_valid[0])
        self.assertFalse(repeated.q_valid)
        with self.assertRaisesRegex(ValueError, "generation cut"):
            self.tick(.6, 3)
