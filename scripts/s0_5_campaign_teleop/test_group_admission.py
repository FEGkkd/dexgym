"""Deterministic controls for the offline simultaneous-commit boundary."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from group_admission import (
    GROUPS, GroupStatus, GroupedLockstep, RECOVERY_HAND_STEP_RAD,
    RECOVERY_SUBCYCLES, admit_groups, verified_groups,
)
from dexgym.tasks.direct.wholebody_s05.v1.rmb.offline_nominal import OfflineNominalStatus
from dexgym.tasks.direct.wholebody_s05.v1.task.safety import Stop
from swept_live_guard import Reason


AXES = (np.arange(0, 7), np.arange(7, 14), np.arange(14, 34), np.arange(34, 54))
FIRST_AXES = [indices[0] for indices in AXES]
STEP = .03125


def command():
    q = np.zeros(54)
    q[FIRST_AXES] = STEP
    return q


def decision(allowed):
    return (Reason.PERMIT, {}) if allowed else (
        Reason.SWEEP_UNPROVEN, {"termination": "MAX_SUBDIVISION_DEPTH"})


def lockstep(proof, *, goal=None, hard_fault=False, feedback_valid=True, event=0, substeps=2):
    """Exercise the real coordinator with explicit fake native and physical seams."""
    step = object.__new__(GroupedLockstep)
    step.fault = None
    step.issued = np.zeros(54)
    step.branch = np.array([.1, .2])
    step.indices = AXES
    step.recovery = np.full(4, RECOVERY_SUBCYCLES, dtype=np.int16)
    step.physics = SimpleNamespace(filtered=np.zeros((1, 54)))
    measured = np.zeros(54)
    if not feedback_valid:
        measured[-1] = np.nan
    step.state = Mock(return_value=(measured, np.zeros(54)))
    step.nominal = SimpleNamespace(propose=Mock(return_value=SimpleNamespace(
        hard_fault=hard_fault, q_requested=command() if goal is None else goal,
        q_branch=np.array([.11, .22]), status=np.full(4, OfflineNominalStatus.ACTIVE),
    )))
    step._checked_proof = Mock(side_effect=lambda q, state: proof(q))
    step._physical_steps = Mock(return_value=(
        SimpleNamespace(event=np.array([event]), substeps=np.array([substeps])), (), (), (),
    ))
    return step


class AdmissionTests(unittest.TestCase):
    def test_named_groups_must_preserve_nominal_order_and_transmissions(self):
        names = tuple(f"q{i}" for i in range(54))
        groups = tuple(tuple(names[i] for i in axes) for axes in AXES)
        joints = tuple(SimpleNamespace(id=i) for i in range(54))
        actuators = tuple(SimpleNamespace(trnid=[j.id, -1]) for j in joints)
        robot = SimpleNamespace(names=names, groups=groups, joints=joints, actuators=actuators)
        for actual, expected in zip(verified_groups(robot), AXES, strict=True):
            np.testing.assert_array_equal(actual, expected)
        robot.groups = (groups[1], groups[0], *groups[2:])
        with self.assertRaises(ValueError):
            verified_groups(robot)
        robot.groups = groups
        robot.actuators = (actuators[1], actuators[0], *actuators[2:])
        with self.assertRaises(ValueError):
            verified_groups(robot)

    def test_one_hand_rejection_keeps_other_movements_and_has_one_commit(self):
        issued = np.zeros(54)
        requested = command()
        proof = Mock(side_effect=lambda q: decision(q[AXES[3][0]] == 0))
        result = admit_groups(issued, requested, AXES, [True] * 4, [False] * 4, proof)
        self.assertEqual([entry["choice"] for entry in result.outcomes], ["EXACT"] * 3 + [None])
        self.assertEqual(len(result.outcomes[3]["trials"]), 3)
        np.testing.assert_array_equal(result.issued[FIRST_AXES], [STEP, STEP, STEP, 0.])
        np.testing.assert_array_equal(proof.call_args.args[0], result.issued)
        np.testing.assert_array_equal(issued, 0.)
        np.testing.assert_array_equal(requested, command())
        step = lockstep(lambda q: decision(q[AXES[3][0]] == 0))
        actual = step.advance(object())
        step._physical_steps.assert_called_once()
        np.testing.assert_array_equal(actual.issued[FIRST_AXES], [STEP, STEP, STEP, 0.])
        self.assertEqual([g["status"] for g in actual.groups],
                         [GroupStatus.ACTIVE] * 3 + [GroupStatus.GROUP_HOLD])

    def test_individual_proofs_do_not_authorize_an_unsafe_union(self):
        proof = Mock(side_effect=lambda q: decision(not (q[AXES[0][0]] and q[AXES[1][0]])))
        single = np.zeros(54)
        single[AXES[1][0]] = STEP
        self.assertEqual(proof(single)[0], Reason.PERMIT)
        proof.reset_mock()
        result = admit_groups(np.zeros(54), command(), AXES, [True] * 4, [False] * 4, proof)
        rejected = result.outcomes[1]
        self.assertEqual(rejected["trials"][0]["proof"], Reason.SWEEP_UNPROVEN.name)
        attempted_union = proof.call_args_list[1].args[0]
        np.testing.assert_array_equal(attempted_union[FIRST_AXES], [STEP, STEP, 0., 0.])
        np.testing.assert_array_equal(result.issued[FIRST_AXES], [STEP, 0., STEP, STEP])
        np.testing.assert_array_equal(proof.call_args.args[0], result.issued)
        step = lockstep(lambda q: decision(not (q[AXES[0][0]] and q[AXES[1][0]])))
        step.advance(object())
        np.testing.assert_array_equal(step.branch, [.11, .2])

    def test_final_veto_requires_independent_hold_proof_and_discards_branches(self):
        proof = Mock(side_effect=[decision(True)] * 4 + [decision(False), decision(True)])
        step = lockstep(proof)
        result = step.advance(object())
        self.assertEqual(proof.call_count, 6)
        np.testing.assert_array_equal(proof.call_args.args[0], 0.)
        np.testing.assert_array_equal(result.issued, 0.)
        np.testing.assert_array_equal(step.branch, [.1, .2])
        self.assertEqual(result.proof, Reason.SWEEP_UNPROVEN.name)
        self.assertEqual(result.hold_proof, Reason.PERMIT.name)
        self.assertTrue(result.used_hold)
        self.assertTrue(all(group["committed"] is None for group in result.groups))
        step._physical_steps.assert_called_once()

    def test_final_and_hold_veto_prevent_any_physical_step(self):
        proof = Mock(side_effect=[decision(True)] * 4 + [decision(False), decision(False)])
        step = lockstep(proof)
        result = step.advance(object())
        self.assertEqual(proof.call_count, 6)
        self.assertIsNone(result.issued)
        self.assertIsNone(result.interval)
        self.assertEqual(result.fault, "HOLD_INFEASIBLE:SWEEP_UNPROVEN")
        np.testing.assert_array_equal(step.branch, [.1, .2])
        step._physical_steps.assert_not_called()

    def test_bounded_hand_rejoin_never_counts_as_exact_active(self):
        requested = command()
        hand_axis = AXES[3][0]
        step = lockstep(lambda q: decision(abs(q[hand_axis]) <= .01), goal=requested)
        step.recovery[3] = 0
        result = step.advance(object())
        hand = result.groups[3]
        attempts = hand["trials"]
        self.assertEqual([entry["proof"] for entry in attempts], ["SWEEP_UNPROVEN", "PERMIT"])
        self.assertAlmostEqual(attempts[0]["candidate"][0], RECOVERY_HAND_STEP_RAD)
        self.assertAlmostEqual(result.issued[hand_axis], RECOVERY_HAND_STEP_RAD * .5)
        self.assertEqual(hand["committed"], "BOUNDED_REJOIN")
        self.assertEqual(hand["status"], GroupStatus.REJOINING)
        self.assertEqual(hand["rejoin_streak"], 0)
        self.assertEqual(requested[hand_axis], STEP)

    def test_rejoin_only_unlocks_after_the_complete_exact_streak(self):
        goal = np.zeros(54)
        goal[AXES[3][0]] = .00390625
        step = lockstep(lambda q: decision(True), goal=goal)
        step.recovery[3] = 0
        for previous in range(RECOVERY_SUBCYCLES):
            hand = step.advance(object()).groups[3]
            self.assertEqual(hand["status"], GroupStatus.REJOINING)
            self.assertEqual(hand["rejoin_streak"], previous + 1)
        hand = step.advance(object()).groups[3]
        self.assertEqual(hand["label"], "RIGHT_HAND_ACTIVE")

    def test_native_hard_fault_and_invalid_feedback_stop_before_proof(self):
        for kwargs, failure in (({"hard_fault": True}, "NATIVE_PAIR_HARD_FAULT"),
                                ({"feedback_valid": False}, "FEEDBACK_INVALID")):
            with self.subTest(failure=failure):
                step = lockstep(lambda q: decision(True), **kwargs)
                result = step.advance(object())
                self.assertEqual(result.fault, failure)
                self.assertIsNone(result.issued)
                np.testing.assert_array_equal(step.branch, [.1, .2])
                step._checked_proof.assert_not_called()
                step._physical_steps.assert_not_called()
                self.assertEqual(step.nominal.propose.call_count, int(kwargs.get("hard_fault", False)))

    def test_sampled_gap_codes_are_distinct_from_continuous_unproven(self):
        step = object.__new__(GroupedLockstep)
        step.robot = SimpleNamespace(clearance_m=.0005)
        distance = Mock()
        step.sweep = SimpleNamespace(native=SimpleNamespace(pair_distances=distance))
        step._proof = Mock(return_value=decision(False))
        for gap, reason, termination in (
                (-.0001, Reason.COLLISION, "PENETRATION"),
                (.0004, Reason.COLLISION, "CLEARANCE_VIOLATION"),
                (np.inf, Reason.INTERNAL, "INVALID_ENDPOINT_GEOMETRY"),
                (.001, Reason.SWEEP_UNPROVEN, "MAX_SUBDIVISION_DEPTH")):
            with self.subTest(gap=gap):
                distance.return_value = np.array([gap, .002])
                got, metrics = step._checked_proof(command(), np.zeros(54))
                self.assertIs(got, reason)
                self.assertEqual(metrics["termination"], termination)
        step._proof.assert_called_once()

    def test_incomplete_physics_does_not_grant_a_rejoin_step(self):
        step = lockstep(lambda q: decision(True), event=Stop.BAD_FEEDBACK, substeps=1)
        result = step.advance(object())
        self.assertEqual(result.fault, "FEEDBACK_INVALID")
        self.assertTrue(all(group["status"] == GroupStatus.FEEDBACK_INVALID for group in result.groups))
        np.testing.assert_array_equal(step.recovery, 0)
        with self.assertRaises(RuntimeError):
            step.advance(object())
        step._physical_steps.assert_called_once()


if __name__ == "__main__":
    unittest.main()
