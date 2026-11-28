"""Deterministic controls for the offline simultaneous-commit boundary."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from build_offline_rmb_diagnostic import RAW_HMB_FIELDS, active, blank, publish_partition, record_execution
from group_admission import (
    GROUPS, GroupStatus, GroupedLockstep, GroupedSubstep, RECOVERY_HAND_STEP_RAD,
    RECOVERY_SUBCYCLES, admit_groups, combine_intervals, verified_groups,
)
from dexgym.tasks.direct.wholebody_s05.v1.rmb.offline_nominal import OfflineNominalStatus
from dexgym.tasks.direct.wholebody_s05.v1.task.physics import Interval
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


def physical_sample(number, *, event=0):
    target = torch.full((1, 1, 54), float(number))
    return Interval(
        torch.full((1, 108), float(number)), target[:, 0],
        target, target * 2, target * 3, torch.zeros(1, 13),
        torch.zeros(1), torch.tensor([.01 / number]),
        torch.ones(1, dtype=torch.int32), torch.tensor([int(event)]),
    )


def builder_fields():
    data = dict.fromkeys(RAW_HMB_FIELDS, np.zeros(1))
    data.update(tick_ns=np.zeros(1), joint_valid=np.ones(1, bool), segment_id=np.zeros(1, int),
                source_index=np.zeros((1, 4, 1), int))
    with patch("build_offline_rmb_diagnostic.hmb_five_rows", return_value=np.ones(1, bool)):
        return blank(data, grouped=True)


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

    def test_physical_calls_share_one_command_and_stop_at_first_event(self):
        for event in (0, Stop.PHYSICAL_CONSTRAINT):
            with self.subTest(event=event):
                step = object.__new__(GroupedLockstep)
                step.issued = command()
                step.safety = object()
                step.state = Mock(side_effect=[
                    (np.full(54, 1.), np.full(54, -1.)), (np.full(54, 2.), np.full(54, -2.)),
                ])
                interval = Mock(side_effect=[physical_sample(1, event=event), physical_sample(2)])
                step.physics = SimpleNamespace(interval=interval)
                combined, positions, velocities, executed = step._physical_steps()
                count = 1 if event else 2
                self.assertEqual(len(executed), count)
                self.assertEqual(int(combined.substeps[0]), count)
                self.assertEqual(int(combined.event[0]), event)
                self.assertEqual(interval.call_count, count)
                self.assertEqual(step.state.call_count, count)
                np.testing.assert_array_equal(combined.applied[0, :, 0].numpy(), np.arange(1, count + 1))
                np.testing.assert_array_equal(np.asarray(positions)[:, 0], np.arange(1, count + 1))
                np.testing.assert_array_equal(np.asarray(velocities)[:, 0], -np.arange(1, count + 1))
                for call in interval.call_args_list:
                    applied, allowed, safety = call.args
                    self.assertEqual(call.kwargs, {"count": 1})
                    self.assertEqual(applied.dtype, torch.float32)
                    np.testing.assert_array_equal(applied.numpy(), step.issued[None])
                    self.assertTrue(allowed.item())
                    self.assertIs(safety, step.safety)

    def test_builder_full_active_requires_every_group_and_all_physical_positions(self):
        proposal = SimpleNamespace(q_requested=command(), status=np.full(4, OfflineNominalStatus.ACTIVE),
                                   native_arms=(), raw_hand_q=np.zeros((2, 20)), raw_hand_valid=np.ones(2, bool),
                                   filtered_hand_q=np.zeros((2, 20)), hand_limited=np.zeros(40, bool))
        groups = tuple({"status": GroupStatus.ACTIVE} for _ in GROUPS)
        history = []
        for first in range(1, 11, 2):
            numbers = (first, first + 1)
            intervals = tuple(physical_sample(number) for number in numbers)
            history.append(GroupedSubstep(proposal, "PERMIT", {}, command(), combine_intervals(intervals),
                          False, None, groups=groups, physical_steps=intervals,
                          physical_q=tuple(np.full(54, number) for number in numbers),
                          physical_qd=tuple(np.full(54, -number) for number in numbers)))
        geometry = SimpleNamespace(wrists=np.zeros((2, 3)), rotations=np.array([np.eye(3)] * 2),
                                   local=np.zeros((2, 5, 4, 3)), landmarks=np.zeros((2, 5, 4, 3)))
        executor = SimpleNamespace(state=lambda: (np.full(54, 10.), np.full(54, -10.)),
                                   robot=SimpleNamespace(geometry=lambda q: geometry))
        self.assertTrue(active(history))
        positive = builder_fields()
        record_execution(positive, 0, history, executor)
        self.assertTrue(positive["q_nominal_mask"][0])
        self.assertEqual(positive["applied_substeps"][0], 10)
        self.assertTrue(positive["physical_state_mask"][0].all())
        np.testing.assert_array_equal(positive["physical_q"][0, :, 0], np.arange(1, 11))
        np.testing.assert_array_equal(positive["physical_qd"][0, :, 0], -np.arange(1, 11))
        for name, changed in (
                ("partial", replace(history[-1], groups=(*groups[:3], {"status": GroupStatus.GROUP_HOLD}))),
                ("rejoin", replace(history[-1], groups=(*groups[:3], {"status": GroupStatus.REJOINING}))),
                ("hold", replace(history[-1], used_hold=True))):
            with self.subTest(name=name):
                result = [*history[:-1], changed]
                self.assertFalse(active(result))
                fields = builder_fields()
                record_execution(fields, 0, result, executor)
                self.assertEqual(fields["applied_substeps"][0], 10)
                self.assertFalse(fields["q_nominal_mask"][0])
        self.assertFalse(active(history[:-1]))
        fields = builder_fields()
        record_execution(fields, 0, history[:-1], executor)
        self.assertEqual(fields["applied_substeps"][0], 8)
        self.assertFalse(fields["q_nominal_mask"][0])
        final = history[-1]
        short = replace(final, interval=final.physical_steps[0], physical_steps=final.physical_steps[:1],
                        physical_q=final.physical_q[:1], physical_qd=final.physical_qd[:1])
        fields = builder_fields()
        record_execution(fields, 0, [*history[:-1], short], executor)
        self.assertEqual(fields["applied_substeps"][0], 9)
        self.assertFalse(fields["q_nominal_mask"][0])

    def test_raw_missing_payload_survives_without_entering_finite_rmb_fields(self):
        fields = builder_fields()
        sentinel = np.asarray([0x7ff80000000000ca], dtype="<u8")
        fields["hmb_pose"] = sentinel.view("<f8")
        with TemporaryDirectory() as directory:
            root = Path(directory) / "part"
            result = publish_partition(root, "signed-source", fields)
            self.assertTrue(result["arrays"]["hmb_pose"]["raw_diagnostic_only"])
            self.assertNotIn("raw_diagnostic_only", result["arrays"]["q_measured"])
            np.testing.assert_array_equal(np.load(root / "hmb_pose.npy", allow_pickle=False).view("<u8"), sentinel)


if __name__ == "__main__":
    unittest.main()
