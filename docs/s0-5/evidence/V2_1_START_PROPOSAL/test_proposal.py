"""Synthetic design controls. No robot, recording, replay, or training entry points."""

import copy
import json
import math
import unittest
from pathlib import Path

from check_proposal import (
    APPROVALS, BASELINE, DOCS, PHASES, RECEIPTS_AT_ENTRY,
    cohort_from_history, git_blob, synthetic_stage_step, validate_candidate,
)


def add(left, right):
    return tuple(a + b for a, b in zip(left, right))


def difference(left, right):
    return tuple(a - b for a, b in zip(left, right))


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def rotate(rotation, point):
    return tuple(dot(row, point) for row in rotation)


def midpoint(left, right):
    return tuple((a + b) / 2 for a, b in zip(left, right))


def transpose(matrix):
    return tuple(zip(*matrix))


def cross(left, right):
    x, y, z = left
    a, b, c = right
    return y * c - z * b, z * a - x * c, x * b - y * a


class ProposalControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proposal = json.loads(Path(__file__).with_name("design_candidate.json").read_bytes())
        signed_index = json.loads(git_blob(BASELINE, DOCS + "evidence/P01_E_E1/build_index.json"))
        cls.cohort = cohort_from_history(signed_index)

    def test_pending_proposal_and_signed_population(self):
        validate_candidate(self.proposal, self.cohort)
        self.assertEqual(tuple(self.cohort["total"].values()), (32, 308, 116613, 114532, 110796))
        for split, min_active, min_history in (
            ("Train", 72575, 57792), ("Validation", 24778, 19766),
        ):
            self.assertEqual(math.ceil(self.cohort[split]["joint_valid_ticks"] * 85 / 100), min_active)
            self.assertEqual(
                math.ceil(self.cohort[split]["original_history_candidates"] * 70 / 100), min_history
            )

    def reject_edit(self, path, value):
        proposal = copy.deepcopy(self.proposal)
        fields = path.split(".")
        container = proposal
        for field in fields[:-1]:
            container = container[field]
        container[fields[-1]] = value
        with self.assertRaises(ValueError):
            validate_candidate(proposal, self.cohort)

    def test_negative_program_permissions(self):
        for key in sorted(APPROVALS):
            with self.subTest(key=key):
                self.reject_edit(key, True)
        self.reject_edit("status", "APPROVED")
        self.reject_edit("approval_ref", "fabricated.json")

    def test_negative_geometry_calibration_and_training_data(self):
        edits = {
            "geometry.physical_calibration_status": "PHYSICAL_QUALIFIED",
            "geometry.approved_rotation_ref": "M1",
            "geometry.rotation_source": "TRAIN_FIT_M1",
            "geometry.scale": 0.9,
            "geometry.per_recording_rotation": True,
            "geometry.per_side_recenter": True,
            "geometry.force_home_palm_direction": True,
            "geometry.geometry_from_commands_or_feedback": True,
            "geometry.robot_feedback_anchor_role": "USE_FEEDBACK_AS_WHOLE_INTENT",
            "geometry.partition_restarts_native_session": True,
            "geometry.anchor_policy": "FIRST_VALID_FRAME",
            "geometry.ineligible_observed_start": "USE_FIRST_VALID_FRAME",
            "geometry.historical_labels": ["M0_TARGET"],
            "future_pilot.fit_on_validation": True,
            "future_pilot.roots": 32,
            "cohort.Validation.planned_ticks": 10000,
        }
        for key, value in edits.items():
            with self.subTest(key=key):
                self.reject_edit(key, value)
        self.reject_edit("geometry.scale", True)

    def test_negative_denominators_and_tracking_credit(self):
        self.reject_edit("coverage.active_percent", 84)
        self.reject_edit("coverage.history_percent", 69)
        self.reject_edit("coverage.active_denominator", "ARRIVED_AT_START")
        self.reject_edit("coverage.new_numerator_approved", True)
        self.reject_edit("coverage.count_additional_cross_partition_history", True)
        exclusions = self.proposal["coverage"]["active_exclusions"]
        for failed_case in exclusions:
            with self.subTest(credit=failed_case):
                self.reject_edit("coverage.active_exclusions", [key for key in exclusions if key != failed_case])


class SyntheticTransitionControls(unittest.TestCase):
    def test_all_adjacent_edges_with_cumulative_fixture_receipts(self):
        receipts = set()
        for source, target in zip(PHASES, PHASES[1:]):
            receipts.update(RECEIPTS_AT_ENTRY.get(target, set()))
            with self.subTest(target=target):
                self.assertEqual(synthetic_stage_step(source, target, receipts), target)

    def test_each_required_receipt_is_independently_necessary(self):
        receipts = set()
        for source, target in zip(PHASES, PHASES[1:]):
            receipts.update(RECEIPTS_AT_ENTRY.get(target, set()))
            for required in sorted(receipts):
                with self.subTest(target=target, missing=required), self.assertRaisesRegex(ValueError, required):
                    synthetic_stage_step(source, target, receipts - {required})

    def test_reject_phase_jump_training_and_hardware_even_with_fixtures(self):
        receipts = set().union(*RECEIPTS_AT_ENTRY.values())
        for target in ("PILOT_VALIDATION", "FULL_RMB_BUILD", "WORLD_MODEL_TRAINING", "REAL_ROBOT"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                synthetic_stage_step("DCR_PENDING_HUMAN_REVIEW", target, receipts)


class SyntheticAnchorControls(unittest.TestCase):
    @staticmethod
    def stable_terminal(eligible, generation, cutoff, observed_start=None):
        """Synthetic three-sample stability support, not the proposed 11-tick estimator."""
        if observed_start is not None and observed_start > cutoff:
            return "PENDING", None
        candidates = [observed_start] if observed_start is not None else range(2, min(cutoff, 5) + 1)
        for terminal in candidates:
            first = terminal - 2
            if first >= 0 and all(eligible[first:terminal + 1]) and len(set(generation[first:terminal + 1])) == 1:
                return ("RECORDED_START" if observed_start is not None else "RECONSTRUCTED_START"), terminal
        return "START_UNQUALIFIED", None

    def test_first_complete_trailing_window_and_future_suffix(self):
        generations = (0,) * 8
        prefix = (True, False, True, True, True)
        expected = ("RECONSTRUCTED_START", 4)
        for suffix in ((False, False, False), (True, True, True)):
            self.assertEqual(self.stable_terminal(prefix + suffix, generations, 4), expected)
        self.assertEqual(self.stable_terminal(prefix, generations, 3), ("START_UNQUALIFIED", None))

    def test_explicit_invalid_start_cannot_be_replaced_by_later_valid_samples(self):
        eligible = (True, False, True, True, True)
        generations = (0,) * 5
        self.assertEqual(self.stable_terminal(eligible, generations, 4, 2), ("START_UNQUALIFIED", None))
        self.assertEqual(self.stable_terminal(eligible, generations, 4, 4), ("RECORDED_START", 4))
        self.assertEqual(self.stable_terminal(eligible, (0, 0, 0, 1, 1), 4), ("START_UNQUALIFIED", None))


class RigidGeometryControls(unittest.TestCase):
    def setUp(self):
        self.rotation = ((1, 0, 0), (0, 0, -1), (0, 1, 0))
        self.human = ((0.2, 1.1, 0.12), (0.75, 1.15, -0.14))
        self.home = ((0.45, 0.40, 0.51), (0.48, -0.37, 0.59))
        self.anchor = midpoint(*self.home)
        self.translation = difference(self.anchor, rotate(self.rotation, midpoint(*self.human)))

    def mapped(self, point):
        return add(rotate(self.rotation, point), self.translation)

    def test_midpoint_gap_and_left_to_right_vector(self):
        left, right = map(self.mapped, self.human)
        self.assertLess(math.dist(midpoint(left, right), self.anchor), 1e-12)
        self.assertAlmostEqual(math.dist(left, right), math.dist(*self.human), places=12)
        expected_vector = rotate(self.rotation, difference(self.human[1], self.human[0]))
        self.assertLess(math.dist(difference(right, left), expected_vector), 1e-12)
        self.assertGreater(math.dist(difference(left, right), expected_vector), 1e-3)
        self.assertGreater(abs(math.dist(*self.home) - math.dist(*self.human)), 1e-3)

    def test_up_down_inward_and_different_palms(self):
        human_pairs = (
            ((0, 1, 0), (0, 1, 0), (0, 0, 1), (0, 0, 1)),
            ((0, -1, 0), (0, -1, 0), (0, 0, -1), (0, 0, -1)),
            ((1, 0, 0), (-1, 0, 0), (1, 0, 0), (-1, 0, 0)),
            ((0, 1, 0), (0, 0, -1), (0, 0, 1), (0, 1, 0)),
        )
        for left, right, expected_left, expected_right in human_pairs:
            with self.subTest(left=left, right=right):
                mapped_left, mapped_right = (rotate(self.rotation, normal) for normal in (left, right))
                self.assertEqual((mapped_left, mapped_right), (expected_left, expected_right))
                self.assertEqual(dot(mapped_left, mapped_right), dot(left, right))
        self.assertNotEqual(rotate(self.rotation, (0, 1, 0)), (0, 0, -1))

    def test_relative_pose_right_multiply_and_reflection_control(self):
        anatomical = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
        mapped_anatomical_columns = tuple(
            rotate(self.rotation, column) for column in transpose(anatomical)
        )
        offset = (0.0, 0.03, -0.07)
        installed = add(self.human[0], rotate(anatomical, offset))
        expected_tcp = add(self.mapped(self.human[0]), rotate(transpose(mapped_anatomical_columns), offset))
        self.assertLess(math.dist(self.mapped(installed), expected_tcp), 1e-12)
        self.assertGreater(math.dist(self.mapped(installed), add(self.mapped(self.human[0]), offset)), 1e-3)
        relative_human = rotate(transpose(anatomical), difference(self.human[1], self.human[0]))
        relative_mapped = rotate(
            mapped_anatomical_columns, difference(self.mapped(self.human[1]), self.mapped(self.human[0]))
        )
        self.assertLess(math.dist(relative_mapped, relative_human), 1e-12)
        columns = transpose(self.rotation)
        self.assertEqual(dot(cross(columns[0], columns[1]), columns[2]), 1)
        self.assertEqual(dot(cross(columns[0], columns[1]), tuple(-x for x in columns[2])), -1)


if __name__ == "__main__":
    unittest.main()
