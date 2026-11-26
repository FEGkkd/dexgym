import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

# These controls only exercise pure array and classifier routines. No HMB
# files are opened and no production file-locking capability is replaced.
needs_fcntl_stub = importlib.util.find_spec("fcntl") is None
if needs_fcntl_stub:
    sys.modules["fcntl"] = types.ModuleType("fcntl")
try:
    from dexgym.tasks.direct.wholebody_s05.hmb_reader import VERSION_FIELDS
    from dexgym.tasks.direct.wholebody_s05.v1.rmb import evaluate_teleop_candidate as evaluation
    from dexgym.tasks.direct.wholebody_s05.v1.rmb.evaluate_teleop_candidate import (
        Classification, classify, count_evidence, output_arrays, prefix_candidates, summarize, witness_offset,
    )
    from dexgym.tasks.direct.wholebody_s05.v1.rmb.schema import GEOMETRY_SHAPES, TICK_NS
finally:
    if needs_fcntl_stub:
        del sys.modules["fcntl"]


def hmb_fixture():
    count = 12
    ticks = np.arange(count, dtype=np.int64) * TICK_NS
    raw = {
        "tick_ns": ticks, "cutoff_ns": ticks.copy(), "source_index": np.zeros((count, 4, 1), dtype=np.int64),
        "capture_bounds_ns": np.zeros((count, 4, 2), dtype=np.int64),
        "arrival_bounds_ns": np.zeros((count, 4, 2), dtype=np.int64),
        "age_ns": np.zeros((count, 4), dtype=np.int64), "reason_code": np.zeros((count, 4), dtype=np.int16),
        "joint_valid": np.ones(count, dtype=bool), "safety_permission": np.ones(count, dtype=bool),
        "pose_valid": np.ones((count, 4), dtype=bool),
        "segment_id": np.r_[np.zeros(5, dtype=np.int64), -1, np.ones(6, dtype=np.int64)],
    }
    raw["joint_valid"][5] = False
    raw.update({name: np.zeros((count, 4), dtype=np.int64) for name in VERSION_FIELDS})
    return raw


class CandidateEvaluationTest(unittest.TestCase):
    def test_stationary_unsettled_is_hold_and_fallback_is_never_active(self):
        categories = classify(np.array([True, True, True, False]), np.array([False, True, False, False]),
                              np.array([False, False, True, False]), np.array([True, True, False, True]),
                              np.array([False, False, False, True]))
        self.assertEqual(categories.tolist(),
                         [Classification.HOLD, Classification.LIMITED_ACTIVE,
                          Classification.ACTIVE_TRACKED, Classification.FALLBACK])

    def test_invalid_and_unknown_rows_remain_in_the_planned_denominator(self):
        raw = hmb_fixture()
        fields = output_arrays(raw, len(raw["tick_ns"]))
        fields["row_class"][[1, 2, 3]] = Classification.ACTIVE_TRACKED
        fields["row_class"][4] = Classification.HOLD
        fields["row_class"][5] = Classification.FAILED
        fields["row_class"][6] = Classification.FALLBACK
        count = count_evidence(fields)
        totals = summarize(count, paired=False)
        self.assertEqual(count["planned"], 12)
        self.assertEqual(count["hmb_valid"], 11)
        self.assertEqual(count["candidate_active_hmb"], 3)
        self.assertEqual(count["FALLBACK"], 1)
        self.assertEqual(count["FAILED"], 7)
        self.assertEqual(totals["candidate_active_coverage_hmb"], 3 / 11)
        self.assertIsNone(totals["direct_nominal_legal_coverage_hmb"])
        self.assertFalse(totals["official_coverage_pass"])

    def test_prefix_plan_uses_signed_input_and_covers_an_available_barrier(self):
        raw = hmb_fixture()
        plan = prefix_candidates(raw, "recording:sample/000", denied=False)
        self.assertIn("ordinary", plan)
        self.assertEqual(plan["generation_barrier"], 9)
        self.assertEqual(prefix_candidates(raw, "recording:sample/000", denied=True), {})
        self.assertEqual(witness_offset("test", 64, 1), 0)

    def test_trace_restarts_strict_velocity_and_matches_mutated_future(self):
        raw = hmb_fixture()
        count = len(raw["tick_ns"])
        raw["pose"] = np.zeros((count, 4, 21, 7))
        raw["sdk_common_pose"] = np.zeros((count, 4, 21, 7))
        target = {key: np.zeros(shape) for key, shape in GEOMETRY_SHAPES.items()}
        target["wrist_xyzw"][:, 3] = 1.

        class Candidate:
            robot = types.SimpleNamespace(geometry=lambda q: types.SimpleNamespace(
                wrists=np.zeros((2, 3)), rotations=np.repeat(np.eye(3)[None], 2, axis=0),
                tips=np.zeros((2, 5, 3)),
            ))
            wrist_frames = [(np.eye(4), np.eye(4)) for _ in range(2)]
            filtered_hand = np.zeros((2, 20))

            def cut(self):
                self.previous = np.zeros(54)

            def step(self, _target, _common, _order, _index, *, tick_ns):
                phase = self.previous[0] + 1e-4
                self.previous[0] = phase
                active = np.full(4, evaluation.CandidateStatus.ACTIVE)
                arm = types.SimpleNamespace(
                    accepted=True, ik_target_accepted=True, analytic_braking=False, workspace_projected=False,
                    rate_limited=False, analytic_exact_stop_used=False,
                    pair_accepted=True, pair_hard_fault=False, safety_code=0, safety_reason="",
                    joint_candidate=np.zeros(7), joint_command=np.zeros(7), projected_axis=-1, backoff_scale=1.,
                    workspace_target=np.eye(4), step_target=np.eye(4), attempt_target=np.eye(4),
                )
                steps = [types.SimpleNamespace(
                    issue_tick_ns=tick_ns + j * evaluation.ARM_SUBSTEP_NS, q_issued=self.previous.copy(),
                    arm_branches_rad=np.zeros(2), status=active, whole_body_path_accepted=True,
                    kernel_proposed=True, hand_candidate_limit=np.zeros(40, dtype=bool),
                    hand_limit_applied=np.zeros(40, dtype=bool), native_arms=(arm, arm),
                ) for j in range(evaluation.ARM_SUBSTEPS)]
                return types.SimpleNamespace(
                    q_nominal=self.previous.copy(), qd_nominal=np.ones(54), qd_valid=True, status=active,
                    raw_hand_q_rad=np.zeros((2, 20)), raw_hand_valid=np.ones(2, dtype=bool),
                    hand_failure_reason=(None, None), native_tcp_target=np.repeat(np.eye(4)[None], 2, axis=0),
                    substeps=steps,
                )

        mapper = types.SimpleNamespace(map=lambda *_: {"mapped": target, "pinch_active": np.zeros((2, 4), dtype=bool)})
        candidate = Candidate()
        with patch.object(evaluation, "compiled_topologies", return_value=(None, None)):
            fields = evaluation.evaluate_partition(raw, {}, mapper, candidate, "synthetic", denied=False)
            check = evaluation.compare_prefix(raw, {}, mapper, candidate, "synthetic", False, fields, 9)
        self.assertEqual(check["replacement_differences"], [])
        self.assertTrue(fields["provider_qd_valid"][6])
        self.assertFalse(fields["qd_valid"][6])
        self.assertAlmostEqual(fields["qd_nominal"][7, 0], .005)
        self.assertEqual(fields["row_class"][5], Classification.FAILED)
