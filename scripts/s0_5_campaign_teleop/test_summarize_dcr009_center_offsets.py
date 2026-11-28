from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from summarize_dcr009_center_offsets import GROUPS, JOINT_NAMES, PREPOSITION, STEP_NS, feedback_rows, publication_audit


class CenterOffsetDiagnosticsTest(unittest.TestCase):
    def test_causal_bins_do_not_reuse_feedback_or_extend_past_fault(self):
        def measurement(receipt, header, left_m):
            return {"stage": PREPOSITION, "kind": "start_measurement", "mono_ns": receipt,
                    "value": {"active_task": True, "feedback_header_ns": header,
                              "wrist_error_m_deg": [[left_m, 3], [.04, 4]]}}

        origin = 100
        terminal = origin + 4 * STEP_NS
        trace = [measurement(origin + STEP_NS - 1, 10, .09),
                 measurement(origin + STEP_NS, 11, .08),
                 measurement(origin + STEP_NS + 1, 11, .08),
                 measurement(origin + 3 * STEP_NS, 12, .07),
                 measurement(terminal + 1, 13, 0)]
        bins, observed = feedback_rows(trace, origin, terminal)
        self.assertEqual(set(bins), {1, 3})
        self.assertEqual(bins[1]["engineering_tcp"][0], (80., 3))
        self.assertEqual(bins[3]["engineering_tcp"][0], (70., 3))
        self.assertEqual(len(observed), 3)

    def test_exact_payload_join_and_distinct_no_request_denominator(self):
        with TemporaryDirectory() as temporary:
            sim = Path(temporary) / "sim.log"
            sim.write_text("S05_DCR009_GUARD group=arm session=s generation=2 command=7 "
                           "proof=12 published=1 reason=0 feedback_ns=100\n", encoding="utf-8")
            output = {"kind": "output", "topic": "/tianji/joint_command", "value": {
                "sequence": 7, "session_id": "s", "joint_names": JOINT_NAMES["arm"], "positions": [0.] * 14}}
            reasons = {group: Counter() for group in GROUPS}
            reasons["arm"]["PERMIT"] = 1
            reasons["left"].update(SWEEP_UNPROVEN=1, GROUP_MISMATCH=6)
            proofs = {12: ("arm", [0.] * 14)}
            valid = publication_audit(sim, [output], "s", proofs, reasons, Counter())
            self.assertFalse(valid["publication_mismatches"])
            self.assertEqual(valid["groups"]["left"]["valid_guard_requests"], 1)
            self.assertEqual(valid["groups"]["left"]["permit_rate_over_valid_guard_requests"], 0)
            self.assertIsNone(valid["groups"]["right"]["permit_rate_over_valid_guard_requests"])
            output["value"]["positions"][0] = .001
            invalid = publication_audit(sim, [output, output], "s", proofs, reasons, Counter())
            self.assertEqual(invalid["publication_mismatches"], {
                "observed_topic_without_matching_session_group_and_payload": 2, "duplicate_ros_observation": 1})


if __name__ == "__main__":
    unittest.main()
