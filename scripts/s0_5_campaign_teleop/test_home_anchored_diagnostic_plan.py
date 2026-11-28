from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from home_anchored_diagnostic_plan import CREDIT, FALLBACK, anchor_payload, choose_diagnostic, make_frames
from test_home_anchored_start_anchor import rows


class TestDiagnosticHomeAnchor(unittest.TestCase):
    def test_virtual_rotation_changes_only_the_common_r(self):
        docs = Path(__file__).resolve().parents[2] / "docs/s0-5/campaign/evidence/dcr009-stage-two"
        axis = docs / "virtual_axis_receipt.json"
        archived = np.repeat(np.eye(4)[None], 7, axis=0)
        archived[:, :3, 3] = np.arange(21).reshape(7, 3) * .02
        with TemporaryDirectory() as temp:
            path = Path(temp) / "archive.txt"
            np.savetxt(path, archived[:, :3].reshape(21, 4), fmt="%.17g")
            content = path.read_bytes()
            with patch("home_anchored_diagnostic_plan.ARCHIVED_FRAMES_SHA", sha256(content).hexdigest()):
                frames, output, receipt = make_frames(axis, path)
        np.testing.assert_array_equal(frames[0, :3, :3], [[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
        np.testing.assert_array_equal(frames[:, :3, 3], archived[:, :3, 3])
        self.assertEqual(content.splitlines()[3:], output.splitlines()[3:])
        self.assertEqual(receipt["transforms"]["sha256"], sha256(output).hexdigest())
        self.assertFalse(receipt["stage_credit"])
        self.assertFalse(receipt["anatomical_geometry_qualified"])

    def test_requires_strata_and_uses_only_necessary_fallback(self):
        data = [{"root": f"capture:{i:02}", "split": "Train", "task": f"task{i % 3}",
                 "median_registered_wrist_gap_m": .2 + i * .02, "ineligibility_reason": None,
                 "selected_anchor_status": "RECONSTRUCTED_START" if i < 5 else FALLBACK,
                 "contains_start_stop_or_reversal": i in (2, 5, 6, 7)} for i in range(8)]
        group, cutoffs, blocking = choose_diagnostic(list(reversed(data)))
        self.assertIsNone(blocking)
        self.assertEqual(len(group), 6)
        self.assertEqual(sum(r["selected_anchor_status"] == FALLBACK for r in group), 1)
        self.assertEqual({r["gap_stratum"] for r in group}, {"lower_quartile", "middle", "upper_quartile"})
        self.assertEqual({r["task"] for r in group}, {"task0", "task1", "task2"})
        self.assertLess(cutoffs[0], cutoffs[1])
        for item in data:
            item["contains_start_stop_or_reversal"] = item["root"] == "capture:02"
        self.assertEqual(choose_diagnostic(data)[0], ())
        data[0]["split"] = "Validation"
        with self.assertRaisesRegex(ValueError, "cannot use Validation"):
            choose_diagnostic(data)

    def test_fallback_anchor_does_not_turn_proxy_into_anatomy(self):
        from home_anchored_diagnostic_plan import CORE_STREAMS
        anchor = rows(1)[0]
        assets = {role: {"topic": role} for role in CORE_STREAMS}
        anchor.sources = tuple({
            "record": {"physical_message": i, "cdr_sha256": str(i), "part_id": role, "source_instance": "unit",
                       "writer_sequence": i}, "capture_bounds_ns": [0, 0], "arrival_bounds_ns": [0, 0],
            "clock_generation": "unit",
        } for i, role in enumerate(CORE_STREAMS))
        info = {"root": "capture:test", "source_manifest_sha256": "unit",
                "historical_window": {"source_cutoff_start_ns": 50}, "full_root_denominators": {"plan_ticks": 1},
                "stable_primary": {"status": "START_UNQUALIFIED"}, "stable_sensitivity": {},
                "explicit_event_status": "UNOBSERVED_IN_SIGNED_NATIVE_INPUTS", "selected_anchor_status": FALLBACK,
                "stable_window_qualified_on_registered_wrist_proxy": False, "stable_anatomical_anchor_qualified": False}
        bridge = {"transforms": {"sha256": "unit"}, "virtual_axis_receipt_sha256": "unit"}
        matrices = np.repeat(np.eye(4)[None], 7, axis=0)
        matrices[3:5, 2, 3] = [.1, .2]
        result = anchor_payload(info, anchor, assets, matrices, bridge, "unit", "unit")
        self.assertIsNone(result["anatomical_start_h"])
        self.assertEqual({key: result[key] for key in CREDIT}, CREDIT)
        np.testing.assert_array_equal(result["registered_wrist_start_h"], anchor.wrists)
        self.assertFalse(np.array_equal(result["runtime_engineering_start_h"], anchor.wrists))
        self.assertEqual(result["stable_primary"]["status"], "START_UNQUALIFIED")
        self.assertEqual(result["selected_anchor_status"], FALLBACK)


if __name__ == "__main__":
    unittest.main()
