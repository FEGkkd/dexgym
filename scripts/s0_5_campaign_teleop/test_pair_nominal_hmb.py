import gzip
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from pair_nominal_hmb import PastMeasured, signed_feedback_names, trace_evidence
from summarize_nominal_replay import ACTIVE, FEEDBACK, HISTORICAL


class PairNominalHmbTest(unittest.TestCase):
    def test_exact_driver_aliases_map_to_signed_order_and_reject_mismatch(self):
        signed_names = tuple(f"Joint{i}_{side}" for side in ("L", "R") for i in range(1, 8))
        signed_names += tuple(f"{side}_physical_joint_{i}" for side in ("l", "r") for i in range(20))
        hands = tuple(f"{side}_{finger}{i}" for side in ("l", "r")
                      for finger in ("THJ", "FFJ", "MFJ", "RFJ", "LFJ") for i in range(4))
        robot = SimpleNamespace(names=signed_names,
                                actuators=[SimpleNamespace(name=name) for name in signed_names[:14] + hands])
        aliases = signed_feedback_names(robot)
        self.assertEqual((aliases[0], aliases[7], aliases[14], aliases[53]),
                         ("left_j1", "right_j1", "l_THJ0", "r_LFJ3"))
        scrambled = {aliases[i]: i for i in reversed(range(54))}
        malformed = dict(scrambled)
        malformed["wrong_driver_alias"] = malformed.pop("r_LFJ3")
        candidates = {
            "timebase": {"origin_schedule_ns": 100, "origin_mono_ns": 1000,
                         "origin_log_ns": 2000, "epoch_delta_ns": 3000},
            "last_source_schedule": 120, "segments": [(995, 1)],
            "frames": [(1002, 5001, 1, scrambled), (1004, 5003, 1, malformed)],
        }
        alignment = PastMeasured(candidates, signed_names, 20, aliases)
        status, q, _, _ = alignment.get(102)
        self.assertEqual(status, "COHERENT_MEASURED")
        self.assertEqual(q.tolist(), list(range(54)))
        self.assertEqual(alignment.get(104)[0], "MEASURED_JOINT_NAMES_DIFFER_FROM_SIGNED_MODEL")
        robot.actuators[-1].name = hands[-2]
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            signed_feedback_names(robot)

    def test_only_complete_past_feedback_and_newest_eligible_measurement(self):
        names = tuple(f"joint_{i}" for i in range(54))
        candidates = {
            "timebase": {"origin_schedule_ns": 100, "origin_mono_ns": 1000, "origin_log_ns": 2000, "epoch_delta_ns": 3000},
            "last_source_schedule": 160,
            "frames": [
                (1002, 5001, 1, dict.fromkeys(names, .1)),
                (1004, 5012, 1, dict.fromkeys(names, .2)),
                (1014, 5010, 1, dict.fromkeys(names, .3)),
            ],
            "segments": [(995, 1), (1030, 0), (1040, 2)],
        }
        alignment = PastMeasured(candidates, names, 20)
        self.assertEqual(alignment.get(99)[0], "NOT_RUN_OUTSIDE_REPLAYED_INPUT_TIMELINE")
        self.assertEqual(alignment.get(101)[0], "NO_COMPLETE_MEASURED_STATE_BEFORE_CUTOFF")
        status, q, age, arrival_age = alignment.get(110)
        self.assertEqual(status, "COHERENT_MEASURED")
        self.assertEqual((q[0], age, arrival_age), (.1, 9, 8))
        self.assertEqual(alignment.get(114)[1][0], .2)
        self.assertEqual(alignment.get(130)[0], "NOT_ACTIVE_AT_PLANNED_CUTOFF")
        self.assertEqual(alignment.get(145)[0], "NO_COMPLETE_MEASURED_STATE_BEFORE_CUTOFF")

    def test_trace_sets_group_completion_to_latest_callback(self):
        base = {"stage": HISTORICAL, "origin_schedule_ns": 100, "origin_mono_ns": 1000,
                "origin_log_ns": 2000, "epoch_delta_ns": 3000}
        rows = [{"kind": "model", "stage": "setup", "mono_ns": 980, "path": "/example"},
                {"kind": "timebase", "mono_ns": 1000, **base}]
        for i, (topic, (name, selected)) in enumerate(ACTIVE.items(), 1):
            rows.append({"kind": "output", "stage": HISTORICAL, "mono_ns": 1000 + i,
                         "topic": topic, "value": {name: selected}})
        rows.append({"kind": "input", "mono_ns": 1003, "stage": HISTORICAL, "original_schedule_ns": 103})
        for i, (topic, count) in enumerate(FEEDBACK.items(), 4):
            fields = {"name": [f"{topic}_{j}" for j in range(count)], "position": [0.] * count,
                      "header": {"stamp": {"sec": 1, "nanosec": 17}}}
            if topic.endswith("/state"):
                fields.update(connected=True, position_valid=[True] * count)
            rows.append({"kind": "output", "stage": HISTORICAL, "mono_ns": 1000 + i, "topic": topic, "value": fields})
        with TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = trace_evidence(trace)
            compressed = trace.with_suffix(".jsonl.gz")
            compressed.write_bytes(gzip.compress(trace.read_bytes()))
            self.assertEqual(trace_evidence(compressed), result)
        frame, = result["frames"]
        self.assertEqual(frame[0], 1007)
        self.assertEqual(len(frame[3]), 54)


if __name__ == "__main__":
    unittest.main()
