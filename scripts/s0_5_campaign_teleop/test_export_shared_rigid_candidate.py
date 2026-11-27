import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from export_shared_rigid_candidate import ARM, MAPPING, MAX_AGE_NS, Observations, arm_report, qualified_command, read_native
from summarize_nominal_replay import HISTORICAL, PHASE, WORKFLOW


class SharedRigidCandidateTest(unittest.TestCase):
    def test_samples_require_both_past_receipt_and_past_header(self):
        stream = Observations()
        stream.add(1, 10, "on time")
        stream.add(2, 20, "future header")
        stream.add(6, 9, "late older header")
        self.assertEqual(stream.get(5, 15)[2], "on time")
        self.assertEqual(stream.get(6, 15)[2], "on time")
        self.assertEqual(stream.get(6, 20)[2], "future header")
        self.assertIsNone(stream.get(MAX_AGE_NS + 7, MAX_AGE_NS + 21))

    def test_arm_command_requires_session_deadline_and_exact_names(self):
        names = tuple(f"arm_{i}" for i in range(14))
        payload = {"sequence": 9, "session_id": "current", "joint_names": list(reversed(names)),
                   "positions": list(reversed(range(14))), "valid_until": {"sec": 0, "nanosec": 30}}
        record = 10, 20, payload
        self.assertEqual(qualified_command(record, ARM, "current", 30, names)[0].tolist(), list(range(14)))
        self.assertIsNone(qualified_command(record, ARM, "current", 31, names))
        self.assertIsNone(qualified_command(record, ARM, "prior", 29, names))
        self.assertIsNone(qualified_command(record, ARM, "current", 29, names[:-1] + ("wrong",)))

    def test_hand_command_checks_side_as_well_as_joint_names(self):
        topic = "/left_hand/command"
        names = tuple(f"left_joint_{i}" for i in range(20))
        payload = {"sequence": 9, "joint_names": list(names), "positions": [0.] * 20, "side": "right"}
        self.assertIsNone(qualified_command((10, 20, payload), topic, "current", 20, names))
        payload["side"] = "left"
        self.assertIsNotNone(qualified_command((10, 20, payload), topic, "current", 20, names))

    def test_epoch_requires_shared_session_and_rejects_earlier_command_header(self):
        def output(arrival, header, topic, **fields):
            return dict(kind="output", stage=HISTORICAL, mono_ns=arrival, topic=topic,
                        value=dict(header={"stamp": {"sec": 0, "nanosec": header}}, **fields))

        records = [
            dict(kind="timebase", stage=HISTORICAL, mono_ns=100, origin_schedule_ns=100),
            dict(kind="input", stage=HISTORICAL, mono_ns=100, original_schedule_ns=100),
            output(101, 20, PHASE, phase="TASK", session_id="prior"),
            output(102, 21, WORKFLOW, state_name="ACTIVE", session_id="current"),
            output(103, 22, ARM, sequence=1),
            output(104, 23, PHASE, phase="TASK", session_id="current"),
            output(105, 22, ARM, sequence=2),
            output(106, 24, ARM, sequence=3),
            output(107, 24, "/diagnostics", status=[{"name": "tianji/left_arm/teleop", "values": [
                {"key": "mapping_policy", "value": MAPPING},
            ]}]),
        ]
        with TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            _, _, times, epochs, streams = read_native(path)
        self.assertEqual(times, [104])
        self.assertEqual(epochs, [(1, "current", 23)])
        self.assertEqual(len(streams[1, ARM].rows), 1)
        self.assertEqual(streams[1, ARM].get(107, 24)[2]["sequence"], 3)

    def test_hold_and_failed_diagnostic_never_promote_published_command(self):
        command = np.zeros(14), 9, 100, 200
        diagnostic = {"session": "current", "sequence": "9", "q_cmd_rad": "0,0,0,0,0,0,0",
                      "mapping_policy": MAPPING, "rate_limited": "false", "solve_mode": "full_step",
                      "arm_command_accepted": "true", "ik_target_accepted": "true"}
        self.assertEqual(arm_report((101, 200, diagnostic), "current", command, 0), (1, True))
        self.assertEqual(arm_report((101, 200, dict(diagnostic, rate_limited="true")), "current", command, 0),
                         (2, True))
        self.assertEqual(arm_report((101, 200, dict(diagnostic, recovery_action="hold")), "current", command, 0),
                         (3, False))
        self.assertEqual(arm_report((101, 200, dict(diagnostic, arm_command_accepted="false")),
                                    "current", command, 0), (5, False))
        self.assertEqual(arm_report((101, 200, dict(diagnostic, sequence="8")), "current", command, 0),
                         (0, False))
        self.assertEqual(arm_report((101, 200, dict(diagnostic, session="prior")), "current", command, 0),
                         (0, False))


if __name__ == "__main__":
    unittest.main()
