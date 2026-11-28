from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import replay_dcr009_guarded_diagnostic as replay


class WristFirstFingersTest(unittest.TestCase):
    def exercise(self, *, valid_wrists=True, wrong_ring_stage=False):
        with TemporaryDirectory() as temporary:
            marker = Path(temporary) / "finger-stage"
            marker.write_bytes(b"0 0\n")
            clock = SimpleNamespace(ns=1_000_000_000)
            history = []
            observations = []

            class Replay:
                args = SimpleNamespace(finger_stage_marker=marker)
                start_ready = False
                node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=clock.ns)))

                def write(self, kind, **values):
                    history.append((kind, values))

                def wait(self, seconds):
                    clock.ns += round(seconds * 1e9)

                def send(self, _):
                    stage, deadline = map(int, marker.read_text(encoding="ascii").split())
                    if stage:
                        self.require_lease(deadline)
                    observations.append(stage)

                def require_lease(self, deadline):
                    if not 0 < deadline - clock.ns <= replay.FRESH_NS:
                        raise AssertionError("Motion sent beyond an observed wrist lease")

                def require_no_hard_fault(self):
                    pass

                def measure(self):
                    stage = int(marker.read_text(encoding="ascii").split()[0])
                    if wrong_ring_stage and stage == 1:
                        stage = 2
                    readbacks = {}
                    for side in ("left", "right"):
                        positions = {f"{side[0]}_{finger}{index}": .005
                                     for finger in ("THJ", "FFJ", "MFJ", "RFJ", "LFJ") for index in range(4)}
                        readbacks[side] = {"stage": stage, "measured_error_rad": positions, "issued_error_rad": positions}
                    return {"arms_accepted": True, "wrist_error_m_deg": [[.005 if valid_wrists else .020, 3], [.008, 3]],
                            "measured_speed_rad_s": .2 if 1 <= stage <= 10 else .05,
                            "feedback_header_ns": clock.ns, "hand_error_to_issued_rad": [.02, .03],
                            "hand_error_to_unsegmented_wuji_rad": [.02, .03], "finger_stage_readbacks": readbacks}

            ros_stub = SimpleNamespace(HUMAN={}, deserialize_message=None)
            with (patch.object(replay, "replay_class", return_value=(Replay, None)),
                  patch.dict(sys.modules, {"nominal_live_replay": ros_stub}),
                  patch.object(replay.time, "monotonic_ns", side_effect=lambda: clock.ns)):
                cls, _ = replay.diagnostic_class()
                runner = cls()
                runner.args.start_only_profile = "segmented"
                runner.measure = lambda: Replay.measure(runner)
                if valid_wrists and not wrong_ring_stage:
                    runner.wrist_then_fingers({"packet": None})
                else:
                    with self.assertRaises(replay.StartMiss):
                        runner.wrist_then_fingers({"packet": None})
            self.assertEqual(marker.read_bytes(), b"0 0\n")
            return history, observations

    def test_wrist_then_right_ring_then_final_with_exact_stages(self):
        history, observations = self.exercise()
        self.assertEqual([value["finger_stage"] for kind, value in history if kind == "finger_stage_dwell"], list(range(12)))
        self.assertEqual(next(value["finger"] for kind, value in history
                              if kind == "finger_stage_begin" and value["finger_stage"] == 1), ("right", "RFJ"))
        self.assertEqual(history[-1][1]["status"], "START_REACHED")
        self.assertEqual(sorted(set(observations)), list(range(12)))

    def test_no_hand_if_wrist_dwell_never_qualifies(self):
        history, observations = self.exercise(valid_wrists=False)
        self.assertEqual(set(observations), {0})
        self.assertEqual(history[-1][1]["terminal_finger_stage"], 0)

    def test_wrong_native_finger_stage_cannot_finish_ring(self):
        history, observations = self.exercise(wrong_ring_stage=True)
        self.assertEqual([value["finger_stage"] for kind, value in history if kind == "finger_stage_dwell"], [0])
        self.assertEqual(set(observations), {0, 1})
        self.assertEqual(history[-1][1]["terminal_finger_stage"], 1)


if __name__ == "__main__":
    unittest.main()
