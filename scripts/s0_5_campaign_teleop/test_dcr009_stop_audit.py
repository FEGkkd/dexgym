import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dcr009_stop_audit import (
    INCONCLUSIVE, JOINT_NAMES, PASS, STOP_SERVICE, VIOLATION, audit_streams,
)


SESSION = "s05-dcr009-stop-fixture"
FEEDBACK_NS = 1_999_000_000
COMMAND_NS = 1_999_500_000


def stamp(ns):
    return {"stamp": {"sec": ns // 1_000_000_000, "nanosec": ns % 1_000_000_000}}


def write_rows(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


def run_fixture(*, publish_after=False, observed=True, mismatch=False, missing_feedback=False, missing_barrier=False,
                wrong_generation=False, header_shift=0, hand=False):
    def output(time, topic, value):
        return {"kind": "output", "mono_ns": time, "topic": topic, "value": value}

    home = [0.] * 54
    target = home.copy()
    target[34 if hand else 0] = .001
    group = "right" if hand else "arm"
    receipts = []
    for time, side, begin in ((10, "left", 0), (20, "right", 7)):
        receipts.append(output(time, f"/{side}_arm/joint_states", {
            "name": JOINT_NAMES["arm"][begin:begin + 7], "position": [0.] * 7, "header": stamp(FEEDBACK_NS)}))
    for time, side in ((30, "left"), (40, "right")):
        if missing_feedback and side == "left":
            continue
        receipts.append(output(time, f"/{side}_hand/state", {
            "joint_names": JOINT_NAMES[side], "position": [0.] * 20, "header": stamp(FEEDBACK_NS), "side": side,
            "feedback_source": "mujoco", "source_clock": "simulation", "source_timestamp_valid": True,
            "connected": True, "online": [True] * 20, "stale": [False] * 20, "position_valid": [True] * 20}))
    receipts.extend([{"kind": "service", "mono_ns": 110, "service": STOP_SERVICE, "response": {"accepted": True}},
                     {"kind": "stop_observation_boundary", "mono_ns": 112, "ack_mono_ns": 111, "session": SESSION}])
    if observed:
        received = target[34:] if hand else target[:14]
        if mismatch:
            received[0] = .5
        topic = "/right_hand/command" if hand else "/tianji/joint_command"
        message = {"sequence": 29, "joint_names": JOINT_NAMES[group], "header": stamp(COMMAND_NS + header_shift)}
        message.update({"side": "right", "position": received} if hand else {"session_id": SESSION, "positions": received})
        receipts.append(output(120, topic, message))
    receipts.append({"kind": "pilot_stop", "mono_ns": 140, "session": SESSION, "late_observed_command_count": int(observed)})
    native = (f"[INFO] [2.000000000] [whole_body_teleop]: S05_DCR009_GUARD group={group} session={SESSION} "
              f"generation={5 if wrong_generation else 4} command=29 proof=44 published=1 reason=0 feedback_ns={FEEDBACK_NS} "
              f"command_header_ns={COMMAND_NS}\n")
    barrier = (f"[INFO] [2.001000000] [whole_body_teleop]: S05_DCR009_STOP_BARRIER session={SESSION} generation=4 "
               f"last_arm={0 if publish_after or hand else 29} last_left=0 last_right={29 if hand and not publish_after else 0} mono_ns=100\n")
    proof = {"sequence": 44, "group": 2 if hand else 0, "reason": "PERMIT", "receipt_ns": 80, "proof_complete_ns": 90,
             "issued_measured_proposed_rad": [home, home, target]}
    with TemporaryDirectory() as directory:
        log, trace, sweep = [Path(directory) / name for name in ("sim.log", "observer.jsonl", "sweep.jsonl")]
        log.write_text(native if missing_barrier else barrier + native if publish_after else native + barrier, encoding="utf-8")
        write_rows(trace, receipts)
        write_rows(sweep, [proof])
        return audit_streams(log, trace, sweep)


class StopAuditTest(unittest.TestCase):
    def test_exact_delayed_old_command_preserves_observation(self):
        for hand in (False, True):
            with self.subTest(hand=hand):
                result = run_fixture(hand=hand)
                self.assertEqual(result["status"], PASS)
                self.assertEqual(result["observer_post_boundary"][0]["classification"], "DELAYED_PRE_BARRIER")
                self.assertFalse(result["native_post_barrier_publications"])

    def test_native_post_barrier_publication_even_if_observer_misses_it(self):
        result = run_fixture(publish_after=True, observed=False)
        self.assertEqual(result["status"], VIOLATION)
        self.assertEqual(result["native_post_barrier_publications"][0]["sequence"], 29)
        self.assertFalse(result["observer_post_boundary"])
        result = run_fixture(publish_after=True)
        self.assertEqual(result["observer_post_boundary"][0]["classification"], "NEW_POST_BARRIER")

    def test_changed_payload_header_or_missing_feedback_are_unknown(self):
        for options in ({"mismatch": True}, {"header_shift": 1}, {"missing_feedback": True}, {"wrong_generation": True},
                        {"missing_barrier": True}):
            with self.subTest(options=options):
                result = run_fixture(**options)
                self.assertEqual(result["status"], INCONCLUSIVE)
                self.assertEqual(result["observer_post_boundary"][0]["classification"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
