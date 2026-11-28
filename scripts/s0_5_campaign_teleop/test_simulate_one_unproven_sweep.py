"""Scope and retention checks for the single MuJoCo admission exception."""

from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest

import swept_live_guard as base
from simulate_one_unproven_sweep import SinglePassGuard


class UnprovenSweepTest(unittest.TestCase):
    def exercise(self, stages, groups, statuses):
        with TemporaryDirectory() as directory:
            stage = Path(directory) / "stage"
            guard = SinglePassGuard.__new__(SinglePassGuard)
            guard.stage, guard.archive, guard.used = stage, StringIO(), False
            guard.sequence, guard.issued_baselines = 0, ()
            guard._legal_box = lambda low, high: True
            guard.envelopes = [SimpleNamespace(prove=lambda low, high, deadline, expected=reason: expected)
                               for reason in statuses]
            outcomes = []
            for number, (phase, group) in enumerate(zip(stages, groups, strict=True), start=1):
                stage.write_bytes(phase)
                packet = base.REQUEST.pack(base.MAGIC, number, group, *([0.] * 162))
                outcomes.append(guard.evaluate(packet, time.monotonic_ns() + 1_000_000_000)[1])
            return outcomes, guard.archive.getvalue().splitlines()

    def test_first_arm_unproven_only_after_start(self):
        verdicts, records = self.exercise([b"PREPOSITION\n", b"HISTORY\n", b"HISTORY\n", b"HISTORY\n"], [0, 1, 0, 0],
                                          [base.Reason.SWEEP_UNPROVEN, base.Reason.SWEEP_UNPROVEN])
        self.assertEqual(verdicts, [base.Reason.SWEEP_UNPROVEN, base.Reason.SWEEP_UNPROVEN,
                                    base.Reason.PERMIT, base.Reason.SWEEP_UNPROVEN])
        self.assertEqual(len(records), 1)

    def test_collision_in_second_model_keeps_rejection(self):
        verdicts, records = self.exercise([b"HISTORY\n"], [0],
                                          [base.Reason.SWEEP_UNPROVEN, base.Reason.COLLISION])
        self.assertEqual(verdicts, [base.Reason.COLLISION])
        self.assertFalse(records)


if __name__ == "__main__":
    unittest.main()
