"""Exercise C0 accounting with an isolated ledger and a Python fixture child."""

from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import supervise_c0 as supervisor


CONFIG_SHA = "c" * 64
CONFIG = {
    "budget": {"elapsed_seconds": 1000},
    "authorization": {"scope": ["V1-A", "V1-B", "V1-C0"]},
}
RUN_ID = "test-run"
GATES = {stage: f"{stage} independently reviewed test receipt".encode() for stage in ("V1-A", "V1-B")}


class FixtureLedger:
    def __init__(self, cap=1000):
        accounts = (*supervisor.PHYSICAL_ACCOUNTS, "gpu_seconds")
        self.state = {
            "execution_authorized": True,
            "authorization_sha256": supervisor.ATTACHMENT_SHA,
            "design_sha256": supervisor.FROZEN_SHA,
            "dcr_review": "TECHNICAL_PASS",
            "execution_config_sha256": CONFIG_SHA,
            "resource_usage": dict.fromkeys(accounts, 100),
            "available_budgets": dict.fromkeys(accounts, cap),
            "remaining_budget": dict.fromkeys(accounts, cap - 100),
            "runs": [],
            "stages": {"V1-A": "TECHNICAL_PASS", "V1-B": "TECHNICAL_PASS", "V1-C0": "NOT_STARTED"},
            "stage_gate_evidence": {
                stage: {"receipt_sha256": supervisor.sha(contents)} for stage, contents in GATES.items()
            },
            "started_at": self.utc(),
        }
        self.events = []

    def lock(self, _root):
        return nullcontext()

    def event_history(self, _path):
        return [{"state": self.state}]

    def append(self, _root, _state, event, _run):
        self.events.append(event)

    @staticmethod
    def utc():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def atomic(path, contents):
        path.write_bytes(contents)

    @staticmethod
    def encoded(record):
        return json.dumps(record).encode()

    @staticmethod
    def read_json(path):
        return json.loads(path.read_bytes())


def args(root, *, smoke=False):
    options = SimpleNamespace(run_id=RUN_ID, environments=8 if smoke else 2, warmup=10,
                              iterations=1, timeout_seconds=15, device="cpu", smoke=smoke)
    if not smoke:
        for stage, contents in GATES.items():
            path = root / stage
            path.write_bytes(contents)
            option = "g_" + stage[-1].lower()
            setattr(options, option, path)
            setattr(options, option + "_sha", supervisor.sha(contents))
    return options


def fit_report(wm_gate="NOT_QUALIFIED", *, smoke=False):
    return {
        "physical_transitions": 22, "design_freeze_sha256": supervisor.FROZEN_SHA,
        "expected_hard_upper_bound": 96 if smoke else 24, "actor_updates": 0, "critic_updates": 0,
        "learner_iterations": 1, "environments": 8 if smoke else 2, "wm_gate": wm_gate,
        "resume_class": "RESUME_REFILL", "checkpoint_sha256": "a" * 64,
        "scope": "PHYSICAL_SYNTHETIC_PREACTOR_SMOKE" if smoke else "C0_WORLD_MODEL_FITTING_ONLY",
        "state": "SMOKE_ONLY" if smoke else "FIT_COMPLETE_REQUIRES_INDEPENDENT_QUALIFICATION",
        "actor_locked": True, "critic_locked": True,
        **({} if smoke else {f"g_{stage[-1].lower()}_receipt_sha256": supervisor.sha(contents)
                            for stage, contents in GATES.items()}),
    }


CHILD = """\
import os
from pathlib import Path
import sys

output = Path(sys.argv[1])
output.mkdir()
(output / "metrics.jsonl").write_text('{"physical_transitions": 20}\\n', encoding="utf-8")
(output / "fit_report.json").write_text(sys.argv[2], encoding="utf-8")
print(os.getpid(), flush=True)
sys.exit(int(sys.argv[3]))
"""


class SuperviseC0Tests(unittest.TestCase):
    def test_actual_pid_exit_and_idempotent_recovery(self):
        cases = [(0, "EXITED", 22, "READY_FOR_REVIEW"), (7, "FAILED", 24, "FAILED")]
        for code, status, charged, stage in cases:
            with self.subTest(exit_code=code), TemporaryDirectory() as directory:
                root = Path(directory)
                ledger = FixtureLedger()
                run_dir = root / "runs" / RUN_ID
                command = [sys.executable, "-c", CHILD, str(run_dir / "learner"),
                           json.dumps(fit_report()), str(code)]
                with patch.object(supervisor, "make_command", return_value=command):
                    result = supervisor.launch(ledger, root, CONFIG, CONFIG_SHA, args(root))

                pid = int((run_dir / "process.log").read_text())
                receipt = ledger.read_json(run_dir / "runner.exit.json")
                self.assertGreater(pid, 0)
                self.assertEqual(result["child_pid"], pid)
                self.assertEqual(ledger.state["runs"][0]["pid"], pid)
                self.assertEqual((result["returncode"], receipt["exit_code"]), (code, code))
                self.assertEqual((result["status"], result["charged_physical_transitions"]), (status, charged))
                self.assertEqual(ledger.state["stages"]["V1-C0"], stage)
                self.assertEqual(len(ledger.events), 3)
                self.assertEqual(supervisor.reconcile(ledger, root, RUN_ID, record=True), result)
                self.assertEqual(ledger.state["resource_usage"]["train_total"], 100 + charged)
                self.assertEqual(len(ledger.events), 3)

    def test_successful_exit_cannot_supply_its_own_gate(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "fit_report.json").write_text(json.dumps(fit_report("TECHNICAL_PASS")), encoding="utf-8")
            receipts = {f"g_{stage[-1].lower()}_receipt_sha256": supervisor.sha(contents)
                        for stage, contents in GATES.items()}
            result = supervisor.read_outcome(output, 0, 24, 1, 2, mode="preactor", gate_receipts=receipts)
        self.assertEqual((result["status"], result["charged_physical_transitions"]), ("FAILED", 24))

    def test_overbudget_rejected_before_process_launch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = FixtureLedger(cap=123)
            with (
                patch.object(supervisor, "make_command", return_value=["unused"]),
                patch.object(supervisor.subprocess, "check_output", return_value=CONFIG_SHA),
                patch.object(supervisor.subprocess, "Popen") as popen,
                self.assertRaisesRegex(ValueError, "insufficient"),
            ):
                supervisor.launch(ledger, root, CONFIG, CONFIG_SHA, args(root))
            popen.assert_not_called()
            self.assertFalse((root / "runs").exists())
            self.assertEqual(ledger.events, [])
            self.assertEqual(ledger.state["resource_usage"]["train_total"], 100)

    def test_formal_c0_requires_two_passes_and_their_registered_bytes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = FixtureLedger()
            options = args(root)
            for stage in GATES:
                with self.subTest(stage=stage):
                    ledger.state["stages"][stage] = "READY_FOR_REVIEW"
                    with self.assertRaisesRegex(ValueError, f"{stage} technical pass"):
                        supervisor.check_stage_scope(ledger.state, CONFIG, CONFIG_SHA, options, 24)
                    ledger.state["stages"][stage] = "TECHNICAL_PASS"
            options.g_b.write_bytes(b"altered receipt")
            with self.assertRaisesRegex(ValueError, "exact registered V1-B receipt"):
                supervisor.check_stage_scope(ledger.state, CONFIG, CONFIG_SHA, options, 24)

    def test_b_smoke_uses_b_scope_and_leaves_both_gates_pending(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = FixtureLedger()
            ledger.state["stages"].update({"V1-A": "NOT_STARTED", "V1-B": "NOT_STARTED"})
            ledger.state["b_physical_smoke_budget"] = {
                "execution_config_sha256": CONFIG_SHA, "previous_physical_transitions": 100,
            }
            command = [sys.executable, "-c", CHILD, str(root / "runs" / RUN_ID / "learner"),
                       json.dumps(fit_report(smoke=True)), "0"]
            with patch.object(supervisor, "make_command", return_value=command):
                result = supervisor.launch(ledger, root, CONFIG, CONFIG_SHA, args(root, smoke=True))
            self.assertEqual(result["status"], "EXITED")
            self.assertEqual(ledger.state["runs"][-1]["stage"], "V1-B")
            self.assertEqual(ledger.state["stages"], dict.fromkeys(("V1-A", "V1-B", "V1-C0"), "NOT_STARTED"))
            self.assertEqual(ledger.state["authorization_sha256"], supervisor.ATTACHMENT_SHA)
            self.assertEqual(ledger.state["resource_usage"]["train_total"], 122)

    def test_b_smoke_rejects_missing_scope_drift_and_cumulative_overrun(self):
        ledger = FixtureLedger(cap=100_000)
        state = ledger.state
        options = args(None, smoke=True)
        with self.assertRaisesRegex(ValueError, "frozen campaign authorization"):
            supervisor.check_stage_scope(state, {"authorization": {"scope": ["V1-C0"]}}, CONFIG_SHA, options, 96)
        with self.assertRaisesRegex(ValueError, "bound ledger"):
            supervisor.check_stage_scope(state, CONFIG, CONFIG_SHA, options, 96)
        state["b_physical_smoke_budget"] = {
            "execution_config_sha256": CONFIG_SHA, "previous_physical_transitions": 24_900,
        }
        state["c0_supervised_runs"] = {
            "failed-smoke": {"mode": supervisor.B_SMOKE_MODE, "status": "FAILED", "charged_physical_transitions": 5},
        }
        state["resource_usage"].update(dict.fromkeys(supervisor.PHYSICAL_ACCOUNTS, 24_905))
        with self.assertRaisesRegex(ValueError, "cumulative 25000-transition limit"):
            supervisor.check_stage_scope(state, CONFIG, CONFIG_SHA, options, 96)


if __name__ == "__main__":
    unittest.main()
