"""Exercise fixture reads and the read-only HTTP surface."""

from http.client import HTTPConnection
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import patch

from server import CampaignReader, ThreadingHTTPServer, handler_for, runtime_state


class DashboardTests(unittest.TestCase):
    def setUp(self):
        sandbox = TemporaryDirectory()
        self.addCleanup(sandbox.cleanup)
        self.root = Path(sandbox.name)
        (self.root / "dashboard").mkdir()
        self.reader = CampaignReader(self.root)

    def test_missing_file_keeps_not_run(self):
        snapshot = self.reader.snapshot()
        self.assertEqual(snapshot["inputs"]["training"]["state"], "NOT_RUN")
        self.assertEqual(snapshot["training"], [])
        self.assertIsNone(snapshot["runtime"]["last_training_log_at"])

    def test_actual_evaluation_and_partial_jsonl(self):
        file = self.root / "dashboard" / "evaluation_metrics.jsonl"
        records = [
            {"evidence_kind": "ACTUAL_MUJOCO", "learner_iteration": 1000, "metrics": {"static_p95_mm": 4.8}},
            {"evidence_kind": "IMAGINED", "learner_iteration": 1001, "metrics": {"static_p95_mm": 3.5}},
        ]
        file.write_text("".join(json.dumps(x) + "\n" for x in records) + '{"partial":', encoding="utf-8")
        snapshot = self.reader.snapshot()
        self.assertEqual(snapshot["evaluation"], records[:1])
        self.assertEqual(snapshot["inputs"]["evaluation"]["excluded_nonphysical_records"], 1)
        self.assertEqual(snapshot["inputs"]["evaluation"]["state"], "PARTIAL")

    def test_partial_atomic_snapshot(self):
        file = self.root / "ledger.json"
        file.write_text('{"campaign_id":', encoding="utf-8")
        payload, meta = self.reader.read("ledger", time.time())
        self.assertEqual(payload, {})
        self.assertEqual(meta["state"], "UNREADABLE_OR_PARTIAL")

    def test_stale_progress_does_not_advance(self):
        now = time.time()
        ledger = {"runs": [{"pid": 54321, "status": "TRAINING", "heartbeat_at": now - 180}]}
        training = [{"learner_iteration": 500, "recorded_at": now - 181}]
        with patch("server.linux_process", return_value="OBSERVED"):
            result = runtime_state(ledger, {}, training, now)
        self.assertEqual(result["state"], "STALE")
        self.assertEqual(result["heartbeat_age_seconds"], 180)
        self.assertEqual(training[0]["learner_iteration"], 500)

    def test_reused_pid_is_not_marked_active(self):
        now = time.time()
        ledger = {"active_run": {"pid": 54321, "process_state": "TRAINING", "heartbeat_at": now}}
        with patch("server.linux_process", return_value="PID_REUSED"):
            result = runtime_state(ledger, {}, [], now)
        self.assertEqual(result["state"], "PID_REUSED")

    def test_http_loopback_only_and_read_only(self):
        with ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.reader, {"127.0.0.1"})) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                cases = [
                    ("GET", "/", "127.0.0.1", 200),
                    ("GET", "/dashboard.js", "127.0.0.1", 200),
                    ("GET", "/api/snapshot", "127.0.0.1", 200),
                    ("GET", "/api/snapshot", "rebind.example", 403),
                    ("GET", "/%2e%2e/ledger.json", "127.0.0.1", 404),
                    ("POST", "/api/snapshot", "127.0.0.1", 405),
                ]
                for method, path, host, status in cases:
                    with self.subTest(method=method, path=path, host=host):
                        connection.request(method, path, headers={"Host": host})
                        response = connection.getresponse()
                        self.assertEqual(response.status, status)
                        self.assertIn("frame-ancestors 'none'", response.getheader("Content-Security-Policy"))
                        response.read()
                self.assertFalse((self.root / "ledger.json").exists())
                connection.close()
            finally:
                server.shutdown()
                thread.join(2)


if __name__ == "__main__":
    unittest.main()
