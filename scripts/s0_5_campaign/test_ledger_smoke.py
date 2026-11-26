from importlib.util import find_spec
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

imports = {"fcntl": ModuleType("fcntl")} if find_spec("fcntl") is None else {}
with patch.dict(sys.modules, imports):
    from ledger import record_dcr008_servo, record_physics_smoke


class PhysicsSmokeBudgetTest(unittest.TestCase):
    accounts = ("train_precursor", "train_static_bucket", "train_primary_total", "train_total")

    def setUp(self):
        self.state = {
            "execution_authorized": True, "dcr_review": "TECHNICAL_PASS",
            "execution_config_sha256": "b" * 64,
            "physical_evidence": {"synthetic_smoke_actual_transitions": 272},
            "resource_usage": dict.fromkeys(self.accounts, 272),
            "remaining_budget": dict.fromkeys(self.accounts, 30000),
            "stages": {"V1-B": "BLOCKED", "V1-C0": "NOT_STARTED"},
        }

    def reserve(self, count=136):
        record_physics_smoke(self.state, {
            "action": "reserve", "run_id": "V1-B-fixture-1", "reserved": count,
        })

    def test_preserves_state_and_charges_conservative_bound_once(self):
        self.reserve()
        with self.assertRaises(ValueError):
            self.reserve()
        record_physics_smoke(self.state, {
            "action": "complete", "run_id": "V1-B-fixture-1", "exit": 135,
            "actual_physical_transitions": None,
        })
        self.assertEqual(self.state["resource_usage"]["train_precursor"], 408)
        self.assertEqual(self.state["b_physical_smoke_budget"]["previous_physical_transitions"], 408)
        self.assertEqual(self.state["stages"], {"V1-B": "BLOCKED", "V1-C0": "NOT_STARTED"})

    def test_includes_supervised_smoke_usage_in_stage_cap(self):
        self.state["c0_supervised_runs"] = {"other": {
            "mode": "b_smoke", "status": "FAILED", "charged_physical_transitions": 2499,
        }}
        with self.assertRaises(ValueError):
            self.reserve(23000)
        self.assertEqual(self.state["resource_usage"]["train_total"], 272)

    def test_candidate_servo_consumes_shared_evaluation_only_and_remains_bounded(self):
        self.state["resource_usage"]["eval_total"] = 0
        self.state["remaining_budget"]["eval_total"] = 10000
        attempt = {"action": "reserve", "run_id": "DCR-008-servo-test", "reserved": 6400,
                   "scope": "DCR008_ACTUAL_STATIC_MUJOCO_WITNESS_ONLY"}
        record_dcr008_servo(self.state, attempt)
        with self.assertRaises(ValueError):
            record_dcr008_servo(self.state, dict(attempt, run_id="DCR-008-servo-repeat", reserved=200))
        record_dcr008_servo(self.state, {"action": "complete", "run_id": attempt["run_id"],
                                         "exit": 0, "actual": 6200})
        self.assertEqual(self.state["resource_usage"]["eval_total"], 6400)
        self.assertEqual(self.state["resource_usage"]["train_total"], 272)
        self.assertEqual(self.state["stages"]["V1-C0"], "NOT_STARTED")
