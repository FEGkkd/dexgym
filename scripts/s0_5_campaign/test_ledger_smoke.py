from importlib.util import find_spec
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

imports = {"fcntl": ModuleType("fcntl")} if find_spec("fcntl") is None else {}
with patch.dict(sys.modules, imports):
    from ledger import record_physics_smoke


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
            "mode": "b_smoke", "status": "FAILED", "outcome": {"charged_physical_transitions": 2499},
        }}
        with self.assertRaises(ValueError):
            self.reserve(23000)
        self.assertEqual(self.state["resource_usage"]["train_total"], 272)
