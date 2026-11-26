"""Synthetic positive and negative controls. Never writes any project ledger."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import check_design as checks


class DesignControls(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.policy = checks.read_json(checks.HERE / "design_policy.json")
        costs = dict.fromkeys(self.policy["resource_budget"], 0)
        self.identities = {
            "design_sha256": "a" * 64, "evaluation_sha256": "b" * 64,
            "budget_sha256": checks.canonical_sha(self.policy["resource_budget"]),
        }
        self.campaign = {**self.identities, "campaign_id": "SYNTHETIC"}
        self.ledger = {**self.campaign, "design_approval": "synthetic-design", "execution_authorization": "synthetic-campaign",
                       "states": dict.fromkeys(checks.STAGES, "NOT_STARTED"), "gate_refs": {},
                       "consumed": costs.copy(), "retries": {}, "execution_context_ref": "synthetic-context"}
        self.action = {"kind": "START", "stage": "V1-A", "reservation": costs.copy()}
        self.context = {**self.campaign, "kind": "verified_execution_context", "active": True,
                        "input_fingerprints": dict.fromkeys(checks.STAGES, "c" * 64), "wm_current": None}
        self.trusted = {
            "synthetic-context": self.context,
            "synthetic-design": {**self.identities, "kind": "human_design_approval", "active": True,
                                 "m1_continuous_simulation": True},
            "synthetic-campaign": {**self.campaign, "kind": "human_campaign_authorization", "active": True,
                                   "scope": list(checks.STAGES)},
        }

    def check(self):
        checks.check_transition(self.policy, self.ledger, self.action, self.trusted, self.root)

    def gate(self, stage):
        parents = self.policy["stages"][stage]["gate_requires"]
        for parent in parents:
            if parent not in self.ledger["gate_refs"]:
                self.gate(parent)
        files = {}
        for name in self.policy["stages"][stage]["checks"]:
            path = self.root / f"{stage}_{name}.json"
            path.write_text('{"fixture": true}', encoding="utf-8")
            files[name] = {"path": path.name, "sha256": checks.file_sha(path)}
        ref = f"synthetic-gate-{stage}"
        self.trusted[ref] = {
            **self.campaign, "kind": "technical_gate", "active": True, "stage": stage,
            "verdict": "TECHNICAL_PASS", "exit_code": 0, "reviewer": "reviewer", "core_authors": ["implementer"],
            "artifacts": files, "input_fingerprint": self.context["input_fingerprints"][stage],
            "upstream_refs": {parent: self.ledger["gate_refs"][parent] for parent in parents},
        }
        self.ledger["states"][stage] = "TECHNICAL_PASS"
        self.ledger["gate_refs"][stage] = ref
        if stage == "V1-C0":
            self.context["wm_current"] = ref
        return self.trusted[ref]

    def test_design_only_fixture_has_no_capability(self):
        self.ledger["execution_authorization"] = None
        with self.assertRaisesRegex(ValueError, "unverified human_campaign_authorization"):
            self.check()

    def test_agent_cannot_mint_approval(self):
        self.ledger["execution_authorization"] = "forged"
        self.ledger["approval"] = {"type": "human", "active": True}
        with self.assertRaisesRegex(ValueError, "unverified human_campaign_authorization"):
            self.check()

    def test_m1_static_approval_does_not_authorize_continuous(self):
        self.trusted["synthetic-design"]["m1_continuous_simulation"] = False
        with self.assertRaisesRegex(ValueError, "M1 continuous"):
            self.check()

    def test_scoped_human_authorization_allows_a(self):
        self.check()

    def test_b_preparation_but_no_formal_gate_before_a(self):
        self.action.update(stage="V1-B")
        self.check()
        self.ledger["states"]["V1-B"] = "IN_PROGRESS"
        self.action.update(kind="PASS")
        with self.assertRaisesRegex(ValueError, "dependency V1-A"):
            self.check()

    def test_c0_requires_a_and_b(self):
        self.action.update(stage="V1-C0")
        self.gate("V1-A")
        with self.assertRaisesRegex(ValueError, "dependency V1-B"):
            self.check()
        self.gate("V1-B")
        self.check()

    def test_qualified_actor_stage(self):
        self.action.update(stage="V1-C1")
        self.gate("V1-C0")
        self.check()
        self.context["wm_current"] = None
        with self.assertRaisesRegex(ValueError, "invalidated"):
            self.check()

    def test_d_cannot_bypass_c3(self):
        self.action.update(stage="V1-D")
        self.gate("V1-C0")
        with self.assertRaisesRegex(ValueError, "dependency V1-C3"):
            self.check()

    def test_tampered_gate_metric_evidence(self):
        self.action.update(stage="V1-C0")
        gate = self.gate("V1-A")
        self.gate("V1-B")
        first = next(iter(gate["artifacts"].values()))
        (self.root / first["path"]).write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.check()

    def test_same_author_review_rejected(self):
        self.action.update(stage="V1-C1")
        self.gate("V1-C0")["reviewer"] = "implementer"
        with self.assertRaisesRegex(ValueError, "not independent"):
            self.check()

    def test_changed_scoring_threshold_hash_is_rejected(self):
        self.ledger["evaluation_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "binding differs"):
            self.check()

    def test_resource_cap_includes_reservations(self):
        self.ledger["consumed"]["gpu_seconds"] = self.policy["resource_budget"]["gpu_seconds"]
        self.action["reservation"]["gpu_seconds"] = 1
        with self.assertRaisesRegex(ValueError, "resource exhausted"):
            self.check()

    def test_resource_totals_cannot_hide_subbuckets(self):
        self.ledger["consumed"]["train_dynamic"] = 1
        with self.assertRaisesRegex(ValueError, "bucket mismatch"):
            self.check()

    def test_retry_exhaustion(self):
        self.ledger["states"]["V1-A"] = "FAILED"
        self.ledger["retries"]["V1-A"] = 2
        with self.assertRaisesRegex(ValueError, "retry cap"):
            self.check()

    def test_stale_world_model_inputs_rejected(self):
        self.action.update(stage="V1-C2")
        self.gate("V1-C1")
        self.context["input_fingerprints"]["V1-C0"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "binding differs"):
            self.check()

    def test_changed_upstream_invalidates_transitive_gate(self):
        self.action.update(stage="V1-C2")
        self.gate("V1-C1")
        self.ledger["states"]["V1-A"] = "BLOCKED"
        with self.assertRaisesRegex(ValueError, "dependency V1-A"):
            self.check()

    def test_design_dag_cannot_drop_c3(self):
        manifest = checks.read_json(checks.DOCS / "design_freeze_candidate.json")
        history = checks.read_json(checks.HERE / "historical_inputs.json")
        requirements = checks.read_json(checks.HERE / "requirement_mapping.json")
        self.policy["stages"]["V1-D"]["start_requires"] = ["V1-C0"]
        manifest["stage_dag"] = manifest["technical_gate_definitions"] = self.policy["stages"]
        with self.assertRaisesRegex(ValueError, "invalid DAG: V1-D"):
            checks.validate_design(manifest, self.policy, history, requirements)

    def test_real_deploy_needs_independent_hardware_evidence(self):
        self.action.update(kind="REAL_DEPLOY", stage=None, external_receipts={})
        with self.assertRaisesRegex(ValueError, "unverified physical_robot_base"):
            self.check()

    def test_final_test_and_formal_scoring_are_separate_gates(self):
        self.action.update(kind="FINAL_TEST", stage=None, external_receipts={})
        with self.assertRaisesRegex(ValueError, "unverified final_test_sealed"):
            self.check()
        self.action["kind"] = "FORMAL_SCORING"
        with self.assertRaisesRegex(ValueError, "unverified physical_robot_base"):
            self.check()

    def test_tampered_historical_receipt_rejected(self):
        original = b'{"scope":"historical static"}'
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            checks.verify_digest(original + b" ", checks.sha(original), "fixture receipt")

    def test_historical_phase_events_are_immutable(self):
        policy_state = checks.read_json(checks.DOCS / "phase_state.json")
        legacy = {key: value for key, value in policy_state.items() if key != "policy_first_v2"}
        checks.check_candidate_state(policy_state, legacy)
        changed = deepcopy(policy_state)
        changed["events"][0]["to"] = "ACCEPTED"
        with self.assertRaisesRegex(ValueError, "historical phase ledger changed"):
            checks.check_candidate_state(changed, legacy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
