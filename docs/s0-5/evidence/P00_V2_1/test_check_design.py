"""Synthetic controls only. Artifacts and permissions live in temporary fixtures."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import check_design as checker


class SyntheticCampaign(unittest.TestCase):
    T = "1" * 64
    E = "2" * 64
    REFINED = "3" * 64
    PLAN = "4" * 64

    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.policy = checker.read_json(checker.HERE / "design_policy.json")
        self.caps = self.policy["resource_budget"]
        self.identities = {"design_sha256": "a" * 64, "evaluation_sha256": "b" * 64,
                           "budget_sha256": checker.canonical_sha(self.caps)}
        self.bindings = {**self.identities, "campaign_id": "SYNTHETIC_ONLY"}
        self.ledger = {**self.bindings, "design_approval": "design", "execution_authorization": "campaign",
                       "execution_context_ref": "context", "states": dict.fromkeys(self.policy["stages"], "NOT_STARTED"),
                       "gate_refs": {}, "retries": {}, "consumed": dict.fromkeys(self.caps, 0)}
        self.action = {"kind": "START", "stage": "V1-A", "reservation": dict.fromkeys(self.caps, 0)}
        self.store = {}
        self.trust("design", "human_design_approval", bind=self.identities,
                   m1_continuous_simulation=True, policy_sha256=checker.canonical_sha(self.policy))
        self.trust("campaign", "human_campaign_authorization", scope=list(self.policy["stages"]), authorize_egopro=True)
        inputs = {"target_sha256": "5" * 64, "asset_sha256": "6" * 64, "time_plan_sha256": "7" * 64}
        self.trust("context", "verified_execution_context", regression_lineage_audited=True,
                   training_source_identities=["teleop-training-only"], regression_lineage_closure=["direct-derivative"],
                   input_fingerprints=dict.fromkeys(self.policy["stages"], "8" * 64),
                   run_inputs={"offline": inputs, "replay": inputs.copy()},
                   wm_current="gate-V1-C0", mixture_fingerprint="9" * 64,
                   expanded_wm_current="wm-expanded", pre_actor_domain_ref="domain",
                   diagnosis_ref="diagnosis", refinement_diagnosis_ref="diagnosis",
                   gap_metric_sha256="0" * 64, qualified_teleop_candidates=[self.T, self.REFINED],
                   standalone_receipts={self.T: "standalone-T", self.E: "standalone-E", self.REFINED: "standalone-refined"})
        for policy, actor in (("T", self.T), ("E", self.E), ("refined", self.REFINED)):
            self.evidence(f"standalone-{policy}", "independent_standalone_candidate", {"checkpoint_sha256": actor,
                **dict.fromkeys(("export_parity", "trained_actor", "data_identity", "non_diagnostic",
                                  "function_pass", "causal_pass", "safety_pass", "no_bank"), True)})
        self.evidence("domain", "independent_pre_actor_domain", {
            "domain_sha256": self.PLAN, **dict.fromkeys(("frozen_before_first_actor", "static_witness_audited",
                "dynamic_servo_witness_audited", "unchanged_planned_domain"), True),
        })
        self.evidence("diagnosis", "independent_d2_diagnosis", {"interface_faults_resolved": True,
            "independent_gap_evidence": True, "egopro_supported_gap": True, "bounded_policy_remedy": True,
            "policy_t_sha256": self.T, "planned_minimum_train_transitions": 200000})
        self.evidence("wm-expanded", "independent_expanded_wm", dict.fromkeys(("excitation_54", "identity_and_holdout",
            "one_four_eight", "actual_gradients", "servo_timing", "safety_boundaries", "model_gap_pass"), True),
            bank_fingerprint="e" * 64, mixture_fingerprint="9" * 64)

    def trust(self, ref, kind, bind=None, **fields):
        self.store[ref] = {**(self.bindings if bind is None else bind), **fields, "kind": kind, "active": True}

    def artifact(self, ref, body):
        path = self.root / f"{ref}.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        return {"path": path.name, "sha256": checker.file_sha(path)}

    def evidence(self, ref, kind, data, **fields):
        self.trust(ref, kind, evidence=self.artifact(ref, data), **fields)

    def offline(self, checkpoint=None):
        return {"checkpoint_sha256": checkpoint or self.T, "domain_sha256": self.PLAN,
            **self.store["context"]["run_inputs"]["offline"],
            **dict.fromkeys(checker.FUNCTION_FLAGS + ["pre_actor_domain_frozen", "coverage_met",
                "functional_geometry", "original_timeline", "full_planned_denominator"], True),
            "aggregation": "failure_composite_nearest_rank", "static_targets": 45, "static_roots": 7,
            "dynamic_roots": 7, "dynamic_ticks": 1200, "static_support_fraction": 0.25,
            "dynamic_support_fraction": 0.12, "static_p95_mm": 4.9, "dynamic_p95_mm": 9.8,
            "static_baseline_p95_mm": 7, "static_candidate_p95_mm": 5.5,
            "dynamic_baseline_p95_mm": 14, "dynamic_candidate_p95_mm": 11,
            "full_baseline_p95_mm": 17, "full_candidate_p95_mm": 16.9, "control_failure_delta": 0,
            "hard_violations": 0}

    def replay(self, checkpoint=None):
        pairs = {}
        for row in self.policy["regression_recordings"]:
            pairs[row["id"]] = {"sha256": row["sha256"], **dict.fromkeys(checker.PAIR_FLAGS, True),
                "t1_t2_common_context": True, "t1_t2_only_residual_changed": True,
                "cold_starts": {"T0": 2, "T1": 2, "T2": 2}, "full_unique_ticks": 700,
                "common_unique_ticks": 500, "nonoverlap_windows_at_least_25_ticks": 2, "hard_violations": 0,
                "late_issues": 0, "full_baseline_p95_mm": 12, "full_candidate_p95_mm": 11.4,
                "control_failure_delta": 0, "illustrative_individual_gain": 0.05}
        return {"checkpoint_sha256": checkpoint or self.T, **self.store["context"]["run_inputs"]["replay"],
            **dict.fromkeys(("actual_existing_graph", "actual_new_feedback", "isolated_ros_domain", "nominal_online",
                            "assets_and_initial_state_verified", "legacy_policy_disabled", "independent_mapped_scorer"), True),
            **dict.fromkeys(("recorded_robot_commands_injected", "recorded_feedback_injected", "cached_rmb_injected",
                            "real_driver_active", "mid_run_reset"), False),
            "aggregation": "failure_composite_nearest_rank", "combined_baseline_p95_mm": 15,
            "combined_candidate_p95_mm": 12.5, "bags": pairs}

    def retention(self):
        return {"checkpoint_sha256": self.E, "policy_t_sha256": self.T,
            **dict.fromkeys(("target_gap_gain", "teleop_static_retained", "teleop_dynamic_retained",
                "functional_safety_retained", "both_mcap_retained", "separate_source_metrics"), True),
            "offline": self.offline(self.E), "replay": self.replay(self.E),
            "preregistered_gap_metric_sha256": "0" * 64,
            "gap": {"metric": "tail_tip_mm", "metric_sha256": "0" * 64, "policy_t_p95": 20, "policy_e_p95": 16},
            "retention": {key: {"policy_t_p95_mm": 10, "policy_e_p95_mm": 10.2}
                          for key in ("offline_static", "offline_dynamic", "offline_full",
                                      *(b["id"] for b in self.policy["regression_recordings"]))}}

    def metrics(self, stage):
        if stage == "V1-D0":
            return {"checkpoint_sha256": self.T, "policy_source": "Policy-T", "offline": self.offline()}
        if stage == "V1-D1":
            return {"checkpoint_sha256": self.T, "replay": self.replay()}
        if stage == "V1-D2":
            return {**dict.fromkeys(("interfaces_checked_before_training", "original_checkpoint_retained",
                                   "both_regressions_replayed"), True), "rounds": 1}
        if stage == "V1-D3":
            return {"checkpoint_sha256": self.REFINED, "policy_source": "Policy-T",
                    "offline": self.offline(self.REFINED), "replay": self.replay(self.REFINED)}
        if stage == "V1-E0":
            return dict.fromkeys(("independent_gap_evidence", "interface_faults_resolved",
                                  "egopro_supported_gap", "retained_policy_t"), True)
        if stage == "V1-E1":
            return {**dict.fromkeys(("separate_bank", "provenance_and_split", "no_regression_sources",
                                     "causal_masks_and_geometry"), True), "bank_fingerprint": "e" * 64}
        if stage == "V1-E2":
            return {**dict.fromkeys(("qualified_expanded_wm", "has_policy_t_origin", "both_registries_frozen",
                                   "recorded_true_transitions"), True), "checkpoint_sha256": self.E}
        return self.retention() if stage == "V1-E3" else {}

    def gate(self, stage, body=None):
        row = self.policy["stages"][stage]
        for parent in row["gate_requires"]:
            if parent not in self.ledger["gate_refs"]:
                self.gate(parent)
        ref = f"gate-{stage}"
        self.trust(ref, "technical_gate", stage=stage, reviewer="reviewer", core_authors=["implementer"],
                   verdict="TECHNICAL_PASS", exit_code=0, metrics=self.artifact(f"metrics-{stage}", self.metrics(stage) if body is None else body),
                   input_fingerprint=self.store["context"]["input_fingerprints"][stage],
                   upstream_refs={parent: self.ledger["gate_refs"][parent] for parent in row["gate_requires"]},
                   artifacts={key: self.artifact(f"{stage}-{key}", {"synthetic": True}) for key in row["checks"]})
        self.ledger["states"][stage] = "TECHNICAL_PASS"
        self.ledger["gate_refs"][stage] = ref

    def check(self, **fields):
        self.action.update(fields)
        checker.check_transition(self.policy, self.ledger, self.action, self.store, self.root)

    def reject(self, message, **fields):
        with self.assertRaisesRegex(ValueError, message):
            self.check(**fields)

    def test_unsigned_execution_rejected(self):
        self.ledger["execution_authorization"] = None
        self.reject("unverified human_campaign")

    def test_original_design_cannot_activate_optional_ego(self):
        self.store["campaign"]["scope"] = [stage for stage in self.policy["stages"] if not stage.startswith("V1-E")]
        self.check()
        self.gate("V1-E1")
        self.reject("optional EgoPro scope", stage="V1-E2")

    def test_d0_before_d1_and_exact_candidate(self):
        self.gate("V1-D0")
        self.check(stage="V1-D1", checkpoint_sha256=self.T)
        self.reject("D1 candidate differs", stage="V1-D1", checkpoint_sha256=self.REFINED)

    def test_d0_failed_precision_cannot_recommend(self):
        body = self.metrics("V1-D0")
        body["offline"]["static_p95_mm"] = 5.1
        self.gate("V1-D0", body)
        self.gate("V1-D3")
        self.reject("offline absolute precision failed", kind="RELEASE", grade="TELEOP_SIM_RECOMMENDED",
                    checkpoint_sha256=self.REFINED)

    def test_low_error_support_cannot_cherry_pick(self):
        body = self.metrics("V1-D0")
        body["offline"]["dynamic_support_fraction"] = 0.04
        self.gate("V1-D0", body)
        self.reject("offline coverage insufficient", stage="V1-D1", checkpoint_sha256=self.T)

    def test_near_zero_baseline_not_relative_pass(self):
        body = self.metrics("V1-D0")
        body["offline"].update(static_baseline_p95_mm=0.10, static_candidate_p95_mm=0.01)
        self.gate("V1-D0", body)
        self.reject("relative gate inconclusive", stage="V1-D1", checkpoint_sha256=self.T)

    def test_changed_pre_actor_domain_rejected(self):
        body = self.metrics("V1-D0")
        body["offline"]["domain_sha256"] = "d" * 64
        self.gate("V1-D0", body)
        self.reject("domain identity changed", stage="V1-D1", checkpoint_sha256=self.T)

    def test_simple_simulator_is_not_teleop_acceptance(self):
        body = self.metrics("V1-D1")
        body["replay"]["actual_existing_graph"] = False
        self.gate("V1-D1", body)
        self.reject("actual_existing_graph", stage="V1-D2")

    def test_recorded_commands_feedback_or_rmb_are_rejected(self):
        for key in ("recorded_robot_commands_injected", "recorded_feedback_injected", "cached_rmb_injected"):
            with self.subTest(key=key):
                body = self.metrics("V1-D1")
                body["replay"][key] = True
                self.gate("V1-D1", body)
                self.reject(key, stage="V1-D2")

    def test_regression_source_or_derivative_denied_in_training(self):
        for candidate in ("direct-derivative", self.policy["regression_recordings"][0]["sha256"]):
            with self.subTest(candidate=candidate):
                self.store["context"]["training_source_identities"] = [candidate]
                self.reject("unauthorized training use")

    def test_verified_d2_e0_trigger(self):
        self.gate("V1-D1")
        self.ledger["states"]["V1-D2"] = "IN_PROGRESS"
        self.check(stage="V1-E0")
        self.store["diagnosis"]["active"] = False
        self.reject("inactive or wrong independent_d2", stage="V1-E0")

    def test_exhausted_egopro_escrow_cannot_trigger(self):
        self.gate("V1-D1")
        self.ledger["states"]["V1-D2"] = "IN_PROGRESS"
        self.ledger["consumed"].update(train_egopro=1000000, train_mixed=1000000, train_total=1000000)
        self.reject("EgoPro escrow exhausted", stage="V1-E0")

    def test_expansion_start_needs_e1_bank_qualification(self):
        self.gate("V1-E0")
        self.reject("V1-E1 not passed", stage="V1-E2")

    def test_egopro_wm_only_then_actor_update(self):
        self.gate("V1-E1")
        self.store["context"]["expanded_wm_current"] = None
        self.check(stage="V1-E2")
        self.ledger["states"]["V1-E2"] = "IN_PROGRESS"
        self.reject("unverified independent_expanded_wm", kind="ACTOR_UPDATE", stage="V1-E2")
        self.store["context"]["expanded_wm_current"] = "wm-expanded"
        self.check(kind="ACTOR_UPDATE", stage="V1-E2")

    def test_expansion_must_retain_original_teleop(self):
        body = self.retention()
        body["retention"]["offline_static"]["policy_e_p95_mm"] = 11.0
        self.gate("V1-E3", body)
        mixed = {"checkpoint_sha256": self.E, "policy_source": "Policy-E",
                 "offline": self.offline(self.E), "replay": self.replay(self.E)}
        self.gate("V1-D3", mixed)
        self.reject("Teleop retention failed", kind="RELEASE", grade="TELEOP_SIM_RECOMMENDED",
                    checkpoint_sha256=self.E)

    def test_ego_cannot_charge_other_training_escrow(self):
        self.gate("V1-E1")
        self.action["reservation"].update(train_mixed_initial=1000, train_mixed=1000, train_total=1000)
        self.reject("wrong escrow", stage="V1-E2")

    def test_nested_budget_overrun_despite_total_headroom(self):
        self.gate("V1-E1")
        self.action["reservation"].update(train_egopro=1000001, train_total=1000001, train_mixed=1000001)
        self.reject("resource exhausted: train_egopro", stage="V1-E2")

    def test_diagnostic_export_is_isolated(self):
        self.evidence("diagnostic", "diagnostic_checkpoint", {"checkpoint_sha256": self.T,
            **dict.fromkeys(("existing_checkpoint", "failed_performance_only", "data_identity", "function_pass",
                            "safety_pass", "causal_pass", "export_parity"), True)})
        self.check(kind="DIAGNOSTIC_EXPORT", grade="EXPORTABLE", advance=False, diagnostic_ref="diagnostic")
        self.reject("cannot advance", advance=True)
        self.reject("cannot advance", advance=False, grade="TELEOP_SIM_RECOMMENDED")
        self.reject("diagnostic cannot become normal release", kind="RELEASE", diagnostic=True)

    def test_unauthorized_real_robot_rejected(self):
        self.reject("real robot unavailable", real_robot_requested=True)
        self.reject("unverified physical_robot_base", kind="REAL_DEPLOY")

    def test_changed_gates_cannot_use_old_approval(self):
        self.policy["performance_candidates"]["static_offline_mm"] = 100
        self.reject("binding differs: human_design_approval")

    def test_missing_cold_restart_rejected(self):
        body = self.metrics("V1-D1")
        next(iter(body["replay"]["bags"].values()))["cold_starts"]["T2"] = 1
        self.gate("V1-D1", body)
        self.reject("insufficient cold starts", stage="V1-D2")

    def test_standalone_runtime_is_separately_proven(self):
        self.gate("V1-D0")
        self.check(kind="RELEASE", grade="SIM_DEPLOYABLE", checkpoint_sha256=self.T)
        self.store["context"]["standalone_receipts"].clear()
        self.reject("unverified independent_standalone", kind="RELEASE", grade="SIM_DEPLOYABLE",
                    checkpoint_sha256=self.T)

    def test_pooled_gain_and_separate_bags(self):
        self.gate("V1-D2")
        self.check(stage="V1-D3", policy_source="Policy-T", checkpoint_sha256=self.REFINED)
        self.gate("V1-D3")
        self.check(kind="RELEASE", grade="TELEOP_SIM_RECOMMENDED", checkpoint_sha256=self.REFINED)
        body = self.metrics("V1-D3")
        body["replay"]["combined_candidate_p95_mm"] = 14.9
        self.gate("V1-D3", body)
        self.reject("relative gain failed: combined")

    def test_policy_e_pass_and_policy_t_remains_valid_path(self):
        self.gate("V1-E3")
        body = {"checkpoint_sha256": self.E, "policy_source": "Policy-E",
                "offline": self.offline(self.E), "replay": self.replay(self.E)}
        self.gate("V1-D3", body)
        self.check(kind="RELEASE", grade="TELEOP_SIM_RECOMMENDED", checkpoint_sha256=self.E)
        self.ledger["states"]["V1-E3"] = "FAILED"
        self.reject("V1-E3 not passed")
        self.gate("V1-D3")
        self.check(checkpoint_sha256=self.REFINED)

    def test_d1_functional_grade_without_d3_performance(self):
        self.gate("V1-D1")
        self.check(kind="RELEASE", grade="TELEOP_SIM_VALIDATED", checkpoint_sha256=self.T)
        self.reject("unverified independent_final_functional_replay", checkpoint_sha256=self.REFINED)
        self.evidence("fresh-function", "independent_final_functional_replay", self.replay(self.REFINED))
        self.check(candidate_ref="fresh-function")

    def test_d3_policy_e_start_requires_completed_retention(self):
        self.gate("V1-D2")
        self.reject("V1-E3 not passed", stage="V1-D3", policy_source="Policy-E", checkpoint_sha256=self.E)
        self.gate("V1-E3")
        self.check()

    def test_policy_e_requires_training_checkpoint_identity(self):
        body = self.retention()
        body.update(checkpoint_sha256=self.REFINED)
        body["offline"].update(checkpoint_sha256=self.REFINED)
        body["replay"].update(checkpoint_sha256=self.REFINED)
        self.gate("V1-E3", body)
        self.gate("V1-D3", {"checkpoint_sha256": self.REFINED, "policy_source": "Policy-E",
                            "offline": self.offline(self.REFINED), "replay": self.replay(self.REFINED)})
        self.reject("Policy-E differs from E2", kind="RELEASE", grade="TELEOP_SIM_RECOMMENDED",
                    checkpoint_sha256=self.REFINED)

    def test_changed_final_target_rejected(self):
        body = self.metrics("V1-D3")
        body["offline"]["target_sha256"] = "e" * 64
        self.gate("V1-D3", body)
        self.reject("frozen offline identity differs", kind="RELEASE", grade="TELEOP_SIM_RECOMMENDED",
                    checkpoint_sha256=self.REFINED)

    def test_wrong_post_refinement_artifact_rejected(self):
        body = self.metrics("V1-D3")
        body["replay"]["checkpoint_sha256"] = self.T
        self.gate("V1-D3", body)
        self.reject("replay checkpoint differs", kind="RELEASE", grade="TELEOP_SIM_RECOMMENDED",
                    checkpoint_sha256=self.REFINED)

    def test_frozen_design_status_and_immutable_past(self):
        state = checker.read_json(checker.DOCS / "phase_state.json")
        checker.check_state(state, self.policy)
        edited = deepcopy(state)
        edited["policy_first_v2"]["execution_authorized"] = True
        with self.assertRaisesRegex(ValueError, "historical state changed"):
            checker.check_state(edited, self.policy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
