"""Verify a design-only release and evaluate synthetic permission proposals."""
import argparse
import importlib.util
import json
from math import isfinite
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[1]
OLD = HERE.parent / "P00_V2"
spec = importlib.util.spec_from_file_location("historical_design_checks", OLD / "check_design.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
require = legacy.require
file_sha = legacy.file_sha
git = legacy.git
git_bytes = legacy.git_bytes
read_json = legacy.read_json
sha = legacy.sha
verify_digest = legacy.verify_digest
local_file = legacy.local_file
canonical_sha = legacy.canonical_sha
trusted_receipt = legacy.trusted_receipt
HEX = legacy.HEX

MIGRATION = {
    "V1-D0": ("V1-C0", "V1-C3"), "V1-D1": ("V1-D0",),
    "V1-D2": ("V1-D1",), "V1-E0": ("V1-D0", "V1-D1"),
    "V1-E1": ("V1-E0",), "V1-E2": ("V1-E1",), "V1-E3": ("V1-E2",),
    "V1-D3": ("V1-D0", "V1-D1", "V1-D2"),
}
GRADE_PAIR = {"TELEOP_SIM_VALIDATED": "V1-D1", "TELEOP_SIM_RECOMMENDED": "V1-D3",
              "SIM_RECOMMENDED": "V1-D0"}
FUNCTION_FLAGS = ["function_pass", "safety_pass", "causal_pass", "export_parity", "wm_current"]
PAIR_FLAGS = ["functional_geometry", "timing_pass", "zero_residual_parity"]
ARCHIVE = "f78ee27f4762fedca9529bf697560bb267f8c704"
ARCHIVE_SHA = "dda0c8254d0c8e637bbe1ee14d1bdfbf8b6b7309a8209263fc350a0c3e4561ea"


def flags(data, keys, label):
    for key in keys:
        require(data.get(key) is True, f"{label}: {key}")


def finite(data, key):
    result = data.get(key)
    require(type(result) in {int, float} and isfinite(result), f"invalid finite metric: {key}")
    return result


def checkpoint(data, key="checkpoint_sha256"):
    result = data.get(key)
    require(isinstance(result, str) and HEX.fullmatch(result), "checkpoint identity missing")
    return result


def check_budget(policy, consumed, reserved):
    caps = policy["resource_budget"]
    # The original validator checks every cap and the original three top-level buckets.
    legacy.check_budget(caps, consumed, reserved)
    for ledger in (consumed, reserved):
        for main, children in {
            "train_mixed": ("train_mixed_initial", "train_deploy_refinement", "train_egopro"),
            "eval_release": ("eval_d0", "eval_d1", "eval_d2_e", "eval_d3"),
        }.items():
            require(ledger[main] == sum(ledger[key] for key in children), f"unbalanced allowance: {main}")
        require(ledger["eval_pre_actor_domain"] <= ledger["eval_c0"], "domain replay outside C0 evaluation")


def check_reservation(action):
    counters = action["reservation"]
    stage = action.get("stage")
    train_allowed = {
        "V1-B": {"train_static_bucket", "train_smoke_c0"},
        "V1-C0": {"train_static_bucket", "train_smoke_c0"},
        "V1-C1": {"train_static_bucket"}, "V1-C2": {"train_dynamic"},
        "V1-C3": {"train_mixed", "train_mixed_initial"},
        "V1-D2": {"train_mixed", "train_deploy_refinement"},
        "V1-E2": {"train_mixed", "train_egopro"},
    }
    eval_allowed = {
        "V1-B": {"eval_c0", "eval_pre_actor_domain"}, "V1-C0": {"eval_c0", "eval_pre_actor_domain"},
        "V1-C1": {"eval_training_watch"}, "V1-C2": {"eval_training_watch"}, "V1-C3": {"eval_training_watch"},
        "V1-D0": {"eval_release", "eval_d0"}, "V1-D1": {"eval_release", "eval_d1"},
        "V1-D2": {"eval_release", "eval_d2_e"}, "V1-E0": {"eval_release", "eval_d2_e"},
        "V1-E1": {"eval_release", "eval_d2_e"}, "V1-E2": {"eval_release", "eval_d2_e"},
        "V1-E3": {"eval_release", "eval_d2_e"}, "V1-D3": {"eval_release", "eval_d3"},
    }
    allowed = train_allowed.get(stage, set()) | eval_allowed.get(stage, set()) | {"train_total", "eval_total"}
    for key, amount in counters.items():
        if key.startswith(("train_", "eval_")):
            require(not amount or key in allowed, f"wrong escrow for {stage}: {key}")


def check_training_roots(policy, context):
    denied = {row["sha256"] for row in policy["regression_recordings"]}
    denied |= set(context["regression_lineage_closure"])
    training = set(context["training_source_identities"])
    require(context["regression_lineage_audited"] is True and training.isdisjoint(denied),
            "unauthorized training use of deployment regression lineage")


def unchanged_full(data, limits):
    base = finite(data, "full_baseline_p95_mm")
    result = finite(data, "full_candidate_p95_mm")
    require(base >= 0 and result >= 0, "negative P95")
    allowance = (base * limits["full_domain_max_relative_degradation"]
                 if base >= limits["reference_floor_mm"] else limits["near_zero_max_delta_mm"])
    require(result <= base + allowance and finite(data, "control_failure_delta") <= limits["control_failure_max_delta"],
            "full domain or control failure non-degradation failed")


def relative_gain(data, prefix, limits, threshold):
    base = finite(data, f"{prefix}_baseline_p95_mm")
    result = finite(data, f"{prefix}_candidate_p95_mm")
    require(base >= limits["reference_floor_mm"] and result >= 0, "relative gate inconclusive")
    require((base - result) / base + 1e-12 >= threshold, f"relative gain failed: {prefix}")


def check_offline(data, limits):
    flags(data, FUNCTION_FLAGS + ["pre_actor_domain_frozen", "coverage_met", "functional_geometry",
                                  "original_timeline", "full_planned_denominator"], "offline precision")
    require(finite(data, "static_targets") >= limits["static_min_targets"]
            and finite(data, "static_roots") >= limits["primary_min_validation_roots"]
            and finite(data, "dynamic_roots") >= limits["primary_min_validation_roots"]
            and finite(data, "dynamic_ticks") >= limits["dynamic_min_unique_ticks"], "offline support insufficient")
    require(finite(data, "static_support_fraction") >= limits["static_min_fraction"]
            and finite(data, "dynamic_support_fraction") >= limits["dynamic_min_fraction"], "offline coverage insufficient")
    require(data.get("aggregation") == limits["aggr"], "offline failure composite required")
    for kind in ("static", "dynamic"):
        require(0 <= finite(data, f"{kind}_p95_mm") <= limits[f"{kind}_offline_mm"],
                f"offline absolute precision failed: {kind}")
        relative_gain(data, kind, limits, limits["primary_offline_relative_gain"])
    unchanged_full(data, limits)
    require(finite(data, "hard_violations") == 0, "offline hard safety failed")


def check_replay(data, policy, performance):
    flags(data, ["actual_existing_graph", "actual_new_feedback", "isolated_ros_domain", "nominal_online",
                 "assets_and_initial_state_verified", "legacy_policy_disabled", "independent_mapped_scorer"],
          "existing actual Teleop")
    for key in ("recorded_robot_commands_injected", "recorded_feedback_injected", "cached_rmb_injected",
                "real_driver_active", "mid_run_reset"):
        require(data.get(key) is False, f"disallowed replay input or execution: {key}")
    expected = {row["id"]: row["sha256"] for row in policy["regression_recordings"]}
    require(set(data["bags"]) == set(expected), "both specified recordings required")
    limits = policy["performance_candidates"]
    require(data.get("aggregation") == limits["aggr"], "replay failure composite required")
    if performance:
        relative_gain(data, "combined", limits, limits["teleop_relative_gain"])
    for bag, row in data["bags"].items():
        require(row["sha256"] == expected[bag], "recording identity differs")
        flags(row, PAIR_FLAGS if performance else ["timing_pass", "zero_residual_parity"], "replay pair")
        require(row["t1_t2_common_context"] is True and row["t1_t2_only_residual_changed"] is True,
                "replay pair is confounded")
        required = {"T0", "T1", "T2"}
        require(set(row["cold_starts"]) == required and
                all(type(x) is int and x >= limits["minimum_cold_starts_per_bag_method"]
                    for x in row["cold_starts"].values()), "insufficient cold starts")
        require(finite(row, "full_unique_ticks") >= limits["teleop_min_unique_ticks"]
                and finite(row, "common_unique_ticks") >= limits["teleop_min_unique_ticks"]
                and finite(row, "nonoverlap_windows_at_least_25_ticks") >= limits["teleop_min_nonoverlap_windows"], "insufficient replay support")
        require(finite(row, "hard_violations") == 0 and finite(row, "late_issues") == 0, "replay safety/timing")
        if performance:
            unchanged_full(row, limits)


def retention(data, policy):
    gap = data["gap"]
    require(gap["metric"] in {"tail_tip_mm", "recovery_ms"}
            and gap["metric_sha256"] == data["preregistered_gap_metric_sha256"], "gap metric changed")
    original = finite(gap, "policy_t_p95")
    updated = finite(gap, "policy_e_p95")
    floor = policy["performance_candidates"]["reference_floor_mm"] if gap["metric"] == "tail_tip_mm" else 20.0
    require(original >= floor and 0 <= updated < original and (original - updated) / original + 1e-12 >= 0.10,
            "no measured targeted expansion gain")
    required = {"offline_static", "offline_dynamic", "offline_full"} | {r["id"] for r in policy["regression_recordings"]}
    require(set(data["retention"]) == required, "missing Teleop retention strata")
    for label, pair in data["retention"].items():
        original = finite(pair, "policy_t_p95_mm")
        updated = finite(pair, "policy_e_p95_mm")
        require(original >= 0 and 0 <= updated <= original + max(0.05 * original, 0.25),
                f"Teleop retention failed: {label}")


def validate_gate_metrics(stage, m, policy):
    if stage in {"V1-D0", "V1-D3"}:
        check_offline(m["offline"], policy["performance_candidates"])
        require(checkpoint(m) == checkpoint(m["offline"]), "offline checkpoint differs")
    if stage in {"V1-D1", "V1-D3"}:
        check_replay(m["replay"], policy, performance=stage == "V1-D3")
        require(checkpoint(m) == checkpoint(m["replay"]), "replay checkpoint differs")
    if stage == "V1-D0":
        require(m.get("policy_source") == "Policy-T", "D0 requires Teleop-only candidate")
    if stage == "V1-D2":
        flags(m, ["interfaces_checked_before_training", "original_checkpoint_retained", "both_regressions_replayed"],
              "bounded refinement")
        require(0 <= finite(m, "rounds") <= policy["allowed_autonomous_changes"]["deployment_rounds"],
                "bounded refinement exceeded rounds")
    if stage == "V1-E0":
        flags(m, ["independent_gap_evidence", "interface_faults_resolved", "egopro_supported_gap",
                  "retained_policy_t"], "conditional expansion")
    if stage == "V1-E1":
        flags(m, ["separate_bank", "provenance_and_split", "no_regression_sources",
                  "causal_masks_and_geometry"], "EgoPro bank qualification")
    if stage == "V1-E2":
        flags(m, ["qualified_expanded_wm", "has_policy_t_origin", "both_registries_frozen",
                  "recorded_true_transitions"], "expanded model and training")
    if stage == "V1-E3":
        flags(m, ["target_gap_gain", "teleop_static_retained", "teleop_dynamic_retained", "functional_safety_retained",
                  "both_mcap_retained", "separate_source_metrics"], "EgoPro retention")
        retention(m, policy)
        check_offline(m["offline"], policy["performance_candidates"])
        check_replay(m["replay"], policy, performance=True)
        require(checkpoint(m) == checkpoint(m["offline"]) == checkpoint(m["replay"]), "mixed candidate identity differs")


class Governance:
    """Offline authorization verifier. Trusted records come from an independent source."""
    def __init__(self, policy, ledger, action, trusted, evidence_root):
        self.policy = policy
        self.ledger = ledger
        self.action = action
        self.trusted = trusted
        self.root = evidence_root
        self.identities = {key: ledger[key] for key in legacy.HASHED_FIELDS}
        self.scope = {**self.identities, "campaign_id": ledger["campaign_id"]}
        self.done = {}
        require(all(isinstance(value, str) and HEX.fullmatch(value)
                    for value in self.identities.values()), "invalid identity")

    def trust(self, ref, kind, bindings=None):
        return trusted_receipt(ref, kind, self.trusted, self.scope if bindings is None else bindings)

    def signed_artifact(self, item):
        blob = local_file(self.root, item["path"]).read_bytes()
        verify_digest(blob, item["sha256"], item["path"])
        return json.loads(blob)

    def gate(self, stage, proposal=None):
        if stage in self.done:
            return self.done[stage]
        row = self.policy["stages"][stage]
        if proposal is None:
            require(self.ledger["states"].get(stage) == "TECHNICAL_PASS", f"dependency {stage} not passed")
        ref = proposal or self.ledger["gate_refs"].get(stage)
        parents = {parent: self.ledger["gate_refs"].get(parent) for parent in row["gate_requires"]}
        fingerprint = self.context["input_fingerprints"].get(stage)
        require(isinstance(fingerprint, str) and HEX.fullmatch(fingerprint), "missing stage input fingerprint")
        proof = self.trust(ref, "technical_gate", {**self.scope, "stage": stage, "upstream_refs": parents,
                                                    "input_fingerprint": fingerprint})
        require(proof["reviewer"] not in proof["core_authors"] and proof["reviewer"] and proof["core_authors"],
                "reviewer not independent")
        require(proof["verdict"] == "TECHNICAL_PASS" and proof["exit_code"] == 0, "stage verdict failed")
        require(set(proof["artifacts"]) == set(row["checks"]), "gate evidence mismatch")
        for artifact in proof["artifacts"].values():
            path = local_file(self.root, artifact["path"])
            verify_digest(path.read_bytes(), artifact["sha256"], artifact["path"])
        if stage == "V1-C0":
            require(self.context.get("wm_current") == ref, "Teleop WM qualification invalidated")
        for parent in parents:
            self.gate(parent)
        data = self.signed_artifact(proof["metrics"])
        validate_gate_metrics(stage, data, self.policy)
        if stage in {"V1-D0", "V1-D3", "V1-E3"}:
            self.check_run_identity(data["offline"], "offline")
            self.check_domain_lock(data["offline"]["domain_sha256"])
        if stage == "V1-D1":
            require(checkpoint(data) == checkpoint(self.gate("V1-D0")), "D1 candidate differs from D0")
        if stage in {"V1-D1", "V1-D3", "V1-E3"}:
            self.check_run_identity(data["replay"], "replay")
        if stage == "V1-E0":
            self.check_egopro_trigger()
        if stage == "V1-E3":
            require(checkpoint(data) == checkpoint(self.gate("V1-E2")), "Policy-E differs from E2 training output")
            require(checkpoint(data, "policy_t_sha256") == checkpoint(self.gate("V1-D0")),
                    "retention comparison uses different Policy-T")
            require(data["preregistered_gap_metric_sha256"] == self.context["gap_metric_sha256"],
                    "retention gap target changed")
        if stage in {"V1-E2", "V1-E3"}:
            self.check_expanded_wm()
        if stage == "V1-D3":
            self.check_final_source(data)
        self.done[stage] = data
        return data

    def check_run_identity(self, data, scope):
        expected = self.context["run_inputs"][scope]
        for name in ("target_sha256", "asset_sha256", "time_plan_sha256"):
            actual = data[name]
            require(isinstance(actual, str) and HEX.fullmatch(actual) and actual == expected[name],
                    f"frozen {scope} identity differs: {name}")

    def check_domain_lock(self, expected=None):
        lock = self.trust(self.context.get("pre_actor_domain_ref"), "independent_pre_actor_domain")
        domain = self.signed_artifact(lock["evidence"])
        flags(domain, ["frozen_before_first_actor", "static_witness_audited", "dynamic_servo_witness_audited",
                       "unchanged_planned_domain"], "offline domain lock")
        locked = domain["domain_sha256"]
        require(isinstance(locked, str) and HEX.fullmatch(locked) and (expected is None or locked == expected),
                "pre-Actor domain identity changed")

    def check_egopro_trigger(self, activating=False):
        require(self.egopro_authorized, "optional EgoPro scope not authorized")
        receipt = self.trust(self.action.get("diagnosis_ref") or self.context.get("diagnosis_ref"),
                             "independent_d2_diagnosis")
        data = self.signed_artifact(receipt["evidence"])
        flags(data, ["interface_faults_resolved", "independent_gap_evidence", "egopro_supported_gap"],
              "EgoPro trigger")
        require(checkpoint(data, "policy_t_sha256") == checkpoint(self.gate("V1-D0")),
                "EgoPro trigger points to wrong Policy-T")
        if activating:
            minimum = data["planned_minimum_train_transitions"]
            require(type(minimum) is int and minimum > 0 and
                    self.ledger["consumed"]["train_egopro"] + minimum <= self.policy["resource_budget"]["train_egopro"],
                    "EgoPro escrow exhausted")

    def check_expanded_wm(self):
        e1 = self.gate("V1-E1")
        require("bank_fingerprint" in e1, "missing EgoPro bank fingerprint")
        ref = self.context.get("expanded_wm_current")
        proof = self.trust(ref, "independent_expanded_wm", {
            **self.scope, "bank_fingerprint": e1["bank_fingerprint"],
            "mixture_fingerprint": self.context["mixture_fingerprint"],
        })
        data = self.signed_artifact(proof["evidence"])
        flags(data, ["excitation_54", "identity_and_holdout", "one_four_eight", "actual_gradients",
                     "servo_timing", "safety_boundaries", "model_gap_pass"], "expanded World Model")

    def check_final_source(self, data):
        src = data["policy_source"]
        require(src in {"Policy-T", "Policy-E"}, "unknown policy source")
        if src == "Policy-E":
            require(checkpoint(self.gate("V1-E3")) == checkpoint(data), "D3 did not use retained Policy-E")
        else:
            require(checkpoint(data) in self.context["qualified_teleop_candidates"],
                    "D3 did not use qualified Teleop-only candidate")

    def check_standalone(self, grade, actor_sha):
        candidate_ref = self.context.get("standalone_receipts", {}).get(actor_sha)
        proof = self.trust(candidate_ref, "independent_standalone_candidate")
        candidate = self.signed_artifact(proof["evidence"])
        eligible = ["export_parity", "trained_actor", "data_identity", "non_diagnostic"]
        if grade != "EXPORTABLE":
            eligible += ["function_pass", "causal_pass", "safety_pass", "no_bank"]
        flags(candidate, eligible, "standalone grade")
        require(checkpoint(candidate) == actor_sha, "release checkpoint differs")

    def check(self):
        action = self.action
        kind = action["kind"]
        require(self.scope["campaign_id"] and self.ledger.get("stop_reason") is None, "Campaign stopped")
        require(self.identities["budget_sha256"] == canonical_sha(self.policy["resource_budget"]), "budget changed")
        approval = self.trust(self.ledger["design_approval"], "human_design_approval", {
            **self.identities, "policy_sha256": canonical_sha(self.policy),
        })
        authorization = self.trust(self.ledger["execution_authorization"], "human_campaign_authorization")
        scope = set(authorization["scope"])
        core = set(self.policy["stages"]) - {"V1-E0", "V1-E1", "V1-E2", "V1-E3"}
        extra = scope - core
        optional = set(self.policy["stages"]) - core
        require(core <= scope and extra in (set(), optional), "Campaign scope differs")
        self.egopro_authorized = extra == optional and authorization.get("authorize_egopro") is True
        require(not (self.action.get("stage") or "").startswith("V1-E") or self.egopro_authorized,
                "optional EgoPro scope not authorized")
        check_budget(self.policy, self.ledger["consumed"], action["reservation"])
        check_reservation(action)
        self.context = self.trust(self.ledger["execution_context_ref"], "verified_execution_context")
        check_training_roots(self.policy, self.context)
        require(action.get("real_robot_requested", False) is False or kind == "REAL_DEPLOY", "real robot unavailable")
        if kind in {"REAL_DEPLOY", "FINAL_TEST", "FORMAL_SCORING"}:
            requirements = {
                "REAL_DEPLOY": self.policy["deployment_grades"]["REAL_DEPLOYABLE"],
                "FINAL_TEST": self.policy["final_test_requirements"],
                "FORMAL_SCORING": self.policy["formal_scoring_requirements"],
            }[kind]
            for key in requirements:
                self.trust(action.get("external_receipts", {}).get(key), key, self.identities)
            return
        if kind == "DIAGNOSTIC_EXPORT":
            require(action.get("grade") == self.policy["diagnostic_export_max_grade"]
                    and action.get("advance") is False, "diagnostic export cannot advance or recommend")
            data = self.signed_artifact(self.trust(action.get("diagnostic_ref"), "diagnostic_checkpoint")["evidence"])
            flags(data, ["existing_checkpoint", "failed_performance_only", "data_identity", "function_pass",
                         "safety_pass", "causal_pass", "export_parity"], "diagnostic export")
            checkpoint(data)
            return
        if kind == "RELEASE":
            require(action.get("diagnostic", False) is False, "diagnostic cannot become normal release")
            grade = action["grade"]
            if grade in {"EXPORTABLE", "SIM_DEPLOYABLE"}:
                self.check_standalone(grade, checkpoint(action))
                return
            require(grade in GRADE_PAIR, "grade unrecognized")
            release = self.gate(GRADE_PAIR[grade])
            baseline_candidate = checkpoint(release)
            if grade == "TELEOP_SIM_RECOMMENDED":
                self.check_final_source(release)
            if grade == "TELEOP_SIM_VALIDATED" and baseline_candidate != checkpoint(action):
                proof = self.trust(action.get("candidate_ref"), "independent_final_functional_replay")
                release = self.signed_artifact(proof["evidence"])
                check_replay(release, self.policy, performance=False)
                self.check_run_identity(release, "replay")
            require(checkpoint(release) == checkpoint(action), "release checkpoint differs")
            self.check_standalone(grade, checkpoint(action))
            return
        stage = action["stage"]
        require(stage in self.policy["stages"], "unknown stage")
        row = self.policy["stages"][stage]
        state = self.ledger["states"].get(stage)
        if kind == "PASS":
            require(state == "IN_PROGRESS", "gate requires active stage")
            self.gate(stage, action["gate_ref"])
            return
        require(kind in {"START", "ACTOR_UPDATE"}, "unsupported action")
        if kind == "START":
            retries = self.ledger["retries"].get(stage, 0)
            cap = self.policy["allowed_autonomous_changes"]["retries_per_stage"]
            require(type(retries) is int and ((state == "NOT_STARTED" and retries == 0) or
                    (state in {"FAILED", "BLOCKED"} and 0 <= retries < cap)), "illegal start or exhausted retry")
        for parent in row["start_requires"]:
            self.gate(parent)
        if stage == "V1-A":
            require(approval.get("m1_continuous_simulation") is True, "M1 continuous scope not approved")
        if stage == "V1-B" and self.ledger["states"].get("V1-A") != "TECHNICAL_PASS":
            require(action.get("prepared_synthetic_only") is True, "V1-B early work is synthetic only")
        if stage == "V1-D1":
            require(checkpoint(action) == checkpoint(self.gate("V1-D0")), "D1 candidate differs from D0")
        if stage == "V1-D3":
            self.check_final_source(action)
        if stage == "V1-E0":
            require(self.ledger["states"].get("V1-D2") in {"IN_PROGRESS", "TECHNICAL_PASS"},
                    "D2 diagnosis stage unavailable")
            self.check_egopro_trigger(activating=True)
        if stage in {"V1-C1", "V1-C2", "V1-C3", "V1-E2"}:
            self.check_domain_lock()
        if kind == "ACTOR_UPDATE":
            require(stage in {"V1-C1", "V1-C2", "V1-C3", "V1-D2", "V1-E2"} and state == "IN_PROGRESS",
                    "Actor update needs a live training stage")
            if stage == "V1-E2":
                self.check_expanded_wm()
            if stage == "V1-D2":
                self.gate("V1-C0")
                diag = self.trust(self.context.get("refinement_diagnosis_ref"), "independent_d2_diagnosis")
                flags(self.signed_artifact(diag["evidence"]),
                      ["interface_faults_resolved", "bounded_policy_remedy"], "D2 Actor authorization")


def check_transition(policy, ledger, action, trusted, evidence_root):
    Governance(policy, ledger, action, trusted, evidence_root).check()


def validate_policy(policy, requirements):
    require(policy["design_version"] == "2.0.1" and policy["reviewed_baseline"] == ARCHIVE
            and policy["baseline_candidate_sha256"] == ARCHIVE_SHA, "wrong reviewed freeze")
    require(policy["approval_status"] == "PENDING_HUMAN_REVIEW"
            and policy["execution_authorized"] is False, "candidate expanded privileges")
    base = {k: v for k, v in legacy.DEPENDENCIES.items() if k != "V1-D"}
    expected = {**base, **{k: (v, v) for k, v in MIGRATION.items()}}
    require(set(expected) == set(policy["stages"]), "stage inventory differs")
    for name, row in policy["stages"].items():
        require((tuple(row["start_requires"]), tuple(row["gate_requires"])) == expected[name], f"bad DAG: {name}")
        require(row["checks"] and len(set(row["checks"])) == len(row["checks"]), "incomplete gate")
    for key in ("final_test_requirements", "formal_scoring_requirements"):
        require(policy[key] == read_json(OLD / "design_policy.json")[key], "physical research obligations differ")
    require(policy["deployment_grades"]["REAL_DEPLOYABLE"] ==
            read_json(OLD / "design_policy.json")["deployment_grades"]["REAL_DEPLOYABLE"],
            "hardware requirements differ")
    require(policy["diagnostic_export_max_grade"] == "EXPORTABLE", "diagnostic scope expanded")
    caps = policy["resource_budget"]
    preserved_caps = read_json(OLD / "design_policy.json")["resource_budget"]
    require(all(caps[key] == value for key, value in preserved_caps.items()), "original resource cap changed")
    for key, expected_value in {"train_mixed_initial": 8000000, "train_deploy_refinement": 1000000,
                                "train_egopro": 1000000, "eval_d0": 200000, "eval_d1": 150000,
                                "eval_d2_e": 100000, "eval_d3": 200000}.items():
        require(caps[key] == expected_value, f"unexpected candidate allocation: {key}")
    ids = [f"S05-REQ-{n:03d}" for n in range(1, 42)]
    require([x["id"] for x in requirements] == ids, "requirement migration incomplete")
    for row in requirements:
        require(all(row.values()) and (row["stage"] in expected or row["stage"] in {"全阶段", "后续硬件"}),
                "incomplete requirement")
        local_file(DOCS, row["design"])
    for filename in ("REQUIREMENTS.md", "TRACEABILITY_MATRIX.md"):
        actual = re.findall(r"(?m)^\| (S05-REQ-\d{3}) \| ([A-Z]+) \|", (DOCS / filename).read_text(encoding="utf-8"))
        require(actual == [(row["id"], row["disposition"]) for row in requirements], f"requirement rows: {filename}")


def check_preservation(policy, history):
    old_candidate = git_bytes(ARCHIVE, "docs/s0-5/design_freeze_candidate.json")
    verify_digest(old_candidate, ARCHIVE_SHA, "v2.0.0 frozen candidate")
    old_manifest = json.loads(old_candidate)
    for relative, digest in {**old_manifest["active_documents"], **old_manifest["verification_artifacts"]}.items():
        verify_digest(git_bytes(ARCHIVE, f"docs/s0-5/{relative}"), digest, relative)
    original_paths = ("docs/s0-5/evidence/P00_V2", "docs/s0-5/reports/P00_V2_design_review.md",
                      "source", "tests", "scripts", "docs/s0-5/HISTORICAL_DATA_USAGE.md")
    require(not git("diff", "--name-only", ARCHIVE, "--", *original_paths).strip(), "protected baseline edited")
    old_history = read_json(OLD / "historical_inputs.json")
    legacy.audit_history(old_history, old_manifest["audited_source_sha256"])
    require(history["reviewed_baseline"] == ARCHIVE and history["baseline_candidate_sha256"] == ARCHIVE_SHA,
            "history identity mismatch")
    for row, observed in zip(policy["regression_recordings"], history["bags"], strict=True):
        require(row["role"] == "TELEOP_DEPLOYMENT_REGRESSION" and row["training_admission"] is False
                and (row["id"], row["sha256"]) == (observed["id"], observed["sha256"])
                and row["path"] == f"/home/ubuntu/teleop_ws/data/{row['id']}/{row['id']}_0.mcap"
                and HEX.fullmatch(row["sha256"]), "source role or identity mismatch")
    print("PASS immutable v2.0.0 freeze, original evidence and two MCAP identities")


def check_state(state, policy):
    original = json.loads(git_bytes(ARCHIVE, "docs/s0-5/phase_state.json"))
    version = state["policy_first_v2_1"]
    require({k: v for k, v in state.items() if k != "policy_first_v2_1"} == original, "historical state changed")
    expected = {**legacy.PUBLISHED_STATE, "real_deployment": "BLOCKED", "egopro_activation": None,
                "graded_candidate": None, "diagnostic_exports": [], "offline_frozen_domain": None}
    expected.pop("m1_continuous_use")
    require(all(version.get(k) == v for k, v in expected.items()) and version["design_version"] == "2.0.1",
            "candidate became executable")
    require(version["stage_states"] == dict.fromkeys(policy["stages"], "NOT_STARTED"), "phase already started")


def main():
    from seal_design import active_paths, verification_paths

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", action="store_true", help="ignore final report/evidence inventory while drafting")
    args = parser.parse_args()
    manifest = read_json(DOCS / "design_freeze_candidate.json")
    policy = read_json(HERE / "design_policy.json")
    requirements = read_json(HERE / "requirement_mapping.json")
    history = read_json(HERE / "historical_inputs.json")
    validate_policy(policy, requirements)
    check_preservation(policy, history)
    check_state(read_json(DOCS / "phase_state.json"), policy)
    require(manifest["policy"] == policy and manifest["requirement_mapping"] == requirements
            and manifest["historical_evidence_identity"] == history, "manifest differs from machine contracts")
    require(manifest["design_version"] == "2.0.1" and manifest["approval_status"] == "PENDING_HUMAN_REVIEW"
            and manifest["execution_authorized"] is False, "manifest authorization")
    active = manifest["active_documents"]
    verification = manifest["verification_artifacts"]
    if not args.draft:
        require(set(active) == set(active_paths(policy)), "active inventory differs")
        require(set(verification) == set(verification_paths()), "verification inventory differs")
    for name, digest in {**active, **verification}.items():
        path = local_file(DOCS, name)
        verify_digest(path.read_bytes(), digest, name)
        if path.suffix == ".md" and name in active:
            require("2.0.1" in path.read_text(encoding="utf-8").splitlines()[0], f"wrong document version: {name}")
    dcr = manifest["dcr_identity"]
    require(dcr["id"] == "DCR-006" and active[dcr["path"]] == dcr["sha256"], "DCR identity differs")
    bookmarks = ["specs/HARDWARE_SHARED_MEASUREMENT_PLAN.md", "specs/P01-data-v0.2.1.md", "specs/P01_teleop_data.md"]
    count, externals = legacy.validate_links([*active, *bookmarks])
    print(f"PASS {len(active)} active files, {len(requirements)} requirements, {len(policy['stages'])} stages, {count} links")
    print("PASS DESIGN_READY_FOR_REVIEW; execution_authorized=false; all stages NOT_STARTED")
    for link in sorted(externals):
        print(f"External target: {link}")


if __name__ == "__main__":
    main()
