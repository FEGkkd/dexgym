"""Build the design-only inventory. Run the checker on the resulting candidate."""
import argparse
import json
from pathlib import Path

from check_design import DOCS, HERE, file_sha, git_bytes, read_json, sha

TOP = (
    "README", "MASTER_DESIGN", "REQUIREMENTS", "CONTRACTS", "ROADMAP",
    "EVALUATION_PROTOCOL", "TRACEABILITY_MATRIX", "PROJECT_STATE", "DECISION_LOG",
    "RISK_REGISTER", "OPEN_ISSUES", "MULTI_AGENT_PROTOCOL",
)
DCR = "dcr/DCR-005_policy_first_supermimic.md"
VERIFICATION = (
    "check_design.py", "test_check_design.py", "seal_design.py", "verify_legacy.py",
    "design_policy.json", "requirement_mapping.json", "historical_inputs.json",
    "independent_findings.md", "verification_results.json",
)


def active_paths(policy):
    active = {f"{name}.md" for name in TOP} | {DCR, "phase_state.json", "reports/P00_V2_design_review.md"}
    active |= {str(path.relative_to(DOCS)).replace("\\", "/") for path in (DOCS / "specs").glob("V2_*.md")}
    active |= {row["spec"] for row in policy["stages"].values()}
    for index in range(3, 11):
        choices = list((DOCS / "specs").glob(f"P{index:02}_*.md"))
        if len(choices) != 1:
            raise ValueError(f"ambiguous migration spec P{index:02}")
        active.add(f"specs/{choices[0].name}")
    return sorted(active)


def build(provisional=False):
    policy = read_json(HERE / "design_policy.json")
    history = read_json(HERE / "historical_inputs.json")
    baseline = history["reviewed_baseline"]
    verification = [f"evidence/P00_V2/{name}" for name in VERIFICATION]
    verification += ["agent_work_manifest.json"]
    if provisional:
        verification = [name for name in verification if (DOCS / name).is_file()]
    active = active_paths(policy)
    if provisional:
        active = [name for name in active if (DOCS / name).is_file()]
    manifest = {
        "schema": "s05.v2.design-freeze-candidate.v1",
        "design_version": policy["design_version"],
        "design_status": "DRAFT_READY_FOR_REVIEW",
        "reviewed_baseline_commit": baseline,
        "active_documents": {path: file_sha(DOCS / path) for path in active},
        "historical_evidence_identity": history,
        "dcr_identity": {"id": "DCR-005", "path": DCR, "sha256": file_sha(DOCS / DCR)},
        "requirement_mapping": read_json(HERE / "requirement_mapping.json"),
        "stage_dag": policy["stages"],
        "technical_gate_definitions": policy["stages"],
        "audited_source_sha256": {path: sha(git_bytes(baseline, path)) for path in history["source_paths"]},
        "verification_artifacts": {path: file_sha(DOCS / path) for path in verification},
        "approval_status": "PENDING_HUMAN_REVIEW",
        "execution_authorized": False,
    }
    for key in ("proposed_training_configuration", "resource_budget", "allowed_autonomous_changes",
                "forbidden_changes", "deployment_grades", "formal_scoring_requirements", "final_test_requirements"):
        manifest[key] = policy[key]
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provisional", action="store_true")
    parser.add_argument("--out", type=Path, default=DOCS / "design_freeze_candidate.json")
    args = parser.parse_args()
    args.out.write_bytes((json.dumps(build(args.provisional), ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
