"""Seal the active v2.0.1 design candidate without changing history."""
import argparse
import json
from pathlib import Path

from check_design import ARCHIVE, ARCHIVE_SHA, DOCS, HERE, file_sha, read_json


DCR = "dcr/DCR-006_teleop_validation_egopro.md"
REPORT = "reports/P00_V2_1_design_review.md"
TOP = ("README", "MASTER_DESIGN", "REQUIREMENTS", "CONTRACTS", "ROADMAP", "EVALUATION_PROTOCOL",
       "TRACEABILITY_MATRIX", "PROJECT_STATE", "DECISION_LOG", "RISK_REGISTER", "OPEN_ISSUES",
       "MULTI_AGENT_PROTOCOL")
ARTIFACTS = ("check_design.py", "test_check_design.py", "seal_design.py", "verify_baseline.py",
             "design_policy.json", "historical_inputs.json", "requirement_mapping.json",
             "independent_findings.md", "verification_results.json", "agent_work_manifest.json")


def active_paths(policy):
    paths = {DCR, REPORT, "phase_state.json", *(f"{name}.md" for name in TOP)}
    paths |= {row["spec"] for row in policy["stages"].values()}
    paths |= {f"specs/{path.name}" for path in (DOCS / "specs").glob("V2_*.md")}
    for n in range(3, 11):
        choices = tuple((DOCS / "specs").glob(f"P{n:02}_*.md"))
        if len(choices) != 1:
            raise ValueError(f"ambiguous P{n:02} migration")
        paths.add(f"specs/{choices[0].name}")
    return sorted(paths)


def verification_paths():
    return sorted(f"evidence/P00_V2_1/{path}" for path in ARTIFACTS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--out", type=Path, default=DOCS / "design_freeze_candidate.json")
    args = parser.parse_args()
    policy = read_json(HERE / "design_policy.json")

    def inventory(paths):
        return {path: file_sha(DOCS / path) for path in paths if not args.draft or (DOCS / path).is_file()}

    active = inventory(active_paths(policy))
    artifact_manifest = {
        "schema": "s05.v2.1.design-freeze-candidate.v1",
        "design_version": "2.0.1",
        "design_status": "DRAFT_READY_FOR_REVIEW",
        "reviewed_baseline_commit": ARCHIVE,
        "previous_freeze_candidate_sha256": ARCHIVE_SHA,
        "active_documents": active,
        "dcr_identity": {"id": "DCR-006", "path": DCR, "sha256": active[DCR]},
        "historical_evidence_identity": read_json(HERE / "historical_inputs.json"),
        "requirement_mapping": read_json(HERE / "requirement_mapping.json"),
        "verification_artifacts": inventory(verification_paths()),
        "policy": policy,
        "approval_status": "PENDING_HUMAN_REVIEW",
        "execution_authorized": False,
    }
    args.out.write_bytes((json.dumps(artifact_manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
