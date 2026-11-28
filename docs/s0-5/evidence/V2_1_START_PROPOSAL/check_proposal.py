"""Read-only Stage I publication checks and an offline gate model for synthetic fixtures."""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[4]
DOCS = "docs/s0-5/"
PROPOSAL = DOCS + "evidence/V2_1_START_PROPOSAL/"
BASELINE = "cf07a88bc93d1aa9737b41fb83b9392921449142"
ORIGINAL_HINT = "324d771e29d436b206211f444a6d800338f871ce"
FREEZE_COMMIT = "490770e1df45832d0693380563152bd19eefa3f4"
FREEZE_SHA = "cd5bff29a5b5f683fae328898aa9060b40fc4be00e193a22d9bf18cb2d10b60b"

DOCUMENTS = (
    "dcr/DCR-009_home_anchored_human_intent.md",
    "specs/V2_1_home_anchored_geometry_proposal.md",
    "specs/V2_1_home_anchored_runtime_proposal.md",
    "specs/V2_1_home_anchored_data_gate_proposal.md",
    "reports/V2_1_home_anchored_design_audit.md",
    "reports/V2_1_home_anchored_independent_review.md",
)
ARTIFACTS = {DOCS + document for document in DOCUMENTS} | {
    PROPOSAL + name
    for name in (
        "design_candidate.json",
        "check_proposal.py",
        "test_proposal.py",
        "check_receipt.json",
    )
}
ARTIFACT_MANIFEST = PROPOSAL + "proposal_artifacts.json"
PHASES = (
    "DESIGN_PROPOSAL",
    "DCR_PENDING_HUMAN_REVIEW",
    "DESIGN_APPROVED",
    "IMPLEMENTATION",
    "PILOT_VALIDATION",
    "PILOT_APPROVED",
    "FULL_RMB_BUILD_AUTHORIZED",
    "FULL_RMB_BUILD",
    "INDEPENDENT_AUDIT",
    "READY_FOR_REVIEW",
)
APPROVALS = {
    "execution_authorized",
    "implementation_authorized",
    "pilot_authorized",
    "full_rmb_authorized",
    "training_authorized",
    "physical_actuation_authorized",
    "calibration_execution_authorized",
}
RECEIPTS_AT_ENTRY = {
    "DESIGN_APPROVED": {"DCR_HUMAN_APPROVAL"},
    "IMPLEMENTATION": {"IMPLEMENTATION_SCOPE_AUTHORIZED"},
    "PILOT_VALIDATION": {
        "PILOT_SCOPE_AUTHORIZED", "IMPLEMENTATION_VERIFIED", "CALIBRATION_QUALIFIED",
        "SIX_DISTINCT_TRAIN_ROOTS_LOCKED", "SIMULATION_ONLY",
    },
    "PILOT_APPROVED": {"PILOT_TECHNICAL_PASS", "INDEPENDENT_PILOT_REVIEW", "PILOT_HUMAN_APPROVAL"},
    "FULL_RMB_BUILD_AUTHORIZED": {
        "USER_EXPLICIT_FULL_RMB_AUTHORIZATION", "MODEL_IDENTITY_RESOLVED",
        "HOME_START_SAFETY_PASS", "TARGET_AND_REWARD_FROZEN",
        "COVERAGE_SEMANTICS_HUMAN_APPROVED", "INDEPENDENT_FULL_BUILD_REVIEW",
    },
    "FULL_RMB_BUILD": {"FULL_RUN_PROVENANCE_LOCKED"},
    "INDEPENDENT_AUDIT": {"FULL_RUN_EVIDENCE"},
    "READY_FOR_REVIEW": {"INDEPENDENT_RMB_AUDIT"},
}
COVERAGE = {
    "active_percent": 85,
    "active_denominator": "ORIGINAL_HMB_JOINT_VALID_PER_SPLIT",
    "history_percent": 70,
    "history_denominator": "ORIGINAL_SIGNED_HMB_FIVE_FRAME_WINDOWS_PER_SPLIT",
    "new_numerator_approved": False,
    "active_exclusions": [
        "PUBLISH_ONLY", "START_ONLY", "PREPOSITION", "HOLD", "FALLBACK",
        "FAILED", "UNKNOWN", "UNREACHABLE", "MISSING_FEEDBACK",
    ],
    "count_additional_cross_partition_history": False,
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.PIPE)


def git_blob(commit, relative_path):
    return git("show", f"{commit}:{relative_path}")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def cohort_from_history(index):
    splits = {}
    for split in ("Train", "Validation"):
        rows = [row for row in index["partitions"] if row["partition"]["split"] == split]
        causal = [row["evidence"]["counts"]["causal"] for row in rows]
        require(all(row["status"] == "PUBLISHED" for row in rows), f"{split} E1 status")
        splits[split] = {
            "roots": len({row["partition"]["parent"] for row in rows}),
            "partitions": len(rows),
            "planned_ticks": sum(row["partition"]["planned_ticks"] for row in rows),
            "joint_valid_ticks": sum(row["joint_valid"] for row in causal),
            "original_history_candidates": sum(
                max(length - 4, 0) for row in causal for length in row["segment_lengths_ticks"]
            ),
        }
    splits["total"] = {key: splits["Train"][key] + splits["Validation"][key] for key in splits["Train"]}
    require(splits["total"]["planned_ticks"] == index["planned_ticks"], "E1 partition denominator")
    return splits


def validate_candidate(candidate, cohort):
    identity = {
        "schema": "s05.start_home_anchored.stage_one_proposal.v1",
        "repository": "FEGkkd/dexgym",
        "branch": "s0-5",
        "reviewed_remote_head": BASELINE,
        "user_baseline_hint": ORIGINAL_HINT,
        "active_design_commit": FREEZE_COMMIT,
        "active_design_freeze_sha256": FREEZE_SHA,
        "design_version": "2.1.0-proposal",
        "dcr_id": "DCR-009",
        "status": "PENDING_HUMAN_REVIEW",
        "stage": "DCR_PENDING_HUMAN_REVIEW",
        "approval_ref": None,
        "machine_check_scope": "READ_ONLY_PHASE_ONE_PROPOSAL_AND_SYNTHETIC_TRANSITION_FIXTURES",
    }
    for key, expected in identity.items():
        require(key in candidate and candidate[key] == expected, f"incorrect proposal field {key}")
    require(all(candidate.get(key) is False for key in APPROVALS), "premature execution authority")
    require(candidate["allowed_stage_one_actions"] == [
        "READ_ONLY_SOURCE_AND_HISTORICAL_AUDIT", "DESIGN_PROPOSAL_EDIT",
        "INDEPENDENT_DESIGN_REVIEW", "SYNTHETIC_DESIGN_CHECK", "PUBLISH_PROPOSAL",
    ], "scope exceeds Stage I")
    require(candidate["cohort"] == cohort, "signed HMB cohort or original denominator mismatch")
    require(candidate["coverage"] == COVERAGE, "coverage percentages, denominators or exemptions changed")
    require(candidate["coverage"]["new_numerator_approved"] is False, "new coverage approved")
    require(tuple(candidate["phase_sequence"]) == PHASES, "unauthorized phase bypass")
    require(candidate["proposal_documents"] == [DOCS + path for path in DOCUMENTS], "proposal document list")
    require(candidate["future_pilot"] == {
        "roots": 6, "allowed_split": "Train", "fit_on_validation": False,
        "max_cold_starts_proposed": 8, "launch_requires_future_stage_authority": True,
    }, "Pilot fit, cap or authority violation")

    geometry = candidate["geometry"]
    required_fields = {
        "target_version": "START_HOME_ANCHORED_V1",
        "planned_target_role": "INDEPENDENT_SCORING_AND_REWARD_AFTER_APPROVAL_AND_REQUALIFICATION",
        "rotation_source": "FIXED_PHYSICAL_CALIBRATION",
        "physical_calibration_status": "MISSING_INDEPENDENT_QUALIFICATION",
        "approved_rotation_ref": None,
        "origin_alignment": "ONE_SHARED_MATCHED_ANATOMICAL_BILATERAL_HOME_MIDPOINT",
        "scale": 1,
        "robot_feedback_anchor_role": "MEASURED_HOME_MIDPOINT_ONLY",
        "anchor_policy": "QUALIFIED_EXPLICIT_OR_EARLIEST_COMPLETE_CAUSAL_STABLE_WINDOW_TERMINAL",
        "ineligible_observed_start": "START_UNQUALIFIED_WITHOUT_RECONSTRUCTION",
        "historical_labels": ["M0_TARGET", "M1_TARGET"],
        "separate_state_layers": ["START_TARGET", "Q_NOMINAL", "Q_ISSUED", "Q_APPLIED", "Q_MEASURED"],
    }
    for key, expected in required_fields.items():
        require(key in geometry and geometry[key] == expected, f"geometry field {key}")
    require(type(geometry["scale"]) in (int, float), "geometry scale must be numeric")
    for key in (
        "per_recording_rotation", "per_side_recenter", "force_home_palm_direction",
        "geometry_from_commands_or_feedback", "partition_restarts_native_session",
    ):
        require(geometry.get(key) is False, f"invalid geometry or epoch option {key}")


def synthetic_stage_step(source, target, synthetic_receipts):
    """Model prerequisites on dummy receipt kinds, without granting or validating real authority."""
    require(source in PHASES and target in PHASES, "out-of-scope phase")
    target_index = PHASES.index(target)
    require(target_index == PHASES.index(source) + 1, "phase jump")
    required = set().union(*(RECEIPTS_AT_ENTRY.get(phase, set()) for phase in PHASES[:target_index + 1]))
    missing = required - synthetic_receipts
    require(not missing, "missing synthetic prerequisites: " + ", ".join(sorted(missing)))
    return target


def check_file_links():
    checked = 0
    for document in DOCUMENTS:
        path = ROOT / DOCS / document
        contents = path.read_text(encoding="utf-8")
        require(not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", contents), f"unexpected control in {document}")
        for href in re.findall(r"\]\(([^)\n]+)\)", contents):
            link = urlsplit(href)
            if link.scheme or not link.path:
                continue
            local_path = (path.parent / unquote(link.path)).resolve()
            require(local_path.is_relative_to(ROOT) and local_path.is_file(), f"broken local link: {document} > {href}")
            checked += 1
    return checked


def check_repository():
    git("merge-base", "--is-ancestor", ORIGINAL_HINT, BASELINE)
    git("merge-base", "--is-ancestor", FREEZE_COMMIT, BASELINE)
    git("merge-base", "--is-ancestor", BASELINE, "HEAD")
    published = git("ls-tree", "--name-only", f"{BASELINE}:{DOCS}dcr").decode()
    historical_ids = [int(match) for match in re.findall(r"^DCR-(\d+)_", published, re.MULTILINE)]
    require(max(historical_ids) == 8, "unexpected DCR sequence at reviewed baseline")
    changed = set(git("diff", "--name-only", "--no-renames", "-z", BASELINE, "--").decode().split("\0"))
    untracked = set(git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0"))
    scope = ARTIFACTS | {ARTIFACT_MANIFEST}
    require((changed | untracked) - {""} == scope, "proposal path scope differs from reviewed baseline")

    freeze_path = DOCS + "design_freeze_candidate.json"
    freeze = git_blob(FREEZE_COMMIT, freeze_path)
    require(sha(freeze) == FREEZE_SHA and git_blob(BASELINE, freeze_path) == freeze, "v2.0.1 freeze identity")
    approved_index = json.loads(freeze)
    frozen_files = approved_index["active_documents"]
    frozen_verification = approved_index["verification_artifacts"]
    historical_index = DOCS + approved_index["historical_evidence_identity"]["historical_contracts"]
    history = json.loads(git_blob(FREEZE_COMMIT, historical_index))["required_evidence"]
    for relative, digest in (frozen_files | frozen_verification | history).items():
        path = DOCS + relative
        content = git_blob(FREEZE_COMMIT, path)
        require(sha(content) == digest and git_blob(BASELINE, path) == content, f"frozen evidence: {path}")

    hmb_manifest = json.loads(git_blob(BASELINE, DOCS + "evidence/P01_E_E1/hmb_manifest.json"))
    build_index = git_blob(BASELINE, DOCS + "evidence/P01_E_E1/build_index.json")
    require(sha(build_index) == hmb_manifest["index_sha256"], "E1 signed partition index hash")
    candidate = json.loads((ROOT / PROPOSAL / "design_candidate.json").read_bytes())
    validate_candidate(candidate, cohort_from_history(json.loads(build_index)))
    link_count = check_file_links()
    if "--verify-artifacts" in sys.argv[1:]:
        manifest = json.loads((ROOT / ARTIFACT_MANIFEST).read_bytes())
        require(manifest["reviewed_remote_head"] == BASELINE, "artifact manifest baseline")
        require(manifest["files"].keys() == ARTIFACTS, "artifact manifest bundle mismatch")
        for relative, digest in manifest["files"].items():
            require(PurePosixPath(relative).as_posix() == relative, "artifact path")
            require(sha((ROOT / relative).read_bytes()) == digest, f"artifact hash mismatch: {relative}")
    print(f"PASS Stage I scope, ancestral baseline and DCR ordering; {len(frozen_files)} design hashes, "
          f"{len(frozen_verification)} verification artifacts, {len(history)} historical receipts, "
          f"{link_count} local document links.")
    print("PASS original HMB split denominators, immutable 85%/70% gates, pending human review, "
          "and execution authorization false.")
    if "--verify-artifacts" in sys.argv[1:]:
        print(f"PASS {len(ARTIFACTS)} proposal artifact hashes.")


if __name__ == "__main__":
    try:
        check_repository()
    except (KeyError, TypeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"FAIL: {error}")
