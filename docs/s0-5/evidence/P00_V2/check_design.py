"""Read-only V2 design and synthetic governance verification. No execution entry point."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

DOCS = Path(__file__).resolve().parents[2]
REPO = DOCS.parents[1]
HERE = Path(__file__).resolve().parent
STAGES = ("V1-A", "V1-B", "V1-C0", "V1-C1", "V1-C2", "V1-C3", "V1-D")
DEPENDENCIES = {
    "V1-A": ((), ()), "V1-B": ((), ("V1-A",)),
    "V1-C0": (("V1-A", "V1-B"), ("V1-A", "V1-B")),
    "V1-C1": (("V1-C0",), ("V1-C0",)),
    "V1-C2": (("V1-C0", "V1-C1"), ("V1-C0", "V1-C1")),
    "V1-C3": (("V1-C0", "V1-C2"), ("V1-C0", "V1-C2")),
    "V1-D": (("V1-C0", "V1-C3"), ("V1-C0", "V1-C3")),
}
PUBLISHED_STATE = {
    "design_status": "DRAFT_READY_FOR_REVIEW", "handoff_status": "DESIGN_READY_FOR_REVIEW",
    "approval_status": "PENDING_HUMAN_REVIEW", "design_approval": None,
    "execution_authorized": False, "execution_authorization": None,
    "campaign_id": None, "m1_continuous_use": "PENDING_HUMAN_REVIEW",
    "formal_physical_scoring": "PENDING_HUMAN_REVIEW", "final_test": "BLOCKED",
}
HEX = re.compile(r"[0-9a-f]{64}\Z")
HASHED_FIELDS = ("design_sha256", "budget_sha256", "evaluation_sha256")
LINK = re.compile(r"\]\((?:<([^>]+)>|([^)\s]+))\)")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return sha(path.read_bytes())


def read_json(path):
    return json.loads(path.read_bytes())


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args])


def git_bytes(commit, path):
    return git("show", f"{commit}:{path}")


def canonical_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def local_file(root, path):
    candidate = (root / path).resolve()
    require(candidate.is_relative_to(root.resolve()) and candidate.is_file(), f"missing or external: {path}")
    return candidate


def verify_digest(actual_bytes, expected, name):
    require(bool(HEX.fullmatch(expected)) and sha(actual_bytes) == expected, f"hash mismatch: {name}")


def check_candidate_state(state, historical):
    v2 = state["policy_first_v2"]
    legacy = {key: value for key, value in state.items() if key != "policy_first_v2"}
    require(legacy == historical, "historical phase ledger changed")
    require(v2["schema"] == "s05.phase-state.v2" and v2["design_version"] == "2.0.0", "phase-state version")
    require(all(v2.get(key) == value for key, value in PUBLISHED_STATE.items()), "design-only state has expanded authority")
    require(v2["stage_states"] == dict.fromkeys(STAGES, "NOT_STARTED"), "stage started during design revision")
    require(v2["hardware"] == dict.fromkeys(("P01-T", "C2-RT", "robot-base", "real_deployment"), "BLOCKED"),
            "hardware gate changed")
    require(v2["historical_P02_limited_acceptance"]["scope"] == "HISTORICAL_SIMULATION_STATIC_RESEARCH_ONLY",
            "P02 scope changed")


def trusted_receipt(identifier, kind, trusted, bindings):
    """The caller supplies independently authenticated records, never ledger-supplied approvals."""
    require(identifier in trusted, f"unverified {kind}")
    receipt = trusted[identifier]
    require(receipt.get("kind") == kind and receipt.get("active") is True, f"inactive or wrong {kind}")
    require(all(receipt.get(key) == value for key, value in bindings.items()), f"binding differs: {kind}")
    return receipt


def check_budget(budget, used, reserve):
    require(set(used) == set(reserve) == set(budget), "incomplete resource accounting")
    total = {}
    for key, cap in budget.items():
        values = (used[key], reserve[key])
        require(all(type(value) is int and value >= 0 for value in values), f"invalid accounting: {key}")
        total[key] = sum(values)
        require(total[key] <= cap, f"resource exhausted: {key}")
    for total_key, fields in {
        "train_total": ("train_static_bucket", "train_dynamic", "train_mixed"),
        "eval_total": ("eval_c0", "eval_training_watch", "eval_release"),
    }.items():
        for ledger in (used, reserve):
            require(ledger[total_key] == sum(ledger[field] for field in fields), f"bucket mismatch: {total_key}")
    require(total["train_smoke_c0"] <= total["train_static_bucket"], "C0 budget outside static bucket")
    require(total["blocked_wait_seconds"] <= total["wall_seconds"], "wait time outside elapsed time")


def check_technical_gate(policy, stage, gate_ref, trusted, bindings, evidence_root):
    gate = trusted_receipt(gate_ref, "technical_gate", trusted, {**bindings, "stage": stage})
    require(gate["reviewer"] and gate["reviewer"] not in gate["core_authors"] and bool(gate["core_authors"]),
            "reviewer is not independent")
    require(gate["verdict"] == "TECHNICAL_PASS" and gate["exit_code"] == 0, "gate failed")
    expected = set(policy["stages"][stage]["checks"])
    require(set(gate["artifacts"]) == expected, "gate evidence missing or unexpected")
    for name, item in gate["artifacts"].items():
        path = local_file(evidence_root, item["path"])
        verify_digest(path.read_bytes(), item["sha256"], name)
    return gate


def check_transition(policy, ledger, action, trusted, evidence_root):
    """Check a proposal in memory. Only a separate authorized runner may act on it."""
    require(action["kind"] in {"START", "PASS", "REAL_DEPLOY", "FINAL_TEST", "FORMAL_SCORING"},
            "unknown transition")
    stage = action.get("stage")
    identities = {key: ledger[key] for key in HASHED_FIELDS}
    require(all(HEX.fullmatch(value) for value in identities.values()), "invalid immutable identity")
    require(ledger.get("campaign_id") and ledger.get("stop_reason") is None, "campaign stopped or unnamed")
    require(identities["budget_sha256"] == canonical_sha(policy["resource_budget"]), "modified budget")
    approved = trusted_receipt(ledger["design_approval"], "human_design_approval", trusted, identities)
    campaign = {**identities, "campaign_id": ledger["campaign_id"]}
    authorization = trusted_receipt(ledger["execution_authorization"], "human_campaign_authorization",
                                    trusted, campaign)
    require(set(authorization["scope"]) == set(STAGES), "wrong campaign scope")
    if stage == "V1-A":
        require(approved.get("m1_continuous_simulation") is True, "M1 continuous use lacks approval")
    check_budget(policy["resource_budget"], ledger["consumed"], action["reservation"])
    if action["kind"] in {"REAL_DEPLOY", "FINAL_TEST", "FORMAL_SCORING"}:
        obligations = {
            "REAL_DEPLOY": policy["deployment_grades"]["REAL_DEPLOYABLE"],
            "FINAL_TEST": policy["final_test_requirements"],
            "FORMAL_SCORING": policy["formal_scoring_requirements"],
        }[action["kind"]]
        for obligation in obligations:
            trusted_receipt(action["external_receipts"].get(obligation), obligation, trusted, identities)
        return
    require(stage in STAGES, "stage outside V1")
    context = trusted_receipt(ledger.get("execution_context_ref"), "verified_execution_context", trusted, campaign)
    row = policy["stages"][stage]
    dependency_key = "gate_requires" if action["kind"] == "PASS" else "start_requires"
    reviewed = set()

    def verify_gate(name, reference):
        fingerprint = context["input_fingerprints"].get(name)
        require(isinstance(fingerprint, str) and HEX.fullmatch(fingerprint), f"unverified input identity: {name}")
        parents = policy["stages"][name]["gate_requires"]
        parent_refs = {parent: ledger["gate_refs"].get(parent) for parent in parents}
        gate = check_technical_gate(policy, name, reference, trusted, {
            **campaign, "input_fingerprint": fingerprint, "upstream_refs": parent_refs,
        }, evidence_root)
        if name == "V1-C0":
            require(context.get("wm_current") == reference, "WM qualification invalidated")
        for parent in parents:
            verify_parent(parent)
        return gate

    def verify_parent(parent):
        if parent in reviewed:
            return
        require(ledger["states"].get(parent) == "TECHNICAL_PASS", f"dependency {parent} not passed")
        verify_gate(parent, ledger["gate_refs"].get(parent))
        reviewed.add(parent)

    for parent in row[dependency_key]:
        verify_parent(parent)
    if action["kind"] == "PASS":
        require(ledger["states"].get(stage) == "IN_PROGRESS", "gate without active stage")
        verify_gate(stage, action["gate_ref"])
    else:
        state = ledger["states"].get(stage)
        require(state in {"NOT_STARTED", "FAILED", "BLOCKED"}, "illegal start or resume")
        retries = ledger["retries"].get(stage, 0)
        require(type(retries) is int and retries >= 0, "invalid retry accounting")
        require((state == "NOT_STARTED" and retries == 0) or
                (state in {"FAILED", "BLOCKED"} and retries < policy["allowed_autonomous_changes"]["retries_per_stage"]),
                "retry cap exhausted")


def validate_links(files):
    count = 0
    externals = set()
    for relative in sorted(files):
        path = local_file(DOCS, relative)
        if path.suffix != ".md":
            continue
        # The active package uses GFM inline links. Code fences hold illustrative commands.
        body = re.sub(r"(?ms)^ *\x60{3,}.*?^ *\x60{3,}[^\n]*", "", path.read_text(encoding="utf-8"))
        for target in LINK.finditer(body):
            parsed = urlsplit(target[1] or target[2])
            link_path = unquote(parsed.path)
            if parsed.scheme:
                require(parsed.scheme == "https", f"unsupported link scheme in {relative}")
                pinned = re.fullmatch(r"/FEGkkd/dexgym/(?:blob|tree)/([a-f0-9]{40})/(.+)", link_path)
                if parsed.netloc == "github.com" and pinned:
                    git("cat-file", "-e", f"{pinned[1]}:{pinned[2]}")
                else:
                    externals.add(parsed.geturl())
            else:
                leaf = (path.parent / link_path).resolve() if link_path else path
                require(leaf.is_relative_to(REPO) and leaf.exists(), f"broken link: {relative} -> {link_path}")
                if parsed.fragment and leaf.suffix == ".md":
                    anchor = unquote(parsed.fragment)
                    headings = re.findall(r"(?m)^#{1,6}\s+(.+?)\s*#*$", leaf.read_text(encoding="utf-8"))
                    slugs = {re.sub(r"[^\w -]", "", item.lower()).replace(" ", "-") for item in headings}
                    require(anchor in slugs, f"missing anchor: {relative} -> {anchor}")
            count += 1
    return count, externals


def validate_design(manifest, policy, historical_inputs, requirements):
    require(manifest["design_version"] == policy["design_version"] == "2.0.0", "design version")
    require(manifest["reviewed_baseline_commit"] == policy["reviewed_baseline"] == historical_inputs["reviewed_baseline"],
            "baseline mismatch")
    require(manifest["historical_evidence_identity"] == historical_inputs, "historical evidence manifest mismatch")
    require(manifest["approval_status"] == policy["approval_status"] == "PENDING_HUMAN_REVIEW"
            and manifest["execution_authorized"] is policy["execution_authorized"] is False,
            "candidate authorization boundary")
    require(manifest["requirement_mapping"] == requirements, "requirement map differs")
    for key in ("deployment_grades", "proposed_training_configuration", "resource_budget",
                "allowed_autonomous_changes", "forbidden_changes", "formal_scoring_requirements", "final_test_requirements"):
        require(manifest[key] == policy[key], f"policy content differs: {key}")
    require(manifest["stage_dag"] == manifest["technical_gate_definitions"] == policy["stages"], "stage DAG differs")
    require(tuple(policy["stages"]) == STAGES, "unknown stage order")
    for stage, row in policy["stages"].items():
        actual = (tuple(row["start_requires"]), tuple(row["gate_requires"]))
        require(actual == DEPENDENCIES[stage], f"invalid DAG: {stage}")
        require(row["checks"] and len(set(row["checks"])) == len(row["checks"]), f"missing gate checks: {stage}")
    require({"physical_robot_base", "C2-RT", "P01-T", "hardware_servo", "external_metrology",
             "site_safety", "shadow", "hil", "human_real_approval"} <=
            set(policy["deployment_grades"]["REAL_DEPLOYABLE"]), "real deployment gates missing")
    require({"physical_robot_base", "C2-RT", "P01-T", "hardware_servo", "external_metrology",
             "formal_scoring_human_approval"} <= set(policy["formal_scoring_requirements"]), "formal scoring gates missing")
    require({"final_test_sealed", "access_log", "independent_lineage", "final_test_human_approval"} <=
            set(policy["final_test_requirements"]), "Final Test gates missing")
    expected = [f"S05-REQ-{i:03}" for i in range(1, 36)]
    require([row["id"] for row in requirements] == expected, "missing or repeated requirements")
    required_fields = {"id", "disposition", "title", "stage", "module", "unit", "integration", "metric", "design", "gate", "evidence"}
    for row in requirements:
        require(set(row) == required_fields and all(row.values()), f"incomplete requirement: {row['id']}")
        local_file(DOCS, row["design"])
    for filename in ("REQUIREMENTS.md", "TRACEABILITY_MATRIX.md"):
        body = (DOCS / filename).read_text(encoding="utf-8")
        found = re.findall(r"(?m)^\| (S05-REQ-\d{3}) \| ([A-Z]+) \|", body)
        require(found == [(row["id"], row["disposition"]) for row in requirements], f"requirement table differs: {filename}")


def audit_history(history, sources):
    base = history["reviewed_baseline"]
    sim = history["sim0"]
    git("merge-base", "--is-ancestor", base, sim)
    for path in (history["sim0_approval"], history["sim0_dcr"]):
        git("cat-file", "-e", f"{sim}:{path}")
    approval_bytes = git_bytes(sim, history["sim0_approval"])
    verify_digest(approval_bytes, history["sim0_approval_sha256"], history["sim0_approval"])
    approval = approval_bytes.decode("utf-8")
    require(history["required_evidence"]["evidence/P02/final_handoff/execution_receipt.json"] in approval,
            "SIM0 approval receipt mismatch")
    for relative, expected in history["required_evidence"].items():
        path = f"docs/s0-5/{relative}"
        verify_digest(git_bytes(base, path), expected, path)
        verify_digest(local_file(DOCS, relative).read_bytes(), expected, relative)
    # Every tracked historical report, evidence payload, historical spec and DCR is immutable.
    frozen_paths = git("ls-tree", "-rz", "--name-only", base, "--", "docs/s0-5").decode("utf-8").split("\0")
    migration_specs = {f"P{i:02}_" for i in range(3, 11)}
    preserved = []
    for path in frozen_paths:
        rel = path.removeprefix("docs/s0-5/")
        if rel.startswith(("reports/", "evidence/", "dcr/")) or (
            rel.startswith("specs/") and not any(Path(rel).name.startswith(p) for p in migration_specs)
        ) or rel in {"EXECUTION_BASELINE.md", "REPOSITORY_AUDIT.md", "HISTORICAL_DATA_USAGE.md"}:
            preserved.append(path)
    # Byte compare using Git's tree comparison rather than spawning once per historical artifact.
    changed = git("diff", "--name-only", base, "--", "docs/s0-5").decode("utf-8").splitlines()
    source_changes = git("diff", "--name-only", base, "--", "source", "tests", "scripts", "pyproject.toml").decode("utf-8")
    require(not source_changes, "campaign implementation changed during design revision")
    require(not (set(changed) & set(preserved)), "historical report/spec/receipt changed")
    require(set(sources) == set(history["source_paths"]), "missing source audit")
    for path, digest in sources.items():
        verify_digest(git_bytes(base, path), digest, path)
        verify_digest(local_file(REPO, path).read_bytes(), digest, path)
    print(f"PASS historical P00–P02: {len(preserved)} preserved paths, scoped SIM0 object, {len(sources)} audited source SHAs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DOCS / "design_freeze_candidate.json")
    parser.add_argument("--provisional", action="store_true", help="during local drafting only")
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    policy = read_json(HERE / "design_policy.json")
    requirements = read_json(HERE / "requirement_mapping.json")
    history = read_json(HERE / "historical_inputs.json")
    validate_design(manifest, policy, history, requirements)
    base = history["reviewed_baseline"]
    state = read_json(DOCS / "phase_state.json")
    check_candidate_state(state, json.loads(git_bytes(base, "docs/s0-5/phase_state.json")))
    require(state["policy_first_v2"]["historical_P02_limited_acceptance"]["git_commit"] == history["sim0"],
            "historical signoff identity changed")
    active = manifest["active_documents"]
    from seal_design import VERIFICATION, active_paths
    if not args.provisional:
        require(set(active) == set(active_paths(policy)), "incomplete active document inventory")
        expected_artifacts = {f"evidence/P00_V2/{name}" for name in VERIFICATION} | {"agent_work_manifest.json"}
        require(set(manifest["verification_artifacts"]) == expected_artifacts, "incomplete verification inventory")
    dcr = manifest["dcr_identity"]
    require(dcr["id"] == "DCR-005" and dcr["sha256"] == active[dcr["path"]], "DCR identity")
    for path, digest in {**active, **manifest["verification_artifacts"]}.items():
        verify_digest(local_file(DOCS, path).read_bytes(), digest, path)
    for row in policy["stages"].values():
        require(row["spec"] in active, f"unsealed stage: {row['spec']}")
        spec = local_file(DOCS, row["spec"]).read_text(encoding="utf-8")
        require(len(re.findall(r"(?m)^## ", spec)) >= 9 and "v2.0.0" in spec.lower(), f"incomplete stage: {row['spec']}")
    for filename in active:
        if filename.endswith(".md"):
            title = local_file(DOCS, filename).read_text(encoding="utf-8").splitlines()[0]
            require("v2.0.0" in title.lower(), f"active version mismatch: {filename}")
    cross_version_links = ["specs/HARDWARE_SHARED_MEASUREMENT_PLAN.md", "specs/P01-data-v0.2.1.md",
                           "specs/P01_teleop_data.md"]
    count, external = validate_links([*active, *cross_version_links])
    audit_history(history, manifest["audited_source_sha256"])
    print(f"PASS {len(active)} active files, {count} links, {len(requirements)} requirements, {len(STAGES)} V1 stages")
    print(f"PASS evidence/hash integrity and design-only state; external URLs: {len(external)}")
    for link in sorted(external):
        print(f"  {link}")


if __name__ == "__main__":
    main()
