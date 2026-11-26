#!/usr/bin/env python3
"""Independently recheck DCR-008 evidence without running the candidate or physics."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


SPLIT_TOTALS = {"Train": (24, 228, 86_338, 85_382), "Validation": (8, 80, 30_275, 29_150)}
ACTIVE_TRACKED, LIMITED_ACTIVE, HOLD, FALLBACK, FAILED = range(1, 6)
STATUS = {"1": "ACTIVE_TRACKED", "2": "LIMITED_ACTIVE", "3": "HOLD", "4": "FALLBACK", "5": "FAILED"}
GROUPS = (slice(0, 7), slice(7, 14), slice(14, 34), slice(34, 54))
ROLES = ("left_arm", "right_arm", "left_hand", "right_hand")
SCOPE = "DCR_008_CANDIDATE_ONLY"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_coverage(entry, paired):
    count = entry["counts"]
    planned, valid = count["planned"], count["hmb_valid"]
    require(all(type(value) is int and value >= 0 for value in count.values()), "invalid count")
    require(sum(count[label] for label in STATUS.values()) == planned, "planned rows do not reconcile")
    require(sum(count[label + "_HMB_VALID"] for label in STATUS.values()) == valid <= planned,
            "HMB-valid rows do not reconcile")
    active = count["ACTIVE_TRACKED_HMB_VALID"] + count["LIMITED_ACTIVE_HMB_VALID"]
    require(active == count["candidate_active_hmb"], "incorrect active numerator")
    require(count["candidate_active"] == count["ACTIVE_TRACKED"] + count["LIMITED_ACTIVE"],
            "incorrect planned active numerator")
    require(entry["candidate_active_coverage_hmb"] == (active / valid if valid else None),
            "incorrect active fraction")
    require(entry["shadow_85_percent_candidate_active"] == (100 * active >= 85 * valid if valid else None),
            "incorrect shadow result")
    require(entry["official_coverage_pass"] is False, "candidate claimed official coverage")
    for label in STATUS.values():
        require(entry["rates_per_planned"][label] == (count[label] / planned if planned else None),
                f"incorrect planned fraction for {label}")
        require(entry["rates_per_hmb_valid"][label] == (count[label + "_HMB_VALID"] / valid if valid else None),
                f"incorrect HMB-valid fraction for {label}")
    for field, key in (("direct_nominal_legal_coverage_hmb", "direct_nominal_legal"),
                       ("direct_exact_wrist_coverage_hmb", "direct_lr_exact")):
        require(entry[field] == (count[key] / valid if paired and valid else None), f"incorrect {field}")
    histories = count["hmb_five_frame"]
    require(entry["candidate_five_frame_coverage"] == (count["candidate_five_frame"] / histories if histories else None),
            "incorrect five-frame fraction")


def check_traces(trace, expected):
    """Use only saved, hashed rows. Recompute numerical eligibility and counts."""
    with np.load(trace, allow_pickle=False) as arrays:
        fields = {key: arrays[key] for key in (
            "row_class", "group_class", "path_accepted", "accepted", "fallback", "held", "sdk_status",
            "substep_status", "kernel_proposed", "native_flags", "q_nominal", "q_issued", "q_changed",
            "limit_applied", "original_arm_pose_error", "hand_request_rad", "hmb_joint_valid", "mapping_valid",
            "provider_observed", "hmb_five_frame", "candidate_five_frame", "direct_observed", "direct_arm_sdk_valid",
            "direct_arm_exact", "direct_nominal_legal", "witness_sampled", "witness_ok", "witness_q",
            "witness_error_m", "cause", "witness_reason", "tick_ns", "cutoff_ns", "issue_ns", "generation",
            "capture_bounds_ns", "arrival_bounds_ns", "qd_valid", "qd_nominal", "native_rate_limit_applied",
            "native_rate_limit_proposed", "hand_limit_applied",
        )}
    rows, groups = fields["row_class"], fields["group_class"]
    hmb, observed = fields["hmb_joint_valid"], fields["provider_observed"]
    q = fields["q_nominal"]
    size = len(rows)
    for field, shape in (("group_class", (size, 4)), ("q_nominal", (size, 54)),
                         ("path_accepted", (size, 5)), ("q_issued", (size, 5, 54)),
                         ("native_flags", (size, 5, 2, 8)), ("substep_status", (size, 5, 4))):
        require(fields[field].shape == shape, f"incorrect {field} shape")
    require(np.isin(groups, tuple(map(int, STATUS))).all(), "unknown group state")
    require(np.array_equal(rows, groups.max(axis=1)), "row state precedence differs from group evidence")
    require(np.isfinite(q).all() and np.isfinite(fields["q_issued"]).all(), "nonfinite command")
    require(np.array_equal(q[observed], fields["q_issued"][observed, -1]), "final command not in issue trace")
    require(np.array_equal(fields["tick_ns"], fields["cutoff_ns"]), "information cutoff changed")
    times = fields["tick_ns"][observed]
    require((fields["capture_bounds_ns"][observed, :, 1] <= times[:, None]).all(), "used future capture")
    require((fields["arrival_bounds_ns"][observed, :, 1] <= times[:, None]).all(), "used future arrival")
    require(np.array_equal(fields["issue_ns"][observed], times[:, None] + np.arange(5) * 4_000_000),
            "issue times differ from the causal schedule")

    accepted = (fields["sdk_status"] == 1) & fields["path_accepted"].all(axis=1)[:, None]
    fallback = (fields["sdk_status"] == 7) | (fields["substep_status"] == 7).any(axis=1)
    failed = (fields["sdk_status"] == 9) | (fields["substep_status"] == 9).any(axis=1)
    require(np.array_equal(fields["accepted"], accepted), "accepted mask mismatch")
    require(np.array_equal(fields["fallback"], fallback), "fallback mask mismatch")
    require(np.array_equal(fields["held"], groups == HOLD), "hold mask mismatch")
    limited = np.column_stack([fields["limit_applied"][:, section].any(axis=1) for section in GROUPS])
    changed = np.column_stack([fields["q_changed"][:, section].any(axis=1) for section in GROUPS])
    arm_exact = (fields["original_arm_pose_error"] <= [1e-6, 1e-5]).all(axis=2)
    hands = q[:, 14:].reshape(size, 2, 20)
    hand_exact = (np.abs(hands - fields["hand_request_rad"]) <= 1e-6).all(axis=2)
    exact = np.column_stack((arm_exact, hand_exact))
    calculated = np.full((size, 4), FAILED, dtype=np.uint8)
    calculated[observed] = HOLD
    calculated[accepted & exact & ~limited] = ACTIVE_TRACKED
    calculated[accepted & (changed | exact) & limited] = LIMITED_ACTIVE
    calculated[fallback] = FALLBACK
    calculated[failed] = FAILED
    require(np.array_equal(groups, calculated), "classifications differ from independently derived eligibility")

    active = rows <= LIMITED_ACTIVE
    require(not (active & (~hmb | ~observed | ~fields["mapping_valid"])).any(), "credit for an inadmissible row")
    require(not (active & fields["native_flags"][..., 7].any(axis=(1, 2))).any(), "credit after native hard fault")
    issue = fields["issue_ns"].ravel()
    generation = np.repeat(fields["generation"], 5)
    adjacent = (generation[:-1] >= 0) & (np.diff(generation) == 0) & (np.diff(issue) == 4_000_000)
    substeps = fields["substep_status"].reshape(-1, 4)
    pair_hold = np.isin(substeps, (8, 9)).all(axis=1)
    issued = fields["q_issued"].reshape(-1, 54)
    require((np.diff(issued, axis=0)[adjacent & pair_hold[1:]] == 0).all(), "paired hold or failure changed command")
    after_failure = adjacent & (substeps[:-1] == 9).all(axis=1)
    require((substeps[1:][after_failure] == 9).all() and not fields["kernel_proposed"].ravel()[1:][after_failure].any(),
            "native hard failure failed to latch until the next generation")
    previous = np.r_[False, (np.diff(fields["tick_ns"]) == 20_000_000) & (np.diff(fields["generation"]) == 0)]
    if size:
        stored_delta = np.abs(q[1:] - q[:-1]) > 1e-6
        require(np.array_equal(fields["q_changed"][1:][previous[1:]], stored_delta[previous[1:]]),
                "recorded motion exceeds its mask or is unrecorded")
        require(np.array_equal(fields["qd_valid"], active & previous & np.r_[False, active[:-1]]),
                "velocity accepted with gap, reset or ineligible state")
        require(np.allclose(fields["qd_nominal"][1:][fields["qd_valid"][1:]],
                            (q[1:] - q[:-1])[fields["qd_valid"][1:]] / .020, rtol=0, atol=1e-9),
                "velocity differs from the admitted consecutive command")
    histories = np.zeros(size, dtype=bool)
    for i in range(4, size):
        if fields["hmb_five_frame"][i]:
            histories[i] = bool(active[i - 4:i + 1].all() and previous[i - 3:i + 1].all())
    require(np.array_equal(fields["candidate_five_frame"], histories), "incorrect active-history admission")

    proved = fields["witness_ok"]
    require(not (proved & (~active | ~fields["witness_sampled"])).any(), "witness outside active sampled domain")
    require((np.abs(fields["witness_q"][proved] - q[proved])
             <= np.deg2rad(np.repeat([8., 5.], [14, 40])) + 1e-9).all(), "witness outside residual envelope")
    computed = derive_counts(fields, active)
    require(Counter(expected) == computed, "trace counts differ from its partition summary")


def derive_counts(fields, active):
    rows, groups, hmb = fields["row_class"], fields["group_class"], fields["hmb_joint_valid"]
    count = Counter(planned=len(rows), hmb_valid=int(hmb.sum()), candidate_active=int(active.sum()),
                    candidate_active_hmb=int((hmb & active).sum()))
    for name, field in (
        ("mapped", "mapping_valid"), ("candidate_observed", "provider_observed"),
        ("direct_mapped_observed", "direct_observed"), ("direct_nominal_legal", "direct_nominal_legal"),
        ("hmb_five_frame", "hmb_five_frame"), ("candidate_five_frame", "candidate_five_frame"),
        ("witness_selected", "witness_sampled"), ("witness_proven", "witness_ok"),
    ):
        count[name] = int(fields[field].sum())
    for kind in ("sdk", "exact"):
        pairs = fields["direct_arm_sdk_valid" if kind == "sdk" else "direct_arm_exact"]
        count[f"direct_lr_{kind}"] = int(pairs.all(axis=1).sum())
        for side, label in enumerate(("left", "right")):
            count[f"direct_{label}_{kind}"] = int(pairs[:, side].sum())
    for key, field in (("joint_limit_reported", "limit_applied"),
                       ("native_rate_limit_proposed", "native_rate_limit_proposed"),
                       ("native_rate_limit_applied", "native_rate_limit_applied"),
                       ("hand_slew_or_joint_limit_applied", "hand_limit_applied")):
        value = fields[field]
        count[key] = int(value.any(axis=tuple(range(1, value.ndim))).sum())
    for key, index in (("braking_proposed", 2), ("exact_stop_proposed", 5)):
        value = fields["native_flags"][..., index] & fields["kernel_proposed"][..., None]
        count[key] = int(value.any(axis=(1, 2)).sum())
    count["static_5mm_witness"] = int((fields["witness_ok"] & (fields["witness_error_m"] <= .005)).sum())
    for raw_code, label in STATUS.items():
        code = int(raw_code)
        count[label] = int((rows == code).sum())
        count[label + "_HMB_VALID"] = int(((rows == code) & hmb).sum())
        for side, role in enumerate(ROLES):
            count[f"{role}_{label}"] = int((groups[:, side] == code).sum())
    count.update(f"cause:{reason}" for reason in fields["cause"] if reason)
    count.update(f"witness:{reason}" for reason in fields["witness_reason"][fields["witness_sampled"]])
    return count


def audit(directory, *, metadata_only=False, allow_bounded=False):
    summary_path = directory / "candidate_comparison.json"
    summary = read_json(summary_path)
    inputs = read_json(directory / "candidate_inputs.json")
    prefix = read_json(directory / "candidate_prefix_controls.json")
    require(summary["schema"] == "s05.dcr008.nominal_evaluation.v1", "unknown summary schema")
    require(summary["scope"] == inputs["scope"] == prefix["scope"] == SCOPE, "unexpected scope")
    require(summary["identity_sha256"] == inputs["identity_sha256"], "run identity changed")
    require(summary["status_dictionary"] == STATUS, "status dictionary changed")
    require(summary["candidate_native_flags_order"] == [
        "accepted", "ik_target_accepted", "analytic_braking", "workspace_projected", "rate_limited",
        "analytic_exact_stop_used", "pair_accepted", "pair_hard_fault",
    ], "native flag ordering changed")
    for field in ("official_admission", "eligible_for_training", "formal_contract_frozen"):
        require(summary[field] is False, f"candidate cannot set {field}")
    require(summary["online_teleop_and_measured_feedback"] == "UNKNOWN_REQUIRES_ACTUAL_D1", "scope exceeds seam")
    require(summary["m1_continuity_gate"] == "STILL_FAILS_INDEPENDENTLY", "old continuity gate omitted")
    full = summary["full_comparison"]
    require(full or allow_bounded, "full comparison is absent")
    if full:
        require(set(summary["split_results"]) == set(SPLIT_TOTALS) and inputs["split"] == "Both",
                "full comparison must include both signed splits")
    paired = "direct_bank" in inputs["identity_sha256"]
    require(not full or paired, "original paired reference missing")

    roots_seen, traces_seen, partitions_seen = set(), set(), set()
    output = {}
    for split, entry in summary["split_results"].items():
        verify_coverage(entry, paired)
        accumulated = Counter()
        roots = defaultdict(Counter)
        for partition in entry["partitions"]:
            verify_coverage(partition, paired)
            root, key = partition["root"], partition["partition"]
            require(key not in partitions_seen, f"duplicate partition {key}")
            partitions_seen.add(key)
            path = (directory / partition["trace"]).resolve()
            relative = path.relative_to(directory)
            require(relative.parts[0] == split and path.suffix == ".npz", f"wrong trace path for {key}")
            require(path not in traces_seen, f"reused trace path for {key}")
            traces_seen.add(path)
            if not metadata_only:
                require(read_json(path.with_suffix(".json")) == partition, f"partition sidecar mismatch for {key}")
                require(digest(path) == partition["trace_sha256"], f"trace digest mismatch for {key}")
                try:
                    check_traces(path, partition["counts"])
                except ValueError as error:
                    raise ValueError(f"{split}/{key}: {error}") from error
            accumulated.update(partition["counts"])
            roots[root].update(partition["counts"])
        require(set(roots) == set(entry["per_root"]) and roots_seen.isdisjoint(roots), "root identity or split leakage")
        roots_seen.update(roots)
        for root, count in roots.items():
            recorded = entry["per_root"][root]
            verify_coverage(recorded, paired)
            require(Counter(recorded["counts"]) == count, f"root counts differ for {root}")
        require(Counter(entry["counts"]) == accumulated, f"split totals differ for {split}")
        population = (len(roots), len(entry["partitions"]), accumulated["planned"], accumulated["hmb_valid"])
        require(population[:2] == (entry["root_count"], entry["partition_count"]), "declared index size differs")
        if full:
            require(entry["full_signed_split"] and population == SPLIT_TOTALS[split], "signed denominator differs")
        output[split] = {"roots": population[0], "partitions": population[1], "planned": population[2],
                         "hmb_joint_valid": population[3], "hmb_active": accumulated["candidate_active_hmb"],
                         "active_coverage": entry["candidate_active_coverage_hmb"],
                         "direct_nominal_legal": accumulated["direct_nominal_legal"],
                         "shadow_85_percent": entry["shadow_85_percent_candidate_active"]}

    checks = prefix["roots"]
    require(set(checks) == roots_seen, "prefix ledger roots differ from comparison")
    for root, outcomes in checks.items():
        for finding in outcomes.values():
            require(finding["compared_fields"] > 0 and not finding["deletion_differences"]
                    and not finding["replacement_differences"], f"prefix check not clean for {root}")
    ordinary = sum("ordinary" in outcomes for outcomes in checks.values())
    barrier = sum("generation_barrier" in outcomes for outcomes in checks.values())
    require((ordinary, barrier) == (prefix["ordinary_checked_roots"], prefix["generation_barrier_checked_roots"]),
            "prefix support count mismatch")
    return {"summary_sha256": digest(summary_path), "full_comparison": full, "paired": paired,
            "verified_trace_files": 0 if metadata_only else len(traces_seen), "metadata_only": metadata_only,
            "ordinary_prefix_roots": ordinary, "barrier_prefix_roots": barrier, "splits": output,
            "official_admission": False, "physical_or_online_qualification": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing the three candidate JSON reports")
    parser.add_argument("--metadata-only", action="store_true", help="Check indexes without reading trace bytes")
    parser.add_argument("--allow-bounded", action="store_true", help="Permit a smoke output with fewer than all rows")
    args = parser.parse_args()
    result = audit(args.directory.resolve(), metadata_only=args.metadata_only, allow_bounded=args.allow_bounded)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
