#!/usr/bin/env python3
"""Seal the six-source DCR-009 diagnostic-only native MuJoCo input plan."""

import argparse
from dataclasses import asdict
from hashlib import sha256
from io import StringIO
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from home_anchored_start_anchor import (
    ANCHOR_SCHEMA, CORE_STREAMS, MOTION, PLAN_SCHEMA, PRIMARY, REPLAY_NS, SEARCH_NS, SENSITIVITY,
    file_sha, json_bytes, proper, scan_root, train_reader, verify_native,
)


RUN_ID = "HOME-ANCHORED-DIAGNOSTIC-DCR009-01"
SCOPE = "DIAGNOSTIC_ONLY"
VIRTUAL_SHA = "4798dfbdaf80178b265a0c325bce1b96b09221d35b4b11c86d35c650c5f3782a"
ARCHIVED_FRAMES_SHA = "33cbc93c4a2db925986f596b4b1730c7eaca026a9facfdf26aa4aaa8032c8f94"
FALLBACK = "FIRST_CAUSAL_JOINT_VALID_DIAGNOSTIC"
STABLE = {"RECONSTRUCTED_START", "EXPLICIT_START"}
LABELS = ("world_from_registered_tracker", "world_from_base_left", "world_from_base_right",
          "tracker_to_engineering_reference_left", "tracker_to_engineering_reference_right",
          "engineering_reference_from_native_tcp_left", "engineering_reference_from_native_tcp_right")
CREDIT = {"stage_credit": False, "formal_coverage_credit": False,
          "tracker_to_anatomical_mount_status": "UNKNOWN", "physical_calibration_qualified": False}


def make_frames(axis_path, archive_path):
    axis_sha, archived_sha = file_sha(axis_path), file_sha(archive_path)
    if axis_sha != VIRTUAL_SHA or archived_sha != ARCHIVED_FRAMES_SHA:
        raise ValueError("Exact virtual axis receipt and archived seven-block bridge are required")
    axis = json.loads(axis_path.read_bytes())
    if (axis["scope"] != "VIRTUAL_SCENE_COORDINATE_CONVENTION_ONLY" or axis["status"] != "PASS"
            or axis["source_convention"]["H_frame_literal"] != "openvr_world"
            or not axis["exact_model_support"]["pass"] or not all(axis["controls"].values())
            or not all(item["pass"] for item in axis["holdouts"])
            or axis["capture_heading_qualified"] or axis["physical_calibration_qualified"]
            or axis["tracker_to_anatomical_palm_qualified"]):
        raise ValueError("Virtual axis qualification exceeds the permitted diagnostic convention")
    transforms = np.repeat(np.eye(4)[None], 7, axis=0)
    transforms[:, :3] = np.loadtxt(archive_path).reshape(7, 3, 4)
    proper(transforms)
    transforms[0, :3, :3] = np.asarray(axis["virtual_world_from_tracker_rotation"], dtype=float)
    proper(transforms)
    # Preserve every archived field except the three common rotation rows.
    lines = archive_path.read_bytes().splitlines(keepends=True)
    if len(lines) != 21 or any(len(line.split()) != 4 for line in lines):
        raise ValueError("Expected 21 unchanged native bridge rows")
    for i in range(3):
        original = lines[i].split()
        row = " ".join(format(float(value), ".17g") for value in transforms[0, i, :3]).encode("ascii")
        eol = b"\r\n" if lines[i].endswith(b"\r\n") else b"\n"
        lines[i] = row + b" " + original[3] + eol
    generated = b"".join(lines)
    if not np.array_equal(np.loadtxt(StringIO(generated.decode("ascii"))), transforms[:, :3].reshape(21, 4)):
        raise ValueError("Diagnostic bridge serialization changed a numeric transform")
    scope = {"schema_version": 1, "scope": SCOPE, "status": "DIAGNOSTIC_BRIDGE_ONLY",
             "run_id": RUN_ID, "virtual_axis_receipt_sha256": axis_sha,
             "archived_shared_frames_sha256": archived_sha,
             "live_model_sha256": axis["exact_model_support"]["live_model_sha256"],
             "transforms": {"path": "diagnostic_runtime_frames.txt", "sha256": sha256(generated).hexdigest(),
                            "order": list(LABELS), "replaced_fields": ["world_from_registered_tracker[:3,:3]"]},
             "virtual_world_from_tracker_rotation": axis["virtual_world_from_tracker_rotation"],
             "registered_wrist_frame": "openvr_world", "anatomical_geometry_qualified": False, **CREDIT}
    return transforms, generated, scope


def choose_diagnostic(roots):
    """Prefer six stable roots, then minimize use of the explicitly authorized fallback."""
    if any(row["split"] != "Train" for row in roots):
        raise ValueError("Diagnostic selection cannot use Validation or test features")
    population = [row["median_registered_wrist_gap_m"] for row in roots if row["median_registered_wrist_gap_m"] is not None]
    if not population:
        return (), None, "NO_SIGNED_REGISTERED_WRIST_GAP_SUPPORT"
    low, high = np.quantile(population, [.25, .75], method="linear")
    quartiles = [float(low), float(high)]
    if low >= high:
        return (), quartiles, "REGISTERED_WRIST_GAP_STRATA_DEGENERATE"
    for row in roots:
        gap = row["median_registered_wrist_gap_m"]
        row["gap_stratum"] = ("unknown" if gap is None else "lower_quartile" if gap <= low else
                              "upper_quartile" if gap >= high else "middle")
    eligible = sorted((row for row in roots if row["ineligibility_reason"] is None
                       and row["selected_anchor_status"] in STABLE | {FALLBACK}), key=lambda row: row["root"])
    stable_available = sum(row["selected_anchor_status"] in STABLE for row in eligible)
    fallback_cap = max(0, 6 - stable_available)
    best, best_score = (), None
    for group in combinations(eligible, 6):
        fallback = sum(row["selected_anchor_status"] == FALLBACK for row in group)
        if (fallback > fallback_cap or not {"lower_quartile", "upper_quartile"} <= {row["gap_stratum"] for row in group}
                or sum(row["contains_start_stop_or_reversal"] for row in group) < 2):
            continue
        score = fallback, -len({row["task"] for row in group})
        if best_score is None or score < best_score:
            best, best_score = group, score
    return best, quartiles, None if best else "NO_SIX_ROOTS_WITH_FROZEN_GAP_AND_MOTION_SUPPORT"


def anchor_payload(row, anchor, assets, frames, bridge, bridge_sha, cohort_sha):
    inputs, sources = {}, {}
    for role, source in zip(CORE_STREAMS, anchor.sources, strict=True):
        topic = assets[role]["topic"]
        record = source["record"]
        inputs[topic] = {"record_index": record["physical_message"], "cdr_sha256": record["cdr_sha256"]}
        sources[role] = {"topic": topic, "part_id": record["part_id"], "source_instance": record["source_instance"],
                         "writer_sequence": record["writer_sequence"], "capture_bounds_ns": source["capture_bounds_ns"],
                         "arrival_bounds_ns": source["arrival_bounds_ns"], "declared_clock_scope": source["clock_generation"]}
    registered = proper(anchor.wrists)
    runtime = proper(registered @ frames[3:5])
    rotation = np.eye(4)
    rotation[:3, :3] = frames[0, :3, :3]
    return {"schema": ANCHOR_SCHEMA, "scope": SCOPE, "run_id": RUN_ID, **CREDIT,
            "cohort_sha256": cohort_sha, "shared_frames_sha256": bridge["transforms"]["sha256"],
            "calibration_receipt_sha256": bridge_sha, "virtual_axis_receipt_sha256": bridge["virtual_axis_receipt_sha256"],
            "root": row["root"], "split": "Train", "source_manifest_sha256": row["source_manifest_sha256"],
            "partition": anchor.part, "partition_tick": anchor.tick, "hmb_cutoff_ns": anchor.time,
            "source_cutoff_ns": row["historical_window"]["source_cutoff_start_ns"],
            "inputs": inputs, "selected_source_evidence": sources,
            "anatomical_start_h": None, "registered_wrist_start_h": registered.tolist(),
            "runtime_engineering_start_h": runtime.tolist(),
            "registered_wrist_start_world_rotation_only": (rotation @ registered).tolist(),
            "runtime_engineering_start_world_rotation_only": (rotation @ runtime).tolist(),
            "world_translation_status": "REQUIRES_SAME_SESSION_MEASURED_HOME",
            "engineering_geometry_lineage": "historical controller bridge only, unqualified as human anatomy",
            "stable_basis": "REGISTERED_WRIST_PROXY_AND_SIGNED_MANUS_LOCAL_TIPS",
            **{key: row[key] for key in ("full_root_denominators", "stable_primary", "stable_sensitivity",
                 "explicit_event_status", "selected_anchor_status", "stable_window_qualified_on_registered_wrist_proxy",
                 "stable_anatomical_anchor_qualified", "historical_window")}}


def build(args):
    from prepare_start_handoff import read_identity

    frames, frames_bytes, bridge = make_frames(args.virtual_axis_receipt, args.archived_frames)
    bridge_bytes = json_bytes(bridge)
    bridge_sha = sha256(bridge_bytes).hexdigest()
    candidates, reader, cohort_sha = train_reader(args.cohort, args.docs, args.hmb_products)
    rows, anchors = [], {}
    identity_mounts = np.repeat(np.eye(4)[None], 2, axis=0)
    for item in candidates:
        _, root, _, assets, origin, end = read_identity(args.cohort, item["root"])
        if root["registered_wrist_frame"] != bridge["registered_wrist_frame"]:
            raise ValueError(f"Registered wrist frame differs: {root['root']}")
        stats, anchor = scan_root(reader, root, identity_mounts, frames[0, :3, :3], None, origin, end,
                                  "UNOBSERVED_IN_SIGNED_NATIVE_INPUTS", diagnostic=True)
        rows.append(stats)
        if anchor is not None:
            anchors[stats["root"]] = anchor, assets, root
    denom_keys = ("partitions", "plan_ticks", "hmb_joint_valid_ticks", "hmb_five_frame_candidate_ticks")
    denom = {key: sum(row["full_root_denominators"][key] for row in rows) for key in denom_keys}
    if tuple(denom.values()) != (228, 86_338, 85_382, 82_559):
        raise ValueError(f"Original Train denominators differ: {denom}")
    chosen, quartiles, blocking = choose_diagnostic(rows)
    selected, failures, anchor_files = [], [], {}
    for row in chosen:
        root = row["root"]
        anchor, assets, signed = anchors[root]
        data = anchor_payload(row, anchor, assets, frames, bridge, bridge_sha, cohort_sha)
        try:
            verify_native(data["inputs"], data, anchor, assets, args.cohort, cohort_sha, signed)
        except (OSError, KeyError, ValueError) as error:
            row["ineligibility_reason"] = "NATIVE_EVENT_BINDING_FAILED"
            failures.append({"root": root, "reason": str(error)})
            continue
        relative = f"anchors/{root.split(':')[-1]}.json"
        raw = json_bytes(data)
        anchor_files[relative] = raw
        rec = {key: row[key] for key in ("root", "split", "task", "stable_primary", "stable_sensitivity",
                                        "contains_start_stop_or_reversal", "gap_stratum", "selected_anchor_status",
                                        "full_root_denominators", "historical_window")}
        rec.update(anchor_path=relative, anchor_sha256=sha256(raw).hexdigest(), repeat_cold_start=False,
                   replay_planned_ticks=row["historical_window"]["plan_ticks"], stage_credit=False)
        selected.append(rec)
    if failures:
        blocking = "BLOCKED_NATIVE_EVENT_BINDING"
        selected, anchor_files = [], {}
    chosen_ids = {row["root"] for row in selected}
    for row in rows:
        row["selection_disposition"] = ("SELECTED" if row["root"] in chosen_ids else
                                        "INELIGIBLE" if row["ineligibility_reason"] else "NOT_SELECTED")
    plan = {"schema": PLAN_SCHEMA, "run_id": RUN_ID, "scope": SCOPE, **CREDIT,
            "selection_status": "PILOT_PLAN_READY_FOR_REVIEW" if selected else "DIAGNOSTIC_PLAN_BLOCKED",
            "blocking_reason": blocking, "cohort_sha256": cohort_sha,
            "shared_frames_sha256": bridge["transforms"]["sha256"], "calibration_receipt_sha256": bridge_sha,
            "virtual_axis_receipt_sha256": bridge["virtual_axis_receipt_sha256"],
            "shared_frames_path": bridge["transforms"]["path"], "calibration_receipt_path": "diagnostic_calibration_receipt.json",
            "selector_sha256": file_sha(Path(__file__)),
            "stable_selector_sha256": file_sha(Path(__file__).with_name("home_anchored_start_anchor.py")),
            "full_signed_train_denominators": denom, "selected_roots": selected, "repeat_roots": [], "attempt_limit": 6,
            "original_frozen_full_cohort_denominators": {"roots": 32, "partitions": 308, "plan_ticks": 116_613,
                "hmb_joint_valid_ticks": 114_532, "hmb_five_frame_candidate_ticks": 110_796,
                "validation_denominators_from_prior_signed_evidence": {"roots": 8, "partitions": 80, "plan_ticks": 30_275,
                    "hmb_joint_valid_ticks": 29_150, "hmb_five_frame_candidate_ticks": 28_237}},
            "selected_replay_plan_ticks": sum(row["replay_planned_ticks"] for row in selected),
            "proposed_roots_before_native_readback": [row["root"] for row in chosen], "native_event_binding_failures": failures,
            "roots": rows, "selection_policy": {"root_split": "Train", "gap_quartiles_m": quartiles,
                "gap_population": "original HMB joint-valid registered Tracker wrist marker separation",
                "motion_criteria": MOTION, "minimum_roots_with_start_stop_or_reversal": 2,
                "task_preference": "maximize distinct signed task labels, then lexicographic signed root IDs",
                "fallback": "only if fewer than six stable A candidates, minimum needed, no qualified stable anchor credit"},
            "stable_primary_rule": asdict(PRIMARY), "stable_sensitivity_rule": asdict(SENSITIVITY),
            "search_ns": SEARCH_NS, "replay_limit_ns": REPLAY_NS, "swept_safety_required_before_replay": True,
            "selection_consumed_execution_outcomes": False, "validation_opened": False, "pilot_approved": False,
            "full_execution_authorized": False, "training_authorized": False, "hardware_authorized": False}
    return plan, {**anchor_files, "diagnostic_runtime_frames.txt": frames_bytes,
                  "diagnostic_calibration_receipt.json": bridge_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cohort", "archived-frames", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--virtual-axis-receipt", type=Path, required=True)
    parser.add_argument("--docs", type=Path, default=Path(__file__).resolve().parents[2] / "docs/s0-5")
    parser.add_argument("--hmb-products", type=Path)
    args = parser.parse_args()
    plan, files = build(args)
    files["pilot_plan.json"] = json_bytes(plan)
    if any((args.out / name).exists() for name in files):
        raise FileExistsError("One or more diagnostic plan artefacts already exist")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, contents in files.items():
        output = args.out / name
        output.parent.mkdir(exist_ok=True)
        with output.open("xb") as handle:
            handle.write(contents)
    print(plan["selection_status"], len(plan["selected_roots"]), file_sha(args.out / "pilot_plan.json"))
    return 0 if plan["selected_roots"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
