#!/usr/bin/env python3
"""Audit exact grouped command transactions and matched full/active geometry."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_root_five_geometry_layers import shape_metrics, stats
from dexgym.tasks.direct.wholebody_s05.v1.rmb.offline_nominal import OfflineNominalStatus
from dexgym.tasks.direct.wholebody_s05.v1.rmb.schema import file_sha, write_json
from group_admission import GROUPS, RECOVERY_SUBCYCLES, GroupStatus, verified_groups
from offline_rmb_lockstep import ACTIVE
from swept_live_guard import verified_robot


def read_arrays(run, partition, fields):
    directory = run / partition["directory"]
    result = {}
    for field in fields:
        path = directory / (field + ".npy")
        if file_sha(path) != partition["arrays"][field]["sha256"]:
            raise ValueError(f"Archive field identity differs: {path}")
        result[field] = np.load(path, allow_pickle=False)
    return result


def pose_stats(geometry, world_tip, local, xyz, xyzw):
    data = shape_metrics(geometry.tips, geometry.local, world_tip, local)
    data["wrist_position_mm_by_side"] = 1000 * np.linalg.norm(geometry.wrists - xyz, axis=-1)
    data["wrist_rotation_deg_by_side"] = np.rad2deg(
        (Rotation.from_quat(xyzw).inv() * Rotation.from_matrix(geometry.rotations)).magnitude())
    return data


def check_transaction(event, target, previous, requested, actual, axes):
    groups = event["groups"]
    if len(groups) != 4:
        raise ValueError("Issued command has no four-group evidence")
    combined = previous.copy()
    for label, group, index in zip(GROUPS, groups, axes, strict=True):
        if group["group"] != label:
            raise ValueError("Proof group identity differs")
        trial = next((r for r in group["trials"] if r["proof"] == "PERMIT"), None)
        if (trial is not None) != bool(group["choice"]) or (trial is not None and trial != group["trials"][-1]):
            raise ValueError("Selected group did not terminate at its first permit")
        if trial is not None:
            combined[index] = trial["candidate"]
            if (group["choice"] == "EXACT") != np.array_equal(combined[index], requested[index]):
                raise ValueError("Exact group credit does not match the original native proposal")
    if not np.array_equal(combined, target):
        raise ValueError("Final simultaneous proof target differs from admitted group union")
    if event["proof"] == "PERMIT" and np.array_equal(actual, target):
        return "GROUPED_PERMIT"
    if event["hold"] and event["hold_proof"] == "PERMIT" and np.array_equal(actual, previous):
        return "FINAL_VETO_WITH_PROVED_HOLD"
    raise ValueError("Actual issued vector lacks a matching continuous proof")


def audit(args):
    receipt_path = args.run / "receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    baseline = json.loads((args.baseline / "receipt.json").read_bytes())
    if (receipt["schema"] != "s05.rmb.offline_diagnostic.v2" or receipt["formal_coverage_credit"]
            or any(receipt[k] != baseline[k] for k in ("root", "full", "window", "input_cohort_sha256",
                "input_plan_sha256", "anchor_sha256", "robot_model_servo", "asset_sha256"))):
        raise ValueError("Candidate and baseline differ in archived scope, model, controller or denominators")
    identity = json.loads(args.model_receipt.read_bytes())["model_identity"]["signed"]
    robot = verified_robot(args.model, identity["xml_sha256"], identity)
    axes = verified_groups(robot)
    event_path = args.run / "diagnostic-events.jsonl"
    if file_sha(event_path) != receipt["diagnostic_events_sha256"]:
        raise ValueError("Candidate event trace identity changed")
    starts = args.run / "start.json", args.baseline / "start.json"
    start_maps = [json.loads(p.read_bytes())["locked_transform"] for p in starts]
    if start_maps[0] != start_maps[1]:
        raise ValueError("Root-five START mapping changed")
    counters, group_codes, trial_codes = Counter(), [Counter() for _ in range(4)], [Counter() for _ in range(4)]
    timing, hold_start = [], [None] * 4
    proof_failures = Counter()
    group_pacing = [Counter() for _ in range(4)]
    error = defaultdict(lambda: defaultdict(list))
    gap_samples = []
    recovery = np.full(4, RECOVERY_SUBCYCLES, dtype=int)
    active_streak = 0
    start_trace = args.run / "start-events.jsonl"
    if file_sha(start_trace) != receipt["start_events_sha256"]:
        raise ValueError("START state lineage differs")
    start_tail = json.loads(start_trace.read_text(encoding="utf-8").splitlines()[-1])
    previous_issued = np.asarray(start_tail["issued"]) if start_tail["issued"] is not None else None
    previous_measured = np.asarray(start_tail["measured"]) if start_tail["measured"] is not None else None
    previous_applied = None
    events = iter(event_path.read_text(encoding="utf-8").splitlines())
    target_sha_fields = ("tick_ns", "control_cutoff_ns", "raw_tracker", "registered", "mapped", "raw_manus_local",
                         "hmb_joint_valid", "hmb_five_frame_candidate", "hmb_source_index")
    for part, old in zip(receipt["partitions"], baseline["partitions"], strict=True):
        common = set(target_sha_fields) | {key for key in part["arrays"] if key.startswith("target_") or key.startswith("raw_") and key.endswith("_mask") and "wuji" not in key and "marvin" not in key}
        for key in common:
            if part["arrays"][key] != old["arrays"][key]:
                raise ValueError(f"Original or independent target bytes changed: {part['partition']} {key}")
        provenance = args.run / part["directory"] / "hmb-source-identities.json"
        if file_sha(provenance) != part["hmb_source_identities_sha256"]:
            raise ValueError("HMB provenance sidecar hash differs")
        a = read_arrays(args.run, part, ("tick_ns", "q_issued", "q_issued_mask", "nominal_proposal",
            "proof_inputs", "proof_inputs_mask", "combined_trial", "combined_trial_mask", "group_status",
            "physical_q", "physical_state_mask", "applied_substeps", "substep_contact", "q_applied", "nominal_status",
            "hmb_joint_valid", "hmb_five_frame_candidate", "hmb_segment",
            "substep_clearance_m", "substep_penetration_m", "substep_event", "q_nominal_mask",
            "history_eligible", "target_world_tip_xyz", "target_local_landmark_xyz", "target_wrist_xyz",
            "target_wrist_xyzw", "target_wrist_xyz_mask", "q_measured"))
        old_a = read_arrays(args.baseline, old, ("q_measured", "applied_substeps"))
        previous_segment = None
        for i, tick in enumerate(a["tick_ns"]):
            if not receipt["window"]["hmb_cutoff_start_ns"] <= tick < receipt["window"]["hmb_cutoff_stop_exclusive_ns"]:
                continue
            row = json.loads(next(events))
            if (row["partition"], row["tick"], row["time_ns"]) != (part["partition"], i, tick):
                raise ValueError("Candidate proof trace differs from original time axis")
            counters["planned_rows"] += 1
            segment = a["hmb_segment"][i]
            if previous_segment is not None and segment != previous_segment:
                active_streak = 0
            previous_segment = segment
            n = int(a["applied_substeps"][i])
            counters["physical_substeps"] += n
            if n:
                gap_samples.extend(a["substep_clearance_m"][i, :n].tolist())
                counters["observed_contact_substeps"] += int(np.any(a["substep_contact"][i, :n], axis=1).sum())
                counters["observed_penetrating_substeps"] += int(np.count_nonzero(a["substep_penetration_m"][i, :n] > 0))
                counters["physical_event_substeps"] += int(np.count_nonzero(a["substep_event"][i, :n]))
            complete = n == 10 and a["physical_state_mask"][i].all()
            active = complete and bool(a["hmb_joint_valid"][i]) and len(row["safety"]) == 5
            for sub, event in enumerate(row["safety"]):
                good_substeps = (a["physical_state_mask"][i, 2*sub:2*sub+2].all()
                                 and not a["substep_event"][i, 2*sub:2*sub+2].any() and not event["fault"])
                hold_detail = event.get("hold_metrics") or {}
                if "original_hold_proof" in hold_detail:
                    if (event["proof"] == "PERMIT" or not event["hold"]
                            or hold_detail["original_hold_proof"] != "SWEEP_UNPROVEN"
                            or hold_detail["original_hold_metrics"].get("termination") != "MAX_SUBDIVISION_DEPTH"
                            or hold_detail["escalation_max_depth"] != receipt.get("unchanged_hold_proof_max_depth")
                            or (event["hold_proof"] == "PERMIT") != bool(a["q_issued_mask"][i, sub])):
                        raise ValueError("Extra proof budget escaped a depth-limited unchanged HOLD")
                    counters["unchanged_hold_depth_escalations"] += 1
                    counters["unchanged_hold_depth_escalations_proved"] += int(event["hold_proof"] == "PERMIT")
                active &= bool(good_substeps and event["proof"] == "PERMIT" and len(event["groups"]) == 4)
                for g, group in enumerate(event["groups"]):
                    status = group.get("label", "NO_COMMIT")
                    nominal = OfflineNominalStatus(int(a["nominal_status"][i, sub, g]))
                    if nominal.name != group["nominal_status"]:
                        raise ValueError("Group native status differs from recorded proposal")
                    exact = good_substeps and event["proof"] == "PERMIT" and group.get("committed") == "EXACT" and nominal in ACTIVE
                    active_group = exact and recovery[g] >= RECOVERY_SUBCYCLES
                    if bool(active_group) != (group.get("status") == GroupStatus.ACTIVE):
                        raise ValueError("Group active status violates the independently counted recovery gate")
                    if "status" in group and group["status"] != a["group_status"][i, sub, g]:
                        raise ValueError("Group code and trace differ")
                    recovery[g] = min(recovery[g] + 1, RECOVERY_SUBCYCLES) if exact else 0
                    active &= bool(active_group)
                    group_codes[g][status] += 1
                    for trial in group["trials"]:
                        result = trial["proof"]
                        cause = trial["metrics"].get("termination", result)
                        trial_codes[g][f"{result}:{cause}"] += 1
                        if result != "PERMIT":
                            for pair in trial["metrics"].get("unresolved_pair_indices", []):
                                proof_failures[(GROUPS[g], pair)] += 1
                    if group.get("committed") and group["delta_max_rad"] > 0:
                        group_pacing[g]["nonzero_admissions"] += 1
                    now = (int(tick) - receipt["window"]["hmb_cutoff_start_ns"]) / 1e9 + .004 * sub
                    if status in ("GROUP_HOLD", "INPUT_HOLD") and hold_start[g] is None:
                        hold_start[g] = now
                    if status.endswith("_ACTIVE") and hold_start[g] is not None:
                        group_pacing[g]["completed_rejoin_intervals"] += 1
                        group_pacing[g]["maximum_rejoin_wall_in_sim_s"] = max(
                            group_pacing[g]["maximum_rejoin_wall_in_sim_s"], now - hold_start[g])
                        hold_start[g] = None
                if not a["q_issued_mask"][i, sub]:
                    continue
                if not a["proof_inputs_mask"][i, sub] or not a["combined_trial_mask"][i, sub]:
                    raise ValueError("Issued command is missing its exact proof inputs")
                old_q, filtered, measured, requested = a["proof_inputs"][i, sub]
                actual = a["q_issued"][i, sub]
                if (previous_issued is not None and not np.array_equal(old_q, previous_issued)
                        or previous_measured is not None and not np.array_equal(measured, previous_measured)
                        or previous_applied is not None and not np.array_equal(filtered.astype(np.float32), previous_applied)
                        or not np.array_equal(requested, a["nominal_proposal"][i, sub])):
                    raise ValueError("Immutable 4 ms proof inputs differ from adjacent physical records")
                counters[check_transaction(event, a["combined_trial"][i, sub], old_q, requested, actual, axes)] += 1
                if a["physical_state_mask"][i, 2*sub + 1]:
                    previous_issued, previous_measured = actual, a["physical_q"][i, 2*sub + 1]
                    previous_applied = a["q_applied"][i, 2*sub + 1]
                if "elapsed_ns" in event["metrics"]:
                    timing.append(event["metrics"]["elapsed_ns"] / 1e6)
            if bool(a["q_nominal_mask"][i]) != bool(active):
                raise ValueError("Partial, recovery or missing feedback received complete active credit")
            active_streak = active_streak + 1 if active else 0
            five = active_streak >= 5 and a["hmb_five_frame_candidate"][i]
            if bool(a["history_eligible"][i]) != bool(five):
                raise ValueError("Five-frame stored credit violates the signed candidate fraction")
            counters["complete_rows"] += int(complete)
            counters["active_rows"] += int(active)
            counters["five_frame_rows"] += int(five)
            if complete and a["target_wrist_xyz_mask"][i].all():
                geometric = (a["target_world_tip_xyz"][i], a["target_local_landmark_xyz"][i],
                             a["target_wrist_xyz"][i], a["target_wrist_xyzw"][i])
                stages = {"nominal": a["nominal_proposal"][i, -1], "issued": a["q_issued"][i, -1],
                          "applied": a["q_applied"][i, -1], "measured": a["q_measured"][i]}
                if old_a["applied_substeps"][i] == 10:
                    stages["baseline_measured_same_rows"] = old_a["q_measured"][i]
                poses = {stage: robot.geometry(q) for stage, q in stages.items()}
                layers = {stage: pose_stats(pose, *geometric) for stage, pose in poses.items()}
                for origin, destination in (("nominal", "issued"), ("issued", "applied"), ("applied", "measured")):
                    g, h = poses[origin], poses[destination]
                    shift = np.rad2deg(abs(stages[destination] - stages[origin]))
                    delta = shape_metrics(h.tips, h.local, g.tips, g.local)
                    delta["joint_max_delta_deg_by_group"] = [float(shift[index].max()) for index in axes]
                    layers[f"{origin}_to_{destination}"] = delta
                for layer, metrics in layers.items():
                    for key, value in metrics.items():
                        error["all_complete_observed"][f"{layer}.{key}"].append(value)
                        if active:
                            error["complete_active_observed"][f"{layer}.{key}"].append(value)
    stored = receipt["historical"]
    if (next(events, None) is not None or counters["planned_rows"] != receipt["window"]["plan_ticks"]
            or counters["active_rows"] != stored["legal_active_nominal_ticks"]
            or counters["five_frame_rows"] != stored["diagnostic_five_frame_ticks"]
            or counters["physical_substeps"] != stored["physical_substeps"]):
        raise ValueError("Planned denominator differs")
    write_json(args.out, {"scope": "DIAGNOSTIC_ONLY", "formal_coverage_credit": False,
             "candidate_receipt_sha256": file_sha(receipt_path), "baseline_receipt_sha256": file_sha(args.baseline / "receipt.json"),
             "counters": dict(counters), "group_status": dict(zip(GROUPS, map(dict, group_codes))),
             "group_trials": dict(zip(GROUPS, map(dict, trial_codes))),
             "group_pacing": dict(zip(GROUPS, map(dict, group_pacing))),
             "open_hold_start_seconds": dict(zip(GROUPS, hold_start)),
             "unresolved_group_pair_counts": [{"group": key[0], "pair": key[1], "count": value}
                                              for key, value in proof_failures.most_common()],
             "final_proof_latency_ms": stats(timing), "minimum_sampled_gap_m": min(gap_samples, default=None),
             "geometric_error": {domain: {key: stats(values) for key, values in layers.items()}
                                 for domain, layers in error.items()}})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("run", "baseline", "model", "model-receipt", "out"):
        p.add_argument("--" + option, type=Path, required=True)
    audit(p.parse_args())
