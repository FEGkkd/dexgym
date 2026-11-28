#!/usr/bin/env python3
"""Build one signed Train source as a diagnostic, non-admissible offline RMB."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import monotonic, process_time

import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.v1.rmb.home_anchored_intent import (
    SignedHumanPartition, SimulationIntent, lock_registered_start,
)
from dexgym.tasks.direct.wholebody_s05.v1.rmb.mapping import compiled_topologies, packed_geometry
from dexgym.tasks.direct.wholebody_s05.v1.rmb.nominal import Wuji
from dexgym.tasks.direct.wholebody_s05.v1.rmb.offline_nominal import OfflineTeleopNominal
from dexgym.tasks.direct.wholebody_s05.v1.rmb.schema import (
    TICK_NS, add_arrays, empty_geometry, file_sha, put_geometry, write_json, write_partition,
)
from dexgym.tasks.direct.wholebody_s05.v1.rmb.teleop_candidate import ARM_SUBSTEPS
from dexgym.tasks.direct.wholebody_s05.v1.rmb.teleop_native import TeleopNative
from dexgym.tasks.direct.wholebody_s05.v1.task.physics import signature
from home_anchored_start_anchor import hmb_five_rows, role_identity, train_reader
from offline_rmb_lockstep import ACTIVE, PROOF, Lockstep
from replay_dcr009_guarded_diagnostic import ROOT_FIVE, ROOT_FIVE_PLAN_SHA, SIGNED_SHA, check_scope
from replay_home_anchored_pilot import palm_poses, read_frames
from swept_live_guard import EXACT_MODEL_RECEIPT_SHA, verified_robot


SCOPE = "DIAGNOSTIC_ONLY"
DOCS = Path(__file__).resolve().parents[2] / "docs/s0-5"


def hand_input(data, meta, sources, i):
    token = tuple(role_identity(sources[j]) for j in data["source_index"][i, :2, 0])
    _, order = compiled_topologies(meta, data["topology_id"][i, :2])
    return data["sdk_common_pose"][i, :2], order, token


def valid_input(data, i):
    return bool(data["joint_valid"][i] and data["safety_permission"][i]
                and data["pose_valid"][i, :4].all() and (data["source_index"][i, :4, 0] >= 0).all()
                and (data["capture_bounds_ns"][i, :4, 1] <= data["tick_ns"][i]).all()
                and (data["arrival_bounds_ns"][i, :4, 1] <= data["cutoff_ns"][i]).all())


def active(history):
    return (len(history) == ARM_SUBSTEPS and all(
        not sub.used_hold and not sub.fault and (np.isin(sub.proposal.status, ACTIVE)).all()
        for sub in history))


def record_execution(fields, row, history, executor):
    cursor = 0
    for arm_tick, result in enumerate(history):
        proposal = result.proposal
        fields["nominal_proposal"][row, arm_tick] = proposal.q_requested
        fields["nominal_status"][row, arm_tick] = proposal.status
        fields["raw_wuji"][row, arm_tick] = proposal.raw_hand_q
        fields["raw_wuji_mask"][row, arm_tick] = proposal.raw_hand_valid
        for side, native in enumerate(proposal.native_arms):
            if native.ik_target_accepted and np.isfinite(native.joint_candidate).all():
                fields["raw_marvin"][row, arm_tick, side] = native.joint_candidate
                fields["raw_marvin_mask"][row, arm_tick, side] = True
        if result.issued is not None:
            fields["q_issued"][row, arm_tick] = result.issued
            fields["q_issued_mask"][row, arm_tick] = True
        if result.interval is None:
            continue
        item = result.interval
        count = int(item.substeps[0])
        fields["q_applied"][row, cursor:cursor + count] = item.applied[0, :count].numpy()
        fields["actuator_ctrl"][row, cursor:cursor + count] = item.ctrl[0, :count].numpy()
        fields["physical_contact"][row] |= item.contact[0].numpy().astype(bool)
        cursor += count
    fields["applied_substeps"][row] = cursor
    if cursor:
        q, qd = executor.state()
        if np.isfinite(q).all() and np.isfinite(qd).all():
            fields["q_measured"][row], fields["qd_measured"][row] = q, qd
            fields["q_measured_mask"][row] = True
            actual = executor.robot.geometry(q)
            put_geometry(fields["actual"], row, packed_geometry(actual.wrists, actual.rotations,
                                                                 actual.local, actual.landmarks))
    if active(history) and cursor == 10:
        fields["q_nominal"][row] = history[-1].proposal.q_requested
        fields["q_nominal_mask"][row] = True


def blank(data):
    n = len(data["tick_ns"])
    fields = {"tick_ns": data["tick_ns"].copy(), "hmb_joint_valid": data["joint_valid"].copy(),
              "hmb_five_frame_candidate": hmb_five_rows(data), "hmb_segment": data["segment_id"].copy(),
              "hmb_source_index": data["source_index"][:, :4, 0].copy(),
              "history_eligible": np.zeros(n, bool),
              "target": empty_geometry(n), "actual": empty_geometry(n)}
    for key, shape in {"raw_tracker": (2, 4, 4), "registered": (2, 4, 4), "mapped": (2, 4, 4),
                       "raw_manus_local": (2, 5, 4, 3), "q_nominal": (54,), "qd_nominal": (54,),
                       "nominal_proposal": (5, 54), "q_measured": (54,), "qd_measured": (54,),
                       "q_issued": (5, 54), "q_applied": (10, 54), "actuator_ctrl": (10, 54),
                       "raw_wuji": (5, 2, 20), "raw_marvin": (5, 2, 7)}.items():
        fields[key] = np.zeros((n, *shape), "<f4")
    for key, shape in {"raw_tracker": (), "registered": (), "mapped": (), "raw_manus_local": (2,),
                       "q_nominal": (), "qd_nominal": (), "q_measured": (), "q_issued": (5,),
                       "raw_wuji": (5, 2), "raw_marvin": (5, 2)}.items():
        fields[key + "_mask"] = np.zeros((n, *shape), bool)
    fields["applied_substeps"] = np.zeros(n, dtype="u1")
    fields["physical_contact"] = np.zeros((n, 13), dtype=bool)
    fields["nominal_status"] = np.zeros((n, 5, 4), dtype="<u2")
    return fields


def record_target(fields, i, mapped):
    for dst, src in (("raw_tracker", "raw_tracker"), ("registered", "registered_wrist_proxy"),
                     ("mapped", "sim_mapped_wrist_proxy")):
        if mapped[src] is not None:
            fields[dst][i], fields[dst + "_mask"][i] = mapped[src], True
    for side, hand in enumerate(mapped["raw_manus_local"]):
        if hand is not None:
            fields["raw_manus_local"][i, side] = hand
            fields["raw_manus_local_mask"][i, side] = True
    if mapped["robot_morphology_target"] is not None:
        put_geometry(fields["target"], i, mapped["robot_morphology_target"])


def mark_error(errors, fields, row, scoring):
    a, t = fields["actual"], fields["target"]
    if not fields["q_measured_mask"][row] or not t["wrist_xyz_mask"][row].all() or not scoring:
        return
    errors["wrist_mm"].append(1000 * np.linalg.norm(a["wrist_xyz"][row] - t["wrist_xyz"][row], axis=-1))
    delta = (Rotation.from_quat(t["wrist_xyzw"][row]).inv() * Rotation.from_quat(a["wrist_xyzw"][row]))
    errors["wrist_deg"].append(np.rad2deg(delta.magnitude()))
    errors["ten_tip_mm"].append(1000 * np.linalg.norm(a["world_tip_xyz"][row] - t["world_tip_xyz"][row], axis=-1))
    valid = fields["q_issued_mask"][row]
    if valid.any():
        errors["issued_measured_joint_deg"].append(
            np.rad2deg(abs(fields["q_issued"][row, valid][-1] - fields["q_measured"][row])))


def publish_partition(output, partition, fields):
    for prefix in ("target", "actual"):
        add_arrays(fields, prefix, fields.pop(prefix))
    return {"partition": partition, "arrays": write_partition(output, fields),
            "directory": output.name, "ticks": len(fields["tick_ns"])}


def build(args, plan, root, reader, robot, kernel, hand, frames, anchor):
    begin = monotonic()
    cpu_begin = process_time()
    selection = next(row for row in plan["selected_roots"] if row["root"] == args.root)
    window = selection["historical_window"]
    anchor_data, anchor_meta, anchor_sources, _ = reader.read(anchor["partition"])
    anchor_tick = anchor["partition_tick"]
    start_human = SignedHumanPartition(anchor_data, anchor_meta, anchor_sources)[anchor_tick]
    if (not valid_input(anchor_data, anchor_tick) or start_human.registered_wrist is None
            or not np.allclose(start_human.registered_wrist, anchor["registered_wrist_start_h"], atol=1e-6)
            or int(anchor_data["tick_ns"][anchor_tick]) != window["hmb_cutoff_start_ns"]):
        raise ValueError("START does not bind the valid original signed anchor")
    initial = robot.geometry(robot.home)
    frames[0] = lock_registered_start(start_human.registered_wrist, palm_poses(initial), align_yaw=True)
    mapper = SimulationIntent(frames[0], initial, proxy_from_model_palm=frames[3:5])
    nominal = OfflineTeleopNominal(robot, kernel, hand, frames)
    executor = Lockstep(robot, nominal)
    mapped_start = mapper.map(start_human)
    start = executor.preposition(start_human.registered_wrist, mapped_start["simulation_scoring_tcp"],
                                 *hand_input(anchor_data, anchor_meta, anchor_sources, anchor_tick))
    write_json(args.out / "start.json", {"scope": SCOPE, "locked_transform": frames[0].tolist(), **start})
    print(args.root, start["status"], start["stages"][-1], flush=True)
    full, historical, causes, proofs, nominal_codes = (Counter() for _ in range(5))
    errors = {key: [] for key in ("wrist_mm", "wrist_deg", "ten_tip_mm", "issued_measured_joint_deg")}
    index = []
    last_tick = None
    previous_q = None
    consecutive = 0
    active_previous = False
    with (args.out / "diagnostic-events.jsonl").open("x", encoding="utf-8") as log:
        for part_i, part in enumerate(root["hmb_partitions"]):
            data, meta, sources, _ = reader.read(part["key"])
            identity = meta["identity"]
            if (identity["source_manifest_sha256"] != root["manifest_sha256"]
                    or identity["root_episode_id"] != root["root"] or identity["development_split"] != "Train"
                    or not np.array_equal(data["tick_ns"], data["cutoff_ns"])
                    or len(data["tick_ns"]) != part["planned_ticks"]):
                raise ValueError("Signed HMB source, clock or denominators disagree")
            human = SignedHumanPartition(data, meta, sources)
            fields = blank(data)
            full.update(partitions=1, plan_ticks=len(data["tick_ns"]), hmb_joint_valid_ticks=int(data["joint_valid"].sum()),
                        hmb_five_frame_candidate_ticks=int(fields["hmb_five_frame_candidate"].sum()))
            for i, raw_tick in enumerate(data["tick_ns"]):
                tick = int(raw_tick)
                sample = human[i]
                target = mapper.map(sample)
                record_target(fields, i, target)
                if not window["hmb_cutoff_start_ns"] <= tick < window["hmb_cutoff_stop_exclusive_ns"]:
                    continue
                historical.update(plan_ticks=1, hmb_joint_valid_ticks=int(data["joint_valid"][i]),
                                  hmb_five_frame_candidate_ticks=int(fields["hmb_five_frame_candidate"][i]),
                                  independent_geometry_ticks=int(target["robot_morphology_target"] is not None))
                if last_tick is not None and tick != last_tick + TICK_NS:
                    executor.fault = "UNBRIDGED_HMB_TIME_GAP"
                last_tick = tick
                history = []
                if start["status"] != "START_REACHED":
                    reason = "START_UNQUALIFIED"
                elif executor.fault:
                    reason = executor.fault
                else:
                    registered = sample.registered_wrist if valid_input(data, i) else None
                    if registered is not None:
                        nominal.refresh(registered, *hand_input(data, meta, sources, i), tick_ns=tick)
                    for _ in range(ARM_SUBSTEPS):
                        result = executor.advance(registered)
                        history.append(result)
                        proofs[result.proof] += 1
                        nominal_codes.update(map(int, result.proposal.status))
                        if result.fault:
                            break
                    record_execution(fields, i, history, executor)
                    reason = (executor.fault or ("INPUT_HOLD" if registered is None else
                              ("ACTIVE" if fields["q_nominal_mask"][i] else "NOMINAL_OR_PROOF_HOLD")))
                historical.update(processed_ticks=1, physical_ticks=int(bool(fields["applied_substeps"][i])),
                                  full_physical_ticks=int(fields["applied_substeps"][i] == 10),
                                  physical_substeps=int(fields["applied_substeps"][i]),
                                  legal_active_nominal_ticks=int(fields["q_nominal_mask"][i]),
                                  contact_ticks=int(fields["physical_contact"][i].any()))
                if fields["q_nominal_mask"][i]:
                    if active_previous:
                        fields["qd_nominal"][i] = (fields["q_nominal"][i] - previous_q) / .02
                        fields["qd_nominal_mask"][i] = True
                    previous_q = fields["q_nominal"][i].copy()
                    consecutive += 1
                else:
                    consecutive = 0
                active_previous = bool(fields["q_nominal_mask"][i])
                fields["history_eligible"][i] = consecutive >= 5 and fields["hmb_five_frame_candidate"][i]
                historical["diagnostic_five_frame_ticks"] += int(fields["history_eligible"][i])
                mark_error(errors, fields, i, target["robot_morphology_target"] is not None)
                causes[reason] += 1
                log.write(json.dumps({"partition": part["key"], "tick": i, "time_ns": tick, "status": reason,
                    "safety": [dict(proof=item.proof, hold=item.used_hold, fault=item.fault,
                                    metrics=item.request_metrics) for item in history]},
                    allow_nan=False, separators=(",", ":")) + "\n")
            index.append(publish_partition(args.out / f"part-{part_i:03}", part["key"], fields))
            print(part["key"], dict(historical), flush=True)
    if (dict(full) != selection["full_root_denominators"]
            or any(historical[key] != window[key] for key in
                   ("plan_ticks", "hmb_joint_valid_ticks", "hmb_five_frame_candidate_ticks"))):
        raise ValueError("Offline execution changed the original HMB denominators")
    digest = lambda name: file_sha(args.out / name)
    ready = (start["status"] == "START_REACHED" and historical["full_physical_ticks"] == window["plan_ticks"]
             and not any(fields["contact_ticks"] for fields in [historical])
             and historical["legal_active_nominal_ticks"] > 0 and not executor.fault)
    return {"schema": "s05.rmb.offline_diagnostic.v1", "scope": SCOPE, "formal_coverage_credit": False,
            "root": args.root, "full": dict(full), "window": window, "historical": dict(historical),
            "statuses": dict(causes), "proofs": dict(proofs), "nominal_status_codes": dict(nominal_codes),
            "start_status": start["status"], "recorded_physical_seconds": historical["physical_substeps"] * .002,
            "root_technical_gate": "DIAGNOSTIC_PASS" if ready else "BLOCKED", "physical_fault": executor.fault,
            "proof_budget": PROOF,
            "statistics": {name: {"ticks": len(data), "p50_p95_p99": np.percentile(data, [50, 95, 99], axis=0).tolist()}
                           for name, data in errors.items() if data},
            "robot_model_servo": signature(robot, executor.profile), "input_cohort_sha256": file_sha(args.cohort),
            "input_plan_sha256": file_sha(args.plan), "anchor_sha256": file_sha(args.plan.parent / selection["anchor_path"]),
            "asset_sha256": {name: file_sha(getattr(args, name)) for name in ("bridge", "wuji", "frames", "config")},
            "start_sha256": digest("start.json"), "diagnostic_events_sha256": digest("diagnostic-events.jsonl"),
            "partitions": index, "wall_seconds": monotonic() - begin, "process_cpu_seconds": process_time() - cpu_begin}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "cohort", "model", "model-receipt", "frames", "bridge", "config", "phase1", "wuji", "out"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--root", default=ROOT_FIVE)
    parser.add_argument("--root-five-receipt", type=Path)
    args = parser.parse_args()
    if file_sha(args.plan) != ROOT_FIVE_PLAN_SHA or file_sha(args.model_receipt) != EXACT_MODEL_RECEIPT_SHA:
        raise ValueError("Require the exact preregistered Train plan and model identity receipt")
    plan = json.loads(args.plan.read_bytes())
    selected = {row["root"]: row for row in check_scope(plan)}
    if args.root not in selected or file_sha(args.frames) != plan["shared_frames_sha256"]:
        raise ValueError("Root or engineering attachment is outside the signed diagnostic")
    if args.root != ROOT_FIVE:
        previous = json.loads(args.root_five_receipt.read_bytes()) if args.root_five_receipt else {}
        if (previous.get("root") != ROOT_FIVE or previous.get("root_technical_gate") != "DIAGNOSTIC_PASS"
                or previous.get("input_plan_sha256") != ROOT_FIVE_PLAN_SHA):
            raise ValueError("Six-root diagnostic requires the root-five physical technical check")
    roots, reader, cohort_sha = train_reader(args.cohort, DOCS, None)
    if cohort_sha != plan["cohort_sha256"]:
        raise ValueError("Original cohort identity differs")
    root, = [row for row in roots if row["root"] == args.root]
    selection = selected[args.root]
    anchor_path = args.plan.parent / selection["anchor_path"]
    if file_sha(anchor_path) != selection["anchor_sha256"]:
        raise ValueError("Original anchor hash differs")
    anchor = json.loads(anchor_path.read_bytes())
    identity = json.loads(args.model_receipt.read_bytes())["model_identity"]["signed"]
    robot = verified_robot(args.model, SIGNED_SHA, identity)
    args.out.mkdir(exist_ok=False)
    with TeleopNative(args.bridge, args.config, args.phase1) as kernel:
        hand = Wuji(args.wuji)
        try:
            report = build(args, plan, root, reader, robot, kernel, hand, read_frames(args.frames), anchor)
        finally:
            hand.close()
    write_json(args.out / "receipt.json", report)
    print(report["root_technical_gate"], report["historical"], flush=True)


if __name__ == "__main__":
    main()
