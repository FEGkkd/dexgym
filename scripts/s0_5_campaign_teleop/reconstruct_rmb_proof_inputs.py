#!/usr/bin/env python3
"""Reconstruct recorded controller boxes from issued-only deterministic CPU replay."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import monotonic, monotonic_ns

import numpy as np
import torch

from dexgym.tasks.direct.wholebody_s05.p02.model import COUPLING_LIMIT
from dexgym.tasks.direct.wholebody_s05.v1.rmb.schema import file_sha, write_json
from dexgym.tasks.direct.wholebody_s05.v1.task.native_sweep import NativeSweep
from dexgym.tasks.direct.wholebody_s05.v1.task.physics import CpuServo, ServoProfile, signature
from dexgym.tasks.direct.wholebody_s05.v1.task.safety import SafetyProjection
from swept_live_guard import Reason, verified_robot
from swept_live_guard_v2 import SweptEnvelopeV2


PROOF = {"max_queries": 8191, "max_depth": 18}
ROOT = "capture:b63bc72d2e8b2ee395e966f6"


def verified(path, expected):
    if file_sha(path) != expected:
        raise ValueError(f"Archive hash mismatch: {path}")
    return path


def require_equal(actual, expected, label):
    if not np.array_equal(actual, expected):
        raise ValueError(f"Deterministic replay mismatch at {label}, max difference {np.max(abs(actual - expected))}")


def data_fields(run, part, names):
    return {name: np.load(verified(run / part["directory"] / (name + ".npy"),
                                  part["arrays"][name]["sha256"]), allow_pickle=False)
            for name in names}


def observations(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def reconstruct(args):
    begin = monotonic()
    receipt = json.loads((args.run / "receipt.json").read_bytes())
    identity = json.loads(args.model_receipt.read_bytes())["model_identity"]["signed"]
    if receipt["root"] != ROOT or receipt["formal_coverage_credit"] or receipt["historical"]["full_physical_ticks"] != 4804:
        raise ValueError("Expected the completed original root-five diagnostic archive")
    if file_sha(args.run / "receipt.json") != "d6d83d40b44d29447e1d4a1e357bc0424b123dbabfcc113eef8218d330c605df":
        raise ValueError("Expected original DCR-010 run02 receipt")
    robot = verified_robot(args.model, identity["xml_sha256"], identity)
    profile = ServoProfile()
    if signature(robot, profile) != receipt["robot_model_servo"]:
        raise ValueError("Archive robot/Servo/runtime identity mismatch")
    start_path = verified(args.run / "start-events.jsonl", receipt["start_events_sha256"])
    path = verified(args.run / "diagnostic-events.jsonl", receipt["diagnostic_events_sha256"])
    physics = CpuServo(robot, 1, profile)
    physical = SafetyProjection(robot, device="cpu", max_step=[.027] * 14 + [.10] * 40,
                                sweep=NativeSweep(robot, **PROOF))
    allowed = torch.ones(1, dtype=torch.bool)
    issued = robot.home.copy()
    state = physics.data[0]

    def step(command):
        interval = physics.interval(torch.as_tensor(command[None], dtype=torch.float32), allowed, physical, count=2)
        if interval.substeps[0] != 2 or interval.event[0]:
            raise ValueError("Issued-only reconstruction diverged or produced a physical fault")
        return interval

    start_count = 0
    for entry in observations(start_path):
        if entry["issued"] is None or entry["physical"]["event"]:
            raise ValueError("Expected proved and complete START commands")
        issued = np.asarray(entry["issued"], dtype=float)
        step(issued)
        require_equal(state.qpos[robot.q_indices], entry["measured"], "START position")
        require_equal(state.qvel[robot.v_indices], entry["velocity"], "START velocity")
        start_count += 1

    box_rows, coordinate, parts = [], [], []
    obs = iter(observations(path))
    decisions = Counter()
    # Attach a second proof only when requested. The old start cache is not present;
    # compare rejection causes and the reported leaf, while measuring any cold-cache differences.
    checker = SweptEnvelopeV2(robot, **PROOF) if args.prove else None
    recomputed = Counter()
    original_rejection = Counter()
    names = ("tick_ns", "q_issued", "q_issued_mask", "nominal_proposal", "q_applied", "actuator_ctrl",
             "actuator_torque", "q_measured", "qd_measured", "applied_substeps")
    first_ns = receipt["window"]["hmb_cutoff_start_ns"]
    end_ns = receipt["window"]["hmb_cutoff_stop_exclusive_ns"]
    for part_index, part in enumerate(receipt["partitions"]):
        arrays = data_fields(args.run, part, names)
        parts.append(part["partition"])
        for i, tick in enumerate(arrays["tick_ns"]):
            if not first_ns <= tick < end_ns:
                continue
            event = next(obs)
            if (event["partition"], event["tick"], event["time_ns"]) != (part["partition"], i, tick):
                raise ValueError("Archive clock does not align with proof trace")
            if not (arrays["q_issued_mask"][i].all() and arrays["applied_substeps"][i] == 10):
                raise ValueError("Archive interval did not retain all five actually issued commands")
            for j, evidence in enumerate(event["safety"]):
                proposal = arrays["nominal_proposal"][i, j].astype(np.float64)
                cmd = arrays["q_issued"][i, j].astype(np.float64)
                coords = np.stack((issued, physics.filtered[0], state.qpos[robot.q_indices], proposal))
                box_rows.append(coords)
                coordinate.append((part_index, i, j, int(tick)))
                decision = evidence["proof"]
                decisions[decision] += 1
                applied_verdict = evidence["hold_proof"] if evidence["hold"] else decision
                require_equal(cmd, issued if evidence["hold"] else proposal, (part_index, i, j, "command owner"))
                if applied_verdict != "PERMIT":
                    raise ValueError("Unproved archive command would be executed")
                if decision != "PERMIT":
                    metrics = evidence["metrics"]
                    original_rejection[(decision, metrics["termination"], tuple(metrics["unresolved_pair_indices"]))] += 1
                if checker is not None:
                    low, high = coords.min(axis=0), coords.max(axis=0)
                    if (np.any(low < robot.bounds[:, 0] - 1e-9) or np.any(high > robot.bounds[:, 1] + 1e-9)
                            or np.any(np.maximum(robot.coupling * low, robot.coupling * high).sum(axis=1)
                                      > COUPLING_LIMIT + 1e-9)):
                        outcome, term, leaf = Reason.LIMIT_OR_COUPLING.name, None, ()
                    else:
                        outcome = checker.prove(low, high, monotonic_ns() + 2_000_000_000).name
                        term = checker.last_metrics["termination"]
                        leaf = tuple(checker.last_metrics["unresolved_pair_indices"])
                    same_leaf = leaf == tuple(evidence["metrics"].get("unresolved_pair_indices", ()))
                    recomputed[(decision, outcome, term, same_leaf)] += 1
                    if evidence["hold"]:
                        held = coords.copy()
                        held[-1] = issued
                        if checker.prove(held.min(axis=0), held.max(axis=0), monotonic_ns() + 2_000_000_000) != Reason.PERMIT:
                            raise ValueError("Reconstructed hold became unproved during audit")
                interval = step(cmd)
                for key, value in (("q_applied", interval.applied), ("actuator_ctrl", interval.ctrl),
                                   ("actuator_torque", interval.torque)):
                    require_equal(value[0].numpy(), arrays[key][i, 2*j:2*j+2], (part_index, i, j, key))
                issued = cmd
            require_equal(state.qpos[robot.q_indices].astype(np.float32), arrays["q_measured"][i],
                          (part_index, i, "measured position"))
            require_equal(state.qvel[robot.v_indices].astype(np.float32), arrays["qd_measured"][i],
                          (part_index, i, "measured velocity"))
        print(f"{part_index + 1}/{len(receipt['partitions'])} {len(box_rows)} commands", flush=True)
    if next(obs, None) is not None or len(box_rows) != 24020 or dict(decisions) != receipt["proofs"]:
        raise ValueError("Archive command denominator did not close")
    args.out.mkdir(exist_ok=False)
    path_boxes = args.out / "proof-inputs.npz"
    np.savez_compressed(path_boxes, states=np.asarray(box_rows), locations=np.asarray(coordinate, dtype="<i8"))
    write_json(args.out / "receipt.json", {
        "scope": "DIAGNOSTIC_ONLY", "formal_coverage_credit": False,
        "mode": "ISSUED_ONLY_RECONSTRUCTION_NO_NOMINAL_RECOMPUTATION",
        "original_receipt_sha256": file_sha(args.run / "receipt.json"), "events_sha256": file_sha(path),
        "model_receipt_sha256": file_sha(args.model_receipt), "audit_sha256": file_sha(Path(__file__)),
        "reconstructed_start_commands_equal_f64": start_count, "historical_commands_equal_f32": len(box_rows),
        "verified_servo_ctrl_torque_substeps": 2 * len(box_rows), "verified_feedback_after_hmb_rows": len(box_rows)//5,
        "proof_inputs_sha256": file_sha(path_boxes), "state_row_order": ["issued", "servo", "measured", "proposal"],
        "location_row_order": ["partition_index", "tick_index", "subcycle", "tick_ns"], "partitions": parts,
        "original_decisions": dict(decisions),
        "archived_rejections": [{"reason": key[0], "termination": key[1], "leaf_pairs": key[2], "count": n}
                                for key, n in original_rejection.items()],
        "fresh_proof_tally": [{"archived": key[0], "decision": key[1], "termination": key[2],
                               "same_leaf": key[3], "count": n} for key, n in recomputed.items()],
        "elapsed_seconds": monotonic() - begin,
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "model", "model-receipt", "out"):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--prove", action="store_true")
    reconstruct(cli.parse_args())
