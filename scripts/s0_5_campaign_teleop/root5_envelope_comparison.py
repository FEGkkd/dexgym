#!/usr/bin/env python3
"""Compare A–E on three reconstructed original proof states without actuation."""
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys
from time import monotonic_ns

import mujoco as mj
import numpy as np

from dexgym.tasks.direct.wholebody_s05.p02.model import COUPLING_LIMIT, RobotModel
from dexgym.tasks.direct.wholebody_s05.v1.task.native_sweep import NativeSweep
from dexgym.tasks.direct.wholebody_s05.v1.task.physics import ServoProfile
from swept_live_guard_v2 import SweptEnvelopeV2


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def timing(fn):
    start = monotonic_ns()
    value = fn()
    return value, (monotonic_ns()-start)/1e6


def compare(checkpoint, ledger, xml, output):
    checkpoint_receipt = json.loads((checkpoint.parent/"receipt.json").read_text())
    ledger_receipt = json.loads((ledger.parent/"summary.json").read_text())
    assert sha(checkpoint) == checkpoint_receipt["proof_inputs_sha256"]
    assert sha(ledger) == ledger_receipt["request_ledger_sha256"]
    assert sha(xml) == ledger_receipt["model_sha256"]
    assert ledger_receipt["receipt_sha256"] == checkpoint_receipt["original_receipt_sha256"]
    box_archive = np.load(checkpoint, allow_pickle=False)
    states = box_archive["states"]
    grid = box_archive["locations"]
    rows = [json.loads(raw) for raw in ledger.read_bytes().splitlines()]
    candidate_gap = lambda row: row["independent_right_pair_gaps_m"][2]
    selected = {"first_rejection": rows[0], "worst_candidate_endpoint": min(rows, key=candidate_gap),
                "widest_candidate_endpoint": max(rows, key=candidate_gap)}
    robot = RobotModel(xml)
    arm_names = {name: i for i,name in enumerate(robot.names)}
    groups = [np.asarray([arm_names[k] for k in grp], dtype=int) for grp in robot.groups]
    proof_args = {"max_queries": 8191, "max_depth": 18}
    path_guard = NativeSweep(robot, **proof_args)
    hi_proof_args = dict(proof_args, max_depth=24)
    a, b = robot.collision_pairs[2425]
    pair_axes = np.flatnonzero(path_guard.weights[2425])
    d = mj.MjData(robot.model)

    def phase_gap(q):
        d.qpos[robot.q_indices] = q
        mj.mj_kinematics(robot.model, d)
        return float(mj.mj_geomDistance(robot.model, d, a, b, .1, None))

    def envelope(base, proposed, opts):
        full = np.append(base, proposed[None], axis=0)
        lo = full.min(axis=0)
        hi = full.max(axis=0)
        if (np.any(lo < robot.bounds[:, 0] - 1e-9) or np.any(hi > robot.bounds[:, 1] + 1e-9)
                or np.any(np.maximum(robot.coupling * lo, robot.coupling * hi).sum(axis=1) > COUPLING_LIMIT + 1e-9)):
            return {"verdict": "LIMIT_OR_COUPLING"}
        solver = SweptEnvelopeV2(robot, **opts)
        verdict, dt = timing(lambda: solver.prove(lo, hi, monotonic_ns()+2_000_000_000).name)
        return {"verdict": verdict, "proof_ms": dt, **solver.last_metrics}

    report = {}
    for name,row in selected.items():
        i = 5*(row["line"]-1)+row["subcycle"]
        part, tick, sub, t = grid[i]
        assert (checkpoint_receipt["partitions"][part], int(tick), int(sub), int(t)) == (
            row["partition"], row["tick"], row["subcycle"], row["time_ns"]
        )
        issued, filtered, measured, candidate = states[i]
        initial = states[i,:3]
        thin = (issued, candidate)
        reference = [filtered]
        alpha = ServoProfile().alpha
        for j in range(2):
            reference.append(reference[-1] + alpha*(candidate-reference[-1]))
        reduced = np.copy(issued)
        stages = []
        for label,axes in zip(("left_arm", "right_arm", "left_hand", "right_hand"), groups, strict=True):
            trial = reduced.copy()
            trial[axes] = candidate[axes]
            verdict = envelope(initial, trial, proof_args)
            stages.append({"group": label, **verdict})
            if verdict["verdict"] == "PERMIT":
                reduced = trial
        lo = states[i].min(axis=0)
        hi = states[i].max(axis=0)
        # Explicit counterexample witness only. Positive corner distances do not certify the box.
        point = .5*(lo+hi)
        gaps = []
        for vertex in product((0,1), repeat=len(pair_axes)):
            point[pair_axes] = np.where(vertex, hi[pair_axes], lo[pair_axes])
            gaps.append(phase_gap(point))
        report[name] = {"source": {key: row[key] for key in ("partition", "tick", "subcycle", "time_ns")},
                        "four_input_points_gap_m": [phase_gap(q) for q in states[i]],
                        "right_pair_sampled_vertex_min_m": min(gaps),
                        "A_full_54D": envelope(initial, candidate, proof_args),
                        "B_servo_reference_segments_only": {
                            "issued_to_proposed_segment": timing(lambda: path_guard.permits(*thin)),
                            "filter_reference_segments": timing(lambda: [path_guard.permits(x,y) for x,y in zip(reference,reference[1:])]),
                            "authority": "setpoint / hypothetical segment only, not dynamics or admission"
                        },
                        "C_archived_held_steps": "all actual executed substep proofs passed; no inference on rejected proposal",
                        "D_same_box_deeper_pair_adaptive": envelope(initial, candidate, hi_proof_args),
                        "E_groupwise_full_body_candidates": stages,
                        "E_combined_final_reproof": envelope(initial, reduced, proof_args)}
    result = {"scope": "DIAGNOSTIC_ONLY_NO_MJ_STEP", "source_proof_inputs_sha256": sha(checkpoint),
              "source_ledger_sha256": sha(ledger), "source_model_sha256": sha(xml),
              "script_sha256": sha(Path(__file__)), "samples": report}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: {"A": sample["A_full_54D"]["verdict"],
                             "D": sample["D_same_box_deeper_pair_adaptive"]["verdict"],
                             "E": [s["verdict"] for s in sample["E_groupwise_full_body_candidates"]],
                             "combined": sample["E_combined_final_reproof"]["verdict"],
                             "corner_min_mm": 1000*sample["right_pair_sampled_vertex_min_m"],
                             "B": sample["B_servo_reference_segments_only"]["issued_to_proposed_segment"][0]}
                      for name,sample in result["samples"].items()}))


if __name__ == "__main__":
    compare(*(Path(value).resolve() for value in sys.argv[1:]))
