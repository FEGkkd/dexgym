#!/usr/bin/env python3
"""Paired, fixed arm-target MuJoCo rollouts for the isolated root-five parity audit."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from time import monotonic_ns
from types import SimpleNamespace

import numpy as np
import torch

from audit_grouped_rmb_diagnostic import read_arrays
from audit_root_five_geometry_layers import stats
from dexgym.tasks.direct.wholebody_s05.p02.model import COUPLING_LIMIT
from dexgym.tasks.direct.wholebody_s05.v1.rmb.offline_nominal import OfflineTeleopNominal
from dexgym.tasks.direct.wholebody_s05.v1.rmb.schema import file_sha, write_json
from dexgym.tasks.direct.wholebody_s05.v1.task.physics import ServoProfile, signature
from group_admission import verified_groups
from offline_rmb_lockstep import Lockstep, PROOF
from swept_live_guard import Reason, verified_robot
from swept_live_guard_v2 import SweptEnvelopeV2


ROOT = "capture:b63bc72d2e8b2ee395e966f6"
MODES = ("A", "B", "C", "D")


def vector(row, name):
    hands = row[name]
    value = np.asarray([[np.nan] * 20 if hand is None else hand for hand in hands], dtype=np.float64)
    if value.shape != (2, 20):
        raise ValueError(f"{name} requires the two ordered 20-axis hand rows")
    return value


def valid(row, name, value):
    declared = np.asarray(row[name], bool)
    if declared.shape != (2,) or np.any(declared & ~np.isfinite(value).all(axis=1)):
        raise ValueError(f"Invalid {name} mask or selected SDK output")
    return declared


def safety(robot, engine, state, target):
    if not np.isfinite(state).all() or not np.isfinite(target).all():
        return Reason.NONFINITE, {}
    low, high = np.minimum(state.min(axis=0), target), np.maximum(state.max(axis=0), target)
    if (np.any(low < robot.bounds[:, 0] - 1e-9) or np.any(high > robot.bounds[:, 1] + 1e-9)
            or np.any(np.maximum(robot.coupling * low, robot.coupling * high).sum(axis=1) > COUPLING_LIMIT + 1e-9)):
        return Reason.LIMIT_OR_COUPLING, {}
    distances = engine.native.pair_distances(target)
    if not np.isfinite(distances).all():
        return Reason.INTERNAL, {}
    pair = int(np.argmin(distances))
    gap = float(distances[pair])
    if gap < robot.clearance_m:
        return Reason.COLLISION, {"termination": "PENETRATION" if gap < 0 else "CLEARANCE_VIOLATION",
                                  "endpoint_gap_m": gap, "witness_pair_index": pair}
    result = engine.prove(low, high, monotonic_ns() + 2_000_000_000)
    return result, engine.last_metrics


class NativeHandUpdates:
    """Apply published native-track updates on their recorded receive clock."""

    def __init__(self, path):
        self.rows = iter(map(json.loads, path.read_text(encoding="utf-8").splitlines()))
        self.pending = next(self.rows, None)
        self.q = [None, None]
        self.source = [None, None]
        self.received = [None, None]
        self.last_clock = None

    def at(self, cutoff):
        while self.pending is not None and self.pending["receive_ns"] <= cutoff:
            row = self.pending
            if self.last_clock is not None and row["receive_ns"] < self.last_clock:
                raise ValueError("Native MANUS update clock reversed")
            self.last_clock = row["receive_ns"]
            if row["published"]:
                side = ("left", "right").index(row["side"])
                self.q[side] = row["filtered_rad"]
                self.source[side] = row["source_index"]
                self.received[side] = row["receive_ns"]
            self.pending = next(self.rows, None)
        return {"a_filtered_rad": self.q.copy(), "a_valid": [x is not None for x in self.q],
                "a_source_index": self.source.copy(), "a_receive_ns": self.received.copy(),
                "physical_selection_ns": cutoff}


class HandRollout:
    def __init__(self, robot, mode):
        self.mode = mode
        self.sim = Lockstep(robot, SimpleNamespace(reset=lambda _: np.zeros(2)))
        self.robot = robot
        self.hand_ctrl_range = np.asarray([a.ctrlrange for a in robot.actuators[14:]], dtype=np.float64)
        self.hand_controller = OfflineTeleopNominal(robot, None, None, np.repeat(np.eye(4)[None], 7, axis=0))
        self.common = SweptEnvelopeV2(robot, max_queries=PROOF["max_queries"], max_depth=24)
        self.proof_stats = Counter()
        self.physics_stats = Counter()
        self.gaps = []
        self.timing = []
        self.stopped = None
        self.shadow = mode in ("C", "D")

    def advance(self, row, subcycle):
        stage = "c" if self.mode == "D" else self.mode.lower()
        filtered = vector(row, f"{stage}_filtered_rad")
        usable = valid(row, f"{stage}_valid", filtered)
        if self.mode in ("C", "D") and not row["nominal_pair_eligible"]:
            usable.fill(False)
        if self.stopped is not None:
            return {"status": "NOT_EXECUTED", "cause": self.stopped}
        measured, _ = self.sim.state()
        previous = self.sim.issued
        proposed = previous.copy()
        limited = np.zeros(40, bool)
        if self.mode in ("C", "D"):
            self.hand_controller.hand_valid = usable
            self.hand_controller.filtered_hand = filtered
            groups = self.hand_controller._propose_hands(previous, proposed, limited, np.ones(40, bool))
            statuses = [value.name for value in groups]
        else:
            statuses = ["INPUT_HOLD"] * 2
            for side in range(2):
                if usable[side]:
                    group = slice(14 + 20 * side, 34 + 20 * side)
                    low, high = self.hand_ctrl_range[side * 20:(side + 1) * 20].T
                    proposed[group] = np.clip(filtered[side], low, high)
                    statuses[side] = ("NATIVE_DRIVER_CLAMPED" if not np.array_equal(proposed[group], filtered[side])
                                      else "NATIVE_SMOOTHED")
        target = proposed.astype(np.float32).astype(np.float64)
        state = np.stack((previous.copy(), self.sim.physics.filtered[0].copy(), measured))
        time0 = monotonic_ns()
        common, common_detail = safety(self.robot, self.common, state, target)
        self.timing.append((monotonic_ns() - time0) / 1e6)
        # C stores the exact same pre-issue state and proposal for D's one-step shadow.
        shadow, shadow_detail = self.sim._proof(target, measured) if self.shadow else (None, {})
        self.proof_stats[f"common:{common.name}:{common_detail.get('termination', '')}"] += 1
        if self.shadow:
            self.proof_stats[f"shadow:{common.name}:{shadow.name}:{shadow_detail.get('termination', '')}"] += 1
        allowed = common == Reason.PERMIT and (self.mode != "D" or shadow == Reason.PERMIT)
        hold = not allowed or not usable.any()
        if hold:
            hold_proof, hold_detail = safety(self.robot, self.common, state, previous)
            if self.mode == "D" and hold_proof == Reason.PERMIT:
                hold_proof, hold_detail = self.sim._proof(previous, measured)
        else:
            hold_proof, hold_detail = None, {}
        receipt = {"status": "HOLD" if hold else "ISSUED", "subcycle": subcycle,
                   "fixed_arm_joint_target": self.robot.home[:14].tolist(), "native_hand_status": statuses,
                   "handoff_mask": usable.tolist(), "python_step_limited": limited.tolist(),
                   "command_source_index": row["a_source_index"] if self.mode == "A" else row["source_index"],
                   "physical_selection_ns": row["physical_selection_ns"],
                   "source_receive_ns": row.get("a_receive_ns") if self.mode == "A" else None,
                   "sdk_selected_control_cutoff_ns": row["control_cutoff_ns"],
                   "snapshot": state.tolist(), "nominal": target.tolist(),
                   "common_proof": common.name, "common_proof_detail": common_detail,
                   "guard_v2": shadow.name if self.shadow else None, "guard_v2_detail": shadow_detail,
                   "hold_proof": hold_proof.name if hold else None, "hold_detail": hold_detail}
        if hold and hold_proof != Reason.PERMIT:
            self.stopped = f"HOLD_INFEASIBLE:{hold_proof.name}"
            return {**receipt, "status": "NOT_EXECUTED", "cause": self.stopped}
        if not hold:
            self.sim.issued = target
        applied = []
        for _ in range(2):
            interval = self.sim.physics.interval(torch.as_tensor(self.sim.issued[None], dtype=torch.float32),
                                                 torch.ones(1, dtype=torch.bool), self.sim.safety, count=1)
            if interval.substeps[0] != 1:
                self.stopped = "PHYSICS_COUNT"
                break
            q, qd = self.sim.state()
            contacts = bool(interval.contact[0].any())
            event = int(interval.event[0])
            gap = float(interval.minimum_gap[0])
            self.gaps.append(gap)
            self.physics_stats.update(substeps=1, contact_substeps=int(contacts), event_substeps=int(bool(event)),
                                      penetrating_substeps=int(interval.depth[0] > 0))
            model = self.sim.physics.model
            data = self.sim.physics.data[0]
            hits = [{"geom1": model.geom(int(x.geom[0])).name, "geom2": model.geom(int(x.geom[1])).name,
                     "gap_m": float(x.dist)} for x in data.contact]
            applied.append({"applied": interval.applied[0, 0].tolist(), "measured": q.tolist(),
                            "velocity": qd.tolist(), "minimum_gap_m": gap, "event": event,
                            "contact_pairs": hits, "depth_m": float(interval.depth[0])})
            if event:
                self.stopped = f"PHYSICS_EVENT:{event}"
                break
        self.physics_stats["hold_intervals"] += int(hold)
        self.physics_stats["moved_intervals"] += int(not hold and not np.array_equal(previous, self.sim.issued))
        self.physics_stats["native_clipped_intervals"] += int("NATIVE_DRIVER_CLAMPED" in statuses)
        return {**receipt, "issued": self.sim.issued.tolist(), "physical": applied, "cause": self.stopped}


def main(args):
    reference_path = args.baseline / "receipt.json"
    baseline = json.loads(reference_path.read_bytes())
    if baseline["root"] != ROOT or baseline["historical"]["plan_ticks"] != 4804:
        raise ValueError("Incorrect root or diagnostic scope")
    signed = json.loads(args.model_receipt.read_bytes())["model_identity"]["signed"]
    robot = verified_robot(args.model, signed["xml_sha256"], signed)
    axes = verified_groups(robot)
    if signature(robot, ServoProfile()) != baseline["robot_model_servo"]:
        raise ValueError("Fixed-arm control differs from the signed model or Servo")
    sdk_summary = json.loads(args.sdk_summary.read_bytes())
    if (sdk_summary["output_sha256"] != {p.name: file_sha(p) for p in (args.sdk_ticks, args.sdk_updates)}
            or any(list(sdk_summary["joint_names"][side]) != [robot.actuators[i].name for i in axes[index + 2]]
                   for index, side in enumerate(("left", "right")))):
        raise ValueError("SDK output artifacts or named actuator order differ from the physical boundary")
    home = robot.geometry(robot.home)
    streams = {mode: HandRollout(robot, mode) for mode in MODES}
    a_updates = NativeHandUpdates(args.sdk_updates)
    geometry = {mode: defaultdict(list) for mode in MODES}
    ledger = args.out / "fixed-arm-parity.jsonl.gz"
    args.out.mkdir()
    score_fields = ("tick_ns", "target_local_landmark_xyz", "target_local_landmark_xyz_mask")
    rows = iter(args.sdk_ticks.read_text(encoding="utf-8").splitlines())
    mode_masks = defaultdict(Counter)
    paired = defaultdict(list)
    count = 0
    with ledger.open("xb") as raw, gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0, compresslevel=3) as out:
        for part in baseline["partitions"]:
            a = read_arrays(args.baseline, part, score_fields)
            for i, tick in enumerate(a["tick_ns"]):
                row = json.loads(next(rows))
                if (row["partition"], row["partition_tick"], row["tick_ns"]) != (part["partition"], i, tick):
                    raise ValueError("SDK parity trace clock or partition differs from signed HMB")
                if not baseline["window"]["hmb_cutoff_start_ns"] <= tick < baseline["window"]["hmb_cutoff_stop_exclusive_ns"]:
                    continue
                count += 1
                target = a["target_local_landmark_xyz"][i]
                target_valid = bool(a["target_local_landmark_xyz_mask"][i].all())
                evidence = {}
                actual = {}
                native_rows = []
                sampled_rows = []
                for sub in range(5):
                    cut = row["control_cutoff_ns"] + 4_000_000 * sub
                    native_rows.append({**row, **a_updates.at(cut)})
                    sampled_rows.append({**row, "physical_selection_ns": cut})
                for mode, rollout in streams.items():
                    schedule = native_rows if mode == "A" else sampled_rows
                    traces = [rollout.advance(values, sub) for sub, values in enumerate(schedule)]
                    evidence[mode] = traces
                    samples = [sample for trial in traces for sample in trial.get("physical", [])]
                    status = ("COMPLETE" if len(samples) == 10 else "PARTIAL" if samples else "NOT_EXECUTED")
                    mode_masks[mode][status] += 1
                    if status != "COMPLETE":
                        continue
                    actual[mode] = robot.geometry(samples[-1]["measured"])
                    g = actual[mode]
                    geometry[mode]["actual_wrist_drift_mm"].append(1000 * np.linalg.norm(g.wrists - home.wrists, axis=-1))
                    geometry[mode]["actual_thumb_index_mm"].append(1000 * np.linalg.norm(g.local[:, 1, -1] - g.local[:, 0, -1], axis=-1))
                    evidence[mode][-1]["measured_fk"] = {"world_tips_m": g.tips.tolist(), "local_landmarks_m": g.local.tolist()}
                    if target_valid:
                        geometry[mode]["actual_to_frozen_target_local_tip_mm"].append(
                            1000 * np.linalg.norm(g.local[:, :, -1] - target[:, :, -1], axis=-1))
                for first, second in (("A", "B"), ("B", "C"), ("C", "D")):
                    if first in actual and second in actual:
                        paired[first + second].append(1000 * np.linalg.norm(
                            actual[first].local[:, :, -1] - actual[second].local[:, :, -1], axis=-1))
                payload = {"partition": part["partition"], "partition_tick": i, "tick_ns": int(tick),
                           "source_index": row["source_index"], "target_local_m": target.tolist() if target_valid else None,
                           "experiments": evidence}
                out.write((json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n").encode())
    if next(rows, None) is not None or count != 4804:
        raise ValueError("Fixed-wrist diagnostic did not preserve the declared row counts")
    write_json(args.out / "fixed-arm-parity-summary.json", {
        "scope": "DIAGNOSTIC_ONLY", "formal_coverage_credit": False, "root": ROOT,
        "full": baseline["full"], "window": baseline["window"], "robot_signature": baseline["robot_model_servo"]["sha256"],
        "groups_by_named_actuator": [[robot.names[j] for j in group] for group in axes],
        "fixed_arm_reference": "Signed Home joint targets; measured wrist drift reported independently",
        "input_latching": "A/B hold their last published value. C/D use signed pair eligibility. Historical driver watchdog unobserved.",
        "a_clock": "Reconstructed MANUS callback outputs at recorder receive time, consumed on the shared 4 ms issue grid.",
        "same_snapshot_counterfactual": "C.guard_v2 sees the exact C nominal vector/pre-issue state; D has separate physical feedback",
        "common_continuous_proof_max_depth": 24, "extra_v2_proof_max_depth": PROOF["max_depth"],
        "sha256": {"baseline_receipt": file_sha(reference_path), "sdk_ticks": file_sha(args.sdk_ticks),
                    "sdk_updates": file_sha(args.sdk_updates),
                    "sdk_summary": file_sha(args.sdk_summary),
                    "model": file_sha(args.model), "model_receipt": file_sha(args.model_receipt),
                    "ledger": file_sha(ledger), "script": file_sha(Path(__file__))},
        "experiments": {mode: {"row_status": dict(mode_masks[mode]), "physics": dict(run.physics_stats),
            "proofs": dict(run.proof_stats), "min_observed_gap_m": min(run.gaps, default=None), "stop": run.stopped,
            "common_proof_latency_ms": stats(run.timing),
            "geometry": {name: stats(samples) for name, samples in geometry[mode].items()}}
                        for mode, run in streams.items()},
        "paired_actual_local_tip_mm": {variant: stats(samples) for variant, samples in paired.items()}})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "model", "model-receipt", "sdk-ticks", "sdk-updates", "sdk-summary", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    main(p.parse_args())
