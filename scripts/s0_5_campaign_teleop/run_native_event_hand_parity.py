#!/usr/bin/env python3
"""Fixed-arm A/B/C/D diagnostic using native hand commands on the 500 Hz clock."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from audit_root_five_geometry_layers import stats
from dexgym.tasks.direct.wholebody_s05.v1.rmb.schema import file_sha, write_json
from dexgym.tasks.direct.wholebody_s05.v1.task.physics import ServoProfile, signature
from native_hand_driver_probe import build_driver, controls
from offline_rmb_lockstep import Lockstep, PROOF
from swept_live_guard import Reason, verified_robot
from swept_live_guard_v2 import SweptEnvelopeV2
from teleop_parity_hand_physics import MODES, ROOT, safety


DT_NS = 2_000_000
SUBSTEPS = 10
FROZEN_WINDOWS_SHA = "0931007455bf9aaf178f6260726e037e74cc69a404b9345439c186263c9b822c"


class RecordedUpdates:
    def __init__(self, stream):
        self.events = iter(map(json.loads, stream))
        self.pending = next(self.events, None)
        self.previous = None

    def through(self, cutoff):
        updates = []
        while self.pending is not None and self.pending["receive_ns"] <= cutoff:
            event = self.pending
            if self.previous is not None and event["receive_ns"] < self.previous:
                raise ValueError("Native events reversed")
            self.previous = event["receive_ns"]
            self.pending = next(self.events, None)
            if event["published"]:
                updates.append(event)
        return updates


class AxisRun:
    def __init__(self, robot, driver, mode, start):
        self.mode = mode
        self.robot = robot
        self.start = start
        self.sim = Lockstep(robot, SimpleNamespace(reset=lambda _: np.zeros(2)))
        self.common = SweptEnvelopeV2(robot, max_queries=PROOF["max_queries"], max_depth=24)
        ctrl = np.stack([a.ctrlrange for a in robot.actuators])
        self.driver = driver.create(robot.home, ctrl, start)
        self.receive_sequence = [0, 0]
        self.source = [None, None]
        self.stopped = None
        self.counts = Counter()
        self.gaps = []
        self.streak = 0
        self.was_held = False

    def close(self):
        self.driver.close()

    def dispatch(self, cutoff, updates):
        admitted_events = []
        for event in updates:
            stamp = event["receive_ns"]
            if not self.start <= stamp <= cutoff:
                continue
            side = ("left", "right").index(event["side"])
            if self.source[side] and self.source[side]["source_index"] == event["source_index"]:
                continue
            self.receive_sequence[side] += 1
            accepted = self.driver.receive(side, stamp, stamp, self.receive_sequence[side], event["filtered_rad"])
            admitted_events.append({"source_index": event["source_index"], "receive_ns": stamp,
                                    "side": event["side"], "driver_accepted": accepted})
            self.counts["driver_accepted" if accepted else "driver_rejected"] += 1
            if accepted:
                self.source[side] = {key: event[key] for key in ("source_index", "receive_ns", "raw_rad", "filtered_rad")}
        return admitted_events

    def advance(self, cutoff, updates):
        if self.stopped:
            return {"status": "NOT_EXECUTED", "cause": self.stopped, "cutoff_ns": cutoff}
        measured, _ = self.sim.state()
        admitted = self.dispatch(cutoff, updates)
        driver_q, driver_fault = self.driver.snapshot(cutoff, measured)
        prior = self.sim.issued
        target = driver_q.astype(np.float32).astype(np.float64)
        snapshot = np.stack((prior, self.sim.physics.filtered[0], measured))
        base, base_detail = safety(self.robot, self.common, snapshot, target)
        extra, extra_detail = self.sim._proof(target, measured) if self.mode in ("C", "D") else (None, None)
        self.counts["candidate:" + base.name + ":" + str(base_detail.get("termination"))] += 1
        if extra is not None:
            self.counts["c_d_snapshot:" + base.name + ":" + extra.name] += 1
        proper = base == Reason.PERMIT and (self.mode != "D" or extra == Reason.PERMIT)
        held = not proper or driver_fault.any()
        result = {"cutoff_ns": cutoff, "sources": self.source.copy(), "admitted_events": admitted,
                  "driver_watchdog": driver_fault.tolist(), "q_driver_command": driver_q.tolist(),
                  "pre_issue_signed_snapshot": snapshot.tolist(), "q_candidate": target.tolist(),
                  "common_proof": base.name, "common_detail": base_detail,
                  "current_extra_v2": extra.name if extra is not None else None, "extra_detail": extra_detail}
        if not proper:
            hold, hold_detail = safety(self.robot, self.common, snapshot, prior)
            if self.mode == "D" and hold == Reason.PERMIT:
                hold, hold_detail = self.sim._proof(prior, measured)
            result.update(hold_proof=hold.name, hold_detail=hold_detail)
            if hold != Reason.PERMIT:
                self.stopped = "HOLD_INFEASIBLE:" + hold.name
                return {**result, "status": "FAILED", "cause": self.stopped}
        else:
            self.sim.issued = target
        interval = self.sim.physics.interval(torch.as_tensor(self.sim.issued[None], dtype=torch.float32),
                                             torch.ones(1, dtype=torch.bool), self.sim.safety, count=1)
        if interval.substeps[0] != 1:
            self.stopped = "PHYSICS_COUNT"
            return {**result, "status": "FAILED", "cause": self.stopped}
        q, qd = self.sim.state()
        status = "HOLD" if held else "REJOINING" if self.was_held else "ISSUED"
        self.was_held = held
        gap, depth = float(interval.minimum_gap[0]), float(interval.depth[0])
        event = int(interval.event[0])
        self.gaps.append(gap)
        self.counts.update(physics_steps=1, signed_contacts=int(bool(interval.contact[0].any())),
                           penetrating_steps=int(depth > 0), physics_events=int(bool(event)),
                           held_steps=int(held), moved_steps=int(not np.array_equal(prior, self.sim.issued)),
                           rejoining_steps=int(status == "REJOINING"))
        unproved = (Reason.SWEEP_UNPROVEN, Reason.TIME_BUDGET)
        self.streak = self.streak + 1 if base in unproved or self.mode == "D" and extra in unproved else 0
        if event:
            self.stopped = f"PHYSICS_EVENT:{event}"
        elif self.streak >= 16:
            self.stopped = "SIXTEEN_CONSECUTIVE_UNPROVED_2MS_TARGETS"
        hits = [{"geom1": self.sim.physics.model.geom(int(x.geom[0])).name,
                 "geom2": self.sim.physics.model.geom(int(x.geom[1])).name, "gap_m": float(x.dist)}
                for x in self.sim.physics.data[0].contact if x.dist <= 0]
        return {**result, "status": status, "cause": self.stopped, "event": event, "q_issued": self.sim.issued.tolist(),
                "q_applied": interval.applied[0, 0].tolist(), "q_measured": q.tolist(), "qd_measured": qd.tolist(),
                "torque_nm": interval.torque[0, 0].tolist(), "minimum_gap_m": gap,
                "depth_m": depth, "contact_pairs": hits, "feedback_available_ns": cutoff + DT_NS}


def b_updates(row):
    return [{"side": side, "source_index": row["source_index"][i], "receive_ns": row["control_cutoff_ns"],
             "raw_rad": row["b_raw_rad"][i], "filtered_rad": row["b_filtered_rad"][i]}
            for i, side in enumerate(("left", "right")) if row["b_valid"][i]]


def available(row, leading_cutoff):
    # The endpoint of a control interval first becomes available for the next tick.
    return [s for s in row if "q_measured" in s and s["feedback_available_ns"] <= leading_cutoff]


def run(args):
    reference = json.loads(args.baseline.read_bytes())
    if reference["root"] != ROOT or reference["window"]["plan_ticks"] != 4804 or file_sha(args.windows) != FROZEN_WINDOWS_SHA:
        raise ValueError("Original root, denominator or frozen seven windows changed")
    windows = json.loads(args.windows.read_bytes())["physical_windows"]
    signed = json.loads(args.model_receipt.read_bytes())["model_identity"]["signed"]
    robot = verified_robot(args.model, signed["xml_sha256"], signed)
    if signature(robot, ServoProfile()) != reference["robot_model_servo"]:
        raise ValueError("Signed physical model or Servo differs")
    sdk = json.loads(args.sdk_summary.read_bytes())
    if (sdk["output_sha256"][args.native_a.name] != file_sha(args.native_a)
            or sdk["output_sha256"][args.sdk_ticks.name] != file_sha(args.sdk_ticks)):
        raise ValueError("Frozen A/B SDK artifacts changed")
    if file_sha(args.native_c) != file_sha(args.native_a):
        # The independent replay also certifies row-level L2-L4 equality.
        # A byte mismatch must be investigated before a dynamics comparison.
        raise ValueError("Recovered C native updates are not byte-identical to baseline A")
    args.out.mkdir()
    api, driver_evidence = build_driver(args.teleop, args.out / "driver-build")
    if api.hand_aliases != [a.name for a in robot.actuators[14:]]:
        raise ValueError("Native 40 hand aliases do not match the signed actuator transmissions")
    qa = controls(api)
    start = reference["window"]["hmb_cutoff_start_ns"]
    stop = reference["window"]["hmb_cutoff_stop_exclusive_ns"]
    geometry_home = robot.geometry(robot.home)
    totals = {m: Counter() for m in MODES}
    statuses = {m: Counter() for m in MODES}
    drift = defaultdict(list)
    maximum_speed = defaultdict(list)
    gaps = defaultdict(list)
    window_results = []
    matches = Counter()
    fault_steps = []
    count = 0
    active = None
    streams = {}
    prior = {m: [] for m in MODES}
    history_path = args.out / "fixed-arm-native.jsonl.gz"
    with ExitStack() as stack:
        stack.callback(lambda: [engine.close() for engine in streams.values()])
        ticks = stack.enter_context(args.sdk_ticks.open(encoding="utf-8"))
        a = RecordedUpdates(stack.enter_context(args.native_a.open(encoding="utf-8")))
        c = RecordedUpdates(stack.enter_context(args.native_c.open(encoding="utf-8")))
        raw = stack.enter_context(history_path.open("xb"))
        ledger = stack.enter_context(gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=3))
        for row in map(json.loads, ticks):
            t = row["tick_ns"]
            if not start <= t < stop:
                continue
            count += 1
            lead = row["control_cutoff_ns"]
            w = next((i for i, span in enumerate(windows)
                      if span["start_tick_ns"] <= t < span["stop_exclusive_tick_ns"]), None)
            if w != active:
                for mode, engine in streams.items():
                    totals[mode].update(engine.counts)
                    gaps[mode].extend(engine.gaps)
                    window_results.append({"window": active, "mode": mode, "stop": engine.stopped,
                                           "counts": dict(engine.counts)})
                    engine.close()
                streams.clear()
                prior = {m: [] for m in MODES}
                if w is not None:
                    streams.update({m: AxisRun(robot, api, m, lead) for m in MODES})
                active = w
            evidence = {mode: [] for mode in MODES}
            for sub in range(SUBSTEPS):
                cutoff = lead + sub * DT_NS
                ae, ce = a.through(cutoff), c.through(cutoff)
                inputs = {"A": ae, "B": b_updates(row) if sub == 0 else [], "C": ce, "D": ce}
                for mode in MODES:
                    step = (streams[mode].advance(cutoff, inputs[mode]) if streams else
                            {"cutoff_ns": cutoff, "status": "NOT_EXECUTED", "cause": "OUTSIDE_FROZEN_WINDOWS"})
                    evidence[mode].append(step)
                    if "qd_measured" in step:
                        speed = float(np.max(np.abs(step["qd_measured"][14:])))
                        maximum_speed[mode].append(speed)
                        if step["event"]:
                            fault_steps.append({"mode": mode, "tick_ns": t, "substep": sub, "event": step["event"],
                                                "qd": step["qd_measured"], "contacts": step["contact_pairs"]})
                first, recovered = evidence["A"][-1], evidence["C"][-1]
                if streams:
                    for key in ("sources", "q_driver_command", "q_issued", "q_applied", "q_measured", "qd_measured"):
                        if key in first or key in recovered:
                            matches[key + (":equal" if first.get(key) == recovered.get(key) else ":different")] += 1
            sampled = {}
            strict_sampled = {}
            nominal_samples = {}
            measured_fk = {}
            for mode, records in evidence.items():
                eligible = available(prior[mode], lead)
                sampled[mode] = ({key: eligible[-1][key] for key in
                                  ("q_issued", "q_applied", "q_measured", "qd_measured", "feedback_available_ns")}
                                 if len(eligible) == SUBSTEPS and not any(s["event"] for s in eligible) else None)
                signed_eligible = available(prior[mode], int(t))
                strict_sampled[mode] = ({key: signed_eligible[-1][key] for key in
                    ("sources", "q_driver_command", "q_issued", "q_applied", "q_measured", "qd_measured",
                     "feedback_available_ns")}
                    if signed_eligible and not any(s["event"] for s in signed_eligible)
                    and int(t) - signed_eligible[-1]["feedback_available_ns"] < DT_NS else None)
                now = records[0]
                nominal_samples[mode] = ({key: now[key] for key in ("sources", "q_driver_command")}
                                         if "q_driver_command" in now else None)
                physical = [s for s in records if "q_measured" in s]
                statuses[mode]["COMPLETE" if len(physical) == SUBSTEPS else "PARTIAL" if physical else "NOT_EXECUTED"] += 1
                if physical:
                    g = robot.geometry(physical[-1]["q_measured"])
                    drift[mode].append((1000 * np.linalg.norm(g.wrists - geometry_home.wrists, axis=1)).tolist())
                    measured_fk[mode] = {"fingertip_world_m": g.tips.tolist(), "landmark_local_m": g.local.tolist(),
                                         "wrist_xyz_m": g.wrists.tolist(),
                                         "timestamp_ns": physical[-1]["feedback_available_ns"]}
                prior[mode] = records
            payload = {"partition": row["partition"], "partition_tick": row["partition_tick"], "tick_ns": t,
                       "input_source_index": row["source_index"], "control_cutoff_ns": lead, "window": w,
                       "rmb_feedback_at_leading_cutoff": sampled, "rmb_native_target_at_leading_cutoff": nominal_samples,
                       "rmb_feedback_at_exact_signed_tick": strict_sampled,
                       "fk_at_last_observed_substep": measured_fk, "experiments": evidence}
            ledger.write((json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n").encode())
        for mode, engine in streams.items():
            totals[mode].update(engine.counts)
            gaps[mode].extend(engine.gaps)
            window_results.append({"window": active, "mode": mode, "stop": engine.stopped, "counts": dict(engine.counts)})
    if count != 4804 or len(window_results) != 28:
        raise ValueError("Not all original rows and frozen seven windows reached reporting")
    write_json(args.out / "fixed-arm-native-summary.json", {
        "scope": "DIAGNOSTIC_ONLY", "formal_coverage_credit": False, "root": ROOT, "full": reference["full"],
        "window": reference["window"], "windows": windows, "source_clock": "RECORDED_EVENT_REPLAY",
        "physical_clock": "Signed START-anchored lockstep grid plus the previous reconstruction's 1 us cutoff margin",
        "strict_RMB_sample": "Latest previous observed 500 Hz endpoint no later than exact signed HMB tick",
        "signed_joints": robot.names, "physics_hz": 500, "rmb_hz": 50, "driver_source": driver_evidence,
        "driver_controls": qa, "experiment_status": {m: dict(statuses[m]) for m in MODES},
        "physics": {m: {**totals[m], "sampled_minimum_gap_m": min(gaps[m], default=None),
                         "max_hand_speed_rad_s": max(maximum_speed[m], default=None),
                         "wrist_position_drift_mm": stats(drift[m])} for m in MODES},
        "paired_A_C": dict(matches), "all_fault_steps": fault_steps, "window_results": window_results,
        "sha256": {**{name: file_sha(getattr(args, name)) for name in
                      ("baseline", "model", "model_receipt", "windows", "sdk_summary", "sdk_ticks", "native_a", "native_c")},
                   "ledger": file_sha(history_path), "source": file_sha(Path(__file__))},
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("baseline", "teleop", "model", "model-receipt", "windows", "sdk-summary", "sdk-ticks",
                   "native-a", "native-c", "out"):
        p.add_argument("--" + option, required=True, type=Path)
    run(p.parse_args())
