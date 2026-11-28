#!/usr/bin/env python3
"""Independent exhaustive archived request audit. No dynamics or controller runs."""
from collections import Counter, defaultdict
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter

import mujoco as mj
import numpy as np

from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel, asset_files
from dexgym.tasks.direct.wholebody_s05.v1.task.native_sweep import NativeSweep


EXPECTED_RECEIPT = "d6d83d40b44d29447e1d4a1e357bc0424b123dbabfcc113eef8218d330c605df"
FIELD_NAMES = (
    "nominal_proposal", "q_issued", "q_issued_mask", "q_measured",
    "q_measured_mask", "q_applied", "applied_substeps", "physical_interval_mask",
    "physical_clearance_m", "physical_penetration_m", "physical_contact", "physical_event",
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def statistics(values, *, mm=False):
    return {key: float(value) * (1000. if mm else 1.) for key, value in zip(
        ("min", "p50", "p95", "p99", "max"), np.percentile(values, [0, 50, 95, 99, 100]), strict=True
    )}


def audit(run, model, out):
    received = run / "receipt.json"
    if digest(received) != EXPECTED_RECEIPT:
        raise ValueError("Unexpected frozen run02 receipt")
    receipt = json.loads(received.read_bytes())
    trace, start_trace = run / "diagnostic-events.jsonl", run / "start-events.jsonl"
    if digest(trace) != receipt["diagnostic_events_sha256"] or digest(start_trace) != receipt["start_events_sha256"]:
        raise ValueError("Frozen raw trace mismatch")
    model_assets = {Path(path).relative_to(model.parent).as_posix(): value for path, value in asset_files(model).items()}
    if model_assets != receipt["robot_model_servo"]["assets"] or mj.__version__ != receipt["robot_model_servo"]["mujoco"]:
        raise ValueError("Model assets or MuJoCo version mismatch")
    robot = RobotModel(model)
    sweep = NativeSweep(robot)
    m, d = sweep.model, sweep.data
    a, b = robot.collision_pairs[2425]
    geom_names = (m.geom(a).name, m.geom(b).name)
    if geom_names != ("r_finger1_tip_sensor_col", "r_finger2_tip_sensor_col"):
        raise ValueError("Unexpected signed pair 2425 identity")
    axes = np.flatnonzero(sweep.weights[2425])
    if len(axes) != 8:
        raise ValueError("Unexpected common-parent cancellation")
    q = robot.home.copy()

    @lru_cache(maxsize=65536)
    def right_pair(angles):
        q[axes] = angles
        d.qpos[robot.q_indices] = q
        mj.mj_kinematics(m, d)
        return float(mj.mj_geomDistance(m, d, a, b, .1, None))

    def gap(q):
        return right_pair(tuple(q[axes]))

    signed_parts = {part["partition"]: part for part in receipt["partitions"]}
    ca = Counter()
    collision_ca = Counter()
    last_part = None
    q_prev = q_filter = q_feedback = None
    original_start = [json.loads(line) for line in start_trace.read_bytes().splitlines()][-1]
    q_prev = q_filter = np.asarray(original_start["issued"], dtype=np.float64)
    q_feedback = np.asarray(original_start["measured"], dtype=np.float64)
    ignored = Counter()
    observed = defaultdict(list)
    sample_cases = {}
    keys, physical_intervals = set(), set()
    t0 = perf_counter()
    n_phys = 0
    out.mkdir(exist_ok=False, parents=True)
    ledger_file = out / "request_ledger.jsonl"
    with trace.open("rb") as rows, ledger_file.open("w", newline="\n", encoding="utf-8") as ledger:
        for line_number, raw in enumerate(rows, 1):
            row = json.loads(raw)
            part, tick = row["partition"], row["tick"]
            key = (part, tick)
            if key in keys:
                raise ValueError(f"Duplicate schedule identity {key}")
            keys.add(key)
            if part != last_part:
                spec = signed_parts[part]
                directory = run / spec["directory"]
                arr = {}
                for name in FIELD_NAMES:
                    fp = directory / (name + ".npy")
                    signed = spec["arrays"][name]
                    if digest(fp) != signed["sha256"]:
                        raise ValueError(f"Array receipt mismatch: {fp}")
                    arr[name] = np.load(fp, allow_pickle=False, mmap_mode="r")
                    if list(arr[name].shape) != signed["shape"] or arr[name].dtype.str != signed["dtype"]:
                        raise ValueError(f"Array shape or dtype mismatch: {fp}")
                last_part = part
            if len(row["safety"]) != 5 or int(arr["applied_substeps"][tick]) != 10:
                raise ValueError(f"Unexpected original schedule at {key}")
            if not arr["q_measured_mask"][tick] or not arr["q_issued_mask"][tick].all():
                raise ValueError(f"Missing originally stored feedback/command at {key}")
            for sub, telemetry in enumerate(row["safety"]):
                seen = telemetry["metrics"]
                status, termination = telemetry["proof"], seen["termination"]
                if telemetry["fault"] or not arr["physical_interval_mask"][tick, sub]:
                    raise ValueError(f"Unexpected physical stop/fault at {key} subcycle {sub}")
                used_hold = telemetry["hold"]
                if (telemetry["hold_proof"] if used_hold else status) != "PERMIT":
                    raise ValueError(f"Final original issuance was not proved at {key} subcycle {sub}")
                issued = np.asarray(arr["q_issued"][tick, sub], dtype=np.float64)
                proposed = np.asarray(arr["nominal_proposal"][tick, sub], dtype=np.float64)
                if not np.isfinite(issued).all() or not np.isfinite(proposed).all():
                    raise ValueError(f"Non-finite command at {key} subcycle {sub}")
                if not np.array_equal(issued, q_prev if used_hold else proposed):
                    raise ValueError(f"Unexpected archived selection at {key} subcycle {sub}")
                if status == "PERMIT":
                    ignored[status] += 1
                else:
                    ca[(status, termination, tuple(seen["unresolved_pair_indices"]))] += 1
                    gap_issued, gap_filtered, gap_candidate = map(gap, (q_prev, q_filter, proposed))
                    gap_measured_sample = gap(q_feedback)
                    if not all(np.isfinite([gap_issued, gap_filtered, gap_candidate, gap_measured_sample])):
                        raise ValueError(f"Non-finite independently computed point gap at {key}, {sub}")
                    for name, value in (("issued", gap_issued), ("filtered", gap_filtered),
                                        ("candidate", gap_candidate), ("last_archived_measurement", gap_measured_sample),
                                        ("runtime_min_probe", seen["min_sample_gap_m"])):
                        observed[name].append(value)
                    for name, value in (("candidate", gap_candidate), ("issued", gap_issued),
                                        ("filtered", gap_filtered), ("last_archived_measurement", gap_measured_sample)):
                        if value < robot.clearance_m:
                            collision_ca[(name, "PENETRATION" if value < 0 else "CLEARANCE_VIOLATION")] += 1
                    row_id = (line_number, tick, sub)
                    marker = {"line": line_number, "partition": part, "tick": tick, "subcycle": sub,
                              "time_ns": row["time_ns"], "control_cutoff_ns": row["control_cutoff_ns"],
                              "status": status, "termination": termination, "held": used_hold,
                              "hold_proof": telemetry["hold_proof"], "last_unresolved_pairs": seen["unresolved_pair_indices"],
                              "queries": seen["queries"], "depth": seen["max_depth"],
                              "sampled_clearance_violation": seen["sampled_clearance_violation"],
                              "witness_pair_index": seen["witness_pair_index"],
                              "runtime_min_probe_gap_m": seen["min_sample_gap_m"],
                              "independent_right_pair_gaps_m": [gap_issued, gap_filtered, gap_candidate, gap_measured_sample],
                              "precontrol_measured_exact": False,
                              "q_measured_samples": "latest_20ms_export_or_exact_start_JSON",
                              "source_event_row_sha256": sha256(raw).hexdigest()}
                    ledger.write(json.dumps(marker, separators=(",", ":"), allow_nan=False) + "\n")
                    if not sample_cases or gap_candidate < sample_cases["candidate_min"]["values"][0]:
                        sample_cases["candidate_min"] = {"values": [gap_candidate], "row": marker}
                    if "first" not in sample_cases:
                        sample_cases["first"] = {"values": [gap_candidate], "row": marker}
                    observed["queries"].append(seen["queries"])
                    observed["depth"].append(seen["max_depth"])
                q_prev = issued
                q_filter = np.asarray(arr["q_applied"][tick, 2*sub+1], dtype=np.float64)
                physical = (float(arr["physical_clearance_m"][tick, sub]),
                            float(arr["physical_penetration_m"][tick, sub]),
                            bool(arr["physical_contact"][tick].any()),
                            int(arr["physical_event"][tick, sub]))
                physical_intervals.add(physical)
                n_phys += 1
            q_feedback = np.asarray(arr["q_measured"][tick], dtype=np.float64)
    if sum(ca.values()) + sum(ignored.values()) != 24020 or len(keys) != 4804 or n_phys != 24020:
        raise ValueError("Frozen run denominators do not conserve")
    visuals = [m.geom("r_visual_13").id, m.geom("r_visual_26").id]
    model_pair = {
        "index": 2425, "geom_ids": [a, b], "geom_names": geom_names,
        "same_mesh_as_visual": [bool(m.geom_dataid[i] == m.geom_dataid[j]) for i,j in zip((a,b), visuals)],
        "same_body_as_visual": [bool(m.geom_bodyid[i] == m.geom_bodyid[j]) for i,j in zip((a,b), visuals)],
        "same_local_transform_as_visual": [
            bool(np.array_equal(m.geom_pos[i], m.geom_pos[j]) and np.array_equal(m.geom_quat[i], m.geom_quat[j]))
            for i,j in zip((a,b), visuals)
        ],
        "contype_conaffinity": [[int(m.geom_contype[i]), int(m.geom_conaffinity[i])] for i in (a,b)],
        "collision_shape": "MuJoCo mesh narrowphase convex collision representation; shared mesh asset with visual renderer",
        "relative_joint_names": [robot.names[i] for i in axes],
    }
    summary = {
        "scope": "DIAGNOSTIC_ONLY_ARCHIVED_NO_DYNAMICS",
        "runtime_commit": "c2b5720e663d54d30ae937ecf710baac20112271",
        "receipt_sha256": digest(received), "trace_sha256": digest(trace), "script_sha256": digest(Path(__file__)),
        "request_ledger_sha256": digest(ledger_file), "model_sha256": digest(model),
        "mujoco": mj.__version__, "asset_count": len(model_assets), "verified_arrays": len(FIELD_NAMES)*len(signed_parts),
        "original_ticks": len(keys), "candidate_requests": sum(ca.values()) + sum(ignored.values()),
        "admitted_candidate": dict(ignored),
        "failed_original_telemetry_counts": [{"status": status, "termination": term, "last_unresolved_pairs": indices,
                                             "count": n} for (status,term,indices),n in sorted(ca.items())],
        "right_pair_geometry": model_pair,
        "independent_point_gap_statistics_mm": {name: statistics(values, mm=True) for name,values in observed.items()
                                                 if name not in ("queries", "depth")},
        "query_statistics": statistics(observed["queries"]), "depth_statistics": statistics(observed["depth"]),
        "point_threshold_failures": [{"point": name, "reason": reason, "count": n}
                                     for (name,reason),n in sorted(collision_ca.items())],
        "original_physics": {"intervals": n_phys, "substeps": 2*n_phys,
                             "minimum_sampled_gap_mm": min(p[0] for p in physical_intervals)*1000.,
                             "maximum_penetration_mm": max(p[1] for p in physical_intervals)*1000.,
                             "any_contact_across_13_channels_per_hmb_tick": any(p[2] for p in physical_intervals),
                             "any_event": any(p[3] for p in physical_intervals)},
        "cases": sample_cases, "wall_seconds": perf_counter()-t0,
        "limitations": [
            "Last unresolved-pair list identifies the original stopping leaf, not all unresolved boxes.",
            "Independent mesh computations are point geometry of all archived rejected proposals and available stored references.",
            "Exact per-request measured f64, exact internal filtered f64, and proof tree were not archived. Full historical proof replay needs verified deterministic physical state reconstruction.",
            "No admission, no interpretation of SWEEP_UNPROVEN as penetration, and no active coverage credit."
        ],
    }
    (out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key,value in summary.items() if key not in ("cases", "limitations")}, separators=(",", ":")))


if __name__ == "__main__":
    audit(*(Path(value).resolve() for value in sys.argv[1:]))
