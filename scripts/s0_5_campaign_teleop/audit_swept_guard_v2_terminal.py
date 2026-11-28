#!/usr/bin/env python3
"""Offline geometric follow-up on the three terminal requests in root five's V2 replay."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

import audit_swept_guard_v2 as archive
import swept_live_guard as v1
import swept_live_guard_v2 as v2
from diagnose_swept_hand_rejection import _check_sources, _named_geom


TERMINAL = (7243, 7244, 7245)


def select(path):
    with path.open(encoding="utf-8") as stream:
        result = {row["sequence"]: row for line in stream
                  if (row := json.loads(line))["sequence"] in TERMINAL}
    if set(result) != set(TERMINAL):
        raise ValueError(f"Missing terminal request in {path}")
    return result


def geometry_pairs(envelope, pairs, width):
    native = envelope.native
    pairs = sorted(pairs, key=lambda index: float(envelope._variation(width, [index])[0]), reverse=True)
    return [{
        "pair_index": index,
        "geometry": [_named_geom(native.model, ids[index]) for ids in (native.a, native.b)],
        "root_relative_displacement_bound_mm": float(envelope._variation(width, [index])[0]) * 1e3,
    } for index in pairs[:8]]


def prove(robot, low, high, profile):
    envelope = v2.SweptEnvelopeV2(robot, max_queries=profile["max_queries"], max_depth=profile["max_depth"])
    tick = time.monotonic_ns()
    result = envelope.prove(low, high, tick + profile["budget_ms"] * 1_000_000)
    elapsed_ms = (time.monotonic_ns() - tick) / 1e6
    metrics = envelope.last_metrics.copy()
    pending = metrics.pop("unresolved_pair_indices")
    return {"reason": result.name, "duration_ms": elapsed_ms, "metrics": metrics,
            "unresolved_pair_count": len(pending), "unresolved_pairs_largest_root_bounds":
                geometry_pairs(envelope, pending, (high - low) * .5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = archive.read(args.run_receipt)
    run, = receipt["attempts"]
    if (receipt["scope"] != "DIAGNOSTIC_ONLY" or receipt["formal_coverage_credit"]
            or receipt["plan_sha256"] != archive.PLAN_SHA or run["root"] != archive.ROOT
            or not run["same_domain_empty_after"] or not run["owned_groups_stopped"]):
        raise ValueError("Expected the single completed and isolated root-five diagnostic receipt")
    command = run["guard_command"]
    paths = {key: Path(command[command.index(f"--{key}") + 1]).resolve(strict=True)
             for key in ("trace", "ready", "metrics", "v2-receipt", "live-model", "signed-model", "model-receipt")}
    bound = {"ready": "guard_ready_sha256", "metrics": "sweep_v2_metrics_sha256",
             "v2-receipt": "sweep_v2_ready_sha256"}
    if (any(v1.digest(paths[key]) != run[name] for key, name in bound.items())
            or v1.digest(paths["trace"].parent / "evidence/trace.jsonl") != run["trace_sha256"]
            or archive.read(paths["v2-receipt"])["implementation_sha256"] != v1.digest(v2.__file__)):
        raise ValueError("Terminal proof under audit differs from the open-loop source or run identity")
    _, models = _check_sources(paths["ready"], paths["model-receipt"])
    robots = (v1.verified_robot(paths["live-model"], v1.LIVE_SHA, models["live"]),
              v1.verified_robot(paths["signed-model"], v1.SIGNED_SHA, models["signed"]))
    raw, meters = select(paths["trace"]), select(paths["metrics"])
    result = []
    baseline = v1.Guard(robots)
    for seq in TERMINAL:
        row, meter = raw[seq], meters[seq]
        states = np.asarray(row["issued_measured_proposed_rad"])
        low, high = states.min(axis=0), states.max(axis=0)
        if (row["reason"] != "SWEEP_UNPROVEN" or meter["reason"] != row["reason"]
                or row["group"] != meter["group"] or not np.isfinite(states).all()
                or not baseline._legal_box(low, high)):
            raise ValueError(f"Recorded request {seq} lacks an eligible sweep denial")
        widths = [{"joint": robots[0].names[i], "box_width_deg": float(np.rad2deg(high[i] - low[i]))}
                  for i in np.argsort(high - low)[::-1][:8]]
        item = {"sequence": seq, "group": row["group"], "runtime_reason": row["reason"],
                "runtime_duration_ms": (row["proof_complete_ns"] - row["receipt_ns"]) / 1e6,
                "largest_joint_intervals": widths, "models": {}}
        for model, robot in zip(archive.MODELS, robots, strict=True):
            runtime = meter["models"][model]
            item["models"][model] = {
                "runtime": runtime,
                "v2_offline_25ms_cold": prove(robot, low, high, dict(budget_ms=25, max_queries=63, max_depth=7)),
                "v2_offline_extended_cold": prove(robot, low, high, archive.HIGH),
                "v1_offline_extended_cold": archive.high_v1(robot, states, (None, None)),
                "dense_check": archive.dense_diagnostic(robot, states),
            }
        result.append(item)
    report = {"scope": "DIAGNOSTIC_ONLY_OFFLINE_TERMINAL_REQUESTS", "root": archive.ROOT,
              "formal_coverage_credit": False, "extended_proof_limits": archive.HIGH,
              "ready_sha256": v1.digest(paths["ready"]), "source_trace_sha256": v1.digest(paths["trace"]),
              "runtime_metrics_sha256": v1.digest(paths["metrics"]), "paired_audit_sha256": v1.digest(archive.__file__),
              "implementation_sha256": v1.digest(v2.__file__), "run_receipt_sha256": v1.digest(args.run_receipt),
              "terminal_requests": result}
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
