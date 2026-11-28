#!/usr/bin/env python3
"""Attribute first per-group sweep rejections without changing the live guard."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import mujoco
import numpy as np

import swept_live_guard as guard


HAND_GROUPS = {1: "left_hand", 2: "right_hand"}
ARM_GROUP = {0: "arms"}


def _check_sources(ready_path, model_receipt_path):
    model_sha = guard.digest(model_receipt_path)
    source_sha = guard.digest(guard.__file__)
    ready = json.loads(ready_path.read_bytes())
    frozen = {
        "scope": "DIAGNOSTIC_ONLY_MUJOCO_COMMAND_ENVELOPE",
        "guard_script_sha256": source_sha,
        "model_receipt_sha256": model_sha,
        "live_model_sha256": guard.LIVE_SHA,
        "signed_model_sha256": guard.SIGNED_SHA,
        "max_box_depth": guard.MAX_BOX_DEPTH,
        "max_box_queries": guard.MAX_BOX_QUERIES,
        "distance_rounding_margin_m": guard.DISTANCE_ROUNDING_MARGIN_M,
        "budget_ms": 25,
    }
    if model_sha != guard.EXACT_MODEL_RECEIPT_SHA or any(ready.get(key) != value for key, value in frozen.items()):
        raise ValueError("The baseline readiness does not bind the exact guard, settings, and models")
    return ready, json.loads(model_receipt_path.read_bytes())["model_identity"]


def _cache_copy(envelope):
    return (None, None) if envelope.anchor is None else (envelope.anchor.copy(), envelope.distances.copy())


def _snapshot(row):
    values = row.get("issued_measured_proposed_rad")
    if values is None:
        return None
    states = np.asarray(values, dtype=np.float64)
    if states.shape != (3, guard.JOINTS) or not np.isfinite(states).all():
        raise ValueError("Malformed finite 54-axis sweep trace snapshot")
    return states


def replay_prefix(path, original, budget_ms, groups=HAND_GROUPS, sequence_filter=None):
    selected, mismatches, counts = {}, [], Counter()
    budget_ns = budget_ms * 1_000_000
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, start=1):
            row = json.loads(line)
            sequence, group, name = row["sequence"], row["group"], row["reason"]
            counts[name] += 1
            states = _snapshot(row)
            target = (group in groups and name == "SWEEP_UNPROVEN" and group not in selected
                      and (sequence_filter is None or sequence_filter == sequence))
            if states is not None:
                if type(sequence) is not int or type(group) is not int or sequence <= 0 or group not in (0, 1, 2):
                    raise ValueError(f"Malformed finite row at line {lineno}")
                caches = tuple(_cache_copy(envelope) for envelope in original.envelopes) if target else None
                packed = guard.REQUEST.pack(guard.MAGIC, sequence, group, *states.reshape(-1))
                _, replay_reason, _, _ = original.evaluate(packed, time.monotonic_ns() + budget_ns)
                if name != replay_reason.name:
                    mismatches.append({"trace_line": lineno, "sequence": sequence,
                                       "recorded": name, "replayed": replay_reason.name})
                # Recover the previously committed possibility from the genuine
                # source trail. Allow timing differences to appear as mismatches.
                original.publish_decision(guard.Reason[name], states)
                if target:
                    selected[group] = (row, states, caches, replay_reason.name, lineno)
            elif target:
                raise ValueError(f"The selected sweep rejection lacks a state at line {lineno}")
            if selected.keys() == groups.keys():
                break
    if selected.keys() != groups.keys():
        raise ValueError(f"The trace has no SWEEP_UNPROVEN row for groups {groups.keys() - selected.keys()}")
    return selected, mismatches, dict(counts)


class ProbeObserver:
    """Observe unchanged production narrow-phase calls and reconstruct their nodes."""

    def __init__(self, envelope, low, high):
        self.envelope = envelope
        self.stack = [((low + high) * .5, (high - low) * .5, 0)]
        self.calls = 0
        self.last_unresolved = None
        self.last_node = None
        self.original_probe = envelope.native._probe

    def observe(self, center, subset, required):
        expected_center, half, depth = self.stack.pop()
        if not np.array_equal(center, expected_center):
            raise ValueError("Instrumented production proof does not match the node sequence")
        remaining = self.original_probe(center, subset, required)
        self.calls += 1
        self.last_node = {"query": self.calls, "depth": depth, "pairs_tested": len(subset)}
        if remaining is None or not remaining.size:
            return remaining
        axis = int(np.argmax(self.envelope.sensitivity * half))
        self.last_unresolved = {
            **self.last_node, "center": center.copy(), "half": half.copy(), "axis": axis,
            "pair_indices": subset[remaining], "required_m": required[remaining],
        }
        if depth < guard.MAX_BOX_DEPTH and half[axis] > 1e-12:
            smaller = half.copy()
            smaller[axis] *= .5
            offset = np.zeros(guard.JOINTS)
            offset[axis] = smaller[axis]
            self.stack.append((center - offset, smaller, depth + 1))
            self.stack.append((center + offset, smaller, depth + 1))
        return remaining


def _named_geom(model, geom_id):
    geom = model.geom(int(geom_id))
    body = model.body(int(model.geom_bodyid[geom_id]))
    return {"geom_id": int(geom_id), "geom_name": geom.name or None,
            "body_id": body.id, "body_name": body.name or "world"}


def _unresolved_details(envelope, hit, robot):
    native = envelope.native
    native.data.qpos[robot.q_indices] = hit["center"]
    mujoco.mj_kinematics(native.model, native.data)
    pairs = []
    clearance = native.clearance + guard.DISTANCE_ROUNDING_MARGIN_M
    for pair, required in zip(hit["pair_indices"], hit["required_m"], strict=True):
        a, b = (int(native.a[pair]), int(native.b[pair]))
        gap = mujoco.mj_geomDistance(native.model, native.data, a, b, float(required + 1e-8), None)
        pairs.append({
            "catalogue_pair_index": int(pair), "geometry": [_named_geom(native.model, a), _named_geom(native.model, b)],
            "center_signed_gap_m": float(gap), "necessary_probe_gap_m": float(required),
            "proof_deficit_m": float(required - gap), "guard_clearance_m": native.clearance,
            "numerical_reserve_m": guard.DISTANCE_ROUNDING_MARGIN_M,
            "conservative_displacement_bound_m": float(required - clearance),
        })
    axis = hit["axis"]
    return {
        "query": hit["query"], "depth": hit["depth"], "pairs_tested": hit["pairs_tested"],
        "unresolved_pair_count": len(pairs),
        "split_axis_index": axis, "split_joint_name": robot.names[axis],
        "split_axis_half_width_rad": float(hit["half"][axis]),
        "center_rad": hit["center"].tolist(), "coordinate_half_width_rad": hit["half"].tolist(),
        "pairs": sorted(pairs, key=lambda item: item["proof_deficit_m"], reverse=True),
    }


def diagnose_model(robot, states, cache):
    low, high = states.min(axis=0), states.max(axis=0)
    envelope = guard.SweptEnvelope(robot)
    envelope.anchor, envelope.distances = cache
    observer = ProbeObserver(envelope, low, high)
    envelope.native._probe = observer.observe
    # Timing was replayed separately. Keep the production 7/63 geometry bounds
    # here, while preventing instrumentation overhead from choosing the result.
    result = envelope.prove(low, high, time.monotonic_ns() + 1_000_000_000)
    hit = observer.last_unresolved
    termination = result.name
    if result == guard.Reason.SWEEP_UNPROVEN:
        if hit is None:
            raise ValueError("A geometry budget stop did not retain an unresolved witness")
        if hit["query"] == observer.calls and hit["depth"] >= guard.MAX_BOX_DEPTH:
            termination = "MAX_SUBDIVISION_DEPTH"
        elif hit["query"] == observer.calls and hit["half"][hit["axis"]] <= 1e-12:
            termination = "MIN_SPLIT_WIDTH"
        elif observer.calls >= envelope.max_queries and observer.stack:
            termination = "MAX_SUBDIVISION_QUERIES"
        else:
            raise ValueError("Instrumented proof returned an unattributable geometry stop")
    return {
        "result": result.name, "termination": termination, "subdivision_queries": observer.calls,
        "last_probe": observer.last_node,
        "last_unresolved_probe": _unresolved_details(envelope, hit, robot) if hit else None,
        "next_unvisited_depth": observer.stack[-1][2] if observer.stack else None,
    }


def _joint_delta(joints, states, count=8):
    issued, measured, proposed = states
    order = np.argsort(np.ptp(states, axis=0))[::-1][:count]
    return [{"joint": joints[i], "axis": int(i), "issued_rad": float(issued[i]), "measured_rad": float(measured[i]),
             "proposed_rad": float(proposed[i]), "envelope_width_rad": float(np.ptp(states[:, i]))} for i in order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("trace", "live-model", "signed-model", "model-receipt", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--ready", type=Path, help="Original sweep-ready.json beside the trace by default")
    parser.add_argument("--arms", action="store_true", help="Attribute the first whole-arm failure instead of both hands")
    parser.add_argument("--sequence", type=int, help="Attribute only this exact logged SWEEP_UNPROVEN request")
    args = parser.parse_args()
    ready_path = args.ready or args.trace.with_name("sweep-ready.json")
    ready, models = _check_sources(ready_path, args.model_receipt)
    live = guard.verified_robot(args.live_model, guard.LIVE_SHA, models["live"])
    signed = guard.verified_robot(args.signed_model, guard.SIGNED_SHA, models["signed"])
    code = guard.Guard((live, signed))
    if ready["joint_names"] != list(code.names) or ready["actuator_names"] != list(code.actuators):
        raise ValueError("The reconstructed named 54-axis order differs from the original")
    groups = ARM_GROUP if args.arms else HAND_GROUPS
    if args.sequence is not None:
        all_groups = {**ARM_GROUP, **HAND_GROUPS}
        with args.trace.open(encoding="utf-8") as stream:
            matches = [row for row in map(json.loads, stream) if row["sequence"] == args.sequence]
        if len(matches) != 1 or matches[0]["reason"] != "SWEEP_UNPROVEN" or matches[0]["group"] not in all_groups:
            raise ValueError("Exact request is missing or is not a sweep-proof rejection")
        selected_group = matches[0]["group"]
        groups = {selected_group: all_groups[selected_group]}
    selected, mismatches, counts = replay_prefix(args.trace, code, ready["budget_ms"], groups, args.sequence)
    witnesses = []
    for group, (row, states, caches, same_budget, lineno) in sorted(selected.items()):
        analysis = {key: diagnose_model(robot, states, cache) for key, robot, cache in
                    zip(("live", "signed"), (live, signed), caches, strict=True)}
        first_reject = next((key for key, result in analysis.items() if result["result"] != "PERMIT"), None)
        reproduced = (same_budget == row["reason"] and first_reject is not None
                      and analysis[first_reject]["result"] == row["reason"] and not mismatches)
        witnesses.append({
            "sequence": row["sequence"], "trace_line": lineno,
            ("group" if group in ARM_GROUP else "hand"): groups[group],
            "recorded_reason": row["reason"], "same_budget_offline_replay_reason": same_budget,
            "recorded_proof_duration_ms": (row["proof_complete_ns"] - row["receipt_ns"]) / 1e6,
            "original_first_failing_model_reproduced": first_reject if reproduced else None,
            "instrumented_first_failing_model": first_reject, "models": analysis,
            "widest_input_axes": _joint_delta(code.names, states),
        })
    receipt = {
        "scope": "OFFLINE_REPLAY_OF_PREPUBLICATION_PROOF_ONLY",
        "trace_sha256": guard.digest(args.trace), "ready_sha256": guard.digest(ready_path),
        "guard_sha256": ready["guard_script_sha256"], "live_model_sha256": guard.LIVE_SHA,
        "signed_model_sha256": guard.SIGNED_SHA, "original_budget_ms": ready["budget_ms"],
        "proof_bounds": {"maximum_depth": guard.MAX_BOX_DEPTH, "maximum_subdivision_queries": guard.MAX_BOX_QUERIES},
        "prefix_reason_counts": counts, "prefix_replay_mismatches": mismatches, "witnesses": witnesses,
    }
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return 0 if all(witness["original_first_failing_model_reproduced"] for witness in witnesses) else 2


if __name__ == "__main__":
    raise SystemExit(main())
