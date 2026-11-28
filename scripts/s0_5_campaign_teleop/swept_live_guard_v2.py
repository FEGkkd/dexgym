#!/usr/bin/env python3
"""Fail-closed pair-adaptive geometric certification for the same MuJoCo guard box."""

import argparse
import json
import os
from pathlib import Path
import signal
import time

import mujoco
import numpy as np

import swept_live_guard as base
from dexgym.tasks.direct.wholebody_s05.v1.task.native_sweep import NativeSweep


MODE = "DIAGNOSTIC_ONLY_MUJOCO_SWEEP_V2"
BOUND_ROUNDING_M = 1e-12


def _chord(radius):
    """Unit-radius orbit length bound for every angle in [-radius, radius]."""
    return np.nextafter(2.0 * np.sin(np.minimum(radius, np.pi) * 0.5), np.inf)


def _relative_levers(sweep):
    """Reduce just the first hinge lever using the invariant axis offset.

    Between each geom and its first ancestor hinge, all bodies are fixed.
    Its center's perpendicular offset from that axis therefore has invariant
    length. Together with the original collider sphere radius it bounds all
    points on the collider. Farther hinges keep the V1 triangle-inequality
    radius because downstream joint rotation can move the geom center.
    A hinge shared by both geoms contributes zero to their relative distance,
    as in NativeSweep. An exact ancestor test verifies that it stays zero.
    """
    model, data = sweep.model, sweep.data
    mujoco.mj_kinematics(model, data)
    axes = {joint.id: i for i, joint in enumerate(sweep.robot.joints)}
    geometry_axes = np.zeros((model.ngeom, base.JOINTS), dtype=bool)
    nearest = np.full(model.ngeom, -1, dtype=int)
    radius = np.zeros(model.ngeom)
    for geom in np.unique(np.r_[sweep.a, sweep.b]):
        body = int(model.geom_bodyid[geom])
        while body:
            count = int(model.body_jntnum[body])
            if count:
                hinge = int(model.body_jntadr[body])
                if count != 1 or model.jnt_type[hinge] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError("V2 requires the reviewed hinge-only model")
                index = axes[hinge]
                geometry_axes[geom, index] = True
                if nearest[geom] < 0:
                    axis = data.xaxis[hinge]
                    center = data.geom_xpos[geom] - data.xanchor[hinge]
                    axis_norm = np.linalg.norm(axis)
                    radial = np.linalg.norm(np.cross(center, axis)) / axis_norm
                    value = radial + model.geom_rbound[geom] + BOUND_ROUNDING_M
                    if not np.isfinite(value) or value <= 0:
                        raise ValueError("Cannot bound collider radius about its nearest hinge")
                    nearest[geom] = index
                    radius[geom] = value
            body = int(model.body_parentid[body])

    relative_axes = geometry_axes[sweep.a] ^ geometry_axes[sweep.b]
    if not np.array_equal(sweep.weights > 0, relative_axes):
        raise ValueError("V1 lever support disagrees with the exact relative ancestor topology")
    weights = sweep.weights.copy()
    for own, other in ((sweep.a, sweep.b), (sweep.b, sweep.a)):
        index = nearest[own]
        eligible = (index >= 0) & ~geometry_axes[other, np.maximum(index, 0)]
        rows = np.flatnonzero(eligible)
        weights[rows, index[rows]] = np.minimum(weights[rows, index[rows]], radius[own[rows]])
    return weights


class SweptEnvelopeV2:
    """Continuous same-box proof, subdividing only axes that affect each pair."""

    def __init__(self, robot, *, max_queries=base.MAX_BOX_QUERIES, max_depth=base.MAX_BOX_DEPTH):
        if max_queries < 0 or max_depth < 0:
            raise ValueError("Invalid subdivision limits")
        self.native = NativeSweep(robot)
        self.max_queries = max_queries
        self.max_depth = max_depth
        self.weights = _relative_levers(self.native)
        self.anchor = self.distances = self.last_proven_box = None
        self.last_metrics = None

    def _variation(self, radius, subset=None):
        weights = self.weights if subset is None else self.weights[subset]
        return np.einsum("ij,j->i", weights, _chord(radius)) + BOUND_ROUNDING_M

    def _record_gap(self, observed):
        prior = self.last_metrics["min_sample_gap_m"]
        self.last_metrics["min_sample_gap_m"] = float(observed if prior is None else min(prior, observed))

    def _classify_gap(self, gap):
        if not np.isfinite(gap):
            self.last_metrics["termination"] = "INVALID_GEOMETRY"
            return base.Reason.INTERNAL
        if gap < self.native.clearance:
            self.last_metrics["sampled_clearance_violation"] = (
                "PENETRATION" if gap < 0.0 else "CLEARANCE_VIOLATION")
            self.last_metrics["termination"] = self.last_metrics["sampled_clearance_violation"]
            return base.Reason.COLLISION
        return base.Reason.PERMIT

    def _probe(self, center, pairs, required):
        sweep = self.native
        model, data = sweep.model, sweep.data
        data.qpos[sweep.robot.q_indices] = center
        mujoco.mj_kinematics(model, data)
        first, second = sweep.a[pairs], sweep.b[pairs]
        broad = np.linalg.norm(data.geom_xpos[first] - data.geom_xpos[second], axis=1) - sweep.radii[pairs]
        if not np.isfinite(broad).all():
            self.last_metrics["termination"] = "INVALID_GEOMETRY"
            return base.Reason.INTERNAL, None
        possible = np.flatnonzero((broad <= required) | sweep.has_plane[pairs])
        unresolved = []
        for index in possible:
            cap = float(required[index] + base.DISTANCE_ROUNDING_MARGIN_M)
            gap = mujoco.mj_geomDistance(model, data, int(first[index]), int(second[index]), cap, None)
            status = self._classify_gap(gap)
            if np.isfinite(gap):
                self._record_gap(gap)
            if status != base.Reason.PERMIT:
                self.last_metrics["witness_pair_index"] = int(pairs[index])
                return status, None
            if gap <= required[index]:
                unresolved.append(pairs[index])
        return base.Reason.PERMIT, np.asarray(unresolved, dtype=int)

    def _split(self, stack, low, high, pairs, depth):
        self.last_metrics["unresolved_pair_indices"] = pairs.tolist()
        if depth >= self.max_depth:
            self.last_metrics["termination"] = "MAX_SUBDIVISION_DEPTH"
            return False
        middle = low + (high - low) * 0.5
        half = np.maximum(middle - low, high - middle)
        # For each pair, choose the axis giving its greatest reduction in the
        # displacement bound at a bisection. Unrelated axes have zero weight.
        gain = self.weights[pairs] * (_chord(half) - _chord(half * 0.5))
        gain[:, (low >= middle) | (middle >= high)] = 0.0
        axes = np.argmax(gain, axis=1)
        if np.any(gain[np.arange(len(axes)), axes] <= 0.0):
            self.last_metrics["termination"] = "UNRESOLVED_WITHOUT_SPLITTABLE_RELATIVE_AXIS"
            return False
        for axis in np.unique(axes):
            active = pairs[axes == axis]
            lower_child_end, upper_child_start = high.copy(), low.copy()
            lower_child_end[axis] = upper_child_start[axis] = middle[axis]
            stack.append((upper_child_start, high, active, depth + 1))
            stack.append((low, lower_child_end, active, depth + 1))
        return True

    def _search(self, low, high, deadline_ns):
        sweep = self.native
        clearance = sweep.clearance + base.DISTANCE_ROUNDING_MARGIN_M
        if time.monotonic_ns() >= deadline_ns:
            return base.Reason.TIME_BUDGET
        if self.last_proven_box is not None:
            previous_low, previous_high = self.last_proven_box
            if np.all(previous_low <= low) and np.all(high <= previous_high):
                self.last_metrics.update(cache_hit=True, cache_kind="CERTIFIED_BOX_CONTAINMENT")
                return base.Reason.PERMIT
        if self.anchor is not None:
            radius = np.maximum(np.abs(low - self.anchor), np.abs(high - self.anchor))
            if np.all(self.distances - self._variation(radius) >= clearance):
                self.last_metrics.update(cache_hit=True, cache_kind="ROOT_DISTANCE_LOWER_BOUND")
                return base.Reason.PERMIT

        center = low + (high - low) * 0.5
        radius = np.maximum(center - low, high - center)
        distances = sweep.pair_distances(center)
        self.last_metrics["root_queries"] = 1
        if not np.isfinite(sweep.data.geom_xpos).all():
            self.last_metrics["termination"] = "INVALID_GEOMETRY"
            return base.Reason.INTERNAL
        index = int(np.argmin(distances))
        gap = distances[index]
        status = self._classify_gap(gap)
        if np.isfinite(gap):
            self._record_gap(gap)
        if status != base.Reason.PERMIT:
            self.last_metrics["witness_pair_index"] = index
            return status
        self.anchor, self.distances = center, distances
        pairs = np.flatnonzero(distances - self._variation(radius) < clearance)
        if not pairs.size:
            return base.Reason.PERMIT
        stack = []
        if not self._split(stack, low, high, pairs, 0):
            return base.Reason.SWEEP_UNPROVEN
        while stack:
            if time.monotonic_ns() >= deadline_ns:
                return base.Reason.TIME_BUDGET
            if self.last_metrics["queries"] >= self.max_queries:
                self.last_metrics["termination"] = "MAX_SUBDIVISION_QUERIES"
                self.last_metrics["unresolved_pair_indices"] = sorted(
                    {int(pair) for _, _, pending, _ in stack for pair in pending})
                return base.Reason.SWEEP_UNPROVEN
            lower, upper, pairs, depth = stack.pop()
            center = lower + (upper - lower) * 0.5
            radius = np.maximum(center - lower, upper - center)
            self.last_metrics["queries"] += 1
            self.last_metrics["max_depth"] = max(depth, self.last_metrics["max_depth"])
            status, unresolved = self._probe(center, pairs, clearance + self._variation(radius, pairs))
            if status != base.Reason.PERMIT:
                return status
            if unresolved.size and not self._split(stack, lower, upper, unresolved, depth):
                return base.Reason.SWEEP_UNPROVEN
        self.last_metrics["unresolved_pair_indices"] = []
        return base.Reason.PERMIT

    def prove(self, low, high, deadline_ns):
        self.last_metrics = {"cache_hit": False, "cache_kind": None, "root_queries": 0, "queries": 0, "max_depth": 0,
                             "termination": None, "min_sample_gap_m": None, "sampled_clearance_violation": None,
                             "witness_pair_index": None, "unresolved_pair_indices": []}
        if (low.shape != (base.JOINTS,) or high.shape != low.shape or not np.isfinite(low).all()
                or not np.isfinite(high).all() or np.any(low > high)):
            status = base.Reason.NONFINITE
        else:
            status = self._search(low, high, deadline_ns)
        if status == base.Reason.PERMIT:
            if time.monotonic_ns() >= deadline_ns:
                status = base.Reason.TIME_BUDGET
            else:
                self.last_proven_box = low.copy(), high.copy()
        self.last_metrics["termination"] = self.last_metrics["termination"] or status.name
        return status


class GuardV2(base.Guard):
    """V1 request validation, issue continuity, and dual-model ordering."""

    def __init__(self, robots, *, max_queries=base.MAX_BOX_QUERIES, max_depth=base.MAX_BOX_DEPTH, metrics=None):
        super().__init__(robots)
        self.envelopes = tuple(SweptEnvelopeV2(robot, max_queries=max_queries, max_depth=max_depth) for robot in robots)
        self.metrics_stream = metrics
        self.request_metrics = None

    def evaluate(self, packet, deadline_ns):
        self.request_metrics = None
        for envelope in self.envelopes:
            envelope.last_metrics = None
        sequence, reason, group, states = super().evaluate(packet, deadline_ns)
        self.request_metrics = {"sequence": sequence, "group": group, "reason": reason.name,
                                "models": {name: envelope.last_metrics for name, envelope in
                                           zip(("live", "signed"), self.envelopes, strict=True)}}
        return sequence, reason, group, states

    def publish_decision(self, reason, states):
        super().publish_decision(reason, states)
        metrics, self.request_metrics = self.request_metrics, None
        if self.metrics_stream is not None and metrics is not None:
            self.metrics_stream.write(json.dumps(metrics, separators=(",", ":"), allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("live-model", "signed-model", "model-receipt", "socket", "ready", "trace", "metrics", "v2-receipt"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--budget-ms", type=int, choices=(25,), default=25)
    args = parser.parse_args()
    if (os.environ.get("ROS_DOMAIN_ID") != "213" or os.environ.get("ROS_LOCALHOST_ONLY") != "1"
            or base.digest(args.model_receipt) != base.EXACT_MODEL_RECEIPT_SHA):
        parser.error("Expected the audited isolated simulation domain and model receipt")
    expected = json.loads(args.model_receipt.read_bytes())["model_identity"]
    robots = (base.verified_robot(args.live_model, base.LIVE_SHA, expected["live"]),
              base.verified_robot(args.signed_model, base.SIGNED_SHA, expected["signed"]))
    with args.metrics.open("x", encoding="utf-8", buffering=1) as metrics:
        guard = GuardV2(robots, metrics=metrics)
        with args.v2_receipt.open("x", encoding="utf-8") as receipt:
            json.dump({"scope": MODE, "implementation_sha256": base.digest(__file__),
                       "strict_guard_sha256": base.digest(base.__file__), "max_box_depth": base.MAX_BOX_DEPTH,
                       "max_box_queries": base.MAX_BOX_QUERIES, "budget_ms": args.budget_ms,
                       "state_envelope": "UNCHANGED_COMPONENTWISE_ISSUED_MEASURED_PROPOSED",
                       "formal_coverage_credit": False}, receipt, indent=2, allow_nan=False)
            receipt.write("\n")
        signal.signal(signal.SIGTERM, lambda _signal, _frame: signal.raise_signal(signal.SIGINT))
        try:
            base.serve(args, guard)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
