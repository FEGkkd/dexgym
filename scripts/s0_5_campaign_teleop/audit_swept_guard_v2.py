#!/usr/bin/env python3
"""Read-only comparison of strict swept guards on the eleven root-five archives."""

import argparse
from collections import Counter
from dataclasses import dataclass, fields
import json
from pathlib import Path
import time

import mujoco
import numpy as np

import swept_live_guard as v1
from diagnose_swept_hand_rejection import (
    ProbeObserver, _cache_copy, _check_sources, _unresolved_details, diagnose_model,
)


ROOT = "capture:b63bc72d2e8b2ee395e966f6"
SOURCES = ("centers", "segmented", "j6-margin", "yaw", "wrist-first", "history", "one-unproven")
HIGH = dict(budget_ms=2000, max_queries=8191, max_depth=18)
MODELS = ("live", "signed")
STRICT_REJECTIONS = {"COLLISION", "LIMIT_OR_COUPLING", "GROUP_MISMATCH", "START_UNBOUND", "NONFINITE"}
PLAN_SHA = "3c0268291bdb8916a9a520d3fa73a1340fffd7f9e8a7b7364a1ef9950f9a0004"


def read(path):
    return json.loads(path.read_bytes())


def quantiles(values):
    if not values:
        return None
    return dict(zip(("p50", "p95", "p99", "max"), map(float, np.percentile(values, [50, 95, 99, 100]))))


def guard_trace_digests(directory, source):
    if source == "history":
        return {read(directory / "terminal-sweep-witness.json")["trace_sha256"]}
    if source == "one-unproven":
        return {read(directory / "strict-counterfactual-proof.json")["trace_sha256"]}
    def find_digests(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "guard_trace_sha256":
                    yield value
                else:
                    yield from find_digests(value)
        elif isinstance(node, list):
            for value in node:
                yield from find_digests(value)
    return set(find_digests(read(directory / "comparison.json")))


def check_exception_counterfactual(raw_trace, guard_digests):
    # The committed digest binds the earlier strict counterfactual copy. The
    # original trace retains actual PERMIT at the single recorded exception.
    strict_trace = raw_trace.parents[2] / "strict-exception-counterfactual.jsonl"
    if v1.digest(strict_trace) not in guard_digests:
        raise ValueError(f"Strict counterfactual differs from committed evidence: {strict_trace}")
    with raw_trace.open(encoding="utf-8") as raw, strict_trace.open(encoding="utf-8") as strict:
        for raw_line, strict_line in zip(raw, strict, strict=True):
            actual, counterpart = json.loads(raw_line), json.loads(strict_line)
            if actual["sequence"] == 7020:
                if actual["reason"] != "PERMIT":
                    raise ValueError("The documented exception must remain an actual PERMIT in the raw trace")
                actual["reason"] = "SWEEP_UNPROVEN"
            if actual != counterpart:
                raise ValueError(f"Raw trace differs from the qualified comparison at {actual['sequence']}")


def run_receipts(docs, runs_dir):
    """Bind original run identity, immutable ready receipts and real observer traces."""
    for source in SOURCES:
        receipt_file = docs / "campaign/evidence" / f"dcr009-root-five-{source}" / "run_receipt.json"
        receipt = read(receipt_file)
        if receipt["scope"] != "DIAGNOSTIC_ONLY" or receipt["plan_sha256"] != PLAN_SHA:
            raise ValueError(f"Unexpected scope or root-five plan in {receipt_file}")
        guard_digests = guard_trace_digests(receipt_file.parent, source)
        for run in receipt["attempts"]:
            if run["root"] != ROOT:
                raise ValueError("Attempt is outside the authorized root-five corpus")
            command = run["guard_command"]
            trace = Path(command[command.index("--trace") + 1]).resolve(strict=True)
            trace.relative_to(runs_dir)
            ready = Path(command[command.index("--ready") + 1]).resolve(strict=True)
            if source == "one-unproven":
                check_exception_counterfactual(trace, guard_digests)
            elif v1.digest(trace) not in guard_digests:
                raise ValueError(f"Guard request trace digest absent from committed evidence: {trace}")
            if ready.parent != trace.parent or v1.digest(ready) != run["guard_ready_sha256"]:
                raise ValueError(f"Guard readiness differs from committed attempt: {trace.parent}")
            observer = trace.parent / "evidence/trace.jsonl"
            if v1.digest(observer) != run["trace_sha256"]:
                raise ValueError(f"Observer trace differs from committed attempt: {trace.parent}")
            if source == "one-unproven":
                exception_trace = trace.parent / "guard-exception.jsonl"
                if v1.digest(exception_trace) != run["guard_exception_trace_sha256"]:
                    raise ValueError(f"Recorded exception receipt differs from committed attempt: {exception_trace}")
            yield source, receipt_file, run, trace, ready


class TimedV1(v1.SweptEnvelope):
    """Measure the original implementation without changing its proof decisions."""

    def __init__(self, robot):
        super().__init__(robot)
        self.pair_distances = self.native.pair_distances
        self.probe = self.native._probe
        self.native.pair_distances = self.observe_root
        self.native._probe = self.observe_probe
        self.last_metrics = {}

    def observe_root(self, q):
        gaps = self.pair_distances(q)
        self.last_metrics["root_queries"] += 1
        self.last_metrics["min_sample_gap_m"] = float(gaps.min())
        if self.last_metrics["min_sample_gap_m"] < self.native.clearance:
            self.unsafe_q = q.copy()
        return gaps

    def observe_probe(self, q, subset, required):
        if not self.nodes:
            low, high = self.box
            self.nodes.append(((high - low) * .5, 0))
        half, depth = self.nodes.pop()
        self.last_metrics["queries"] += 1
        self.last_metrics["max_depth"] = max(depth, self.last_metrics["max_depth"])
        result = self.probe(q, subset, required)
        if result is None:
            self.unsafe_q = q.copy()
        elif result.size and depth < v1.MAX_BOX_DEPTH:
            axis = int(np.argmax(self.sensitivity * half))
            if half[axis] > 1e-12:
                child_half = half.copy()
                child_half[axis] *= .5
                self.nodes.extend(((child_half, depth + 1), (child_half, depth + 1)))
                self.last_metrics["splits"] += 1
        return result

    def prove(self, low, high, deadline_ns):
        self.unsafe_q = None
        self.box, self.nodes = (low, high), []
        self.last_metrics = {"queries": 0, "root_queries": 0, "splits": 0, "max_depth": 0}
        result = super().prove(low, high, deadline_ns)
        self.last_metrics["cache_hit"] = result == v1.Reason.PERMIT and self.last_metrics["root_queries"] == 0
        return result


def build_v1(robots):
    guard = v1.Guard(robots)
    guard.envelopes = tuple(TimedV1(robot) for robot in robots)
    return guard


def build_v2(version2, robots):
    def observe_splits(envelope):
        original = envelope._split
        def split(*args):
            ok = original(*args)
            if ok:
                envelope.last_metrics["splits"] = envelope.last_metrics.get("splits", 0) + 1
            return ok
        return split
    guard = version2.GuardV2(robots)
    for envelope in guard.envelopes:
        envelope._split = observe_splits(envelope)
    return guard


@dataclass
class Stats:
    counts: Counter
    durations_ms: list
    fine_queries: list
    root_queries: list
    splits: list
    max_depths: list
    minima_m: list
    by_model_invocations: Counter
    by_model_cache_hits: Counter

    @classmethod
    def empty(cls):
        return cls(Counter(), [], [], [], [], [], [], Counter(), Counter())

    def record(self, reason, elapsed, model_metrics):
        self.counts[reason] += 1
        self.durations_ms.append(elapsed / 1e6)
        self.fine_queries.append(sum(item.get("queries", 0) for item in model_metrics.values()))
        self.root_queries.append(sum(item.get("root_queries", 0) for item in model_metrics.values()))
        self.splits.append(sum(item.get("splits", 0) for item in model_metrics.values()))
        self.max_depths.append(max((item.get("max_depth", 0) for item in model_metrics.values()), default=0))
        for key, item in model_metrics.items():
            self.by_model_invocations[key] += 1
            self.by_model_cache_hits[key] += int(bool(item.get("cache_hit")))
            minimum = item.get("min_sample_gap_m")
            if minimum is not None and np.isfinite(minimum):
                self.minima_m.append(minimum)

    def summary(self):
        attempts = sum(self.counts.values())
        return {
            "denominator": attempts, "counts": dict(self.counts),
            "permit_fraction": self.counts["PERMIT"] / attempts if attempts else None,
            "proof_latency_ms": quantiles(self.durations_ms), "fine_queries": quantiles(self.fine_queries),
            "fine_queries_total": sum(self.fine_queries), "root_queries_total": sum(self.root_queries),
            "subdivisions_total": sum(self.splits), "subdivisions_per_request": quantiles(self.splits),
            "max_depth_per_request": quantiles(self.max_depths),
            "root_or_reported_sample_min_mm": min(self.minima_m) * 1e3 if self.minima_m else None,
            "model_proof_invocations": dict(self.by_model_invocations),
            "model_cache_hits": dict(self.by_model_cache_hits),
            "cache_hit_fraction": {key: self.by_model_cache_hits[key] / count
                                   for key, count in self.by_model_invocations.items()},
        }


def evaluate(guard, packet, row, states):
    # Identify which envelopes the actual serial guard entered. Clearing the
    # counters neither resets a geometry cache nor affects a proof verdict.
    for envelope in guard.envelopes:
        envelope.last_metrics = {}
    start = time.monotonic_ns()
    _, reason, _, _ = guard.evaluate(packet, start + 25_000_000)
    elapsed = time.monotonic_ns() - start
    model_metrics = {key: envelope.last_metrics for key, envelope in zip(MODELS, guard.envelopes, strict=True)
                     if envelope.last_metrics}
    # Both guards see the preceding native admission from the recording,
    # including the one documented diagnostic exception. No extra simulated
    # q is propagated from a counterfactual offline admission.
    guard.publish_decision(v1.Reason[row["reason"]], states)
    return reason.name, elapsed, model_metrics


def exact_gap(sweep, q):
    """Uncensored below 100 mm, preserving non-finite narrow-phase outputs."""
    sweep.data.qpos[sweep.robot.q_indices] = q
    mujoco.mj_kinematics(sweep.model, sweep.data)
    a, b = sweep.a, sweep.b
    broad = np.linalg.norm(sweep.data.geom_xpos[a] - sweep.data.geom_xpos[b], axis=-1) - sweep.radii
    result = .1
    for i in np.flatnonzero((broad <= .1) | sweep.has_plane):
        distance = mujoco.mj_geomDistance(sweep.model, sweep.data, int(a[i]), int(b[i]), .1, None)
        if not np.isfinite(distance):
            return None
        result = min(result, distance)
    return result


def high_v1(robot, states, cache):
    low, high = states.min(axis=0), states.max(axis=0)
    envelope = v1.SweptEnvelope(robot, max_queries=HIGH["max_queries"])
    envelope.anchor, envelope.distances = cache
    observer = ProbeObserver(envelope, low, high)
    first_unsafe = []

    def observe(center, subset, required):
        remaining = observer.observe(center, subset, required)
        if remaining is None:
            first_unsafe.append(center.copy())
        return remaining

    envelope.native._probe = observe
    start = time.monotonic_ns()
    original_depth = v1.MAX_BOX_DEPTH
    try:
        v1.MAX_BOX_DEPTH = HIGH["max_depth"]  # This process runs offline only.
        reason = envelope.prove(low, high, start + HIGH["budget_ms"] * 1_000_000)
    finally:
        v1.MAX_BOX_DEPTH = original_depth
    elapsed_ms = (time.monotonic_ns() - start) / 1e6
    termination = reason.name
    if reason == v1.Reason.SWEEP_UNPROVEN:
        if observer.calls >= envelope.max_queries:
            termination = "QUERY_LIMIT"
        elif observer.last_node and observer.last_node["depth"] >= HIGH["max_depth"]:
            termination = "DEPTH_LIMIT"
        else:
            termination = "ZERO_WIDTH_OR_UNRESOLVED"
    details = _unresolved_details(envelope, observer.last_unresolved, robot) if observer.last_unresolved else None
    if details:
        details["pairs"] = details["pairs"][:3]
    actual_violation_gap = None
    if reason == v1.Reason.COLLISION:
        actual_violation_gap = exact_gap(envelope.native, first_unsafe[0] if first_unsafe else (low + high) * .5)
        if actual_violation_gap is None or actual_violation_gap >= envelope.native.clearance:
            termination = "INVALID_OR_UNREPRODUCIBLE_NARROW_PHASE"
        else:
            termination = "GEOMETRIC_PENETRATION" if actual_violation_gap < 0. else "CLEARANCE_DEFICIT"
    return {"reason": reason.name, "termination": termination, "queries": observer.calls, "elapsed_ms": elapsed_ms,
            "first_unsafe_gap_mm": None if actual_violation_gap is None else actual_violation_gap * 1e3,
            "last_unresolved_probe": details}


def dense_diagnostic(robot, states):
    """Supplementary point measurements, never referenced by an admission decision."""
    sweep = v1.NativeSweep(robot)
    endpoints = [exact_gap(sweep, q) for q in states]
    samples = list(endpoints)
    count = 33
    for ia, ib in ((0, 1), (1, 2), (0, 2)):
        for alpha in np.linspace(0, 1, count)[1:-1]:
            samples.append(exact_gap(sweep, (1 - alpha) * states[ia] + alpha * states[ib]))
    finite = [gap for gap in samples if gap is not None]
    return {"label": "SAMPLED_DIAGNOSTIC_ONLY_NO_CONTINUOUS_CREDIT", "sample_count": 3 + 3 * (count - 2),
            "paths": ["issued_to_measured", "measured_to_proposed", "issued_to_proposed"],
            "minimum_whole_catalogue_gap_mm": min(finite) * 1e3 if finite else None,
            "endpoint_gap_mm": [gap * 1e3 if gap is not None else None for gap in endpoints],
            "invalid_geometry_samples": len(samples) - len(finite)}


def summarize_attempt(source, receipt_file, run, trace, ready, robots, version2, out_dir):
    guards = {"v1": build_v1(robots)}
    if version2:
        guards["v2"] = build_v2(version2, robots)
    stats = {k: Stats.empty() for k in guards}
    target_ids = {"history": {6985}, "one-unproven": {7020, 7023}}.get(source, set())
    earliest = set()
    targeted, transitions, original, mismatches, excluded, bad = [], Counter(), Counter(), [], Counter(), []
    violations = Counter()
    exceptions = {}
    if source == "one-unproven":
        for line in trace.with_name("guard-exception.jsonl").read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            exceptions[event["sequence"]] = event
        if set(exceptions) != {7020} or len(exceptions) != run["guard_exception_count"]:
            raise ValueError("The diagnostic exception archive differs from the historical singleton")
    with trace.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            original[row["reason"]] += 1
            if row["issued_measured_proposed_rad"] is None:
                excluded[row["reason"]] += 1
                continue
            states = np.asarray(row["issued_measured_proposed_rad"], dtype=np.float64)
            if (states.shape != (3, v1.JOINTS) or not np.isfinite(states).all()
                    or row["sequence"] <= 0 or row["group"] not in (0, 1, 2)):
                raise ValueError(f"Invalid snapshot with sequence {row['sequence']}: {trace}")
            seq, group = row["sequence"], row["group"]
            if seq in exceptions:
                exception = exceptions[seq]
                if (exception["group"] != group or exception["root"] != ROOT
                        or not np.array_equal(exception["issued_measured_proposed_rad"], states)):
                    raise ValueError("Exception snapshots differ from the recorded guard inputs")
            selected = seq in target_ids or (row["reason"] == "SWEEP_UNPROVEN" and group not in earliest)
            if selected:
                caches = tuple(_cache_copy(env) for env in guards["v1"].envelopes)
                earliest.add(group)
            packet = v1.REQUEST.pack(v1.MAGIC, seq, group, *states.reshape(-1))
            decisions = {}
            for key, guard in guards.items():
                reason, elapsed, metrics = evaluate(guard, packet, row, states)
                stats[key].record(reason, elapsed, metrics)
                decisions[key] = reason
                if reason == "COLLISION":
                    for model, env in zip(MODELS, guard.envelopes, strict=True):
                        detail = metrics.get(model, {})
                        if key == "v1" and detail and env.unsafe_q is not None:
                            gap = exact_gap(env.native, env.unsafe_q)
                            label = ("INVALID_OR_UNREPRODUCIBLE_NARROW_PHASE" if gap is None or gap >= env.native.clearance
                                     else "GEOMETRIC_PENETRATION" if gap < 0 else "CLEARANCE_DEFICIT")
                            violations[f"{key}/{label}"] += 1
                        if key == "v2" and (label := detail.get("sampled_clearance_violation")):
                            violations[f"{key}/{label}"] += 1
                        if key == "v2" and detail.get("termination") == "INVALID_GEOMETRY":
                            violations[f"{key}/INVALID_GEOMETRY"] += 1
            if decisions["v1"] != row["reason"]:
                mismatches.append({"sequence": seq, "recorded": row["reason"], "reconstructed": decisions["v1"],
                                   "documented_exception": seq in exceptions})
            if version2:
                transitions[f"{decisions['v1']}->{decisions['v2']}"] += 1
                if decisions["v2"] == "PERMIT" and (decisions["v1"] in STRICT_REJECTIONS or row["reason"] in STRICT_REJECTIONS):
                    bad.append({"sequence": seq, "recorded": row["reason"], "paired": decisions})
            if selected:
                targeted.append((row, states.copy(), caches, decisions.copy()))
    targets = []
    for row, states, caches, decisions in targeted:
        seq = row["sequence"]
        issue = {"sequence": seq, "group": row["group"], "recorded": row["reason"], "paired_25ms": decisions,
                 "archived_proof_latency_ms": (row["proof_complete_ns"] - row["receipt_ns"]) / 1e6,
                 "recorded_exception_model_checks": exceptions[seq]["model_checks"] if seq in exceptions else None,
                 "models": {}}
        for model, robot, cache in zip(MODELS, robots, caches, strict=True):
            realtime = diagnose_model(robot, states, cache)
            if realtime["last_unresolved_probe"]:
                realtime["last_unresolved_probe"]["pairs"] = realtime["last_unresolved_probe"]["pairs"][:3]
            result = high_v1(robot, states, cache)
            results = {"v1_warm": result}
            if version2:
                envelope = version2.SweptEnvelopeV2(robot, max_queries=HIGH["max_queries"], max_depth=HIGH["max_depth"])
                start = time.monotonic_ns()
                response = envelope.prove(states.min(axis=0), states.max(axis=0), start + HIGH["budget_ms"] * 1_000_000)
                results["v2_cold"] = {"reason": response.name, "elapsed_ms": (time.monotonic_ns() - start) / 1e6,
                                      "metrics": envelope.last_metrics}
            diagnostic = dense_diagnostic(robot, states)
            issue["models"][model] = {"v1_same_geometry_budget": realtime, "high_budget": results,
                                      "dense_check": diagnostic}
        targets.append(issue)
    if target_ids - {item["sequence"] for item in targets}:
        raise ValueError(f"Mandatory target request missing: {trace}")
    public = {
        "source": source, "attempt": run["attempt"], "root": ROOT, "guard_trace_path": str(trace),
        "guard_trace_sha256": v1.digest(trace), "ready_sha256": v1.digest(ready),
        "committed_run_receipt_sha256": v1.digest(receipt_file), "raw_counts": dict(original),
        "records_without_original_q_snapshot": dict(excluded),
        "paired_25ms": {version: s.summary() for version, s in stats.items()},
        "paired_transitions": dict(transitions), "reconstructed_v1_differences": mismatches,
        "prohibited_v2_permits": bad, "classified_collision_returns": dict(violations), "high_budget_targets": targets,
    }
    out_dir.joinpath(f"{source}-{run['attempt']:02d}.json").write_text(
        json.dumps(public, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"{source}/{run['attempt']}: {public['paired_transitions'] or public['paired_25ms']['v1']['counts']}", flush=True)
    return public, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--docs", type=Path, default=Path(__file__).resolve().parents[2] / "docs/s0-5")
    parser.add_argument("--v1-only", action="store_true")
    parser.add_argument("--source", choices=SOURCES, action="append")
    args = parser.parse_args()
    if args.v1_only:
        version2 = None
    else:
        import swept_live_guard_v2 as version2
    args.outdir.mkdir(exist_ok=False, parents=True)
    sources = list(run_receipts(args.docs, args.runs_dir.resolve(strict=True)))
    _, _, run, _, ready = sources[0]
    command = run["guard_command"]
    paths = {key: Path(command[command.index(f"--{key}") + 1]) for key in ("live-model", "signed-model", "model-receipt")}
    _, models = _check_sources(ready, paths["model-receipt"])
    robots = (v1.verified_robot(paths["live-model"], v1.LIVE_SHA, models["live"]),
              v1.verified_robot(paths["signed-model"], v1.SIGNED_SHA, models["signed"]))
    aggregate, overall, transitions, rejects = [], {}, Counter(), Counter()
    for source, receipt, run, trace, ready in sources:
        if args.source and source not in args.source:
            continue
        ready_data, _ = _check_sources(ready, paths["model-receipt"])
        if ready_data["joint_names"] != list(robots[0].names) or ready_data["actuator_names"] != list(
            actuator.name for actuator in robots[0].actuators
        ):
            raise ValueError("Guard ready layout does not match the signed canonical robot")
        report, stats = summarize_attempt(source, receipt, run, trace, ready, robots, version2, args.outdir)
        aggregate.append({key: report[key] for key in report if key != "high_budget_targets"})
        transitions.update(report["paired_transitions"])
        rejects.update(report["classified_collision_returns"])
        for key, item in stats.items():
            acc = overall.setdefault(key, Stats.empty())
            for field in fields(Stats):
                original = getattr(acc, field.name)
                incoming = getattr(item, field.name)
                if isinstance(original, Counter):
                    original.update(incoming)
                else:
                    original.extend(incoming)
    results = {"scope": "DIAGNOSTIC_ONLY_ROOT_FIVE_READ_ONLY_ARCHIVE", "root": ROOT,
               "real_time_profile": {"budget_ms": 25, "max_queries": 63, "max_depth": 7},
               "offline_high_budget": HIGH, "formal_coverage_credit": False,
               "method": "Paired strict proof on original 54D snapshots; both issued baselines advance only on recorded native admissions.",
               "runtime": {"mujoco": mujoco.__version__, "numpy": np.__version__},
               "models": dict(zip(MODELS, (v1.LIVE_SHA, v1.SIGNED_SHA))),
               "harness_sha256": v1.digest(__file__),
               "v1_sha256": v1.digest(v1.__file__), "v2_sha256": v1.digest(version2.__file__) if version2 else None,
               "aggregate_25ms": {key: item.summary() for key, item in overall.items()},
               "aggregate_transitions": dict(transitions), "classified_collision_returns": dict(rejects),
               "unexpected_unsafe_permit_count": sum(len(item["prohibited_v2_permits"]) for item in aggregate),
               "attempts": aggregate}
    args.outdir.joinpath("paired-summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n",
                                                           encoding="utf-8")
    return int(results["unexpected_unsafe_permit_count"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
