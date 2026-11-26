"""Pre-Actor actual MuJoCo D10 window witnesses with immutable plans and full denominators."""

import argparse
from hashlib import sha256
from io import TextIOBase
import json
from math import ceil, inf
from pathlib import Path

import numpy as np
import torch

from ...p02.model import RobotModel
from ..rmb.domains import residual_witness
from ..rmb.reader import RobotReferenceBank
from ..rmb.schema import GEOMETRY_SHAPES
from .physics import ServoProfile, signature
from .reference import RmbReference
from .smoke import make_task


HEURISTICS = ("NOMINAL_B0", "V1A_STATIC_WITNESS_8")


def file_digest(path):
    hasher = sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def nearest_rank(finite_errors, planned):
    ordered = sorted(finite_errors) + [inf] * (planned - len(finite_errors))
    value = ordered[ceil(planned * .95) - 1]
    return value if value < inf else None


def validate_plan(plan, reference, rmb_sha, budget):
    if (plan["schema"] != "s05.v1.dynamic-servo-witness-plan.v1" or plan["rmb_sha256"] != rmb_sha
            or plan["split"] != reference.split or plan["purpose"] != "PRE_ACTOR_NO_POLICY"):
        raise ValueError("Expected a frozen independent pre-Actor witness plan")
    seen = set()
    planned = 0
    for row in plan["windows"]:
        key, start, stop = row["partition"], row["start"], row["stop"]
        identity = (key, start, stop)
        values = reference.partitions[key].arrays
        if identity in seen or not 0 <= start < stop <= len(values["tick_ns"]):
            raise ValueError("Invalid or duplicate D10 planned window")
        seen.add(identity)
        hypotheses = tuple(row["hypotheses"])
        if not hypotheses or len(hypotheses) != len(set(hypotheses)) or not set(hypotheses) <= set(HEURISTICS):
            raise ValueError("Only the two registered independent hypotheses are legal")
        segment = values["audit_segment"][start:stop]
        generation = values["audit_generation"][start:stop]
        if stop - start < 25 or segment[0] < 0 or not np.all(segment == segment[0]) or not np.all(generation == generation[0]):
            raise ValueError("D10 requires a continuous source/segment/generation window of at least 25 ticks")
        planned += (stop - start) * len(hypotheses)
    if not 0 < planned <= budget:
        raise ValueError("Unreserved physical rollout exceeds the requested D10 evaluation allowance")
    return planned


def one_witness(task, reference, planned, hypothesis, trace: TextIOBase):
    key, start, stop = planned["partition"], planned["start"], planned["stop"]
    values = reference.partitions[key].arrays
    root = reference.partitions[key].root
    record = {"partition": key, "root": root, "start": start, "stop": stop, "planned_ticks": stop-start,
              "hypothesis": hypothesis, "executed_ticks": 0}
    errors = []
    last = reference.current([(key, start)])
    if not last.nominal_valid[0, 0]:
        return {**record, "failure": "invalid_initial_nominal", "p95_mm": None, "passes_threshold": False}
    q = last.nominal
    ids = torch.arange(1, device=q.device)
    try:
        task.reset(ids, q, torch.zeros_like(q))
    except ValueError:
        return {**record, "failure": "illegal_initial_state", "p95_mm": None, "passes_threshold": False}
    failure = None
    for i in range(start, stop):
        target = reference.current([(key, i)])
        action = torch.zeros_like(q)
        if hypothesis == "V1A_STATIC_WITNESS_8" and target.nominal_valid[0, 0] and target.permitted[0]:
            geometry = {name: values[f"mapped_{name}"][i] for name in GEOMETRY_SHAPES}
            witness = residual_witness(task.guard.robot, values["nominal_q_nominal"][i], geometry,
                                       values["mapped_pinch_active"][i], 8, task.guard.sweep)
            absolute = torch.as_tensor(witness.q, device=q.device, dtype=q.dtype)
            action = ((absolute[None] - target.nominal) / task.guard.scale).clamp(-1., 1.)
        task.observe(target)
        result = task.step(action)
        record["executed_ticks"] += int(result.substeps[0] > 0)
        err = None
        if result.valid[0]:
            actual = task.geometry(result.x1_pre_reset[:, :54])
            err = 1000 * torch.linalg.vector_norm(actual.tips[0] - target.geometry.tips[0], dim=-1).max().item()
            errors.append(err)
        trace.write(json.dumps({"partition": key, "root": root, "tick": i, "tick_ns": target.tick_ns[0].item(),
                                "hypothesis": hypothesis, "q_before": result.x[0,:54].tolist(),
                                "q_issued": result.c[0].tolist(), "q_actual": result.x1_pre_reset[0,:54].tolist(),
                                "applied_mean": result.u[0].tolist(), "actual_substeps": result.substeps[0].item(),
                                "hard_event": result.events[0].item(), "score_mm": err}) + "\n")
        if result.terminated[0] or not result.valid[0]:
            failure = "physical_failure" if result.terminated[0] else "invalid_or_fallback"
            break
    error = nearest_rank(errors, stop - start)
    # Dynamic compilation demands complete legal continuous Servo execution; a
    # finite nearest-rank result cannot hide a short omitted tail.
    ok = failure is None and len(errors) == stop - start and error is not None and error <= 10.
    return {**record, "failure": failure, "scored_ticks": len(errors), "p95_mm": error, "passes_threshold": ok}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("rmb", "rmb-review", "model", "plan", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    for key in ("rmb-sha", "rmb-review-sha", "plan-sha"):
        parser.add_argument(f"--{key}", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args()
    if file_digest(args.plan) != args.plan_sha:
        raise ValueError("Witness plan SHA mismatch")
    plan = json.loads(args.plan.read_bytes())
    bank = RobotReferenceBank(args.rmb, expected_sha=args.rmb_sha, review_receipt=args.rmb_review,
                              expected_review_sha=args.rmb_review_sha)
    reference = RmbReference(bank, device=args.device, train=plan["split"] == "Train")
    planned_count = validate_plan(plan, reference, args.rmb_sha, args.budget)
    profile = ServoProfile()
    robot = RobotModel(args.model)
    task, _, _ = make_task(robot, 1, profile, "B54", args.device)
    model_signature = signature(robot, profile)
    if model_signature["sha256"] != plan["model_sha256"]:
        raise ValueError("Physics asset/Servo identity differs from the frozen witness plan")
    args.out.mkdir(parents=True, exist_ok=False)
    trace_path = args.out / "dynamic_servo_witnesses.jsonl"
    windows = []
    with trace_path.open("x", encoding="utf-8") as trace:
        for planned in plan["windows"]:
            for hypothesis in planned["hypotheses"]:
                windows.append(one_witness(task, reference, planned, hypothesis, trace))
                trace.flush()
    supported = {}
    for row in windows:
        if row["passes_threshold"]:
            key = row["partition"]
            supported.setdefault(key, set()).update(range(row["start"], row["stop"]))
    summary = {
        "qualification": "WITNESS_EVIDENCE_ONLY",
        "rmb_sha256": args.rmb_sha, "rmb_review_sha256": args.rmb_review_sha, "plan_sha256": args.plan_sha,
        "model": model_signature, "trace_sha256": file_digest(trace_path), "split": reference.split,
        "planned_physical_ticks": planned_count, "actual_physical_ticks": sum(row["executed_ticks"] for row in windows),
        "unique_supported_ticks": sum(map(len, supported.values())),
        "supported_roots": sorted({row["root"] for row in windows if row["passes_threshold"]}),
        "windows": windows,
    }
    with (args.out / "low_error_domains.json").open("x", encoding="utf-8") as out:
        json.dump(summary, out, indent=2, allow_nan=False)
        out.write("\n")
    print(f"actual_physical_ticks={summary['actual_physical_ticks']} supported={summary['unique_supported_ticks']}")


if __name__ == "__main__":
    main()
