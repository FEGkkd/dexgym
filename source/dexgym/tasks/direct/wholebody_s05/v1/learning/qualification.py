"""Independent actual-physics, same-command 1/4/8-step WM diagnosis.

This runner produces quantitative evidence and complete raw comparison traces.
It has no pathway to mark G-C0 passed. The independently reviewed gradient,
edge, stationary, source coverage and watchdog checks remain separate.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from ......algorithms.supermimic.dex_reference.networks import WorldModel
from ...p02.model import RobotModel
from ..rmb.reader import RobotReferenceBank
from ..task import DT
from ..task.geometry import error6
from ..task.physics import ServoProfile, signature
from ..task.reference import RmbReference
from ..task.smoke import make_task
from .adapter import PhysicalCodec
from .catalog import FrameCatalog
from .train import DESIGN_FREEZE, digest, json_file, read_receipt

HORIZON = 8
REPORT_STEPS = (1, 4, 8)
LIMITS = {
    "q_max_deg": (.5, 1.5, 3.),
    "qd_max_rad_s": (.25, .60, 1.),
    "wrist_max_mm": (2., 5., 9.),
    "wrist_max_degrees": (1., 3., 5.),
    "max_ten_tip_mm": (2., 5., 8.),
    "local_max_mm": (2., 4., 6.),
    "active_pinch_max_mm": (2., 4., 6.),
    "cross_max_mm": (3., 7., 12.),
}


def max_norm(x):
    return x.norm(dim=-1).flatten(1).amax(-1)


def geometry_errors(kinematics, estimated, measured, active_pinch):
    wanted = kinematics(measured[:, :54])
    actual = kinematics(estimated[:, :54])
    rotation = error6(actual.wrist, actual.rotation, wanted.wrist, wanted.rotation)[..., 3:]
    pinch = torch.where(active_pinch, (actual.pinch - wanted.pinch).norm(dim=-1), 0.).flatten(1).amax(-1)
    return {
        "q_max_deg": (estimated[:, :54] - measured[:, :54]).abs().amax(-1).rad2deg(),
        "qd_max_rad_s": (estimated[:, 54:] - measured[:, 54:]).abs().amax(-1),
        "wrist_max_mm": max_norm(actual.wrist - wanted.wrist) * 1000,
        "wrist_max_degrees": max_norm(rotation).rad2deg(),
        "max_ten_tip_mm": max_norm(actual.tips - wanted.tips) * 1000,
        "local_max_mm": max_norm(actual.local - wanted.local) * 1000,
        "active_pinch_max_mm": pinch * 1000,
        "cross_max_mm": max_norm(actual.cross - wanted.cross) * 1000,
    }


@torch.no_grad()
def actual_pair(task, catalog, starts, actions):
    """The physics twin owns guard, command execution, and measured response."""
    ref = catalog.target(starts)
    ids = torch.arange(task.num_envs, device=starts.device)
    task.reset(ids, ref.nominal, torch.zeros_like(ref.nominal))
    x0 = task.physics.state().clone()
    s0 = task.physics.servo().clone()
    trace = {"state": [], "issued": [], "effective": [], "full": [], "eligible": [], "physical": [],
             "contact": [], "depth": [], "pinch": [], "nominal": [], "event": []}
    for tick, action in enumerate(actions):
        ref = catalog.target(starts + tick)
        task.observe(ref)
        tr = task.step(action)
        for key, value in zip(trace, (tr.x1_pre_reset, tr.c, tr.u, tr.full_step, tr.valid, tr.substeps > 0,
                                      tr.contact, tr.contact_depth_m, ref.pinch_active, ref.nominal, tr.events),
                              strict=True):
            trace[key].append(value.clone())
    return x0, s0, {key: torch.stack(values) for key, values in trace.items()}


@torch.no_grad()
def compare(task, wm, start, servo_start, trace):
    """Compare predictions using only executed commands and deterministic Servo recurrence."""
    estimate, s = start, servo_start
    results = {"wm": [], "persistent": [], "constant_velocity": [], "servo_mae": []}
    for i in range(HORIZON):
        executed = trace["issued"][i]
        applied = task.physics.profile.applied_interval(s, executed)
        effective = applied.mean(1)
        actual_effective = trace["effective"][i]
        results["servo_mae"].append((actual_effective - effective).abs().amax(-1))
        input_state = torch.cat((estimate, s, task.sim_context), -1)
        estimate, _ = wm(estimate, effective, input_state)
        s = applied[:, -1]
        velocity_state = torch.cat((start[:, :54] + (i + 1) * DT * start[:, 54:], start[:, 54:]), -1)
        results["wm"].append(estimate)
        results["persistent"].append(start)
        results["constant_velocity"].append(velocity_state)
    return {name: torch.stack(values) for name, values in results.items()}


def scalar_metrics(rows):
    values = rows.detach().cpu().numpy().reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return {"count_finite": int(values.size), "p50": float(np.quantile(values, .5)),
            "p95": float(np.quantile(values, .95)), "p99": float(np.quantile(values, .99)),
            "max": float(values.max())}


@torch.no_grad()
def summarize(kinematics, arrays):
    complete = (arrays["full"] & arrays["eligible"]).long().cumprod(0).bool()
    rows = {}
    for slot, step in enumerate(REPORT_STEPS):
        valid = complete[step - 1]
        scores = {}
        for method in ("wm", "persistent", "constant_velocity"):
            error = geometry_errors(kinematics, arrays[method][step - 1], arrays["state"][step - 1],
                                    arrays["pinch"][step - 1])
            scores[method] = {}
            for key, values in error.items():
                applies = valid
                if key == "active_pinch_max_mm":
                    applies = applies & arrays["pinch"][step - 1].flatten(1).any(-1)
                item = scalar_metrics(values[applies])
                scores[method][key] = item
                if method == "wm" and item is not None:
                    item["threshold"] = LIMITS[key][slot]
                    item["numeric_threshold_met"] = item["p95"] <= item["threshold"]
                    item["nonfinite_on_supported_cases"] = int(applies.sum()) - item["count_finite"]
        rows[str(step)] = {"planned_cases": len(valid), "completed_cases": int(valid.sum()),
                          "unscored_cases": int((~valid).sum()), "comparisons": scores,
                          "max_servo_mean_error_rad": scalar_metrics(arrays["servo_mae"][step - 1][valid])}
    return rows


def initialized_model(args, task):
    if digest(args.checkpoint) != args.checkpoint_sha or digest(args.fit_report) != args.fit_report_sha:
        raise ValueError("Checkpoint or manifest identity differs")
    report = json.loads(args.fit_report.read_bytes())
    if report["checkpoint_sha256"] != args.checkpoint_sha or report.get("design_freeze_sha256") != DESIGN_FREEZE:
        raise ValueError("Checkpoint does not belong to this unchanged fit report")
    if report["model_sha256"] != digest(args.model) or report["physical_profile"]["sha256"] != signature(
        task.guard.robot, task.physics.profile
    )["sha256"]:
        raise ValueError("Comparison physics differs from fit")
    if args.smoke != (report["scope"] == "PHYSICAL_SYNTHETIC_PREACTOR_SMOKE"):
        raise ValueError("Synthetic checkpoint cannot enter production qualification")
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = payload["config"]
    wm = WorldModel(108 + 54 + task.sim_context.shape[1], 13, PhysicalCodec(), 54,
                    [.001] * 54 + [.01] * 54, hidden=tuple(config["wm_hidden"]), amp=False,
                    control_error_features=config.get("control_error_features", False),
                    predict_contact_depth=config.get("contact_depth_weight", 0) > 0).to(args.device)
    wm.load_state_dict(payload["modules"]["wm"])
    return wm.eval(), report


def planned_catalog(args, task, report):
    if args.smoke:
        catalog = FrameCatalog.smoke(task)
        cases = [(int(i), 4900 + int(i), "synthetic") for i in range(8, 16)]
        return catalog, cases
    if digest(args.root_plan) != report["root_plan_sha256"] or digest(args.cases_plan) != args.cases_plan_sha:
        raise ValueError("Roles or preregistered independent case plan differs")
    roots = set(json.loads(args.root_plan.read_bytes())["qualification_roots"])
    rows = json.loads(args.cases_plan.read_bytes())["cases"]
    if args.g_a_sha != report["g_a_receipt_sha256"]:
        raise ValueError("Fit and qualification G-A identities differ")
    receipt = read_receipt(args.g_a, args.g_a_sha, "V1-A")
    if receipt.get("manifest_sha256") != report["rmb_manifest_sha256"]:
        raise ValueError("Qualification bank is outside the approved A scope")
    bank = RobotReferenceBank(args.bank, expected_sha=report["rmb_manifest_sha256"],
                              review_receipt=args.g_a, expected_review_sha=args.g_a_sha)
    provider = RmbReference(bank, device=args.device)
    provider.partitions = {key: row for key, row in provider.partitions.items() if row.root in roots}
    offsets = {}
    total = 0
    for key, row in provider.partitions.items():
        offsets[key] = (total, row)
        total += len(row.arrays["tick_ns"])
    catalog = FrameCatalog.from_rmb(provider, fit_roots=roots, holdout_roots=())
    cases, repeats = [], {}
    support = Counter()
    for row in rows:
        bias, part = offsets[row["partition"]]
        tick, seed, group = row["tick"], row["seed"], row["scenario"]
        if (part.root != row["root"] or group not in {"stationary", "dynamic", "boundary"}
                or type(tick) is not int or not 0 <= tick < len(part.arrays["tick_ns"])
                or int(catalog.remaining[bias + tick]) < HORIZON):
            raise ValueError("Case violates source, timing or contiguous legal horizon")
        identity = (row["partition"], tick)
        key = (bias + tick, seed, group)
        repeated = repeats.setdefault(identity, set())
        if seed in repeated or len(repeated) == 4:
            raise ValueError("Expected exactly four distinct action trials per independent anchor")
        repeated.add(seed)
        cases.append(key)
        support[group] += 1
    if len(repeats) < 1024 or len(cases) < 4096 or min(support[k] for k in ("stationary", "dynamic", "boundary")) < 256:
        raise ValueError("Qualification case plan has insufficient prespecified support")
    by_partition = {}
    for (key, i), seeds in repeats.items():
        if len(seeds) != 4:
            raise ValueError("Every anchor needs exactly four experimental seeds")
        by_partition.setdefault(key, []).append(i)
    if any(np.any(np.diff(sorted(i)) < HORIZON) for i in by_partition.values()):
        raise ValueError("Independent anchors overlap within a partition")
    return catalog, cases


def make_actions(cases, device):
    result = np.empty((HORIZON, len(cases), 54), dtype=np.float32)
    for i, (_, seed, group) in enumerate(cases):
        rng = np.random.default_rng(seed)
        for tick in range(HORIZON):
            action = rng.choice([-1, 1], size=54) * rng.uniform(.1, .35, size=54)
            if group == "stationary":
                action = rng.choice([-1, 1], size=54) * .04
            result[tick, i] = action
    return torch.from_numpy(result).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha", required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--fit-report-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--reserved-evaluation-transitions", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--g-a", type=Path)
    parser.add_argument("--g-a-sha")
    parser.add_argument("--root-plan", type=Path)
    parser.add_argument("--cases-plan", type=Path)
    parser.add_argument("--cases-plan-sha")
    args = parser.parse_args()
    if args.batch < 1 or (not args.smoke and not all((
            args.bank, args.root_plan, args.cases_plan, args.cases_plan_sha, args.g_a, args.g_a_sha,
    ))):
        parser.error("missing real bank and sealed plan, or invalid physical batch")
    robot = RobotModel(args.model)
    task, _, _ = make_task(robot, args.batch, ServoProfile(), "B54", args.device)
    wm, fit = initialized_model(args, task)
    catalog, cases = planned_catalog(args, task, fit)
    if len(cases) % args.batch or args.reserved_evaluation_transitions < len(cases) * HORIZON:
        raise ValueError("Batch divisibility or independent evaluation reservation insufficient")
    args.output.mkdir(parents=True, exist_ok=False)
    grouped = {}
    for i in range(0, len(cases), args.batch):
        chunk = cases[i:i + args.batch]
        starts = torch.tensor([c[0] for c in chunk], device=args.device, dtype=torch.long)
        x0, s0, trace = actual_pair(task, catalog, starts, make_actions(chunk, args.device))
        scores = compare(task, wm, x0, s0, trace)
        for key, value in {**trace, **scores}.items():
            grouped.setdefault(key, []).append(value.cpu())
    all_arrays = {key: torch.cat(value, 1) for key, value in grouped.items()}
    traces = args.output / "same_command_traces.npz"
    np.savez_compressed(traces, **{name: value.numpy() for name, value in all_arrays.items()})
    summary = summarize(task.geometry.cpu(), all_arrays)
    report = {
        "scope": "SYNTHETIC_ACTUAL_PHYSICS_DIAGNOSTIC" if args.smoke else "INDEPENDENT_CASE_PREDICTION_EVIDENCE",
        "wm_gate": "NOT_QUALIFIED_BY_THIS_RUNNER", "actor_updates": 0, "critic_updates": 0,
        "real_evaluation_transitions": int(all_arrays["physical"].sum()), "planned_cases": len(cases),
        "checkpoint_sha256": args.checkpoint_sha, "comparison_trace_sha256": digest(traces),
        "case_plan_sha256": args.cases_plan_sha, "horizon_metrics": summary,
        "remaining_separate_gates": ["static_drift", "boundary_controls", "paired_action_gradients",
                                     "imagination_rollout_gap", "independent_review"],
    }
    json_file(args.output / "prediction_report.json", report)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
