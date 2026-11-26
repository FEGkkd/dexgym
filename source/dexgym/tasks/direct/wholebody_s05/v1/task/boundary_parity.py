"""Run small synthetic safety-boundary fixtures on Native MuJoCo and optional Warp."""

import argparse
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import torch

from ...p02.model import RobotModel
from ..rmb.schema import file_sha, write_json
from . import JOINTS, SUBSTEPS
from .geometry import ResidualKinematics
from .parity import FREE_RESPONSE_LIMITS
from .physics import ServoProfile, signature
from .safety import Stop
from .smoke import make_task, target_frame


JOINT = "l_index_finger_mcp_abd"
SAFE_OFFSET_RAD = .20
CONTACT_OFFSET_RAD = .275
MOVING_OFFSET_RAD = .24
OUTWARD_VELOCITY_RAD_S = 3.
MIN_CONTACT_DEPTH_M = .0001
CONTACT_DEPTH_DIFFERENCE_M = .0005
HOLD_TOL_RAD = 2e-6


@dataclass(frozen=True)
class Case:
    name: str
    event: Stop
    fallback: bool = False
    hard_stop: bool = False


CASES = (
    Case("withdraw_input_permission", Stop.INVALID_INPUT, fallback=True),
    Case("nonfinite_action", Stop.INVALID_ACTION, fallback=True),
    Case("illegal_nominal", Stop.ILLEGAL_NOMINAL, fallback=True),
    Case("legal_action_rate_projection", Stop(0)),
    Case("unsafe_path_safe_hold", Stop.PATH_REJECTED, fallback=True),
    Case("contact_feedback_reject_hold", Stop.PATH_REJECTED | Stop.HOLD_INFEASIBLE, fallback=True, hard_stop=True),
    Case("moving_legal_start_physical_stop", Stop.PHYSICAL_CONSTRAINT, hard_stop=True),
    Case("injected_self_contact_backend_stop", Stop.PHYSICAL_CONSTRAINT, hard_stop=True),
)


def offset(home, axis, value):
    result = home.clone()
    result[:, axis] += value
    return result


def collect_interval(substeps, report):
    report["physical_control_intervals"] += int((substeps > 0).sum())
    report["physical_substeps_completed"] += int(substeps.sum())
    report["kernel_world_substeps_upper"] += len(substeps) * int(substeps.max())


def json_finite(value):
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_finite(item) for item in value]
    return None if isinstance(value, float) and not isfinite(value) else value


def task_stopped(task, home, reset, transition, budget):
    task.observe(target_frame(task, home, 1))
    stopped = task.step(torch.zeros_like(home))
    collect_interval(stopped.substeps, budget)
    latched = (
        stopped.terminated.all() and not stopped.substeps.any() and not stopped.applied.any()
        and torch.equal(stopped.x1_pre_reset, transition.x1_pre_reset)
    )
    reset()
    task.observe(target_frame(task, home, 0))
    restart = task.step(torch.zeros_like(home))
    collect_interval(restart.substeps, budget)
    recovered = restart.valid.all() and not restart.events.any() and (restart.substeps == SUBSTEPS).all()
    return {"stop_latched": bool(latched), "explicit_reset_recovers": bool(recovered),
            "post_stop_substeps": stopped.substeps.tolist(), "after_reset_substeps": restart.substeps.tolist()}


def task_trial(case, task, home, reset, axis, budget):
    action = torch.zeros_like(home)
    ids = torch.arange(len(home), device=home.device)
    target = target_frame(task, home, 0)
    hold = home
    if case.name == "withdraw_input_permission":
        target = target._replace(permitted=torch.zeros_like(target.permitted))
    elif case.name == "nonfinite_action":
        action[:, 0] = torch.nan
    elif case.name == "illegal_nominal":
        nominal = home.clone()
        nominal[:, 0] = task.guard.bounds[0, 1] + .1
        target = target._replace(nominal=nominal)
    elif case.name == "legal_action_rate_projection":
        action[:, 0] = 1
        hold = offset(home, 0, task.guard.max_step[0])
    elif case.name == "unsafe_path_safe_hold":
        hold = offset(home, axis, SAFE_OFFSET_RAD)
        task.reset(ids, hold, torch.zeros_like(hold))
        target = target_frame(task, hold, 0)._replace(nominal=offset(home, axis, CONTACT_OFFSET_RAD))
    elif case.name == "contact_feedback_reject_hold":
        # Deliberate plant fault. This bypasses the reset gate to check its independent command guard.
        task.physics.reset(ids, offset(home, axis, CONTACT_OFFSET_RAD), torch.zeros_like(home))
    elif case.name == "moving_legal_start_physical_stop":
        hold = offset(home, axis, MOVING_OFFSET_RAD)
        velocity = torch.zeros_like(home)
        velocity[:, axis] = OUTWARD_VELOCITY_RAD_S
        task.reset(ids, hold, velocity)
        target = target_frame(task, hold, 0)
    else:
        raise ValueError(f"unrecognized boundary fixture {case.name}")

    observation = task.observe(target)
    transition = task.step(action)
    collect_interval(transition.substeps, budget)
    finite = all(torch.isfinite(value).all() for value in (transition.x1_pre_reset, observation, transition.reward,
                                                            transition.contact_depth_m, transition.minimum_gap_m))
    command_error = (transition.c - hold).abs().max().item()
    used_substeps = transition.substeps
    if case.name == "moving_legal_start_physical_stop":
        step_limit_ok = ((used_substeps > 0) & (used_substeps < SUBSTEPS)).all()
    else:
        step_limit_ok = (used_substeps == (0 if case.hard_stop else SUBSTEPS)).all()
    result = {
        "events": transition.events.tolist(), "substeps": used_substeps.tolist(),
        "fallback": transition.fallback.tolist(), "terminated": transition.terminated.tolist(),
        "contact": transition.contact.tolist(), "depth_m": transition.contact_depth_m.tolist(),
        "minimum_gap_m": transition.minimum_gap_m.tolist(),
        "issued_command_error_rad": command_error, "maximum_reward": transition.reward.max().item(),
        "finite_trace": finite, "recovery": None,
    }
    correct = (
        finite and (transition.events == int(case.event)).all() and step_limit_ok
        and (transition.fallback == case.fallback).all() and (transition.terminated == case.hard_stop).all()
        and (transition.safety_failure == case.hard_stop).all() and command_error <= HOLD_TOL_RAD
        and not transition.contact.any() and not transition.contact_depth_m.any()
    )
    if case.fallback or case.hard_stop:
        correct = correct and not transition.reward.any() and not transition.valid.any()
    else:
        requested_step = (transition.requested[:, 0] - home[:, 0]).abs()
        correct = correct and transition.valid.all() and (requested_step > task.guard.max_step[0]).all()
    if case.hard_stop:
        result["recovery"] = task_stopped(task, home, reset, transition, budget)
        correct = correct and result["recovery"]["stop_latched"] and result["recovery"]["explicit_reset_recovers"]
    result["pass"] = bool(correct)
    return result, transition.x1_pre_reset.detach().cpu()


def contact_trial(task, home, axis, budget):
    collision = offset(home, axis, CONTACT_OFFSET_RAD)
    ids = torch.arange(len(home), device=home.device)
    # Fault injection only. Advance the two unmodified physical engines from the same penetrating state.
    task.physics.reset(ids, collision, torch.zeros_like(home))
    allowed = torch.ones(len(home), device=home.device, dtype=torch.bool)
    interval = task.physics.interval(collision, allowed, task.guard, count=SUBSTEPS)
    collect_interval(interval.substeps, budget)
    finite = all(torch.isfinite(value).all() for value in (interval.state, interval.depth, interval.minimum_gap,
                                                            interval.contact))
    checks = (
        task.guard.positions_valid(collision).all() and finite
        and (interval.event == Stop.PHYSICAL_CONSTRAINT).all() and (interval.substeps == 1).all()
        and (interval.depth >= MIN_CONTACT_DEPTH_M).all() and (interval.contact[:, [1, 2, 10]] == 1).all()
    )
    return {
        "events": interval.event.tolist(), "substeps": interval.substeps.tolist(),
        "contact": interval.contact.tolist(), "depth_m": interval.depth.tolist(),
        "minimum_gap_m": interval.minimum_gap.tolist(), "finite_trace": finite, "pass": bool(checks),
    }, interval.state.detach().cpu()


def backend(robot, profile, envs, device, axis):
    task, home, reset = make_task(robot, envs, profile, "B54", device)
    budget = dict.fromkeys(("physical_control_intervals", "physical_substeps_completed",
                           "kernel_world_substeps_upper"), 0)
    results, states = {}, {}
    for case in CASES:
        reset()
        row, state = (contact_trial(task, home, axis, budget) if case.name == "injected_self_contact_backend_stop"
                      else task_trial(case, task, home, reset, axis, budget))
        results[case.name] = row
        states[case.name] = state
    guard = {"clearance_m": task.guard.sweep.clearance, "per_joint_max_step_rad": task.guard.max_step.tolist(),
             "coupling_limit_rad": task.guard.coupling_limit, "sweep_max_depth": task.guard.sweep.max_depth}
    return {"cases": results, "budget": budget, "guard": guard,
            "pass": all(row["pass"] for row in results.values())}, states


def pair_case(cpu, gpu, state_cpu, state_gpu, fk):
    delta = (state_cpu - state_gpu).abs()
    q_error = delta[:, :JOINTS].max().item()
    qd_error = delta[:, JOINTS:].max().item()
    tip_error = torch.linalg.vector_norm(fk(state_cpu[:, :JOINTS]).tips - fk(state_gpu[:, :JOINTS]).tips,
                                         dim=-1).max().item()
    depth_delta = [abs(a - b) for a, b in zip(cpu["depth_m"], gpu["depth_m"], strict=True)]
    depth_error = max(depth_delta) if all(map(isfinite, depth_delta)) else float("nan")
    maxima = {"q_rad": q_error, "qd_rad_s": qd_error, "tip_m": tip_error, "contact_depth_m": depth_error}
    limits = {name: FREE_RESPONSE_LIMITS[name] for name in ("q_rad", "qd_rad_s", "tip_m")}
    limits["contact_depth_m"] = CONTACT_DEPTH_DIFFERENCE_M
    exact = all(cpu.get(key) == gpu.get(key) for key in ("events", "substeps", "contact", "fallback", "terminated"))
    return {"discrete_exact_match": exact, "maxima": maxima, "limits": limits,
            "pass": cpu["pass"] and gpu["pass"] and exact and all(
                isfinite(value) and value <= limits[name] for name, value in maxima.items())}


@torch.inference_mode()
def compare(robot, *, envs=1, device="auto"):
    profile = ServoProfile()
    axis = robot.names.index(JOINT)
    native, native_state = backend(robot, profile, envs, "cpu", axis)
    warp = parity = warp_error = None
    if device != "cpu" and torch.cuda.is_available():
        try:
            warp, warp_state = backend(robot, profile, envs, "cuda:0", axis)
            fk = ResidualKinematics(robot)
            parity = {case.name: pair_case(native["cases"][case.name], warp["cases"][case.name],
                                           native_state[case.name], warp_state[case.name], fk) for case in CASES}
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            warp_error = {"type": type(error).__name__, "detail": str(error)}
    return json_finite({
        "scope": "SYNTHETIC_BOUNDARY_DIAGNOSTIC", "technical_gate_pass": False,
        "model": signature(robot, profile), "worlds": envs, "requested_device": device,
        "expected_cases": {case.name: {"event_bits": int(case.event), "fallback": case.fallback,
                                      "hard_stop": case.hard_stop} for case in CASES},
        "fixture": {"joint": JOINT, "safe_offset_rad": SAFE_OFFSET_RAD, "contact_offset_rad": CONTACT_OFFSET_RAD,
                    "moving_offset_rad": MOVING_OFFSET_RAD, "outward_velocity_rad_s": OUTWARD_VELOCITY_RAD_S},
        "limits": {"hold_command_rad": HOLD_TOL_RAD, "min_injected_contact_depth_m": MIN_CONTACT_DEPTH_M,
                   "max_physics_substeps_per_control": SUBSTEPS},
        "native": native, "warp": warp, "warp_error": warp_error, "paired_cases": parity,
        "paired_pass": None if parity is None else all(row["pass"] for row in parity.values()),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", type=int, choices=(1, 8, 16), default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"), default="auto")
    args = parser.parse_args()
    result = compare(RobotModel(args.model), envs=args.envs, device=args.device)
    write_json(args.out, result)
    print(f"native_pass={result['native']['pass']} paired_pass={result['paired_pass']} sha256={file_sha(args.out)}")
    if not result["native"]["pass"] or result["paired_pass"] is False or result["warp_error"] is not None:
        return 1
    return 2 if args.device == "cuda:0" and result["paired_pass"] is None else 0


if __name__ == "__main__":
    raise SystemExit(main())
