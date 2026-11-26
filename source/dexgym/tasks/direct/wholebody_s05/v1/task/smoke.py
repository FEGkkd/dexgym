"""Physical smoke for the signed MuJoCo model; never starts a learner or ROS."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ...p02.model import RobotModel
from . import HISTORY, JOINTS, SUBSTEPS
from .env import ResidualTask
from .geometry import GeometryVelocity
from .observation import TargetFrame, geometry_mask, velocity_mask
from .physics import CpuServo, ServoProfile, signature
from .safety import SafetyProjection, Stop


def target_frame(task, q, tick, *, generation=0, permit=True):
    n = q.shape[0]
    zeros = q.new_zeros
    good = torch.full((n,), permit, device=q.device, dtype=torch.bool)
    geometry = task.geometry(q)
    velocity = GeometryVelocity(zeros(n, 2, 3), zeros(n, 2, 3), zeros(n, 2, 5, 3))
    return TargetFrame(q, zeros(n, JOINTS), good[:, None].expand(-1, 2), geometry, velocity,
                       geometry_mask(good), velocity_mask(good), good[:, None].expand(-1, 4),
                       zeros(n, 4), good[:, None].expand(-1, 4), good,
                       torch.ones((n, 2, 4), dtype=torch.bool, device=q.device),
                       torch.full((n,), generation, dtype=torch.long, device=q.device),
                       torch.full((n,), tick * 20_000_000, dtype=torch.long, device=q.device))


def make_task(robot, environments, profile, mode, device):
    if device == "cpu":
        physics = CpuServo(robot, environments, profile)
    else:
        from .warp import WarpServo

        physics = WarpServo(robot, environments, profile, device=device)
    # Same historical sim fallback per-tick envelope. Production profile
    # ownership and tolerance review remain separate from this smoke.
    step = [.027] * 14 + [.10] * 40
    guard = SafetyProjection(robot, device=device, max_step=step)
    task = ResidualTask(robot, physics, guard, mode=mode)
    home = torch.as_tensor(robot.home, device=device, dtype=torch.float32)[None].repeat(environments, 1)
    ids = torch.arange(environments, device=device)

    def reset():
        task.reset(ids, home, torch.zeros_like(home))

    reset()
    return task, home, reset


def scalar(tensor):
    return tensor.detach().cpu().item()


def smoke(robot, *, environments, steps, device):
    profile = ServoProfile()
    task, home, reset = make_task(robot, environments, profile, "B54", device)
    zero = torch.zeros_like(home)
    zero_baseline_task, _, _ = make_task(robot, environments, profile, "B0", device)
    before = perf_counter()
    obs = task.observe(target_frame(task, home, 0))
    if obs.shape != (environments, 1703 * HISTORY) or obs[:, :1703 * 4].count_nonzero():
        raise AssertionError("First actor history did not start with zero padding")
    b54 = task.step(zero)
    zero_baseline_task.observe(target_frame(zero_baseline_task, home, 0))
    b0 = zero_baseline_task.step(torch.full_like(zero, .5))
    if not all(torch.equal(getattr(b54, field), getattr(b0, field))
               for field in ("requested", "c", "events", "x1_pre_reset", "ctrl")):
        raise AssertionError("Legal zero-residual B54 differed from B0")
    if not b54.full_step.all() or not b54.valid.all():
        raise AssertionError(f"Baseline failed actual servo/safety: {b54.events.tolist()}")
    trial_count = int(torch.count_nonzero(b54.substeps) + torch.count_nonzero(b0.substeps))
    pulse = []
    for axis, name in enumerate(robot.names):
        row = {"joint": name}
        for sign in (-1, 1):
            reset()
            command = zero.clone()
            command[:, axis] = sign * .05
            task.observe(target_frame(task, home, 0))
            transition = task.step(command)
            trial_count += int(torch.count_nonzero(transition.substeps))
            delta_position = sign * (transition.x1_pre_reset[:, axis] - b0.x1_pre_reset[:, axis])
            delta_ctrl = sign * (transition.ctrl[:, -1, axis] - b0.ctrl[:, -1, axis])
            row[str(sign)] = {
                "min_response_rad": scalar(delta_position.min()),
                "min_ctrl_response_rad": scalar(delta_ctrl.min()),
                "min_actual_substeps": scalar(transition.substeps.min()),
                "hard_events": transition.events.tolist(),
            }
        pulse.append(row)
    pulse_pass = all(side["min_response_rad"] > 1e-7
                     and side["min_ctrl_response_rad"] > 1e-5 and side["min_actual_substeps"] == SUBSTEPS
                     and all(bits == 0 for bits in side["hard_events"])
                     for row in pulse for side in (row["-1"], row["1"]))
    reset()
    valid_steps = 0
    physics_substeps = 0
    simulated_reward = []
    for i in range(steps):
        task.observe(target_frame(task, home, i))
        result = task.step(zero)
        trial_count += int(torch.count_nonzero(result.substeps))
        valid_steps += int(result.valid.sum())
        physics_substeps += int(result.substeps.sum())
        simulated_reward.append(scalar(result.reward.mean()))
        if result.terminated.any():
            break
    reset()
    task.observe(target_frame(task, home, 0))
    invalid = zero.clone()
    invalid[0, 0] = torch.nan
    fallback = task.step(invalid)
    trial_count += int(torch.count_nonzero(fallback.substeps))
    fallback_pass = bool(fallback.fallback[0] and fallback.events[0] & Stop.INVALID_ACTION
                         and not fallback.terminated[0] and fallback.reward[0] == 0)
    reset()
    task.observe(target_frame(task, home, 0))
    end = task.step(zero, truncated=torch.ones(environments, dtype=torch.bool, device=device))
    trial_count += int(torch.count_nonzero(end.substeps))
    saved = end.x1_pre_reset.clone()
    reset()
    precopy_pass = torch.equal(saved, end.x1_pre_reset) and end.truncated.all().item() and not end.terminated.any().item()
    elapsed = perf_counter() - before
    return {
        "scope": "simulation-only signed-model smoke; synthetic independent target fixture",
        "model": signature(robot, profile), "target": "static fixture generated from signed home pose",
        "device": device, "parallel_envs": environments, "seconds": elapsed,
        "guard": {"clearance_m": task.guard.sweep.clearance, "max_steps_rad": task.guard.max_step.tolist(),
                  "coupling_limit_rad": task.guard.coupling_limit, "max_sweep_depth": task.guard.sweep.max_depth},
        "real_transitions": trial_count, "valid_hold_transitions": valid_steps, "hold_physics_substeps": physics_substeps,
        "hold_reward_range": [min(simulated_reward), max(simulated_reward)],
        "b0_zero_parity_pass": True, "action_pulses_pass": pulse_pass, "axis_pulses": pulse,
        "invalid_action_fallback_pass": fallback_pass, "pre_reset_trace_pass": precopy_pass,
        "pass": pulse_pass and fallback_pass and precopy_pass and valid_steps == environments * steps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args()
    if args.steps < 1 or args.envs < 1:
        raise ValueError("Smoke requires a positive batch and hold duration")
    result = smoke(RobotModel(args.model), environments=args.envs, steps=args.steps, device=args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    payload_sha = sha256(args.out.read_bytes()).hexdigest()
    print(f"pass={result['pass']} real_transitions={result['real_transitions']} sha256={payload_sha}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
