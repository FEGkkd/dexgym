"""Compare native and real batched Warp Servo transitions from identical fixtures."""

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

import torch

from ...p02.model import RobotModel
from . import JOINTS
from .geometry import ResidualKinematics
from .physics import ServoProfile, signature
from .smoke import make_task


POSITION_TOLERANCE_RAD = 0.000872664626
FREE_RESPONSE_LIMITS = {
    "q_rad": POSITION_TOLERANCE_RAD,
    "qd_rad_s": .05,
    "tip_m": .0005,
    "servo_rad": POSITION_TOLERANCE_RAD,
    "applied_rad": POSITION_TOLERANCE_RAD,
    "ctrl_rad": POSITION_TOLERANCE_RAD,
}


def free_response_ok(maxima, *, clean, completed, planned):
    return clean and completed == planned and all(map(isfinite, maxima.values())) and all(
        maxima[name] <= limit for name, limit in FREE_RESPONSE_LIMITS.items()
    )


def update_peak(value, observed):
    return max(value, observed) if isfinite(observed) else float("inf")


def compare(robot, envs):
    profile = ServoProfile()
    cpu, home_cpu, cpu_reset = make_task(robot, envs, profile, "B54", "cpu")
    gpu, home_gpu, gpu_reset = make_task(robot, envs, profile, "B54", "cuda:0")
    fk = ResidualKinematics(robot)
    conditions = []
    active_cpu = torch.ones(envs, dtype=torch.bool)
    active_gpu = active_cpu.cuda()
    cases = ("home_hold", "arm_only", "hand_only", "joint_arm_and_hand", "moving_initial_state")
    expected = (1, 10, 150)
    for name in cases:
        for steps in expected:
            cpu_reset()
            gpu_reset()
            if name == "moving_initial_state":
                position = home_cpu.clone()
                position[:, 0] += .008
                position[:, 34] += .007
                velocity = torch.zeros_like(position)
                velocity[:, 0] = -.03
                velocity[:, 34] = .03
                ids = torch.arange(envs)
                cpu.physics.reset(ids, position, velocity)
                gpu.physics.reset(ids.cuda(), position.cuda(), velocity.cuda())
            start = perf_counter()
            remaining = steps
            q = home_cpu.clone()
            if name in ("arm_only", "joint_arm_and_hand"):
                q[:, 0] += .004
                q[:, 7] -= .004
            if name in ("hand_only", "joint_arm_and_hand"):
                q[:, 14] += .003
                q[:, 34] += .003
            q_gpu = q.cuda()
            maxima = dict.fromkeys(FREE_RESPONSE_LIMITS, 0.)
            maxima.update(torque_Nm=0., contact_depth_m=0., minimum_gap_m=0.)
            completed, warp_attempts, real_calls = 0, 0, 0
            event_match, clean = True, True
            events_cpu, events_gpu = [], []
            while remaining:
                count = min(remaining, 10)
                a = cpu.physics.interval(q, active_cpu, cpu.guard, count=count)
                b = gpu.physics.interval(q_gpu, active_gpu, gpu.guard, count=count)
                real_calls += int((a.substeps > 0).sum()) + int((b.substeps > 0).sum())
                events_cpu.append(a.event.tolist())
                events_gpu.append(b.event.tolist())
                event_match &= torch.equal(a.event, b.event.cpu())
                clean &= not bool(a.event.any() or b.event.any() or a.contact.any() or b.contact.any()
                                  or (a.depth > 0).any() or (b.depth > 0).any())
                completed += min(int(a.substeps.min()), int(b.substeps.min()))
                warp_attempts += int(b.substeps.max()) * envs
                b_state = b.state.cpu()
                diff = (a.state - b_state).abs()
                maxima["q_rad"] = update_peak(maxima["q_rad"], diff[:, :JOINTS].max().item())
                maxima["qd_rad_s"] = update_peak(maxima["qd_rad_s"], diff[:, JOINTS:].max().item())
                tips = torch.linalg.vector_norm(fk(a.state[:, :JOINTS]).tips - fk(b_state[:, :JOINTS]).tips, dim=-1)
                maxima["tip_m"] = update_peak(maxima["tip_m"], tips.max().item())
                for key, field in (("servo_rad", "servo"), ("applied_rad", "applied"), ("ctrl_rad", "ctrl"),
                                   ("torque_Nm", "torque"), ("contact_depth_m", "depth"),
                                   ("minimum_gap_m", "minimum_gap")):
                    delta = (getattr(a, field) - getattr(b, field).cpu()).abs()
                    maxima[key] = update_peak(maxima[key], delta.max().item())
                remaining -= count
                if a.event.any() or b.event.any():
                    break
            match = free_response_ok(maxima, clean=clean and event_match, completed=completed, planned=steps)
            conditions.append({"fixture": name, "planned_substeps": steps, "completed_per_world": completed,
                               "warp_executed_kernel_world_substeps_upper": warp_attempts, "real_guarded_intervals": real_calls,
                               "native_events": events_cpu, "warp_events": events_gpu,
                               "events_equal": event_match, "event_and_contact_free": clean,
                               "maxima": {key: value if isfinite(value) else None for key, value in maxima.items()},
                               "seconds": perf_counter() - start, "pass": match})
    return {"test": "S0-5 signed model CPU/Warp free-response parity", "model": signature(robot, profile),
            "scope": "FREE_RESPONSE_ONLY", "limits": FREE_RESPONSE_LIMITS,
            "warp_parallel_worlds": envs, "conditions": conditions, "pass": all(row["pass"] for row in conditions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=1)
    args = parser.parse_args()
    result = compare(RobotModel(args.model), args.envs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"pass={result['pass']} sha256={sha256(args.out.read_bytes()).hexdigest()}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
