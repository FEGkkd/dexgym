"""Fit single-clip teaching targets by repeated physical MuJoCo rollouts."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .model import ARM_JOINTS, sha256, validate_drive_only_model
from .sim_to_sim import MuJoCoTracking, describe
from .task import DT, MotionLibrary


def inverse_targets(env, q, qd):
    task, model = env.task, env.model
    data = mujoco.MjData(model)
    qdd = np.gradient(qd, DT, axis=0)
    target = np.empty_like(q[1:])
    kp, kd = (v.cpu().numpy() for v in (task.kp, task.kd))
    for i in range(len(target)):
        data.qpos[:] = (q[i] + q[i + 1]) * 0.5
        data.qvel[:] = (qd[i] + qd[i + 1]) * 0.5
        data.qacc[:] = (qdd[i] + qdd[i + 1]) * 0.5
        mujoco.mj_inverse(model, data)
        target[i] = data.qpos + (data.qfrc_inverse + kd * data.qvel) / kp
    return np.clip(target, task.lo.cpu().numpy(), task.hi.cpu().numpy())


def target_rollout(env, targets, qref, qdref, arm):
    env.reset(0)
    q, qd, depths = [env.data.qpos.copy()], [env.data.qvel.copy()], []
    causes = []
    for target in targets:
        depth, _, causes = env.step_target(target)
        q.append(env.data.qpos.copy())
        qd.append(env.data.qvel.copy())
        depths.append(depth)
        if causes:
            break
    q, qd = np.asarray(q), np.asarray(qd)
    err = q - qref[:len(q)]
    vel = qd - qdref[:len(q)]
    score = np.mean(err[:, arm] ** 2) + 0.35 * np.mean(err[:, ~arm] ** 2)
    score += 0.002 * np.mean(vel ** 2)
    return dict(qpos=q, qvel=qd, depth_m=np.asarray(depths), score=float(score),
                complete=not causes and len(q) == len(qref), failure_causes=causes)


def optimize(env, iterations):
    task = env.task
    q, qd = (getattr(task.library, field).cpu().numpy() for field in ("qpos", "qvel"))
    arm = np.asarray([n in ARM_JOINTS for n in task.library.names])
    lo, hi = (v.cpu().numpy() for v in (task.lo, task.hi))
    inverse = inverse_targets(env, q, qd)
    best = None
    history = []

    def trial(label, command):
        run = target_rollout(env, command, q, qd, arm)
        report = {key: run[key] for key in ("complete", "score", "failure_causes")}
        report.update(label=label, steps=len(run["depth_m"]), peak_mm=float(run["depth_m"].max() * 1000))
        history.append(report)
        print(json.dumps(report), flush=True)
        return run

    for mix in (0.0, 0.5, 0.85, 1.0):
        command = q[1:] + mix * (inverse - q[1:])
        run = trial(f"inverse_mix_{mix}", command)
        if run["complete"] and (best is None or run["score"] < best["score"]):
            best, targets = run, command
    if best is None:
        raise RuntimeError("No safe full-clip teaching rollout")
    rate = 0.65
    for iteration in range(iterations):
        correction = (q[1:] - best["qpos"][1:]) + 0.008 * (qd[1:] - best["qvel"][1:])
        correction = np.clip(correction, -0.12, 0.12)
        # Smooth the update across the adjacent commands in this 50 Hz clip.
        correction = np.pad(correction, ((1, 1), (0, 0)), mode="edge")
        correction = (correction[:-2] + 2 * correction[1:-1] + correction[2:]) / 4
        candidate = np.clip(targets + rate * correction, lo, hi)
        run = trial(f"ilc_{iteration}", candidate)
        if run["complete"] and run["score"] < best["score"]:
            best, targets = run, candidate
        else:
            rate *= 0.55
            if rate < 0.02:
                break
    return targets, best, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--reference-model", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(1)
    validate_drive_only_model(args.reference_model, args.model)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    library = MotionLibrary([args.reference], model, args.reference_model, "cpu")
    env = MuJoCoTracking(model, library)
    targets, best, trials = optimize(env, args.iterations)
    pose = env.task.geometry.pose(env.task.tensor(best["qpos"]))
    errors = {key: describe((1000 * (current - ref).norm(dim=-1)).numpy()) for key, current, ref in (
        ("wrist_mm", pose[0], library.ref_wrist_pos), ("elbow_mm", pose[2], library.ref_elbow_pos),
        ("tip_mm", pose[3], library.ref_tip_pos.flatten(1, 2)),
    )}
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(model_sha256=sha256(args.model), motion_sha256=library.sha256[0], joint_names=library.names)
    np.savez_compressed(args.output / "teacher.npz", targets=targets, qpos=best["qpos"], qvel=best["qvel"],
                        depth_m=best["depth_m"], metadata=json.dumps(metadata))
    (args.output / "teacher.json").write_text(json.dumps(
        {**metadata, "chosen_score": best["score"], "geometry": errors, "trials": trials}, indent=2
    ) + "\n")
    print(json.dumps(errors), flush=True)


if __name__ == "__main__":
    main()
