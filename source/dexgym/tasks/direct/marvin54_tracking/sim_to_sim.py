"""Run a trained Marvin policy closed-loop in native MuJoCo."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from .model import drive_parameters, sha256, validate_drive_only_model
from .task import DT, HARD_CONTACT_DEPTH_M, JOINT_OVERFLOW_RAD, JOINTS, TASK_ID, MotionLibrary, TrackingTask


class MuJoCoTracking:
    def __init__(self, model, library, tracking_failure_m=0.25):
        self.model = model
        self.data = mujoco.MjData(model)
        self.device, self.num_envs = library.qpos.device, 1
        self.task = TrackingTask(
            self, model, library, "home", "motion-encoder", "transition", "multiscale",
            first_event=True, tracking_failure_m=tracking_failure_m,
        )
        self.actuator_ids = [p["actuator_id"] for p in drive_parameters(model, library.names).values()]
        self.substeps = round(DT / model.opt.timestep)
        if not np.isclose(model.opt.timestep, 0.002) or self.substeps != 10:
            raise ValueError("Tracking requires a 2 ms physics / 20 ms policy period")
        self.lo = self.task.joint_lo.cpu().numpy()
        self.hi = self.task.joint_hi.cpu().numpy()

    def state(self):
        return self.task.tensor(np.r_[self.data.qpos, self.data.qvel])[None]

    def reset(self, frame, position_noise=None, velocity_noise=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.task.library.qpos[frame].cpu().numpy()
        self.data.qvel[:] = self.task.library.qvel[frame].cpu().numpy()
        if position_noise is not None:
            self.data.qpos[:] = np.clip(self.data.qpos + position_noise, self.lo, self.hi)
        if velocity_noise is not None:
            self.data.qvel[:] += velocity_noise
        self.data.ctrl[self.actuator_ids] = self.data.qpos
        mujoco.mj_forward(self.model, self.data)

    def step_target(self, target):
        """Advance ten physics steps or return the first detected safety event."""
        m, d = self.model, self.data
        d.ctrl[self.actuator_ids] = target
        peak = 0.0
        for substep in range(1, self.substeps + 1):
            previous_time = d.time
            mujoco.mj_step(m, d)
            mujoco.mj_fwdPosition(m, d)
            depth = max(0.0, -float(np.min(d.contact.dist, initial=0.0)))
            peak = max(peak, depth)
            causes = []
            if depth > HARD_CONTACT_DEPTH_M:
                causes.append("collision")
            if (d.qpos < self.lo - JOINT_OVERFLOW_RAD).any() or (d.qpos > self.hi + JOINT_OVERFLOW_RAD).any():
                causes.append("joint_limit")
            if (d.time <= previous_time or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all()
                    or not np.isfinite(depth)):
                causes.append("nonfinite")
            if causes:
                return peak, substep, causes
        return peak, None, []


def describe(values):
    return {name: float(value) for name, value in zip(
        ("mean", "p95", "max"), (values.mean(), np.percentile(values, 95), values.max())
    )}


@torch.inference_mode()
def rollout(env, actor=None, *, clip_index=0, position_noise=None, velocity_noise=None):
    task, lib = env.task, env.task.library
    start = lib.start[clip_index].item()
    end = lib.end[start].item()
    env.reset(start, position_noise, velocity_noise)
    memory = {"frame": torch.tensor([start], device=env.device),
              "action": torch.zeros(1, JOINTS, device=env.device)}
    history = task.policy_observation(env.state(), memory, None)[:, None].repeat(1, task.history_steps, 1)
    records = {"qpos": [env.data.qpos.copy()], "qvel": [env.data.qvel.copy()],
               "time_s": [0.0], "action": [], "target": [], "depth_m": []}
    errors = []
    substep, causes = None, []
    for _ in range(start, end):
        action = (actor(task.observation_from_history(history, memory))
                  if actor is not None else task.baseline_action(memory))
        if not torch.isfinite(action).all():
            raise ValueError("Policy produced a nonfinite action")
        target, _ = task.control_target(action, memory, None)
        depth, substep, causes = env.step_target(target[0].cpu().numpy())
        state = env.state()
        records["qpos"].append(env.data.qpos.copy())
        records["qvel"].append(env.data.qvel.copy())
        records["time_s"].append(env.data.time)
        records["action"].append(action[0].cpu().numpy())
        records["target"].append(target[0].cpu().numpy())
        records["depth_m"].append(depth)
        if not causes or substep == env.substeps:
            e = task.geometric_errors(state, memory)
            errors.append({
                "wrist_mm": (e["wrist"].norm(dim=-1) * 1000)[0],
                "wrist_deg": task.angle(e["wrist_rot"]).rad2deg()[0],
                "tip_mm": (e["tip"].norm(dim=-1) * 1000)[0],
                "tip_deg": task.angle(e["tip_rot"]).rad2deg()[0],
                "elbow_mm": (e["elbow"].norm(dim=-1) * 1000)[0],
                "relative_wrist_mm": (e["relative_wrist"].norm(dim=-1) * 1000)[0],
                "joint_deg": e["q"].abs().rad2deg()[0],
            })
            if not causes and errors[-1]["wrist_mm"].max().item() > 1000 * task.tracking_failure_m:
                causes = ["tracking"]
        if causes:
            break
        memory = task.advance_memory(state, memory, action, target, None, None, soft=False)
        frame_obs = task.policy_observation(state, memory, None)
        history = torch.cat((history[:, 1:], frame_obs[:, None]), 1)
    trace = {key: np.asarray(value) for key, value in records.items()}
    result = {
        "reference_complete": not causes and len(trace["target"]) == end - start,
        "failed": bool(causes), "failure_causes": causes, "failure_substep": substep,
        "failure_time_s": float(env.data.time) if causes else None,
        "executed_steps": len(trace["target"]), "full_clip_steps": end - start,
        "complete_control_steps": len(errors), "penetration_mm": describe(trace["depth_m"] * 1000),
    }
    if errors:
        result.update({
            key: describe(torch.stack([entry[key] for entry in errors]).cpu().numpy()) for key in errors[0]
        })
        result["per_side"] = {
            side: {key: describe(np.split(torch.stack([entry[key] for entry in errors]).cpu().numpy(), 2, axis=1)[i])
                   for key in ("wrist_mm", "wrist_deg", "tip_mm", "tip_deg", "elbow_mm")}
            for i, side in enumerate(("left", "right"))
        }
    return result, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, action="append", type=Path)
    parser.add_argument("--reference-model", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--perturb-seed", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    validate_drive_only_model(args.reference_model, args.model)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    library = MotionLibrary(args.reference, model, args.reference_model, "cpu")
    actor = None
    threshold = 0.25
    model_hash = sha256(args.model)
    if args.checkpoint:
        p = json.loads((args.checkpoint.parent / "provenance.json").read_text())
        expected = dict(task_id=TASK_ID, nominal="home", policy_arch="motion-encoder", task_contract="first-event",
                        model_sha256=model_hash, motion_sha256=library.sha256, joint_names=library.names)
        if any(p.get(key) != value for key, value in expected.items()):
            raise ValueError("Checkpoint provenance does not match the deployed model, motion and policy contract")
        threshold = p["tracking_failure_m"]
    env = MuJoCoTracking(model, library, threshold)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if payload["format_version"] != 1 or payload["task_id"] != TASK_ID:
            raise ValueError("Checkpoint task or format mismatch")
        actor = env.task.make_actor().eval()
        actor.load_state_dict(payload["modules"]["actor"])
    perturb = {}
    if args.perturb_seed is not None:
        rng = np.random.default_rng(args.perturb_seed)
        perturb = dict(position_noise=rng.normal(0, 0.01, JOINTS), velocity_noise=rng.normal(0, 0.05, JOINTS))
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "backend": "native MuJoCo", "model_sha256": model_hash, "motion_sha256": library.sha256,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "perturb_seed": args.perturb_seed, "clip_index": args.clip_index,
        "physics_dt_s": model.opt.timestep, "control_dt_s": DT,
        "joint_names": library.names, "metrics": {},
    }
    controllers = [("reference_pd", None)]
    if actor is not None:
        controllers.append(("policy", actor))
    for name, controller in controllers:
        report, trace = rollout(env, controller, clip_index=args.clip_index, **perturb)
        result["metrics"][name] = report
        np.savez_compressed(args.output / f"{name}.npz", **trace)
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
