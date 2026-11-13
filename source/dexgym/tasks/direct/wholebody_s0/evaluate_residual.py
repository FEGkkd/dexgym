"""Closed-loop held-out evaluation on a native MuJoCo model."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .bag import digest
from .domain import Domain, IMPULSE_PROBABILITY, PHYSICAL_FIELDS
from .intent import GOAL_VERSION, HumanIntentGeometry, IntentTargets
from .native_replay import CRITICAL_WARNINGS, distribution
from .prepare import write_json, write_npz
from .residual import (AUDIT_DEPTH, HARD_DEPTH, DT, HUMAN_INTENT_WEIGHTS, N, POSE_GOAL_VERSION,
                       Geometry, ReferenceBank, ResidualTask, human_intent_score)
from .robot import FINGER_SITES, FIRMWARE_FINGERS


def angles_deg(a, b):
    a = Rotation.from_matrix(a.reshape(-1, 3, 3))
    b = Rotation.from_matrix(b.reshape(-1, 3, 3))
    return np.degrees((a.inv() * b).magnitude())


def lag_ms(reference, measured):
    """Estimate velocity lag on the original wrist time series."""
    if len(reference) < 30:
        return None
    a, b = np.diff(reference, axis=0), np.diff(measured, axis=0)
    if np.sqrt(np.mean(a**2)) < .0002:
        return None
    scores = []
    for delay in range(-5, 6):
        lo, hi = max(0, -delay), min(len(a), len(a) - delay)
        x, y = a[lo:hi].reshape(-1), b[lo + delay:hi + delay].reshape(-1)
        x, y = x - x.mean(), y - y.mean()
        norm = np.linalg.norm(x) * np.linalg.norm(y)
        scores.append(float(x @ y / norm) if norm > 1e-10 else -1.)
    if max(scores) < .2:
        return None
    return 20 * (int(np.argmax(scores)) - 5)


def load_actor(checkpoint, task, *, compare_earlier_library=False):
    if checkpoint is None:
        return None
    bank = task.bank
    meta = json.loads((checkpoint.parent / "provenance.json").read_text())
    if (meta["joint_names"] != bank.names or meta["policy_dim"] != task.policy_dim
            or meta["action_scale_rad"] != task.scale.tolist()
            or meta["command_delta_limit_rad"] != task.slew.tolist()):
        raise ValueError(f"Checkpoint observation/action contract mismatch: {checkpoint}")
    if meta["manifest_sha256"] != bank.manifest_sha256:
        model_sha = bank.library.manifest["provenance"]["robot_model"]["sha256"]
        training_sources = {entry["source_episode"] for entry in bank.library.entries("train")}
        if (not compare_earlier_library or meta["model_sha256"] != model_sha
                or not set(meta["train_episodes"]).issubset(training_sources)
                or digest(meta["manifest"]) != meta["manifest_sha256"]):
            raise ValueError(f"Checkpoint reference contract mismatch: {checkpoint}")
    if set(meta["train_episodes"]) & {entry["source_episode"] for entry in bank.entries}:
        raise ValueError("A source recording was used in both training and held-out evaluation")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=meta["method"] == "ppo")
    if payload["task_id"] != meta["task_id"]:
        raise ValueError(f"Checkpoint task identity mismatch: {checkpoint}")
    actor = task.make_actor()
    actor.load_state_dict(payload["actor"] if meta["method"] == "ppo" else payload["modules"]["actor"])
    return actor.eval().to(task.device)


def checkpoint_geometry(meta):
    profile = meta.get("reward_profile", "v1")
    if profile == "fingertip":
        info = meta["hand_servo"]["fingertip_site_model"]
    elif profile == "human_intent":
        if meta["intent"]["goal_version"] != GOAL_VERSION:
            raise ValueError("Unsupported human-intent target geometry version")
        info = meta["intent"]["geometry_model"]
    elif profile in ("retarget_pose", "tip_precision"):
        if meta["pose_tracking"]["goal_version"] != POSE_GOAL_VERSION:
            raise ValueError("Unsupported retargeted pose target geometry version")
        info = meta["pose_tracking"]["geometry_model"]
    else:
        return None
    if digest(info["path"]) != info["sha256"]:
        raise ValueError("Task-space site model checksum mismatch")
    return mujoco.MjModel.from_xml_path(info["path"])


def task_for_checkpoint(checkpoint, env, model, bank, domain):
    meta = json.loads((checkpoint.parent / "provenance.json").read_text()) if checkpoint else {}
    return ResidualTask(env, model, bank, domain=domain, reward_profile=meta.get("reward_profile", "v1"),
                        hand_model=checkpoint_geometry(meta))


class NativeEvaluator:
    def __init__(self, bank, reference_model, model_path, target_poses, human_tips, checkpoint, output, seed, robust,
                 trace, reference_start, compare_earlier_library):
        self.bank, self.output, self.seed, self.robust, self.trace = bank, output, seed, robust, trace
        self.target_poses = target_poses
        self.human_world_tips = human_tips
        self.reference_start = reference_start
        self.model = m = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(m)
        env = SimpleNamespace(device="cpu")
        arm = torch.as_tensor([name.startswith("Joint") for name in bank.names])
        self.task = task_for_checkpoint(checkpoint, env, reference_model, bank, Domain(1, "cpu", seed, False, arm))
        self.actor = load_actor(checkpoint, self.task, compare_earlier_library=compare_earlier_library)
        if (m.nq, m.nv, m.nu) != (N, N, N) or not np.isclose(m.opt.timestep, .002, rtol=0, atol=1e-12):
            raise ValueError("Evaluation needs the 54 DoF, 2 ms teleoperation model")
        joints = [m.joint(name) for name in bank.names]
        self.qi = [int(j.qposadr[0]) for j in joints]
        self.vi = [int(j.dofadr[0]) for j in joints]
        self.ai = [int(np.flatnonzero(m.actuator_trnid[:, 0] == j.id).item()) for j in joints]
        if not (np.all(m.actuator_gaintype[self.ai] == mujoco.mjtGain.mjGAIN_FIXED)
                and np.all(m.actuator_biastype[self.ai] == mujoco.mjtBias.mjBIAS_AFFINE)
                and np.all(m.actuator_biasprm[self.ai, 1] < 0)
                and m.actuator_forcelimited[self.ai].all() and m.actuator_ctrllimited[self.ai].all()):
            raise ValueError("Evaluation model does not provide the expected position servos")
        self.joints = [j.id for j in joints]
        self.jlo, self.jhi = m.jnt_range[self.joints].T
        self.clo = np.maximum(self.jlo, m.actuator_ctrlrange[self.ai, 0])
        self.chi = np.minimum(self.jhi, m.actuator_ctrlrange[self.ai, 1])
        self.task.lo = torch.maximum(self.task.lo, torch.tensor(self.clo, dtype=torch.float32))
        self.task.hi = torch.minimum(self.task.hi, torch.tensor(self.chi, dtype=torch.float32))
        if ((bank.q < self.task.lo - 1e-6) | (bank.q > self.task.hi + 1e-6)).any():
            raise ValueError("Held-out targets exceed the common train / evaluation command limits")
        self.wrists = [m.site(f"tcp_{side}").id for side in "LR"]
        self.tips = [m.site(f"{side}_{tip}").id for side in "lr" for tip in FINGER_SITES]
        self.finger_joints = [int(m.actuator(f"{side}_{finger}{joint}").trnid[0])
                              for side in "lr" for finger in FIRMWARE_FINGERS for joint in (1, 2, 3)]
        self.palms = [m.body(f"{side}_wrist").id for side in "lr"]
        self.fields = {
            name: getattr(m, name).copy() for name in (
                "actuator_gainprm", "actuator_biasprm", "actuator_forcerange",
                "jnt_actfrcrange", "dof_frictionloss", "geom_friction",
            )
        }

    def configure(self, domain):
        m = self.model
        for name, value in self.fields.items():
            getattr(m, name)[:] = value
        if not domain.robust:
            return
        ap, hp, kd, effort, contact, friction = domain.physical[0].tolist()
        scale = np.where(self.task.arm.numpy(), ap, hp)
        m.actuator_gainprm[self.ai, 0] *= scale
        m.actuator_biasprm[self.ai, 1] *= scale
        m.actuator_biasprm[self.ai, 2] *= kd
        m.actuator_forcerange[self.ai] *= effort
        m.jnt_actfrcrange[self.joints] *= effort
        m.dof_frictionloss[:] *= friction
        m.geom_friction[:] *= contact

    def state(self):
        d = self.data
        return torch.tensor(np.r_[d.qpos[self.qi], d.qvel[self.vi]], dtype=torch.float32).unsqueeze(0)

    def read_sites(self):
        d = self.data
        wrist, wr = d.site_xpos[self.wrists], d.site_xmat[self.wrists].reshape(2, 3, 3)
        tip = d.site_xpos[self.tips].reshape(2, 5, 3)
        tip_rotation = d.site_xmat[self.tips].reshape(2, 5, 3, 3)
        palm, pr = d.xpos[self.palms], d.xmat[self.palms].reshape(2, 3, 3)
        local = np.einsum("ski,sij->skj", tip - palm[:, None], pr)
        joints = d.xanchor[self.finger_joints].reshape(2, 5, 3, 3)
        links = np.einsum("skli,sij->sklj", joints - palm[:, None, None], pr)
        relative = (wrist[1] - wrist[0]) @ wr[0]
        wrist_local = np.einsum("ski,sij->skj", tip - wrist[:, None], wr)
        wrist_local_rotation = wr.transpose(0, 2, 1)[:, None] @ tip_rotation
        return wrist, wr, tip, local, relative, wr[0].T @ wr[1], links, wrist_local, wrist_local_rotation

    def deepest_geoms(self):
        pair = self.data.contact.geom[int(np.argmin(self.data.contact.dist))]
        m = self.model
        return [m.geom(int(i)).name or f"{m.body(int(m.geom_bodyid[i])).name}:geom{int(i)}" for i in pair]

    def step_interval(self, target, count):
        d, m = self.data, self.model
        d.ctrl[self.ai] = target
        peak, shallow, steps, event = 0., 0, 0, None
        for substep in range(1, 11):
            mujoco.mj_step(m, d)
            mujoco.mj_fwdPosition(m, d)
            steps += 1
            distance = d.contact.dist
            depth = max(0., -float(np.min(distance, initial=0.)))
            peak = max(peak, depth)
            shallow += depth > AUDIT_DEPTH
            reasons = []
            if (not np.isfinite(distance).all() or not np.isfinite(d.qpos).all()
                    or not np.isfinite(d.qvel).all() or not np.isfinite(d.qacc).all()
                    or not np.isfinite(d.time) or abs(d.time - ((count - 1) * 10 + substep) * .002) > 1e-8
                    or any(d.warning.number[int(w)] for w in CRITICAL_WARNINGS)):
                reasons.append("invalid_physics")
            if depth > HARD_DEPTH:
                reasons.append("collision")
            q = d.qpos[self.qi]
            if (q < self.jlo - .05).any() or (q > self.jhi + .05).any():
                reasons.append("joint_limit")
            if reasons:
                pair = self.deepest_geoms() if depth > HARD_DEPTH else None
                event = {"reasons": reasons, "frame": count, "substep": substep, "depth_mm": depth * 1000,
                         "time_s": float(d.time) if np.isfinite(d.time) else None, "geoms": pair}
                break
        return peak, shallow, steps, event

    @torch.inference_mode()
    def run(self, index):
        entry = self.bank.entries[index]
        start = int(self.bank.start[index])
        end = int(self.bank.end[start])
        seed = (self.seed + int.from_bytes(sha256(entry["id"].encode()).digest()[:4], "little")) % 2**32
        domain = Domain(1, "cpu", seed, self.robust, self.task.arm)
        self.configure(domain)
        task, d, m = self.task, self.data, self.model
        mujoco.mj_resetData(m, d)
        q_ref, v_ref = self.bank.q[start:start + 1], self.bank.qd[start:start + 1]
        q, v = domain.reset_noise(q_ref, v_ref)
        if self.reference_start:
            # Consume the same random draws so paired velocity impulses stay unchanged.
            q, v = q_ref, v_ref
        q = q.clamp(task.lo, task.hi)
        d.qpos[self.qi], d.qvel[self.vi], d.ctrl[self.ai] = q[0].numpy(), v[0].numpy(), q[0].numpy()
        mujoco.mj_forward(m, d)
        memory = {"frame": torch.tensor([start]), "cmd": q, "action": torch.zeros(1, N), "delay": domain.delay}
        initial_depth = max(0., -float(np.min(d.contact.dist, initial=0.)))
        peak, shallow, substeps, event = initial_depth, 0, 0, None
        if initial_depth > HARD_DEPTH:
            event = {"reasons": ["collision"], "frame": 0, "substep": 0,
                     "depth_mm": initial_depth * 1000, "time_s": 0., "geoms": self.deepest_geoms()}
        poses, projections, residuals, state_trace = [], [], [], []
        command_trace, applied_trace, action_trace, observed_trace = [], [], [], []
        action_steps, command_steps, residual_steps = [], [], []
        impulse_frames = []
        for count in range(1, end - start + 1):
            if event:
                break
            x = self.state()
            obs = task.policy_observation(x, memory, domain.observation_bias)
            action = self.actor(obs) if self.actor is not None else torch.zeros_like(memory["action"])
            if not torch.isfinite(action).all():
                event = {"reasons": ["invalid_action"], "frame": count, "substep": 0, "time_s": float(d.time)}
                break
            command, applied = task.control_target(action, memory)
            goal = task.bank.q[start + count]
            projected = (command - (goal + task.scale * action)).abs()
            projections.append(float((projected > 1e-6).float().mean()))
            residuals.append(float((command - goal).abs().mean()))
            depth, contact_count, steps, event = self.step_interval(applied[0].numpy(), count)
            peak, shallow, substeps = max(peak, depth), shallow + contact_count, substeps + steps
            if steps == 10 and (not event or "invalid_physics" not in event["reasons"]):
                poses.append(self.read_sites())
                state_trace.append(self.state()[0].numpy())
                action_steps.append((action - memory["action"])[0].numpy())
                command_steps.append((command - memory["cmd"])[0].numpy())
                residual_steps.append((command[0] - goal).numpy())
                if self.trace:
                    command_trace.append(command[0].numpy().copy())
                    applied_trace.append(applied[0].numpy().copy())
                    action_trace.append(action[0].numpy().copy())
                    observed_trace.append((x + domain.observation_bias)[0].numpy())
            if not event and self.robust and count % 25 == 0 and float(domain.uniform(())) < 25 * IMPULSE_PROBABILITY:
                d.qvel[self.vi] += domain.velocity_impulse((N,)).numpy()
                impulse_frames.append(count)
            memory = task.advance_memory(None, memory, action, command, None, None)
        errors, wrist_values = {}, None
        if poses:
            wrist, wr, tip, local, relative, rel_r, links, wrist_local, wrist_local_r = (
                np.stack(part) for part in zip(*poses, strict=True))
            goal = slice(start + 1, start + 1 + len(poses))
            wp, wt, fl, rp, rt, gt = [item[goal].numpy() for item in task.goal_geometry]
            tw, twr, t_local, tr, trr, tt, t_wrist_local, t_wrist_local_r = (item[goal] for item in self.target_poses)
            q_error = np.abs(np.stack(state_trace)[:, :N] - task.bank.q[goal].numpy())
            errors = {
                "wrist_mm": np.linalg.norm(wrist - wp, axis=-1) * 1000,
                "wrist_deg": angles_deg(wr, wt).reshape(-1, 2),
                "tip_mm": np.linalg.norm(tip - gt, axis=-1) * 1000,
                "local_tip_mm": np.linalg.norm(local - fl[:, :, :, -1], axis=-1) * 1000,
                "target_tip_mm": np.linalg.norm(tip - tt, axis=-1) * 1000,
                "target_local_tip_mm": np.linalg.norm(local - t_local[..., -1, :], axis=-1) * 1000,
                "target_wrist_mm": np.linalg.norm(wrist - tw, axis=-1) * 1000,
                "target_wrist_deg": angles_deg(wr, twr).reshape(-1, 2),
                "wrist_local_tip_mm": np.linalg.norm(wrist_local - t_wrist_local, axis=-1) * 1000,
                "wrist_local_tip_deg": angles_deg(wrist_local_r, t_wrist_local_r).reshape(-1, 2, 5),
                "target_relative_mm": np.linalg.norm(relative - tr, axis=-1) * 1000,
                "target_relative_deg": angles_deg(rel_r, trr),
                "arm_joint_rad": q_error[:, task.arm.numpy()],
                "hand_joint_rad": q_error[:, ~task.arm.numpy()],
                "relative_mm": np.linalg.norm(relative - rp, axis=-1) * 1000,
                "relative_deg": angles_deg(rel_r, rt),
                "action_step_rms": np.sqrt(np.square(action_steps).mean(axis=1)),
                "command_step_rms_rad": np.sqrt(np.square(command_steps).mean(axis=1)),
                "command_residual_rms_rad": np.sqrt(np.square(residual_steps).mean(axis=1)),
            }
            if self.human_world_tips is not None:
                human = IntentTargets(*(v[goal].numpy() for v in self.bank.intent_targets))
                human_local = human.local[..., -1, :]
                hr = human.rotation[:, 0].transpose(0, 2, 1)
                h_relative = np.einsum("tij,tj->ti", hr, human.wrist[:, 1] - human.wrist[:, 0])
                actual_links = np.concatenate((links, local[..., None, :]), axis=3)
                pose = tuple(torch.as_tensor(v, dtype=torch.float32) for v in
                             (wrist, wr, actual_links, relative, rel_r, tip))
                h_goal = tuple(torch.as_tensor(v, dtype=torch.float32) for v in
                               (*human, h_relative, hr @ human.rotation[:, 1]))
                score, _ = human_intent_score(pose, h_goal)
                errors.update({
                    "intent_loss": (sum(HUMAN_INTENT_WEIGHTS.values()) - score).numpy().clip(min=0.),
                    "intent_wrist_mm": np.linalg.norm(wrist - human.wrist, axis=-1) * 1000,
                    "intent_wrist_deg": angles_deg(wr, human.rotation).reshape(-1, 2),
                    "intent_local_tip_mm": np.linalg.norm(local - human_local, axis=-1) * 1000,
                    "intent_world_tip_mm": np.linalg.norm(tip - self.human_world_tips[goal], axis=-1) * 1000,
                    "intent_relative_mm": np.linalg.norm(relative - h_relative, axis=-1) * 1000,
                    "intent_relative_deg": angles_deg(rel_r, hr @ human.rotation[:, 1]),
                    "intent_thumb_index_gap_mm": 1000 * np.abs(
                        np.linalg.norm(local[:, :, 0] - local[:, :, 1], axis=-1)
                        - np.linalg.norm(human_local[:, :, 0] - human_local[:, :, 1], axis=-1)),
                })
            wrist_values = wp, wrist
            if self.trace:
                write_npz(self.output / "traces" / f"{entry['id']}.npz", {
                    "time_s": np.arange(1, len(poses) + 1) * DT, "state": np.stack(state_trace),
                    "initial_q": q[0].numpy(), "initial_reference_q": task.bank.q[start].numpy(),
                    "observed_state_before": np.stack(observed_trace),
                    "command": np.stack(command_trace), "applied": np.stack(applied_trace),
                    "action": np.stack(action_trace), "reference_q": task.bank.q[goal].numpy(),
                    "wrist": wrist, "tip": tip, "reference_wrist": wp, "reference_tip": gt, **errors,
                })
        report = {
            "id": entry["id"], "episode": entry["source_episode"], "task": entry["task"],
            "planned_intervals": end - start, "measured_intervals": len(poses),
            "complete": event is None and len(poses) == end - start, "failure": event,
            "peak_depth_mm": 1000 * peak, "substeps": substeps, "contact_substeps": shallow,
            "lag_ms": lag_ms(*wrist_values) if wrist_values is not None else None,
            "projection_fraction": float(np.mean(projections)) if projections else None,
            "compensation_rad": float(np.mean(residuals)) if residuals else None,
            "domain": dict(zip(PHYSICAL_FIELDS, domain.physical[0].tolist(), strict=True)),
            "delay_steps": int(domain.delay[0]),
            "velocity_impulse_frames": impulse_frames,
            "metrics": {key: distribution(value) for key, value in errors.items()},
        }
        return report, errors


def worker_init(args):
    global evaluator
    evaluator = NativeEvaluator(*args)


def worker_run(i):
    return evaluator.run(i)


def stats(items):
    if not items:
        return None
    errors = {key: np.concatenate([row[key] for row in items]) for key in items[0]}
    out = {key: distribution(value) for key, value in errors.items()}
    for key in ("wrist_mm", "wrist_deg", "tip_mm", "local_tip_mm", "target_tip_mm", "target_local_tip_mm",
                "target_wrist_mm", "target_wrist_deg", "wrist_local_tip_mm", "wrist_local_tip_deg",
                "intent_wrist_mm", "intent_wrist_deg", "intent_local_tip_mm", "intent_world_tip_mm"):
        if key in errors:
            out[key + "_sides"] = {side: distribution(errors[key][:, i]) for i, side in enumerate(("left", "right"))}
    for key in ("target_local_tip_mm", "intent_local_tip_mm", "wrist_local_tip_mm", "wrist_local_tip_deg"):
        if key not in errors:
            continue
        values = errors[key]
        out[key + "_fingers"] = {
            side: {tip: distribution(values[:, i, j]) for j, tip in enumerate(FINGER_SITES)}
            for i, side in enumerate(("left", "right"))
        }
    return out


def eval_case(args, bank, reference_model, target_poses, human_tips, checkpoint, robust, folder, model_offset):
    folder.mkdir(parents=True)
    if args.trace:
        (folder / "traces").mkdir()
    engine_args = (bank, reference_model, args.model, target_poses, human_tips, checkpoint, folder, args.seed, robust,
                   args.trace, args.reference_start, args.compare_earlier_library)
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init, initargs=(engine_args,)) as pool:
            items = list(pool.map(worker_run, range(len(bank.entries))))
    else:
        worker_init(engine_args)
        items = [worker_run(i) for i in range(len(bank.entries))]
    rows = [row for row, _ in items]
    complete = sum(row["complete"] for row in rows)
    total_steps = sum(row["substeps"] for row in rows)
    measured = sum(row["measured_intervals"] for row in rows)
    planned = sum(row["planned_intervals"] for row in rows)
    failures = Counter(reason for row in rows if row["failure"] for reason in row["failure"]["reasons"])
    summaries = {
        "checkpoint": str(checkpoint) if checkpoint else None, "split": args.split, "robust": robust,
        "reference_start": args.reference_start,
        "checkpoint_sha256": digest(checkpoint) if checkpoint else None,
        "seed": args.seed, "model": str(args.model), "model_sha256": digest(args.model), "mujoco": mujoco.__version__,
        "reference_manifest_sha256": bank.manifest_sha256, "static_model_geometry_difference": model_offset,
        "human_goal_version": GOAL_VERSION if human_tips is not None else None,
        "pose_goal_version": POSE_GOAL_VERSION,
        "episodes": len({row["episode"] for row in rows}), "clips": len(rows), "complete": complete,
        "complete_rate": complete / len(rows), "measured_intervals": measured, "planned_intervals": planned,
        "coverage": measured / planned, "failures": dict(failures),
        "delayed_clips": sum(row["delay_steps"] > 0 for row in rows),
        "velocity_impulses": sum(len(row["velocity_impulse_frames"]) for row in rows),
        "peak_depth_mm": max(row["peak_depth_mm"] for row in rows),
        "contact_substep_rate": sum(row["contact_substeps"] for row in rows) / max(1, total_steps),
        "all_measured": stats([errors for _, errors in items if errors]),
        "completed_only": stats([errors for row, errors in items if row["complete"]]),
        "lag_ms_moving_clips": distribution([row["lag_ms"] for row in rows if row["lag_ms"] is not None]),
    }
    write_json(folder / "summary.json", summaries)
    write_json(folder / "clips.json", rows)
    p95 = summaries["all_measured"]["wrist_mm"]["p95"] if summaries["all_measured"] else None
    print(f"{folder.name} {complete}/{len(rows)} wrist P95 {p95} contact {failures}", flush=True)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--checkpoint", type=Path, action="append", default=[])
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--profile", choices=("nominal", "stress", "both"), default="both")
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--intent-metrics", action="store_true", help="Score MANUS and Tracker intent for every policy")
    p.add_argument("--compare-earlier-library", action="store_true",
                   help="Compare a prior policy after verifying its model and training source episodes")
    p.add_argument("--reference-start", action="store_true",
                   help="Start from the exact reference state while retaining other requested disturbances")
    args = p.parse_args()
    if not (args.baseline or args.checkpoint) or args.workers < 1:
        p.error("Choose --baseline or --checkpoint and a positive worker count")
    torch.set_num_threads(1)
    args.model = args.model.resolve(strict=True)
    manifest = json.loads(args.manifest.read_text())
    info = manifest["provenance"]["robot_model"]
    if digest(info["path"]) != info["sha256"]:
        raise ValueError("Reference asset checksum mismatch")
    ref = mujoco.MjModel.from_xml_path(info["path"])
    evaluation_model = mujoco.MjModel.from_xml_path(str(args.model))
    candidates = ([None] if args.baseline else []) + [path.resolve(strict=True) for path in args.checkpoint]
    intent_metrics = args.intent_metrics
    for checkpoint in args.checkpoint:
        meta = json.loads((checkpoint.resolve().parent / "provenance.json").read_text())
        profile = meta.get("reward_profile")
        geometry_key = {"human_intent": "intent", "retarget_pose": "pose_tracking", "tip_precision": "pose_tracking"}.get(profile)
        if geometry_key and meta[geometry_key]["geometry_model"]["sha256"] != digest(args.model):
            raise ValueError(f"Native evaluation needs the {profile} policy's teleoperation geometry")
        if profile == "human_intent":
            intent_metrics = True
    bank = ReferenceBank(args.manifest, args.split, ref, "cpu", intent_model=evaluation_model if intent_metrics else None)
    human_tips = None
    if intent_metrics:
        targets = IntentTargets(*(v.numpy() for v in bank.intent_targets))
        human_tips = HumanIntentGeometry(evaluation_model).world_tips(targets)
    with torch.no_grad():
        ref_geometry = Geometry(ref, bank.names, "cpu")
        target_geometry = Geometry(evaluation_model, bank.names, "cpu")
        offsets = [[], []]
        target_parts = []
        for q in bank.q.split(1024):
            a, b = ref_geometry.pose(q), target_geometry.pose(q, wrist_local=True)
            offsets[0].append((a[0] - b[0]).norm(dim=-1).numpy() * 1000)
            offsets[1].append((a[5] - b[5]).norm(dim=-1).numpy() * 1000)
            target_parts.append(b)
        target_poses = tuple(torch.cat(group).numpy() for group in zip(*target_parts, strict=True))
        model_offset = {name: distribution(np.concatenate(val)) for name, val in zip(("wrist_mm", "tip_mm"), offsets)}
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    profiles = (False, True) if args.profile == "both" else (args.profile == "stress",)
    for ckpt in candidates:
        label = "zero_residual" if ckpt is None else ckpt.parent.name + "_" + ckpt.stem
        for robust in profiles:
            folder = args.output / (label + ("_stress" if robust else "_nominal"))
            results.append(eval_case(args, bank, ref, target_poses, human_tips, ckpt, robust, folder, model_offset))
    write_json(args.output / "comparison.json", results)


if __name__ == "__main__":
    main()
