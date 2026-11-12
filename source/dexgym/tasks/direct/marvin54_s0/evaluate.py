"""Evaluate a frozen S0 Actor on held-out Hand-centric motions."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from ..marvin54_tracking.task import DT, JOINTS
from .env import HandCentricEnv
from .io import clip_paths, write_json
from .physics import CpuPhysics, FAILURES, PHYSICS_DT
from .task import HandGoalLibrary, TASK_ID, cross_tip_vectors


MODES = ("nominal", "perturbed", "alternate", "reference_pd", "arms_fixed")


def select_paths(paths, limit):
    """Visit one clip per episode before considering a second clip."""
    groups = defaultdict(list)
    for path in paths:
        with np.load(path, allow_pickle=False) as clip:
            groups[str(clip["source_episode"].item())].append((len(clip["time_s"]), path))
    groups = [sorted(values, reverse=True) for _, values in sorted(groups.items())]
    return [row[i][1] for i in range(max(map(len, groups))) for row in groups if i < len(row)][:limit]


def initial_states(model, library, mode, seed):
    q = library.qpos[library.start].cpu().numpy().astype(np.float64)
    qd = library.qvel[library.start].cpu().numpy().astype(np.float64)
    arm = np.asarray([name.startswith("Joint") for name in library.names])
    joint_ids = [model.joint(name).id for name in library.names]
    lo, hi = model.jnt_range[joint_ids].T
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    available = np.ones(len(q), dtype=bool)
    shift = np.zeros(len(q))

    def forward(pos):
        data.qpos[:] = pos
        mujoco.mj_fwdPosition(model, data)
        return np.min(data.contact.dist, initial=0.0) >= -0.0005

    def alternate(pos):
        wrist = [model.body(f"{side}_wrist").id for side in ("l", "r")]
        elbow = [model.body(f"Link4_{side}").id for side in ("L", "R")]
        joint = [np.flatnonzero([n.startswith("Joint") and n.endswith(f"_{side}") for n in library.names])
                 for side in ("L", "R")]
        forward(pos)
        wp = data.xpos[wrist].copy()
        wr = data.xmat[wrist].reshape(2, 3, 3).copy()
        ep = data.xpos[elbow].copy()
        best, change = pos, 0.

        def jac(index, ids):
            jp = np.zeros((3, JOINTS))
            jr = np.zeros_like(jp)
            mujoco.mj_jacBody(model, data, jp, jr, index)
            return np.vstack((jp[:, ids], 0.25 * jr[:, ids]))

        null = [np.linalg.svd(jac(index, ids), full_matrices=True)[2][-1]
                for index, ids in zip(wrist, joint, strict=True)]
        for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            trial = pos.copy()
            for side, ids in enumerate(joint):
                trial[ids] += 0.36 * signs[side] * null[side]
            trial = np.clip(trial, lo, hi)
            for _ in range(5):
                forward(trial)
                for side, (index, ids) in enumerate(zip(wrist, joint, strict=True)):
                    error_r = Rotation.from_matrix(wr[side] @ data.xmat[index].reshape(3, 3).T).as_rotvec()
                    error = np.r_[wp[side] - data.xpos[index], 0.25 * error_r]
                    j = jac(index, ids)
                    trial[ids] += np.clip(j.T @ np.linalg.solve(j @ j.T + 1e-5 * np.eye(6), error), -0.08, 0.08)
                trial = np.clip(trial, lo, hi)
            if not forward(trial):
                continue
            wp_error = np.linalg.norm(data.xpos[wrist] - wp, axis=-1).max()
            wr_error = Rotation.from_matrix(wr @ data.xmat[wrist].reshape(2, 3, 3).swapaxes(-1, -2)).magnitude().max()
            elbow_shift = np.linalg.norm(data.xpos[elbow] - ep, axis=-1).max()
            if wp_error <= 0.004 and wr_error <= 0.025 and elbow_shift > change:
                best, change = trial, float(elbow_shift)
        return best, change

    for i in range(len(q)):
        if mode == "perturbed":
            delta = rng.normal(0., np.where(arm, 0.06, 0.04), JOINTS)
            for scale in (1., 0.5, 0.25):
                candidate = np.clip(q[i] + scale * delta, lo, hi)
                if forward(candidate):
                    shift[i] = np.linalg.norm(candidate - q[i])
                    q[i], qd[i] = candidate, qd[i] + rng.normal(0., 0.12, JOINTS)
                    break
            else:
                available[i] = False
        elif mode == "alternate":
            q[i], shift[i] = alternate(q[i])
            available[i] = shift[i] >= 0.01
    return q, qd, available, shift


def get_errors(task, state, memory):
    pose = task.geometry.hand_pose(state[:, :JOINTS])
    e = task.errors(state, memory, pose)
    goal = task.library.ref_tip_pos[task.current_goal(memory)]
    cross_distance = cross_tip_vectors(pose["tip"]).norm(dim=-1) - cross_tip_vectors(goal).norm(dim=-1)
    return {
        "tip_mm": e["tip_pos"].norm(dim=-1).flatten(1) * 1000,
        "tip_deg": task.angle(e["tip_rot"]).flatten(1).rad2deg(),
        "cross_tip_vector_mm": e["cross_tip"].norm(dim=-1).flatten(1) * 1000,
        "cross_tip_distance_mm": cross_distance.abs().flatten(1) * 1000,
        "shape_mm": e["shape"].norm(dim=-1).flatten(1) * 1000,
        "relative_palm_deg": task.angle(e["palm_relative"])[:, None].rad2deg(),
        "wrist_mm": e["wrist"].norm(dim=-1) * 1000,
        "elbow_mm": e["elbow"].norm(dim=-1) * 1000,
        "joint_deg": e["joint"].abs().rad2deg(),
    }


@torch.inference_mode()
def evaluate(env, actor, mode, seed, paths, trace_dir=None):
    task, library = env.task, env.task.library
    starts = library.start
    lengths = library.end[starts] - starts
    q, qd, eligible, shift = initial_states(env.backend.model, library, mode, seed)
    obs, _ = env.reset(frame=starts, q=task.tensor(q), qd=task.tensor(qd))
    active = torch.as_tensor(eligible, device=env.device).clone()
    arm = torch.tensor([name.startswith("Joint") for name in library.names], device=env.device)
    fixed_arm = env.state()[:, :JOINTS].clone()
    done = torch.zeros_like(active)
    failure_bits = torch.zeros_like(starts)
    steps = torch.zeros_like(starts)
    peak = torch.zeros_like(starts, dtype=torch.float32)
    total_reward = torch.zeros_like(peak)
    saturation = torch.zeros_like(peak)
    records = defaultdict(list)
    if trace_dir:
        trace = dict(qpos=[env.state()[:, :JOINTS].clone()], target=[], depth=[], substep=[])
    for _ in range(int(lengths.max())):
        if not active.any():
            break
        before = env.memory
        action = actor(obs["policy"])
        override = None
        if mode == "reference_pd":
            override = library.qpos[task.current_goal(before)]
        elif mode == "arms_fixed":
            override = task.control_target(action, before, None)[1]
            override[:, arm] = fixed_arm[:, arm]
        obs, reward, terminal, _, event = env.step(action, target_override=override)
        state = env.state().nan_to_num(nan=0., posinf=0., neginf=0.)
        errors = get_errors(task, state, before)
        mask = active & event["full_step"]
        for key, values in errors.items():
            records[key].append(values.masked_fill(~mask[:, None], float("nan")))
        if trace_dir:
            trace["qpos"].append(env.state()[:, :JOINTS].clone())
            trace["target"].append(env.transition["u"].clone())
            trace["depth"].append(env.transition["depth"].clone())
            trace["substep"].append(event["safety_substep"].clone())
        done |= active & event["reference_complete"]
        failure_bits = torch.where(active, event["failure_bits"], failure_bits)
        steps += active
        peak = torch.where(active, torch.maximum(peak, env.transition["depth"]), peak)
        total_reward += torch.where(active, reward, 0.)
        saturation += torch.where(active, (action.abs() > 0.98).float().mean(-1), 0.)
        active &= ~terminal
        inactive = (~active).nonzero().flatten()
        if len(inactive) and active.any():
            obs, _ = env.reset(inactive, starts[inactive])
    errors = {key: torch.stack(values).cpu().numpy() for key, values in records.items()}
    if trace_dir and records:
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace = {key: torch.stack(values).cpu().numpy() for key, values in trace.items()}
    rows = []
    for i, path in enumerate(paths):
        n, bits = int(steps[i]), int(failure_bits[i])
        row = dict(clip=str(path), eligible=bool(eligible[i]), complete=bool(done[i]),
                   initial_shift=float(shift[i]), executed_steps=n, required_steps=int(lengths[i]),
                   depth_mm=float(peak[i] * 1000), reward_per_required_step=float(total_reward[i] / lengths[i]),
                   saturated_action_fraction=float(saturation[i] / max(1, n)),
                   failure_causes=[name for bit, name in FAILURES.items() if bits & bit])
        for key, values in errors.items():
            values = values[:, i]
            good = values[np.isfinite(values).all(-1)]
            row[key] = dict(mean=float(good.mean()), p95=float(np.percentile(good, 95))) if good.size else None
        rows.append(row)
        if trace_dir and n:
            time = np.arange(n + 1) * DT
            if trace["substep"][n - 1, i]:
                time[-1] = time[-2] + trace["substep"][n - 1, i] * PHYSICS_DT
            np.savez_compressed(trace_dir / f"{i:03d}.npz", time_s=time, qpos=trace["qpos"][:n + 1, i],
                                target=trace["target"][:n, i], depth_m=trace["depth"][:n, i])
    eligible_rows = [r for r in rows if r["eligible"]]
    completed = [r for r in eligible_rows if r["complete"]]
    return {
        "eligible": len(eligible_rows), "completed": len(completed),
        "completion": len(completed) / max(1, len(eligible_rows)),
        "step_coverage": sum(r["executed_steps"] for r in eligible_rows)
                         / max(1, sum(r["required_steps"] for r in eligible_rows)),
        "reward_per_required_step": float(np.mean([r["reward_per_required_step"] for r in eligible_rows]))
                                    if eligible_rows else None,
        "completed_means": {key: float(np.mean([r[key]["mean"] for r in completed])) if completed else None
                            for key in errors},
        "observed_prefix_means": {
            key: float(np.mean([r[key]["mean"] for r in eligible_rows if r[key] is not None]))
            if any(r[key] is not None for r in eligible_rows) else None for key in errors
        },
        "max_depth_mm": max((r["depth_mm"] for r in eligible_rows), default=0.),
        "initial_shift_unit": {"alternate": "max_elbow_displacement_m", "perturbed": "joint_l2_rad"}.get(mode),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "warp"), default="cpu")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seed", type=int, default=614)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.max_clips is not None and args.max_clips < 1:
        parser.error("--max-clips must be positive")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    device = "cuda:0" if args.backend == "warp" else "cpu"
    model = mujoco.MjModel.from_xml_path(str(args.model))
    paths = clip_paths(args.library)
    if args.max_clips:
        paths = select_paths(paths, args.max_clips)
    library = HandGoalLibrary(paths, model, args.model, device)
    prov = json.loads((args.checkpoint.parent / "provenance.json").read_text())
    if (prov["task_id"], prov["model_sha256"], prov["joint_names"]) != (TASK_ID, library.model_sha256, library.names):
        raise ValueError("Checkpoint task, asset or joint order differs")
    if args.backend == "warp":
        from .warp_physics import WarpPhysics

        backend = WarpPhysics(model, library.names, len(paths), device)
    else:
        backend = CpuPhysics(model, library.names, len(paths))
    env = HandCentricEnv(backend, library, reset_noise=False)
    actor = env.task.make_actor().to(device).eval()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload["task_id"] != TASK_ID or payload["format_version"] != 1:
        raise ValueError("Unsupported checkpoint task or format")
    actor.load_state_dict(payload["modules"]["actor"])
    report = dict(checkpoint=str(args.checkpoint), iteration=payload["iteration"], backend=args.backend,
                  initial_seed=args.seed, clips=len(paths), metrics={})
    del payload
    args.output.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        result = evaluate(env, actor, mode, args.seed, paths, args.output / mode if args.trace else None)
        report["metrics"][mode] = result
        print(mode, json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
        write_json(args.output / "evaluation.json", report)


if __name__ == "__main__":
    main()
