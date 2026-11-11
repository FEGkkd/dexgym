"""Run the 54 DoF tracking task with native Newton and DexGym SuperMimic."""

import argparse
import json
import random
from pathlib import Path

import mujoco
import numpy as np
import torch
from isaaclab.app import launch_simulation

from dexgym.algorithms.supermimic.dex_reference.config import SuperMimicConfig
from dexgym.algorithms.supermimic.dex_reference.engine import SuperMimicEngine
from dexgym.algorithms.supermimic.dex_reference.networks import Actor
from dexgym.assets.marvin_wuji.robot_cfg import PACKAGED_MJCF

from .native import NewtonTrackingEnv, simulation_cfg
from .task import IMAGINATION_STEPS, JOINTS, TASK_ID, MotionLibrary


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@torch.no_grad()
def check_contract(env):
    task, lib = env.task, env.task.library
    ids = torch.arange(env.num_envs, device=env.device)
    frames = torch.linspace(0, lib.frames - 2, env.num_envs, device=env.device).long()
    obs, _ = env.reset(frame=frames)
    q = env.state()[:, :JOINTS]
    wp, wr, ep, fp, fr = task.geometry.pose(q)
    errors = {
        "reset_rad": (q - lib.qpos[frames]).abs().max(),
        "wrist_m": (wp - lib.ref_wrist_pos[frames]).abs().max(),
        "elbow_m": (ep - lib.ref_elbow_pos[frames]).abs().max(),
        "tip_m": (fp - lib.ref_tip_pos[frames].flatten(1, 2)).abs().max(),
        "wrist_rotation_matrix": (wr - lib.ref_wrist_rot[frames]).abs().max(),
        "tip_rotation_matrix": (fr - lib.ref_tip_rot[frames].flatten(1, 2)).abs().max(),
    }
    wrist_ids = [env.robot.body_names.index(name) for name in ("l_wrist", "r_wrist")]
    newton_wrist = env.robot.data.body_pose_w.torch[:, wrist_ids, :3] - env.scene.env_origins[:, None]
    errors["newton_wrist_m"] = (newton_wrist - wp).abs().max()
    errors = {k: float(v) for k, v in errors.items()}
    if max(errors.values()) > 2e-4 or obs["policy"].shape != (env.num_envs, task.policy_dim):
        raise ValueError(f"Reference and Newton/FK contract mismatch: {errors}")
    memory = env.memory
    action = torch.zeros(env.num_envs, JOINTS, device=env.device)
    _, reward, terminated, _, _ = env.step(action)
    if not reward.isfinite().all() or terminated.any():
        raise RuntimeError("Reference phase test failed during the first physical step")
    if not torch.equal(env.memory["frame"], lib.at(memory["frame"])):
        raise RuntimeError("Physical phase failed to advance by one reference frame")
    env.reset(ids[-1:], frame=lib.end[frames[-1:]] - 1)
    _, _, terminated, truncated, _ = env.step(action)
    if terminated[-1] or not truncated[-1]:
        raise RuntimeError("The final reference step must truncate independently of safety termination")
    return errors


@torch.no_grad()
def evaluate(env, actor, clip_index, steps):
    if env.num_envs != 2:
        raise ValueError("Evaluation requires two environments for a matched rollout")
    task, lib = env.task, env.task.library
    frame = lib.start[clip_index]
    max_steps = int(lib.end[frame] - frame)
    count = max_steps if steps is None else min(max_steps, steps)
    obs, _ = env.reset(frame=frame.repeat(2))
    active = torch.ones(2, dtype=torch.bool, device=env.device)
    terminations = torch.zeros_like(active)
    records = {}
    for i in range(count):
        m = env.memory
        action = actor(obs["policy"])
        action[0] = 0.0
        action[~active] = 0.0
        obs, reward, terminated, truncated, _ = env.step(action)
        e = task.geometric_errors(env.state(), m)
        values = {
            "wrist_mm": e["wrist"].norm(dim=-1) * 1000,
            "elbow_mm": e["elbow"].norm(dim=-1) * 1000,
            "tip_mm": e["tip"].norm(dim=-1) * 1000,
            "wrist_deg": task.angle(e["wrist_rot"]).rad2deg(),
            "tip_deg": task.angle(e["tip_rot"]).rad2deg(),
            "relative_wrist_mm": e["relative_wrist"].norm(dim=-1)[:, None] * 1000,
            "joint_deg": e["q"].abs().rad2deg(),
            "penetration_mm": env.transition["depth"][:, None] * 1000,
            "reward": reward[:, None],
        }
        for key, value in values.items():
            records.setdefault(key, []).append(value.masked_fill(~active[:, None], float("nan")))
        terminations |= active & terminated
        active &= ~(terminated | truncated)
        if not active.any():
            break
        ids = (~active).nonzero(as_tuple=True)[0]
        if len(ids):
            obs, _ = env.reset(ids, frame=lib.at(frame.repeat(len(ids)), i + 1))
    result = {}
    arrays = {k: torch.stack(v).cpu().numpy() for k, v in records.items()}
    for index, name in enumerate(("zero_residual", "policy")):
        entry = {"requested_steps": count, "full_clip_steps": max_steps, "terminated": bool(terminations[index])}
        for key, frames in arrays.items():
            values = frames[:, index]
            valid = np.isfinite(values).all(axis=-1)
            if key == "reward":
                entry["executed_steps"] = int(valid.sum())
            if not valid.any():
                entry[key] = None
                continue
            values = values[valid]
            entry[key] = {"mean": float(values.mean()), "p95": float(np.percentile(values, 95)), "max": float(values.max())}
            if key == "penetration_mm":
                entry["control_steps_penetration_gt_0_5mm"] = int((values > 0.5).sum())
        result[name] = entry
    matched = np.isfinite(arrays["reward"]).all(axis=(1, 2))
    if matched.any():
        result["matched_steps"] = int(matched.sum())
        for i, name in enumerate(("zero_residual", "policy")):
            result[name]["matched_mean"] = {
                key: float(values[matched, i].mean())
                for key, values in arrays.items() if np.isfinite(values[matched, i]).all()
            }
    return result


def optimizer_steps(engine):
    return {
        name: max((int(state["step"]) for state in opt.state.values() if "step" in state), default=0)
        for name, opt in engine.optimizers.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--wm-only-iters", type=int, default=30)
    parser.add_argument("--wm-batches", type=int, default=2)
    parser.add_argument("--wm-batch", type=int, default=192)
    parser.add_argument("--pi-batch", type=int, default=32)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--nominal", choices=("reference", "feedforward"), default="feedforward")
    args = parser.parse_args()
    if args.num_envs < 2 or not 0 <= args.wm_only_iters < args.iterations:
        parser.error("Need at least two environments and at least one Actor iteration")
    if min(args.wm_batches, args.wm_batch, args.pi_batch) < 1 or (args.eval_steps is not None and args.eval_steps < 1):
        parser.error("Batch sizes and evaluation steps must be positive")
    if args.check_only and args.evaluate:
        parser.error("Choose check-only or evaluate")
    if not 0 <= args.clip_index < len(args.reference):
        parser.error("Clip index is outside the Motion Library")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = simulation_cfg()
    model = mujoco.MjModel.from_xml_path(str(PACKAGED_MJCF))
    lib = MotionLibrary(args.reference, model, PACKAGED_MJCF, cfg.device)
    provenance = {"task_id": TASK_ID, "model_sha256": lib.model_sha256, "motion_sha256": lib.sha256,
                  "motion_files": [str(path.resolve()) for path in args.reference], "frames": lib.frames,
                  "joint_names": lib.names, "seed": args.seed, "backend": "Isaac Lab 3 / Newton MJWarp",
                  "nominal": args.nominal}
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.evaluate and not args.check_only and (args.output / "metrics.jsonl").exists():
        parser.error("Select a fresh output directory for training")
    with launch_simulation(cfg):
        env = NewtonTrackingEnv(cfg, 2 if args.evaluate else args.num_envs, model, lib, args.nominal)
        contract = check_contract(env)
        write_json(args.output / "contract.json", contract)
        print("TRACKING_CONTRACT", json.dumps(contract), flush=True)
        if args.check_only:
            return
        if args.evaluate:
            original = json.loads((args.evaluate.parent / "provenance.json").read_text())
            original.setdefault("nominal", "reference")
            if any(original[k] != provenance[k]
                   for k in ("task_id", "model_sha256", "motion_sha256", "joint_names", "nominal")):
                raise ValueError("Checkpoint reference or asset provenance mismatch")
            payload = torch.load(args.evaluate, map_location=env.device, weights_only=False)
            if payload["format_version"] != 1 or payload["task_id"] != TASK_ID:
                raise ValueError("Checkpoint task or format mismatch")
            actor = Actor(env.task.policy_dim, JOINTS).to(env.device)
            actor.load_state_dict(payload["modules"]["actor"])
            result = evaluate(env, actor.eval(), args.clip_index, args.eval_steps)
            metadata = {
                "checkpoint": str(args.evaluate.resolve()),
                "clip_index": args.clip_index,
                "clip": provenance["motion_files"][args.clip_index],
                "clip_sha256": lib.sha256[args.clip_index],
                "nominal": args.nominal,
                "residual_scale_rad": env.task.scale.tolist(),
                "seed": args.seed,
            }
            write_json(args.output / "evaluation.json", {**metadata, "metrics": result})
            print("TRACKING_EVALUATION", json.dumps(result), flush=True)
            return
        write_json(args.output / "provenance.json", provenance)
        recipe = SuperMimicConfig(
            seed=args.seed, max_iterations=args.iterations, buffer_steps=256, warmup_steps=64,
            wm_only_iters=args.wm_only_iters, wm_window=8, wm_batches=args.wm_batches, wm_batch=args.wm_batch,
            wm_hidden=(384, 384, 384), pi_batch=args.pi_batch, horizon=IMAGINATION_STEPS, critic_epochs=5,
            action_noise=0.12, failure_warmup_iters=10, amp=False, compile=False, compile_policy=False,
            save_interval=max(1, min(50, args.iterations)), log_interval=1,
        )
        engine = SuperMimicEngine(env.task, recipe, args.output)
        engine.learn()
        proof = {"sim_transitions": engine.sim_steps, "optimizer_steps": optimizer_steps(engine),
                 "gradients": engine.last_gradients, "train_envs": engine.buffer.train_end,
                 "holdout_envs": env.num_envs - engine.buffer.train_end}
        write_json(args.output / "training_proof.json", proof)
        if not all(proof["optimizer_steps"].values()) or engine.last_gradients["actor"] <= 0:
            raise RuntimeError(f"Incomplete World Model / Actor / Critic updates: {proof}")
        print("TRACKING_TRAINING_PROOF", json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
