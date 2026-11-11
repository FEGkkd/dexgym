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
from dexgym.assets.marvin_wuji.robot_cfg import PACKAGED_MJCF

from .diagnostics import diagnose_model
from .imitation import PhysicalTeacher
from .model import initial_perturbation, sha256, validate_drive_only_model
from .native import FAILURE_CAUSES, PHYSICS_DT, NewtonTrackingEnv, simulation_cfg
from .task import DT, FUTURE_FRAMES, IMAGINATION_STEPS, JOINTS, MOTION_LATENT_DIM, TASK_ID, MotionLibrary


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@torch.no_grad()
def check_contract(env):
    task, lib = env.task, env.task.library
    ids = torch.arange(env.num_envs, device=env.device)
    frames = torch.linspace(0, lib.frames - 2, env.num_envs, device=env.device).long()
    obs, _ = env.reset(frame=frames)
    initial_history = task.history_from_obs(task.replay_observation(obs["policy"]))
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
    if not torch.equal(initial_history, initial_history[:, -1:].expand_as(initial_history)):
        raise RuntimeError("Reset history must contain copies of the initial observation")
    memory = env.memory
    action = torch.zeros(env.num_envs, JOINTS, device=env.device)
    if task.action_mode == "home":
        home_target, _ = task.control_target(action, memory, task.encoder_bias)
        errors["home_target_rad"] = float((home_target - task.home).abs().max())
        action = task.baseline_action(memory)
        target, _ = task.control_target(action, memory, task.encoder_bias)
        errors["reference_pd_target_rad"] = float((target - lib.qpos[lib.at(memory["frame"])]).abs().max())
        upper, _ = task.control_target(torch.ones_like(action), memory, task.encoder_bias)
        lower, _ = task.control_target(-torch.ones_like(action), memory, task.encoder_bias)
        errors["upper_target_rad"] = float((upper - task.hi).abs().max())
        errors["lower_target_rad"] = float((lower - task.lo).abs().max())
        if max(errors[k] for k in ("home_target_rad", "reference_pd_target_rad", "upper_target_rad", "lower_target_rad")) > 1e-6:
            raise RuntimeError(f"Fixed Home target mapping mismatch: {errors}")
    obs, reward, _, _, events = env.step(action)
    if not reward.isfinite().all() or (events["failure_bits"] != 0).any():
        raise RuntimeError("Reference phase test failed during the first physical step")
    if not torch.equal(env.memory["frame"], lib.at(memory["frame"])):
        raise RuntimeError("Physical phase failed to advance by one reference frame")
    stepped_history = task.history_from_obs(task.replay_observation(obs["policy"]))
    if not torch.equal(stepped_history[:, :-1], initial_history[:, 1:]):
        raise RuntimeError("Physical observation history failed to advance by one frame")
    env.reset(ids[-1:], frame=lib.end[frames[-1:]] - 1)
    if not torch.equal(env.history[:-1], stepped_history[:-1]):
        raise RuntimeError("A partial reset changed another environment's observation history")
    step_limit = env.max_episode_steps
    try:
        if task.first_event:
            env.reset(ids[:1], frame=lib.start[:1])
            env.max_episode_steps = 1
        action = task.baseline_action(env.memory)
        _, _, terminated, truncated, events = env.step(action)
        if not events["reference_complete"][-1] or bool(terminated[-1]) != task.first_event:
            raise RuntimeError("The final reference step must complete independently of safety failure")
        if task.first_event and (terminated[0] or not truncated[0] or not events["time_limit"][0] or truncated[-1]):
            raise RuntimeError("A training cap must truncate without replacing clip completion")
        if not task.first_event and not truncated[-1]:
            raise RuntimeError("The legacy contract must truncate at the final reference step")
    finally:
        env.max_episode_steps = step_limit
    return errors


@torch.no_grad()
def evaluate(env, actor, clip_index, steps, *, perturb_seed=None, trace_dir=None):
    if env.num_envs != 2:
        raise ValueError("Evaluation requires two environments for a matched rollout")
    task, lib = env.task, env.task.library
    frame = lib.start[clip_index]
    max_steps = int(lib.end[frame] - frame)
    count = max_steps if steps is None else min(max_steps, steps)
    obs, _ = env.reset(frame=frame.repeat(2), **initial_perturbation(perturb_seed, JOINTS))
    active = torch.ones(2, dtype=torch.bool, device=env.device)
    end_step = torch.zeros(2, dtype=torch.long, device=env.device)
    complete_intervals = torch.zeros_like(end_step)
    endings = {
        key: torch.zeros_like(end_step)
        for key in ("failure_bits", "safety_substep", "reference_complete", "time_limit")
    }
    failure_state = torch.zeros_like(env.state())
    failure_wrist = failure_state.new_zeros((2, 2))
    records = {}
    if trace_dir is not None:
        trace = {"state": [env.state().clone()], "target": [], "action": [], "depth_m": [], "substep": []}
    baseline_name = task.baseline_name
    for i in range(count):
        m = env.memory
        action = actor(obs["policy"])
        action[~active] = 0.0
        action[0] = task.baseline_action(m)[0]
        obs, reward, terminated, truncated, events = env.step(action)
        if trace_dir is not None:
            trace["state"].append(env.state().clone())
            trace["target"].append(env.transition["u"].clone())
            trace["action"].append(action.clone())
            trace["depth_m"].append(env.transition["depth"].clone())
            trace["substep"].append(events["safety_substep"].clone())
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
            sample = active if key in ("penetration_mm", "reward") else active & events["full_step"]
            records.setdefault(key, []).append(value.masked_fill(~sample[:, None], float("nan")))
        complete_intervals += active & events["full_step"]
        ended = active & (terminated | truncated)
        failed = ended & (events["failure_bits"] != 0)
        failure_state = torch.where(failed[:, None], env.state(), failure_state)
        failure_wrist = torch.where(failed[:, None], values["wrist_mm"], failure_wrist)
        end_step = torch.where(ended, i + 1, end_step)
        for key, previous in endings.items():
            endings[key] = torch.where(ended, events[key], previous)
        active &= ~(terminated | truncated)
        if not active.any():
            break
        ids = (~active).nonzero(as_tuple=True)[0]
        if len(ids):
            obs, _ = env.reset(ids, frame=lib.at(frame.repeat(len(ids)), i + 1))
    result = {}
    arrays = {k: torch.stack(v).cpu().numpy() for k, v in records.items()}
    for index, name in enumerate((baseline_name, "policy")):
        bits = int(endings["failure_bits"][index])
        failed = bits != 0
        completed = bool(endings["reference_complete"][index])
        event_substep = int(endings["safety_substep"][index])
        event_step = int(end_step[index])
        entry = {
            "requested_steps": count, "full_clip_steps": max_steps, "terminated": failed,
            "failed": failed, "episode_terminated": failed or (task.first_event and completed),
            "reference_complete": completed,
            "time_limit": bool(endings["time_limit"][index]),
            "failure_causes": [cause for bit, cause in FAILURE_CAUSES.items() if bits & bit],
            "failure_control_step": event_step if failed else None,
            "failure_substep": event_substep or None,
            "failure_time_s": ((event_step - 1) * DT + (event_substep * PHYSICS_DT if event_substep else DT))
            if failed else None,
            "complete_control_steps": int(complete_intervals[index]),
        }
        if failed and torch.isfinite(failure_state[index]).all():
            entry["failure_state"] = {
                "joint_position_rad": failure_state[index, :JOINTS].tolist(),
                "joint_velocity_rad_s": failure_state[index, JOINTS:].tolist(),
            }
            entry["failure_wrist_error_mm"] = failure_wrist[index].tolist()
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
        entry["per_side"] = {}
        for side, label in enumerate(("left", "right")):
            group = {}
            for key in ("wrist_mm", "wrist_deg", "elbow_mm", "tip_mm", "tip_deg"):
                values = arrays[key][:, index]
                values = values[np.isfinite(values).all(axis=-1)]
                if values.size:
                    part = np.split(values, 2, axis=1)[side]
                    group[key] = {"mean": float(part.mean()), "p95": float(np.percentile(part, 95))}
            entry["per_side"][label] = group
        result[name] = entry
    matched = np.isfinite(arrays["reward"]).all(axis=(1, 2))
    if matched.any():
        result["matched_steps"] = int(matched.sum())
        for i, name in enumerate((baseline_name, "policy")):
            shared = {}
            for key, values in arrays.items():
                valid = matched & np.isfinite(values).all(axis=(1, 2))
                if valid.any():
                    shared[key] = float(values[valid, i].mean())
            result[name]["matched_mean"] = shared
    if trace_dir is not None:
        trace = {key: torch.stack(value).cpu().numpy() for key, value in trace.items()}
        for i, name in enumerate((baseline_name, "policy")):
            n = result[name]["executed_steps"]
            time_s = np.arange(n + 1, dtype=float) * DT
            if trace["substep"][n - 1, i]:
                time_s[-1] = (n - 1) * DT + trace["substep"][n - 1, i] * PHYSICS_DT
            np.savez_compressed(
                trace_dir / f"{name}.npz", qpos=trace["state"][:n + 1, i, :JOINTS],
                qvel=trace["state"][:n + 1, i, JOINTS:], time_s=time_s,
                **{k: trace[k][:n, i] for k in ("target", "action", "depth_m")},
            )
    return result


def optimizer_steps(engine):
    return {
        name: max((int(state["step"]) for state in opt.state.values() if "step" in state), default=0)
        for name, opt in engine.optimizers.items()
    }


def replay_coverage(engine):
    buffer = engine.buffer
    data = buffer.data
    train_roles = engine.task.collection_roles[:buffer.train_end]
    valid = data["valid"][:buffer.size, :buffer.train_end]
    failure = data["safety_failure"][:buffer.size, :buffer.train_end]
    full = data["full_step"][:buffer.size, :buffer.train_end]
    result = {}
    for i, name in enumerate(("policy", "reference_pd", "perturbed_reference_pd")):
        selected = valid & (train_roles == i)
        result[name] = {
            "environments": int((train_roles == i).sum()),
            "valid_transitions": int(selected.sum()),
            "safety_failures": int((selected & failure).sum()),
            "partial_intervals": int((selected & ~full).sum()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, action="append", type=Path)
    parser.add_argument("--model-xml", type=Path, help="Packaged Hand2.2 with updated Marvin arm position servos")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--wm-only-iters", type=int, default=30)
    parser.add_argument("--wm-batches", type=int, default=2)
    parser.add_argument("--wm-batch", type=int, default=192)
    parser.add_argument("--wm-width", type=int, default=2048)
    parser.add_argument("--pi-batch", type=int, default=32)
    parser.add_argument("--buffer-steps", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--action-noise", type=float)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--skip-model-probe", action="store_true")
    parser.add_argument("--teacher", type=Path, help="Single-clip teaching trajectory used during training")
    parser.add_argument("--bc-steps", type=int, default=5000)
    parser.add_argument("--init-actor", type=Path, help="Reuse Actor initialization from the same task and motion")
    parser.add_argument("--actor-lr", type=float)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--teacher-anchor", choices=("imagined", "reference"), default="imagined")
    parser.add_argument("--actor-output-only", action="store_true")
    parser.add_argument("--control-error-features", action="store_true")
    parser.add_argument("--collection-noise-mix", action="store_true")
    parser.add_argument("--imagination-steps", type=int, default=IMAGINATION_STEPS)
    parser.add_argument("--model-probe-only", action="store_true")
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--perturb-seed", type=int, help="Match the native MuJoCo evaluation reset perturbations")
    parser.add_argument("--save-eval-trace", action="store_true")
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--nominal", choices=("home", "reference", "feedforward"))
    parser.add_argument("--policy-arch", choices=("motion-encoder", "flat"))
    parser.add_argument("--termination-model", choices=("transition", "legacy-state"))
    parser.add_argument("--task-contract", choices=("first-event", "control-boundary"))
    parser.add_argument("--tracking-failure-m", type=float)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--reward-profile", choices=("multiscale", "precision"))
    parser.add_argument("--matmul-precision", choices=("highest", "high"))
    args = parser.parse_args()
    if args.num_envs < 2 or not 0 <= args.wm_only_iters < args.iterations:
        parser.error("Need at least two environments and at least one Actor iteration")
    if min(args.wm_batches, args.wm_batch, args.pi_batch, args.wm_width) < 1 or (
        args.eval_steps is not None and args.eval_steps < 1
    ):
        parser.error("Batch sizes and evaluation steps must be positive")
    if min(args.warmup_steps, args.buffer_steps) < IMAGINATION_STEPS:
        parser.error(f"Warmup and replay capacity need at least {IMAGINATION_STEPS} steps")
    if min(args.log_interval, args.save_interval) < 1:
        parser.error("Log and checkpoint intervals must be positive")
    if args.action_noise is not None and (not np.isfinite(args.action_noise) or args.action_noise < 0):
        parser.error("Action noise must be finite and nonnegative")
    if args.check_only and args.evaluate:
        parser.error("Choose check-only or evaluate")
    if args.model_probe_only and not args.evaluate:
        parser.error("--model-probe-only requires --evaluate")
    if args.model_probe_only and args.skip_model_probe:
        parser.error("Cannot skip the model probe in probe-only mode")
    if (args.perturb_seed is not None or args.save_eval_trace) and not args.evaluate:
        parser.error("Evaluation reset perturbation and tracing require --evaluate")
    if args.teacher and (args.evaluate or args.check_only or (not args.init_actor and args.bc_steps <= 0)):
        parser.error("--teacher requires training with Actor initialization")
    if (args.init_actor or args.actor_output_only) and not args.teacher:
        parser.error("These Actor initialization options require --teacher")
    if not 1 <= args.imagination_steps <= IMAGINATION_STEPS:
        parser.error(f"--imagination-steps must be in [1, {IMAGINATION_STEPS}]")
    if not np.isfinite(args.teacher_weight) or args.teacher_weight < 0:
        parser.error("--teacher-weight must be finite and nonnegative")
    if args.actor_lr is not None and (not np.isfinite(args.actor_lr) or args.actor_lr <= 0):
        parser.error("--actor-lr must be positive and finite")
    if not 0 <= args.clip_index < len(args.reference):
        parser.error("Clip index is outside the Motion Library")
    if args.max_episode_steps is not None and args.max_episode_steps < 1:
        parser.error("A training episode cap must be positive")
    original = None
    defaults = {"nominal": "home", "policy_arch": "motion-encoder", "termination_model": "transition",
                "reward_profile": "multiscale", "matmul_precision": "high", "task_contract": "first-event",
                "tracking_failure_m": 0.25}
    if args.evaluate:
        original = json.loads((args.evaluate.parent / "provenance.json").read_text())
        if original.get("contact_model") == "hard-contact":
            parser.error("The experimental three-contact checkpoint requires DexGym commit 3147228")
        legacy = {"nominal": "reference", "policy_arch": "flat", "termination_model": "legacy-state",
                  "reward_profile": "precision", "matmul_precision": "highest", "task_contract": "control-boundary",
                  "tracking_failure_m": None}
        for key, value in legacy.items():
            original.setdefault(key, value)
        defaults.update({key: original[key] for key in defaults})
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    first_event = args.task_contract == "first-event"
    if first_event and (
        args.termination_model != "transition" or args.tracking_failure_m is None
        or not np.isfinite(args.tracking_failure_m) or args.tracking_failure_m <= 0
    ):
        parser.error("First-event tracking requires transition termination and a positive finite wrist threshold")
    if (args.control_error_features or args.collection_noise_mix) and not first_event:
        parser.error("Control-error features and mixed exploration require first-event transition semantics")
    tracking_failure_m = args.tracking_failure_m if first_event else None
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    cfg = simulation_cfg()
    xml = args.model_xml or PACKAGED_MJCF
    if args.model_xml:
        validate_drive_only_model(PACKAGED_MJCF, xml)
    model = mujoco.MjModel.from_xml_path(str(xml))
    lib = MotionLibrary(args.reference, model, PACKAGED_MJCF, cfg.device)
    provenance = {"task_id": TASK_ID, "model_sha256": sha256(xml), "motion_sha256": lib.sha256,
                  "reference_model_sha256": lib.model_sha256, "model_xml": str(xml.resolve()),
                  "motion_files": [str(path.resolve()) for path in args.reference], "frames": lib.frames,
                  "joint_names": lib.names, "seed": args.seed, "backend": "Isaac Lab 3 / Newton MJWarp",
                  "nominal": args.nominal, "num_envs": 2 if args.evaluate else args.num_envs,
                  "policy_arch": args.policy_arch, "matmul_precision": args.matmul_precision,
                  "termination_model": args.termination_model, "reward_profile": args.reward_profile,
                  "task_contract": args.task_contract, "tracking_failure_m": tracking_failure_m,
                  "training_episode_steps": args.max_episode_steps if not args.evaluate else None}
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.evaluate and not args.check_only and (args.output / "metrics.jsonl").exists():
        parser.error("Select a fresh output directory for training")
    with launch_simulation(cfg):
        env = NewtonTrackingEnv(
            cfg, 2 if args.evaluate else args.num_envs, model, lib, args.nominal,
            args.policy_arch, args.termination_model, args.reward_profile, first_event=first_event,
            tracking_failure_m=tracking_failure_m, max_episode_steps=None if args.evaluate else args.max_episode_steps,
        )
        provenance.update(history_frames=env.task.history_steps, policy_dim=env.task.policy_dim,
                          future_frames=FUTURE_FRAMES if env.task.goal_dim else 0,
                          motion_latent_dim=MOTION_LATENT_DIM if env.task.goal_dim else 0)
        if args.nominal == "home":
            mapping = {
                "name": "fixed_home_asymmetric_joint_position" if first_event else "fixed_home_joint_position",
                "home_rad": env.task.home.tolist(),
                "lower_rad": env.task.lo.tolist(), "upper_rad": env.task.hi.tolist(),
            }
            if first_event:
                mapping.update(lower_travel_rad=env.task.lower_travel.tolist(),
                               upper_travel_rad=env.task.upper_travel.tolist())
            else:
                mapping["scale_rad"] = env.task.scale.tolist()
            provenance["action_mapping"] = mapping
        contract = check_contract(env)
        write_json(args.output / "contract.json", contract)
        print("TRACKING_CONTRACT", json.dumps(contract), flush=True)
        if args.check_only:
            return
        if original is not None:
            if any(original[k] != provenance[k]
                   for k in ("task_id", "model_sha256", "motion_sha256", "joint_names", "nominal", "policy_arch",
                             "termination_model", "reward_profile", "task_contract", "tracking_failure_m")):
                raise ValueError("Checkpoint reference or asset provenance mismatch")
            if original.get("action_mapping") != provenance.get("action_mapping"):
                raise ValueError("Checkpoint action mapping mismatch")
            payload = torch.load(args.evaluate, map_location=env.device, weights_only=False)
            if payload["format_version"] != 1 or payload["task_id"] != TASK_ID:
                raise ValueError("Checkpoint task or format mismatch")
            actor = env.task.make_actor().to(env.device)
            actor.load_state_dict(payload["modules"]["actor"])
            actor.eval()
            metadata = {
                "checkpoint": str(args.evaluate.resolve()),
                "clip_index": args.clip_index,
                "clip": provenance["motion_files"][args.clip_index],
                "clip_sha256": lib.sha256[args.clip_index],
                "nominal": args.nominal,
                "policy_arch": args.policy_arch,
                "termination_model": args.termination_model,
                "task_contract": args.task_contract,
                "tracking_failure_m": tracking_failure_m,
                "reward_profile": args.reward_profile,
                "matmul_precision": args.matmul_precision,
                "wm_hidden": payload["config"]["wm_hidden"],
                "action_mapping": provenance.get("action_mapping"),
                "residual_scale_rad": env.task.scale.tolist(),
                "seed": args.seed,
                "perturb_seed": args.perturb_seed,
            }
            if not args.skip_model_probe:
                probe = diagnose_model(env, actor, payload, args.clip_index)
                write_json(args.output / "model_probe.json", {**metadata, **probe})
                print("TRACKING_MODEL_PROBE", json.dumps(probe["counts"]), flush=True)
            if args.model_probe_only:
                return
            result = evaluate(env, actor, args.clip_index, args.eval_steps, perturb_seed=args.perturb_seed,
                              trace_dir=args.output if args.save_eval_trace else None)
            write_json(args.output / "evaluation.json", {**metadata, "metrics": result})
            print("TRACKING_EVALUATION", json.dumps(result), flush=True)
            return
        if args.teacher:
            provenance["actor_initialization"] = dict(
                teacher=str(args.teacher.resolve()), teacher_sha256=sha256(args.teacher),
                bc_steps=0 if args.init_actor else args.bc_steps, regularization_weight=args.teacher_weight,
                checkpoint=str(args.init_actor.resolve()) if args.init_actor else None,
                teacher_anchor=args.teacher_anchor, output_layer_only=args.actor_output_only,
                freeze_observation_norm=True,
            )
        provenance["collection_noise_mix"] = args.collection_noise_mix
        env.task.collection_noise_mix = args.collection_noise_mix
        write_json(args.output / "provenance.json", provenance)
        default_noise = 0.045 if args.nominal == "home" else 0.12
        recipe = SuperMimicConfig(
            seed=args.seed, max_iterations=args.iterations, buffer_steps=args.buffer_steps, warmup_steps=args.warmup_steps,
            wm_only_iters=args.wm_only_iters, wm_window=8, wm_batches=args.wm_batches, wm_batch=args.wm_batch,
            wm_hidden=(args.wm_width,) * 3, isolate_wm_init=True, pi_batch=args.pi_batch,
            horizon=args.imagination_steps, critic_epochs=5, control_error_features=args.control_error_features,
            actor_lr=args.actor_lr,
            action_noise=default_noise if args.action_noise is None else args.action_noise,
            failure_warmup_iters=0 if env.task.transition_termination else 10,
            failure_clip=1.0 if env.task.transition_termination else 0.1,
            amp=False, compile=False, compile_policy=False,
            save_interval=args.save_interval, log_interval=args.log_interval,
        )
        engine = SuperMimicEngine(env.task, recipe, args.output)
        if args.teacher:
            teacher = PhysicalTeacher(env.task, args.teacher, provenance["model_sha256"])
            if args.init_actor:
                origin = json.loads((args.init_actor.parent / "provenance.json").read_text())
                if any(origin.get(k) != provenance[k] for k in (
                    "task_id", "model_sha256", "motion_sha256", "joint_names", "action_mapping",
                    "policy_arch", "task_contract",
                )):
                    raise ValueError("Actor initialization provenance differs from this task")
                initial = torch.load(args.init_actor, map_location=env.device, weights_only=False)
                if initial["format_version"] != 1 or initial["task_id"] != TASK_ID:
                    raise ValueError("Actor initialization task or format mismatch")
                engine.actor.load_state_dict(initial["modules"]["actor"])
                del initial
            else:
                losses = teacher.initialize(engine.actor, args.bc_steps)
                write_json(args.output / "actor_initialization.json", losses)
            if args.actor_output_only:
                engine.actor.requires_grad_(False)
                engine.actor.decoder[-1].requires_grad_(True)
            env.task.freeze_actor_norm = True
            if args.teacher_weight:
                if args.teacher_anchor == "reference":
                    env.task.actor_reference_anchor = lambda actor: args.teacher_weight * teacher.anchor(actor)
                else:
                    env.task.actor_regularization = lambda x, m, u: args.teacher_weight * teacher.penalty(x, m, u)
            engine.save(args.output / "model_bc.pt")
        engine.learn()
        proof = {"sim_transitions": engine.sim_steps, "optimizer_steps": optimizer_steps(engine),
                 "gradients": engine.last_gradients, "train_envs": engine.buffer.train_end,
                 "holdout_envs": env.num_envs - engine.buffer.train_end,
                 "peak_torch_memory_bytes": torch.cuda.max_memory_allocated(env.device)}
        if env.task.collection_roles is not None:
            proof["final_replay_coverage"] = replay_coverage(engine)
        write_json(args.output / "training_proof.json", proof)
        if not all(proof["optimizer_steps"].values()) or engine.last_gradients["actor"] <= 0:
            raise RuntimeError(f"Incomplete World Model / Actor / Critic updates: {proof}")
        print("TRACKING_TRAINING_PROOF", json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
