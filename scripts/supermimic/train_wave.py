"""Collect real Newton transitions, fit an 8-step world model, and optimize a 32-step actor."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch
from dexgym.algorithms.supermimic.imagination import critic_td_lambda_loss, imagined_rollout
from dexgym.algorithms.supermimic.networks import DeltaWorldModel, GoalActor, PrivilegedCritic
from dexgym.algorithms.supermimic.replay_buffer import MotionReplay
from dexgym.tasks.marvin_supermimic.env_cfg import MarvinWaveDirectCfg
from isaaclab.app import launch_simulation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preflight import REFERENCE, REFERENCE_SHA256, SOURCE, verify


def state_of(native):
    q = native.robot.data.joint_pos.torch[:, native.arm_ids]
    v = native.robot.data.joint_vel.torch[:, native.arm_ids]
    return torch.cat((q, v, native.command), dim=-1).detach().clone()


def collect(env, actor, replay, steps, teacher_fraction=0.6, noise=0.18, teacher_noise=0.0):
    native = env.unwrapped
    obs, _ = env.reset()
    start = time.perf_counter()
    terminations = 0
    rewards = []
    max_joint_speed = 0.0
    max_finger_drift = 0.0
    contact_steps = 0
    max_speed_event = None
    contact_events = []
    for iteration in range(steps):
        with torch.no_grad():
            state = state_of(native)
            history = obs["policy"][:, :560].reshape(native.num_envs, 10, 56).clone()
            frame = native.motion.frame.clone()
            actor_action = actor(history, obs["policy"][:, 560:].reshape(native.num_envs, 10, 90))
            actor_action = actor_action + torch.randn_like(actor_action) * noise
            next_frame = (frame + 1).clamp(max=native.motion.length - 1)
            next_q = native.motion.data["q_arm"][next_frame]
            teacher = torch.atanh(((next_q - native.action_map.origin) / native.action_map.scale).clamp(-0.98, 0.98))
            teacher = teacher + torch.randn_like(teacher) * teacher_noise
            choose_teacher = torch.rand(native.num_envs, 1, device=native.device) < teacher_fraction
            action = torch.where(choose_teacher, teacher, actor_action)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            replay.append(state, action, state_of(native), history, frame, done)
            speed_tensor = native.robot.data.joint_vel.torch[:, native.arm_ids].abs()
            speed = float(speed_tensor.max())
            if speed > max_joint_speed:
                flat = int(speed_tensor.reshape(-1).argmax())
                max_speed_event = {
                    "iteration": iteration,
                    "environment": flat // 14,
                    "joint": flat % 14,
                    "reference_frame": int(native.motion.frame[flat // 14]),
                }
                max_joint_speed = speed
            max_finger_drift = max(
                max_finger_drift,
                float(
                    (native.robot.data.joint_pos.torch[:, native.hand_ids] - native.home_full[native.hand_ids])
                    .abs()
                    .max()
                ),
            )
            contact = int(native.sim.physics_manager._solver.mjw_data.nacon.numpy()[0] > 0)
            contact_steps += contact
            if contact and len(contact_events) < 10:
                contact_events.append(
                    {
                        "iteration": iteration,
                        "reference_frame_min": int(native.motion.frame.min()),
                        "reference_frame_max": int(native.motion.frame.max()),
                    }
                )
            terminations += int(terminated.sum())
            rewards.append(float(reward.mean()))
    duration = time.perf_counter() - start
    return {
        "transitions": steps * native.num_envs,
        "seconds": duration,
        "samples_per_second": steps * native.num_envs / duration,
        "mean_physical_reward": float(np.mean(rewards)),
        "terminations": terminations,
        "teacher_fraction": teacher_fraction,
        "exploration_noise_std": noise,
        "teacher_noise_std": teacher_noise,
        "max_actual_arm_speed_rad_s": max_joint_speed,
        "max_finger_home_drift_rad": max_finger_drift,
        "newton_contact_steps": contact_steps,
        "max_speed_event": max_speed_event,
        "contact_events": contact_events,
    }


@torch.no_grad()
def model_errors(model, replay, mapping, fk, batch=32, env_range=None):
    result = {}
    for horizon in (1, 8, 32):
        window = replay.sample_windows(batch, horizon, env_range=env_range)
        predicted = window["state"][:, 0]
        baseline = predicted.clone()
        for step in range(horizon):
            predicted = model(predicted, window["action"][:, step], mapping)
            baseline = torch.cat(
                (
                    baseline[:, :14] + 0.02 * baseline[:, 14:28],
                    baseline[:, 14:28],
                    mapping(window["action"][:, step], baseline[:, 28:42]),
                ),
                dim=-1,
            )
        truth = window["next_state"][:, -1]
        pred_tcp, pred_r = fk(predicted[:, :14])
        true_tcp, true_r = fk(truth[:, :14])
        base_tcp, _ = fk(baseline[:, :14])
        from dexgym.algorithms.supermimic.rewards import so3_geodesic

        result[str(horizon)] = {
            "q_rmse_rad": float((predicted[:, :14] - truth[:, :14]).square().mean().sqrt()),
            "qd_rmse_rad_s": float((predicted[:, 14:28] - truth[:, 14:28]).square().mean().sqrt()),
            "tcp_position_mean_mm": float((pred_tcp[:, :, 2] - true_tcp[:, :, 2]).norm(dim=-1).mean() * 1000),
            "tcp_orientation_mean_deg": float(so3_geodesic(pred_r[:, :, 2], true_r[:, :, 2]).mean() * 180 / np.pi),
            "constant_velocity_tcp_mean_mm": float((base_tcp[:, :, 2] - true_tcp[:, :, 2]).norm(dim=-1).mean() * 1000),
        }
    return result


@torch.no_grad()
def actor_command_error(actor, replay, mapping, motion, env_range, batch=128):
    sample = replay.sample_starts(batch, env_range=env_range)
    frame = sample["frame"]
    predicted = actor(sample["history"], motion.goal_at(frame))
    next_q = motion.data["q_arm"][(frame + 1).clamp(max=motion.length - 1)]
    teacher = torch.atanh(((next_q - mapping.origin) / mapping.scale).clamp(-0.98, 0.98))
    previous = sample["state"][:, 28:42]
    difference = (mapping(predicted, previous) - mapping(teacher, previous)).abs()
    return {
        "mean_rad": float(difference.mean()),
        "p95_rad": float(torch.quantile(difference.flatten(), 0.95)),
        "max_rad": float(difference.max()),
    }


def evaluate_full(actor, num_envs=1, seed=140116, output_dir: Path | None = None):
    cfg = MarvinWaveDirectCfg()
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = 28.0
    cfg.random_reference_start = False
    cfg.reset_joint_perturb_rad = 0.0
    cfg.seed = seed
    start = time.perf_counter()
    with launch_simulation(cfg):
        env = gym.make("DexGym-Marvin-SuperMimic-Wave-Direct-v0", cfg=cfg)
        try:
            obs, _ = env.reset()
            native = env.unwrapped
            model_order = torch.argsort(native.model_to_newton)
            physical_q = [native.robot.data.joint_pos.torch[:, model_order].detach().cpu().numpy().copy()]
            physical_qd = [native.robot.data.joint_vel.torch[:, model_order].detach().cpu().numpy().copy()]
            errors_p, errors_r, rewards = [], [], []
            terminations = 0
            first_failure_step = None
            contact_steps = 0
            saturations = 0
            torque_samples = 0
            command_jump_max = 0.0
            for control_step in range(1356):
                with torch.no_grad():
                    action = actor(
                        obs["policy"][:, :560].reshape(num_envs, 10, 56),
                        obs["policy"][:, 560:].reshape(num_envs, 10, 90),
                    )
                    obs, reward, terminated, _truncated, _ = env.step(action)
                    metrics = env.unwrapped.last_tracking
                    physical_q.append(native.last_physical_q[:, model_order].cpu().numpy().copy())
                    physical_qd.append(native.last_physical_qd[:, model_order].cpu().numpy().copy())
                    contact_steps += int(native.sim.physics_manager._solver.mjw_data.nacon.numpy()[0] > 0)
                    applied = native.last_physical_torque[:, native.arm_ids].abs()
                    effort = native.robot.data.joint_effort_limits.torch[:, native.arm_ids]
                    saturations += int((applied >= effort - 1e-3).sum())
                    torque_samples += applied.numel()
                    command_jump_max = max(
                        command_jump_max, float((native.command - native.previous_command).abs().max())
                    )
                    errors_p.append(metrics["tcp_position_error_m"].detach().cpu().numpy() * 1000)
                    errors_r.append(metrics["tcp_rotation_error_rad"].detach().cpu().numpy() * 180 / np.pi)
                    rewards.append(float(reward.mean()))
                    terminations += int(terminated.sum())
                    if bool(terminated.any()):
                        first_failure_step = control_step + 1
                        break
            p = np.stack(errors_p)
            r = np.stack(errors_r)
            q_record = np.stack(physical_q)
            qd_record = np.stack(physical_qd)
            source = mujoco.MjModel.from_xml_path(str(SOURCE))
            source_data = mujoco.MjData(source)
            source_contact_frames = 0
            for q in q_record[:, 0]:
                source_data.qpos[:] = q
                mujoco.mj_forward(source, source_data)
                source_contact_frames += int(source_data.ncon > 0)
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output_dir / "newton_policy_states.npz",
                    time=np.arange(len(q_record)) * 0.02,
                    q=q_record,
                    qd=qd_record,
                    tcp_position_error_mm=np.concatenate((np.zeros((1, num_envs, 2)), p), axis=0),
                    tcp_orientation_error_deg=np.concatenate((np.zeros((1, num_envs, 2)), r), axis=0),
                )
            ref = native.motion.data["q_arm"][: len(q_record)].detach().cpu().numpy()
            arm_source_ids = [
                mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_JOINT, name) for name in native.motion.joint_names_arm
            ]
            return {
                "completed_control_steps": len(errors_p),
                "wall_time_s": time.perf_counter() - start,
                "terminations": terminations,
                "first_failure_control_step": first_failure_step,
                "full_motion_completed": first_failure_step is None and len(errors_p) == 1356,
                "reward_mean": float(np.mean(rewards)),
                "newton_contact_control_steps": contact_steps,
                "source_mjcf_contact_frames": source_contact_frames,
                "torque_saturation_rate": saturations / max(1, torque_samples),
                "max_command_jump_rad": command_jump_max,
                "max_actual_arm_speed_rad_s": float(np.max(abs(qd_record[:, :, arm_source_ids]))),
                "max_actual_arm_accel_rad_s2": float(
                    np.max(abs(np.diff(qd_record[:, :, arm_source_ids], axis=0) / 0.02))
                ),
                "joint_position_rmse_rad": float(np.mean(np.square(q_record[:, 0, arm_source_ids] - ref)) ** 0.5),
                "first_position_over_50mm_control_step": int(np.argmax((p > 50).any(axis=(1, 2))) + 1)
                if (p > 50).any()
                else None,
                "left_position_mm": {
                    k: float(v)
                    for k, v in zip(
                        ("mean", "p95", "max"), (p[..., 0].mean(), np.percentile(p[..., 0], 95), p[..., 0].max())
                    )
                },
                "right_position_mm": {
                    k: float(v)
                    for k, v in zip(
                        ("mean", "p95", "max"), (p[..., 1].mean(), np.percentile(p[..., 1], 95), p[..., 1].max())
                    )
                },
                "left_orientation_deg": {
                    k: float(v)
                    for k, v in zip(
                        ("mean", "p95", "max"), (r[..., 0].mean(), np.percentile(r[..., 0], 95), r[..., 0].max())
                    )
                },
                "right_orientation_deg": {
                    k: float(v)
                    for k, v in zip(
                        ("mean", "p95", "max"), (r[..., 1].mean(), np.percentile(r[..., 1], 95), r[..., 1].max())
                    )
                },
            }
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--collect-steps", type=int, default=320)
    parser.add_argument("--world-updates", type=int, default=100)
    parser.add_argument("--actor-updates", type=int, default=20)
    parser.add_argument("--actor-warmup-updates", type=int, default=0)
    parser.add_argument("--world-batch", type=int, default=128)
    parser.add_argument("--imag-batch", type=int, default=32)
    parser.add_argument("--world-hidden", type=int, default=2048)
    parser.add_argument("--teacher-fraction", type=float, default=0.6)
    parser.add_argument("--exploration-noise", type=float, default=0.18)
    parser.add_argument("--teacher-noise", type=float, default=0.0)
    parser.add_argument("--behavior-prior-weight", type=float, default=0.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("/home/ubuntu/dexgym_data/marvin_supermimic_wave_m4/e1_first")
    )
    args = parser.parse_args()
    preflight = verify()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(140116)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    actor = GoalActor().to(device)
    model = DeltaWorldModel(hidden=args.world_hidden).to(device)
    critic = PrivilegedCritic().to(device)
    optim_model = torch.optim.Adam(model.parameters(), lr=3e-4)
    optim_actor = torch.optim.Adam(actor.parameters(), lr=1e-4)
    optim_critic = torch.optim.Adam(critic.parameters(), lr=3e-4)
    prior_training = None
    if args.resume:
        previous = torch.load(args.resume, map_location=device, weights_only=False)
        if previous["reference_sha256"] != REFERENCE_SHA256:
            raise ValueError("Cannot resume from a different reference")
        actor.load_state_dict(previous["actor"])
        model.load_state_dict(previous["world_model"])
        critic.load_state_dict(previous["critic"])
        optim_actor.load_state_dict(previous["actor_optimizer"])
        optim_model.load_state_dict(previous["world_optimizer"])
        optim_critic.load_state_dict(previous["critic_optimizer"])
        prior_training = previous["training"]
    cfg = MarvinWaveDirectCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 140116
    cfg.reset_joint_perturb_rad = 0.01
    started = time.perf_counter()
    with launch_simulation(cfg):
        env = gym.make("DexGym-Marvin-SuperMimic-Wave-Direct-v0", cfg=cfg)
        try:
            native = env.unwrapped
            replay = MotionReplay(max(128, args.collect_steps + 32), args.num_envs, device)
            train_envs = max(1, int(args.num_envs * 0.8))
            fk = native.fk
            mapping = native.action_map
            motion = native.motion
            collection = collect(
                env,
                actor,
                replay,
                args.collect_steps,
                teacher_fraction=args.teacher_fraction,
                noise=args.exploration_noise,
                teacher_noise=args.teacher_noise,
            )
            model_losses = []
            for _ in range(args.world_updates):
                window = replay.sample_windows(args.world_batch, 8, env_range=(0, train_envs))
                optim_model.zero_grad()
                loss, _ = model.autoregressive_loss(window, mapping)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optim_model.step()
                model_losses.append(float(loss.detach()))
            prediction = model_errors(model, replay, mapping, fk, env_range=(train_envs, args.num_envs))
            print("world model validation", json.dumps(prediction), flush=True)
            trust_gate = (
                prediction["8"]["tcp_position_mean_mm"] < 50
                and prediction["32"]["tcp_position_mean_mm"] < 100
                and collection["newton_contact_steps"] == 0
                and collection["max_actual_arm_speed_rad_s"] < 5
            )
            actor_losses, critic_losses = [], []
            warmup_losses = []
            warmup_action_error = None
            final_action_error = None
            if trust_gate:
                for _ in range(args.actor_warmup_updates):
                    start_batch = replay.sample_starts(args.imag_batch, env_range=(0, train_envs))
                    goal = motion.goal_at(start_batch["frame"])
                    target_q = motion.data["q_arm"][(start_batch["frame"] + 1).clamp(max=motion.length - 1)]
                    teacher_action = torch.atanh(((target_q - mapping.origin) / mapping.scale).clamp(-0.98, 0.98))
                    optim_actor.zero_grad()
                    predicted_action = actor(start_batch["history"], goal)
                    warmup_loss = torch.nn.functional.smooth_l1_loss(predicted_action, teacher_action)
                    warmup_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    optim_actor.step()
                    warmup_losses.append(float(warmup_loss.detach()))
                warmup_action_error = actor_command_error(actor, replay, mapping, motion, (train_envs, args.num_envs))
                torch.save(actor.state_dict(), args.output / "actor_after_auxiliary_warmup.pt")
                prior_actor = copy.deepcopy(actor).eval()
                for parameter in prior_actor.parameters():
                    parameter.requires_grad_(False)
                for _ in range(args.actor_updates):
                    start_batch = replay.sample_starts(args.imag_batch, env_range=(0, train_envs))
                    for module in (model, critic):
                        for parameter in module.parameters():
                            parameter.requires_grad_(False)
                    optim_actor.zero_grad()
                    rollout = imagined_rollout(
                        actor,
                        model,
                        critic,
                        fk,
                        mapping,
                        motion,
                        start_batch,
                        horizon=32,
                        prior_actor=prior_actor,
                        prior_weight=args.behavior_prior_weight,
                    )
                    rollout["actor_loss"].backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    optim_actor.step()
                    actor_losses.append(float(rollout["actor_loss"].detach()))
                    for parameter in critic.parameters():
                        parameter.requires_grad_(True)
                    optim_critic.zero_grad()
                    critic_loss = critic_td_lambda_loss(critic, motion, rollout)
                    critic_loss.backward()
                    optim_critic.step()
                    critic_losses.append(float(critic_loss.detach()))
                    for parameter in model.parameters():
                        parameter.requires_grad_(True)
                final_action_error = actor_command_error(actor, replay, mapping, motion, (train_envs, args.num_envs))
            training = {
                "reference_sha256": REFERENCE_SHA256,
                "reference": str(REFERENCE),
                "preflight": preflight,
                "resume_from": str(args.resume) if args.resume else None,
                "resume_recollected_replay": bool(args.resume),
                "prior_training_updates": {
                    k: prior_training.get(k, 0) for k in ("world_updates", "actor_updates", "critic_updates")
                }
                if prior_training
                else None,
                "configuration": vars(args) | {"output": str(args.output)},
                "collection": collection,
                "world_model_validation_env_ids": [train_envs, args.num_envs - 1],
                "replay_bytes": replay.bytes_allocated,
                "world_updates": len(model_losses),
                "actor_updates": len(actor_losses),
                "actor_auxiliary_warmup_updates": len(warmup_losses),
                "actor_auxiliary_warmup_loss_first_last": [warmup_losses[0], warmup_losses[-1]]
                if warmup_losses
                else None,
                "actor_command_error_after_auxiliary_warmup": warmup_action_error,
                "actor_command_error_after_bptt": final_action_error,
                "critic_updates": len(critic_losses),
                "world_loss_first_last": [model_losses[0], model_losses[-1]],
                "actor_loss_first_last": [actor_losses[0], actor_losses[-1]] if actor_losses else None,
                "world_prediction": prediction,
                "world_model_trust_gate": trust_gate,
                "train_wall_time_s": time.perf_counter() - started,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            }
        finally:
            env.close()
    checkpoint = {
        "actor": actor.state_dict(),
        "world_model": model.state_dict(),
        "critic": critic.state_dict(),
        "actor_optimizer": optim_actor.state_dict(),
        "world_optimizer": optim_model.state_dict(),
        "critic_optimizer": optim_critic.state_dict(),
        "training": training,
        "reference_sha256": REFERENCE_SHA256,
        "replay_saved": False,
        "resume_requires_recollection": True,
    }
    torch.save(checkpoint, args.output / "last.pt")
    if training["actor_updates"]:
        training["evaluation"] = evaluate_full(actor, output_dir=args.output)
        checkpoint["training"] = training
        torch.save(checkpoint, args.output / "best.pt")
    (args.output / "metrics.json").write_text(json.dumps(training, indent=2))
    print(json.dumps(training, indent=2))


if __name__ == "__main__":
    main()
