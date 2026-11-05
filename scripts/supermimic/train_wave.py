"""Collect real Newton transitions, fit an 8-step world model, and optimize a 32-step actor."""

import argparse
import copy
import json
import subprocess
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
from dexgym.algorithms.supermimic.rewards import REWARD_PROFILES
from dexgym.tasks.marvin_supermimic.env_cfg import MarvinWaveDirectCfg
from isaaclab.app import launch_simulation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preflight import REFERENCE, REFERENCE_SHA256, SOURCE, verify

PD_STATES = Path(
    "/home/ubuntu/dexgym_data/marvin_motion_library_v1/bimanual/physics_replays/newton_1env/newton_pd_states.npz"
)


def pd_actor_examples(motion, mapping, device):
    """Verified Newton PD states supply phase-balanced actor initialization examples."""
    source = mujoco.MjModel.from_xml_path(str(SOURCE))
    arm_ids = [mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_JOINT, name) for name in motion.joint_names_arm]
    with np.load(PD_STATES, allow_pickle=False) as replay:
        if replay["q"].shape != (motion.length, 1, 54):
            raise ValueError("Pinned PD replay does not match motion length or 54DoF model")
        q = torch.as_tensor(replay["q"][:, 0, arm_ids].copy(), device=device)
        qd = torch.as_tensor(replay["qd"][:, 0, arm_ids].copy(), device=device)
    reference = motion.data["q_arm"]
    raw = torch.atanh(((reference - mapping.origin) / mapping.scale).clamp(-0.98, 0.98))
    previous_action = raw.tanh().clone()
    previous_action[0] = 0
    frame = torch.cat(
        (
            (q - mapping.origin) / mapping.scale,
            qd / 2,
            previous_action,
            (reference - mapping.origin) / mapping.scale,
        ),
        dim=-1,
    )
    indices = (torch.arange(motion.length, device=device)[:, None] + torch.arange(-9, 1, device=device)[None]).clamp(
        min=0
    )
    history = frame[indices]
    goal = motion.goal_at(torch.arange(motion.length, device=device))
    target = raw[torch.arange(motion.length, device=device).add(1).clamp(max=motion.length - 1)]
    return history, goal, target


def state_of(native):
    q = native.robot.data.joint_pos.torch[:, native.arm_ids]
    v = native.robot.data.joint_vel.torch[:, native.arm_ids]
    return torch.cat((q, v, native.command), dim=-1).detach().clone()


def collect(env, actor, replay, steps, teacher_fraction=0.6, noise=0.18, teacher_noise=0.0, actor_mix=1.0):
    native = env.unwrapped
    obs, _ = env.reset()
    start = time.perf_counter()
    terminations = 0
    rewards = []
    max_joint_speed = 0.0
    max_finger_drift = 0.0
    contact_steps = 0
    active_phase_samples = 0
    early_phase_samples = 0
    actual_teacher_actions = 0
    max_speed_event = None
    contact_events = []
    for iteration in range(steps):
        with torch.no_grad():
            state = state_of(native)
            history = obs["policy"][:, :560].reshape(native.num_envs, 10, 56).clone()
            frame = native.motion.frame.clone()
            active_phase_samples += int(((frame >= 450) & (frame < 850)).sum())
            early_phase_samples += int((frame < 50).sum())
            actor_action = actor(history, obs["policy"][:, 560:].reshape(native.num_envs, 10, 90))
            actor_action = actor_action + torch.randn_like(actor_action) * noise
            next_frame = (frame + 1).clamp(max=native.motion.length - 1)
            next_q = native.motion.data["q_arm"][next_frame]
            teacher = torch.atanh(((next_q - native.action_map.origin) / native.action_map.scale).clamp(-0.98, 0.98))
            teacher = teacher + torch.randn_like(teacher) * teacher_noise
            choose_teacher = torch.rand(native.num_envs, 1, device=native.device) < teacher_fraction
            actual_teacher_actions += int(choose_teacher.sum())
            mixed = teacher + actor_mix * (actor_action - teacher)
            action = torch.where(choose_teacher, teacher, mixed)
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
        "realized_teacher_fraction": actual_teacher_actions / (steps * native.num_envs),
        "active_phase_fraction": active_phase_samples / (steps * native.num_envs),
        "early_phase_fraction": early_phase_samples / (steps * native.num_envs),
        "exploration_noise_std": noise,
        "teacher_noise_std": teacher_noise,
        "actor_mix": actor_mix,
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


def evaluate_full(
    actor,
    num_envs=1,
    seed=140116,
    output_dir: Path | None = None,
    action_fn=None,
    initial_perturb_rad=0.0,
    physics_contact_audit=True,
    reward_profile="marvin_tcp_priority",
):
    cfg = MarvinWaveDirectCfg()
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = 28.0
    cfg.random_reference_start = False
    cfg.reset_joint_perturb_rad = initial_perturb_rad
    cfg.audit_contacts_each_physics_step = physics_contact_audit
    cfg.reward_profile = reward_profile
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
            executed_actions, executed_commands, contact_counts = [], [], []
            link_errors = {
                "position_mm": [],
                "orientation_deg": [],
                "linear_velocity_m_s": [],
                "angular_velocity_rad_s": [],
            }
            terminations = 0
            first_failure_step = None
            contact_steps = 0
            contact_physics_steps = 0
            saturations = 0
            torque_samples = 0
            command_jump_max = 0.0
            for control_step in range(1356):
                with torch.no_grad():
                    action = (
                        action_fn(native, control_step)
                        if action_fn is not None
                        else actor(
                            obs["policy"][:, :560].reshape(num_envs, 10, 56),
                            obs["policy"][:, 560:].reshape(num_envs, 10, 90),
                        )
                    )
                    obs, reward, terminated, _truncated, _ = env.step(action)
                    metrics = env.unwrapped.last_tracking
                    physical_q.append(native.last_physical_q[:, model_order].cpu().numpy().copy())
                    physical_qd.append(native.last_physical_qd[:, model_order].cpu().numpy().copy())
                    executed_actions.append(action.detach().cpu().numpy().copy())
                    executed_commands.append(native.command.detach().cpu().numpy().copy())
                    contact_counts.append(native.last_policy_contact_physics_steps if physics_contact_audit else -1)
                    if physics_contact_audit:
                        contact_steps += int(native.last_policy_contact_physics_steps > 0)
                        contact_physics_steps += native.last_policy_contact_physics_steps
                    else:
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
                    for output_key, source_key, scale in (
                        ("position_mm", "link_position_error_m", 1000),
                        ("orientation_deg", "link_rotation_error_rad", 180 / np.pi),
                        ("linear_velocity_m_s", "link_linear_velocity_error_m_s", 1),
                        ("angular_velocity_rad_s", "link_angular_velocity_error_rad_s", 1),
                    ):
                        link_errors[output_key].append(metrics[source_key].detach().cpu().numpy() * scale)
                    rewards.append(float(reward.mean()))
                    terminations += int(terminated.sum())
                    if bool(terminated.any()):
                        first_failure_step = control_step + 1
                        break
            p = np.stack(errors_p)
            r = np.stack(errors_r)
            q_record = np.stack(physical_q)
            qd_record = np.stack(physical_qd)
            link_errors = {key: np.stack(values) for key, values in link_errors.items()}
            source = mujoco.MjModel.from_xml_path(str(SOURCE))
            source_data = mujoco.MjData(source)
            arm_source_ids = [
                mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_JOINT, name) for name in native.motion.joint_names_arm
            ]
            source_contact_frames = 0
            illegal_joint_states = 0
            for q in q_record[:, 0]:
                source_data.qpos[:] = q
                mujoco.mj_forward(source, source_data)
                source_contact_frames += int(source_data.ncon > 0)
                illegal_joint_states += int(
                    np.any(q[arm_source_ids] < source.jnt_range[arm_source_ids, 0] - 1e-3)
                    or np.any(q[arm_source_ids] > source.jnt_range[arm_source_ids, 1] + 1e-3)
                )
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output_dir / "newton_policy_states.npz",
                    time=np.arange(len(q_record)) * 0.02,
                    q=q_record,
                    qd=qd_record,
                    action=np.stack(executed_actions),
                    q_command=np.stack(executed_commands),
                    contact_physics_steps=np.asarray(contact_counts),
                    tcp_position_error_mm=np.concatenate((np.zeros((1, num_envs, 2)), p), axis=0),
                    tcp_orientation_error_deg=np.concatenate((np.zeros((1, num_envs, 2)), r), axis=0),
                    **{f"link_{key}": value for key, value in link_errors.items()},
                )
            ref = native.motion.data["q_arm"][: len(q_record)].detach().cpu().numpy()
            ref_qd = native.motion.data["qd_arm"][: len(qd_record)].detach().cpu().numpy()
            hand_source_ids = [joint for joint in range(54) if joint not in arm_source_ids]
            home_model = native.home_full[model_order].detach().cpu().numpy()
            hand_drift = np.abs(q_record[:, :, hand_source_ids] - home_model[hand_source_ids])
            tracked_links = {}
            for side_index, side_name in enumerate(("left", "right")):
                tracked_links[side_name] = {}
                for link_index, link_name in enumerate(native.motion.link_names[side_index]):
                    tracked_links[side_name][link_name] = {
                        key: {
                            "mean": float(values[:, 0, side_index, link_index].mean()),
                            "p95": float(np.percentile(values[:, 0, side_index, link_index], 95)),
                            "max": float(values[:, 0, side_index, link_index].max()),
                        }
                        for key, values in link_errors.items()
                    }
            return {
                "completed_control_steps": len(errors_p),
                "initial_perturb_rad": initial_perturb_rad,
                "physics_contact_audit": physics_contact_audit,
                "reward_profile": reward_profile,
                "seed": seed,
                "wall_time_s": time.perf_counter() - start,
                "terminations": terminations,
                "first_failure_control_step": first_failure_step,
                "full_motion_completed": first_failure_step is None and len(errors_p) == 1356,
                "reward_mean": float(np.mean(rewards)),
                "newton_contact_control_steps": contact_steps,
                "newton_contact_physics_steps": contact_physics_steps if physics_contact_audit else None,
                "source_mjcf_contact_frames": source_contact_frames,
                "illegal_joint_state_frames": illegal_joint_states,
                "torque_saturation_rate": saturations / max(1, torque_samples),
                "max_command_jump_rad": command_jump_max,
                "max_actual_arm_speed_rad_s": float(np.max(abs(qd_record[:, :, arm_source_ids]))),
                "max_actual_arm_accel_rad_s2": float(
                    np.max(abs(np.diff(qd_record[:, :, arm_source_ids], axis=0) / 0.02))
                ),
                "joint_position_rmse_rad": float(np.mean(np.square(q_record[:, 0, arm_source_ids] - ref)) ** 0.5),
                "joint_velocity_rmse_rad_s": float(np.mean(np.square(qd_record[:, 0, arm_source_ids] - ref_qd)) ** 0.5),
                "tracked_links": tracked_links,
                "finger_home_drift_rad": {
                    "mean": float(hand_drift.mean()),
                    "p95": float(np.percentile(hand_drift, 95)),
                    "max": float(hand_drift.max()),
                },
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
    parser.add_argument("--warmup-actor-lr", type=float)
    parser.add_argument("--world-batch", type=int, default=128)
    parser.add_argument("--imag-batch", type=int, default=32)
    parser.add_argument("--world-hidden", type=int, default=2048)
    parser.add_argument("--teacher-fraction", type=float, default=0.6)
    parser.add_argument("--exploration-noise", type=float, default=0.18)
    parser.add_argument("--teacher-noise", type=float, default=0.0)
    parser.add_argument("--reset-joint-perturb-rad", type=float, default=0.01)
    parser.add_argument("--actor-mix", type=float, default=1.0)
    parser.add_argument("--active-reset-fraction", type=float, default=0.0)
    parser.add_argument("--active-reset-frame-range", type=int, nargs=2, default=(450, 850))
    parser.add_argument("--early-reset-fraction", type=float, default=0.0)
    parser.add_argument("--early-reset-frame-range", type=int, nargs=2, default=(0, 50))
    parser.add_argument("--behavior-prior-weight", type=float, default=0.0)
    parser.add_argument("--reward-profile", choices=tuple(REWARD_PROFILES), default="marvin_tcp_priority")
    parser.add_argument("--pd-warmup-fraction", type=float, default=0.0)
    parser.add_argument("--reset-actor-on-resume", action="store_true")
    parser.add_argument("--actor-init-state", type=Path)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--early-bptt-fraction", type=float, default=0.0)
    parser.add_argument("--hard-bptt-fraction", type=float, default=0.0)
    parser.add_argument("--hard-bptt-range", type=int, nargs=2, default=(450, 850))
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("/home/ubuntu/dexgym_data/marvin_supermimic_wave_m4/e1_first")
    )
    args = parser.parse_args()
    if not (
        0 <= args.early_bptt_fraction <= 1
        and 0 <= args.hard_bptt_fraction <= 1
        and args.early_bptt_fraction + args.hard_bptt_fraction <= 1
        and 0 <= args.pd_warmup_fraction <= 1
    ):
        raise ValueError("BPTT and PD sample fractions must lie in [0, 1] and sum to at most one")
    if not (0 <= args.hard_bptt_range[0] < args.hard_bptt_range[1] <= 1357):
        raise ValueError("Hard BPTT frame range must lie within the pinned motion")
    if args.actor_lr <= 0:
        raise ValueError("Actor learning rate must be positive")
    if args.warmup_actor_lr is not None and args.warmup_actor_lr <= 0:
        raise ValueError("Actor warmup learning rate must be positive")
    if not 0 <= args.reset_joint_perturb_rad <= 0.1:
        raise ValueError("Reset joint perturbation must lie in [0, 0.1] rad")
    if not (
        0 <= args.teacher_fraction <= 1
        and 0 <= args.actor_mix <= 1
        and 0 <= args.active_reset_fraction <= 1
        and 0 <= args.early_reset_fraction <= 1
        and args.active_reset_fraction + args.early_reset_fraction <= 1
    ):
        raise ValueError("Collection sampling fractions and actor mix must lie in [0, 1]")
    if not (0 <= args.active_reset_frame_range[0] < args.active_reset_frame_range[1] <= 1357):
        raise ValueError("Active reset frame range must lie within the pinned motion")
    if not (0 <= args.early_reset_frame_range[0] < args.early_reset_frame_range[1] <= 1357):
        raise ValueError("Early reset frame range must lie within the pinned motion")
    preflight = verify()
    git_commit = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True
    ).strip()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(140116)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    actor = GoalActor().to(device)
    model = DeltaWorldModel(hidden=args.world_hidden).to(device)
    critic = PrivilegedCritic().to(device)
    optim_model = torch.optim.Adam(model.parameters(), lr=3e-4)
    optim_actor = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
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
        if args.reset_actor_on_resume:
            actor = GoalActor().to(device)
            optim_actor = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    if args.actor_init_state:
        actor.load_state_dict(torch.load(args.actor_init_state, map_location=device, weights_only=True))
        optim_actor = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    cfg = MarvinWaveDirectCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 140116
    cfg.reset_joint_perturb_rad = args.reset_joint_perturb_rad
    cfg.active_reset_fraction = args.active_reset_fraction
    cfg.active_reset_frame_range = tuple(args.active_reset_frame_range)
    cfg.early_reset_fraction = args.early_reset_fraction
    cfg.early_reset_frame_range = tuple(args.early_reset_frame_range)
    cfg.reward_profile = args.reward_profile
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
                actor_mix=args.actor_mix,
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
                warmup_optimizer = (
                    torch.optim.Adam(actor.parameters(), lr=args.warmup_actor_lr or args.actor_lr)
                    if args.actor_warmup_updates
                    else None
                )
                pd_examples = (
                    pd_actor_examples(motion, mapping, device)
                    if args.pd_warmup_fraction > 0 and args.actor_warmup_updates
                    else None
                )
                for _ in range(args.actor_warmup_updates):
                    batch_size = max(args.imag_batch, 128)
                    pd_count = round(batch_size * args.pd_warmup_fraction) if pd_examples is not None else 0
                    replay_count = batch_size - pd_count
                    histories, goals, targets = [], [], []
                    if replay_count:
                        start_batch = replay.sample_starts(replay_count, env_range=(0, train_envs))
                        goal = motion.goal_at(start_batch["frame"])
                        target_q = motion.data["q_arm"][(start_batch["frame"] + 1).clamp(max=motion.length - 1)]
                        teacher_action = torch.atanh(((target_q - mapping.origin) / mapping.scale).clamp(-0.98, 0.98))
                        histories.append(start_batch["history"])
                        goals.append(goal)
                        targets.append(teacher_action)
                    if pd_count:
                        early_count = pd_count // 4
                        frame_ids = torch.cat(
                            (
                                torch.randint(0, min(50, motion.length), (early_count,), device=device),
                                torch.randint(0, motion.length, (pd_count - early_count,), device=device),
                            )
                        )
                        histories.append(pd_examples[0][frame_ids])
                        goals.append(pd_examples[1][frame_ids])
                        targets.append(pd_examples[2][frame_ids])
                    warmup_optimizer.zero_grad()
                    predicted_action = actor(torch.cat(histories), torch.cat(goals))
                    warmup_loss = torch.nn.functional.mse_loss(predicted_action, torch.cat(targets))
                    warmup_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    warmup_optimizer.step()
                    warmup_losses.append(float(warmup_loss.detach()))
                if warmup_optimizer is not None:
                    optim_actor = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
                warmup_action_error = actor_command_error(actor, replay, mapping, motion, (train_envs, args.num_envs))
                torch.save(actor.state_dict(), args.output / "actor_after_auxiliary_warmup.pt")
                prior_actor = copy.deepcopy(actor).eval()
                for parameter in prior_actor.parameters():
                    parameter.requires_grad_(False)
                for actor_update in range(args.actor_updates):
                    early_count = round(args.imag_batch * args.early_bptt_fraction)
                    hard_count = min(round(args.imag_batch * args.hard_bptt_fraction), args.imag_batch - early_count)
                    batches = []
                    if early_count:
                        batches.append(
                            replay.sample_starts(early_count, env_range=(0, train_envs), frame_range=(0, 50))
                        )
                    if hard_count:
                        batches.append(
                            replay.sample_starts(
                                hard_count,
                                env_range=(0, train_envs),
                                frame_range=tuple(args.hard_bptt_range),
                            )
                        )
                    if args.imag_batch - early_count - hard_count:
                        batches.append(
                            replay.sample_starts(
                                args.imag_batch - early_count - hard_count,
                                env_range=(0, train_envs),
                            )
                        )
                    start_batch = {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}
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
                        weights=native.reward_weights,
                        prior_actor=prior_actor,
                        prior_weight=args.behavior_prior_weight,
                    )
                    rollout["actor_loss"].backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    optim_actor.step()
                    actor_losses.append(float(rollout["actor_loss"].detach()))
                    if (actor_update + 1) % 50 == 0:
                        torch.save(actor.state_dict(), args.output / f"actor_bptt_{actor_update + 1:04d}.pt")
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
                "git_commit": git_commit,
                "reference": str(REFERENCE),
                "preflight": preflight,
                "resume_from": str(args.resume) if args.resume else None,
                "resume_recollected_replay": bool(args.resume),
                "prior_training_updates": {
                    k: prior_training.get(k, 0) for k in ("world_updates", "actor_updates", "critic_updates")
                }
                if prior_training
                else None,
                "configuration": {
                    key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                },
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
        "git_commit": git_commit,
        "replay_saved": False,
        "resume_requires_recollection": True,
    }
    torch.save(checkpoint, args.output / "last.pt")
    np.savez_compressed(
        args.output / "learning_curves.npz",
        world_loss=np.asarray(model_losses),
        actor_loss=np.asarray(actor_losses),
        critic_loss=np.asarray(critic_losses),
        auxiliary_loss=np.asarray(warmup_losses),
    )
    if training["actor_updates"]:
        training["evaluation"] = evaluate_full(actor, output_dir=args.output, reward_profile=args.reward_profile)
        checkpoint["training"] = training
        passed = (
            training["evaluation"]["full_motion_completed"]
            and training["evaluation"]["newton_contact_control_steps"] == 0
            and training["evaluation"]["source_mjcf_contact_frames"] == 0
            and training["evaluation"]["illegal_joint_state_frames"] == 0
            and training["evaluation"]["max_actual_arm_speed_rad_s"] <= 5
            and all(
                training["evaluation"][side]["mean"] <= limit
                for side, limit in (
                    ("left_position_mm", 25),
                    ("right_position_mm", 25),
                    ("left_orientation_deg", 8),
                    ("right_orientation_deg", 8),
                )
            )
            and all(
                training["evaluation"][side]["p95"] <= limit
                for side, limit in (
                    ("left_position_mm", 50),
                    ("right_position_mm", 50),
                    ("left_orientation_deg", 15),
                    ("right_orientation_deg", 15),
                )
            )
        )
        training["policy_validation_passed"] = passed
        torch.save(checkpoint, args.output / ("best.pt" if passed else "candidate.pt"))
    (args.output / "metrics.json").write_text(json.dumps(training, indent=2))
    print(json.dumps(training, indent=2))


if __name__ == "__main__":
    main()
