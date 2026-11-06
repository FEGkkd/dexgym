"""Task-owned full Newton evaluator for legacy Marvin motion tracking.

The historical train_wave.py remains unchanged for M4.1 reproducibility.
"""

import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch
from isaaclab.app import launch_simulation

import dexgym.tasks.marvin_supermimic  # register Gym environment
from dexgym.tasks.marvin_supermimic.env_cfg import MarvinWaveDirectCfg
from dexgym.tasks.marvin_supermimic.robot_adapter import DEFAULT_MJCF as SOURCE

def evaluate_full(
    actor,
    num_envs=1,
    seed=140116,
    output_dir: Path | None = None,
    action_fn=None,
    initial_perturb_rad=0.0,
    physics_contact_audit=True,
    reward_profile="marvin_tcp_priority",
    reference_path=None,
    action_schema="legacy_wave_v1",
    initialize_hands_from_pd_replay=True,
):
    cfg = MarvinWaveDirectCfg()
    if reference_path is not None:
        cfg.reference_path = str(reference_path)
    cfg.action_schema = action_schema
    cfg.initialize_hands_from_pd_replay = initialize_hands_from_pd_replay
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
            for control_step in range(native.motion.length - 1):
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
                "full_motion_completed": first_failure_step is None and len(errors_p) == native.motion.length - 1,
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

