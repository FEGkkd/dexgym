"""Hold GRAB right-hand joint references in the task's position controller."""

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from dexgym.assets.marvin_wuji.robot_cfg import make_robot_cfg
from dexgym.tasks import RIGHT_HAND_ID
from dexgym.tasks.direct.right_hand_tracking.drive_experiment_cfg import make_drive_experiment_cfg
from dexgym.tasks.direct.right_hand_tracking.env_cfg import RightHandTrackingDirectCfg
from isaaclab.app import launch_simulation
from isaaclab.cli import load_external_tasks

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--library-path", type=Path, default=None, help="Override the task's motion manifest")
    parser.add_argument("--sample-pairs-csv", type=Path, default=None, help="Use exact motion/start/goal rows")
    parser.add_argument("--starts-per-motion", type=int, default=2)
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    parser.add_argument("--goal-offset", type=int, default=4)
    parser.add_argument("--disable-hand-contacts", action="store_true")
    parser.add_argument("--zero-gravity", action="store_true")
    parser.add_argument("--corrected-drive", action="store_true")
    parser.add_argument("--drive-profile", choices=("source", "hand_x2"), default="source")
    parser.add_argument("--dt-denominator", type=int, choices=(120, 240, 500), default=120)
    parser.add_argument("--hand-kp-scale", type=float, default=1.0)
    parser.add_argument("--hand-kd-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hand_only_position_hold"))
    args = parser.parse_args()
    if args.starts_per_motion < 1 or args.hold_seconds <= 0 or args.goal_offset < 0:
        parser.error("--starts-per-motion and --hold-seconds must be positive; --goal-offset must be nonnegative")
    if args.hand_kp_scale <= 0 or args.hand_kd_scale <= 0:
        parser.error("--hand-kp-scale and --hand-kd-scale must be positive")
    if not args.corrected_drive and (args.hand_kp_scale != 1.0 or args.hand_kd_scale != 1.0):
        parser.error("Hand gain scaling requires --corrected-drive")
    if not args.corrected_drive and args.drive_profile != "source":
        parser.error("--drive-profile hand_x2 requires --corrected-drive")

    load_external_tasks()
    reference_cfg = RightHandTrackingDirectCfg()
    if args.library_path is not None:
        reference_cfg.library_path = str(args.library_path.expanduser().resolve())
    manifest = json.loads(Path(reference_cfg.library_path).read_text())
    records = [record for record in manifest["motions"] if record["split"] == args.split]
    sampled_rows = None
    if args.sample_pairs_csv is not None:
        with args.sample_pairs_csv.expanduser().open(newline="") as stream:
            sampled_rows = list(csv.DictReader(stream))
        selected_names = {record["source"] for record in records}
        sampled_rows = [row for row in sampled_rows if row["motion"] in selected_names]
        if not sampled_rows:
            parser.error("--sample-pairs-csv has no rows for the selected library split")
    num_envs = len(sampled_rows) if sampled_rows is not None else len(records) * args.starts_per_motion
    if args.corrected_drive:
        cfg = make_drive_experiment_cfg(
            physics_dt=1.0 / args.dt_denominator,
            num_envs=num_envs,
            hand_contacts=not args.disable_hand_contacts,
            profile=args.drive_profile,
        )
        cfg.decimation = 1  # Hold the same command while sampling every physics step.
    else:
        if args.dt_denominator != 120:
            parser.error("Non-default physics dt requires --corrected-drive")
        cfg = reference_cfg
        cfg.scene.num_envs = num_envs
        if args.disable_hand_contacts:
            cfg.scene.robot = make_robot_cfg(hand_contacts=False)
    cfg.split = args.split
    cfg.library_path = reference_cfg.library_path
    cfg.seed = args.seed
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    if args.zero_gravity:
        cfg.sim.gravity = (0.0, 0.0, 0.0)
    control_dt = cfg.sim.dt * cfg.decimation
    control_steps = round(args.hold_seconds / control_dt)
    if abs(control_steps * control_dt - args.hold_seconds) > 1e-8:
        parser.error("--hold-seconds must be an integer number of control steps")

    with launch_simulation(cfg):
        raw_env = gym.make(RIGHT_HAND_ID, cfg=cfg)
        try:
            task = raw_env.unwrapped
            raw_env.reset(seed=args.seed)
            generator = torch.Generator(device=task.device).manual_seed(args.seed)
            if sampled_rows is None:
                motion_ids = torch.arange(task.bank.motion_count, device=task.device).repeat_interleave(
                    args.starts_per_motion
                )
                # Keep identical sampled starts between the default four-frame
                # target and the zero-offset hold control.
                max_start_plus_one = task.bank.lengths[motion_ids] - max(cfg.subgoal_stride, args.goal_offset, 1)
                if (max_start_plus_one <= 0).any():
                    raise ValueError("--goal-offset exceeds the length of at least one motion")
                start_frames = torch.floor(
                    torch.rand(len(motion_ids), device=task.device, generator=generator) * max_start_plus_one
                ).long()
                goal_frames = start_frames + args.goal_offset
            else:
                names = {entry["source"]: index for index, entry in enumerate(task.bank.records)}
                motion_ids = torch.tensor([names[row["motion"]] for row in sampled_rows], device=task.device)
                start_frames = torch.tensor([int(row["start_frame"]) for row in sampled_rows], device=task.device)
                goal_frames = torch.tensor([int(row["goal_frame"]) for row in sampled_rows], device=task.device)
                if (
                    (start_frames < 0).any()
                    or (goal_frames < 0).any()
                    or (start_frames >= task.bank.lengths[motion_ids]).any()
                    or (goal_frames >= task.bank.lengths[motion_ids]).any()
                ):
                    raise ValueError("A supplied sample frame is outside its motion")
            q_start = task.bank.joints(motion_ids, start_frames)
            q_goal = task.bank.joints(motion_ids, goal_frames)
            tip_start = task.bank.fingertips(motion_ids, start_frames)
            tip_goal = task.bank.fingertips(motion_ids, goal_frames)
            limits = task.joint_limits
            if ((q_goal < limits[:, 0]) | (q_goal > limits[:, 1])).any():
                raise ValueError("Reference joint target exceeds the task's joint limits")
            robot = task.robot
            scene = task.scene
            sim = task.sim
            if args.corrected_drive and (args.hand_kp_scale != 1.0 or args.hand_kd_scale != 1.0):
                kp = robot.data.joint_stiffness.torch[:, task.right_ids].clone() * args.hand_kp_scale
                kd = robot.data.joint_damping.torch[:, task.right_ids].clone() * args.hand_kd_scale
                robot.write_joint_stiffness_to_sim_index(stiffness=kp, joint_ids=task.right_ids)
                robot.write_joint_damping_to_sim_index(damping=kd, joint_ids=task.right_ids)
                if not torch.allclose(robot.data.joint_stiffness.torch[:, task.right_ids], kp, rtol=1e-5):
                    raise AssertionError("Newton did not accept the isolated hand Kp scale")
                if not torch.allclose(robot.data.joint_damping.torch[:, task.right_ids], kd, rtol=1e-5):
                    raise AssertionError("Newton did not accept the isolated hand Kd scale")
            actual_gravity = tuple(sim.physics_manager._gravity_vector)
            if not np.allclose(actual_gravity, cfg.sim.gravity, atol=1e-9):
                raise AssertionError(f"Newton gravity {actual_gravity} differs from requested {cfg.sim.gravity}")

            def write_pose(right_q: torch.Tensor) -> None:
                full_q = task.home_q[None].repeat(task.num_envs, 1)
                full_q[:, task.right_ids] = right_q
                robot.write_joint_position_to_sim_index(position=full_q)
                robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(full_q))
                scene.write_data_to_sim()
                sim.forward()
                scene.update(0.0)

            # Teleport/forward only for a kinematic consistency measurement.
            # Restore the start pose before applying the actual position command.
            write_pose(q_goal)
            fk_goal_tip_error_mm = (task._current_tips() - tip_goal).norm(dim=-1).detach().cpu().numpy() * 1000
            write_pose(q_start)
            fk_start_tip_error_mm = (task._current_tips() - tip_start).norm(dim=-1).detach().cpu().numpy() * 1000

            target = task.home_q[None].repeat(task.num_envs, 1)
            target[:, task.right_ids] = q_goal
            handle_decimation = task._physics_handles_decimation
            steps_per_call = cfg.decimation if handle_decimation else 1
            joint_errors = []
            joint_speed = []
            tip_errors = []
            active_contact_counts = []
            max_limit_violations = []
            torque_saturation_steps = np.zeros(task.num_envs, dtype=np.int32)
            peak_applied_torque_ratio = np.zeros(task.num_envs, dtype=np.float32)
            effort_limits = robot.data.joint_effort_limits.torch[:, task.right_ids].detach().cpu().numpy()
            if (effort_limits <= 0).any():
                raise ValueError("Right-hand effort limits must be positive to measure saturation")
            solver = sim.physics_manager._solver
            max_tip_displacement_mm = (tip_goal - tip_start).norm(dim=-1).max(dim=-1).values
            with torch.inference_mode():
                for step in range(control_steps + 1):
                    q_actual = robot.data.joint_pos.torch[:, task.right_ids]
                    qd_actual = robot.data.joint_vel.torch[:, task.right_ids]
                    tips_actual = task._current_tips()
                    joint_errors.append((q_actual - q_goal).abs().detach().cpu().numpy().copy())
                    joint_speed.append(qd_actual.abs().detach().cpu().numpy().copy())
                    tip_errors.append((tips_actual - tip_goal).norm(dim=-1).detach().cpu().numpy().copy() * 1000)
                    active_contact_counts.append(int(solver.mjw_data.nacon.numpy()[0]))
                    if step > 0:
                        applied = robot.actuators.applied_effort.torch[:, task.right_ids].abs().detach().cpu().numpy()
                        ratio = applied / effort_limits
                        torque_saturation_steps += (ratio >= 0.99).any(axis=-1)
                        peak_applied_torque_ratio = np.maximum(peak_applied_torque_ratio, ratio.max(axis=-1))
                    violation = torch.maximum(
                        torch.maximum(limits[:, 0] - q_actual, q_actual - limits[:, 1]), torch.zeros_like(q_actual)
                    )
                    max_limit_violations.append(float(violation.max().item()))
                    if step == control_steps:
                        break
                    robot.actuators.target_command.set_position_index(value=target, full_data=True)
                    if not torch.allclose(robot.actuators.target_command.position.torch, target, atol=1e-6):
                        raise AssertionError("Position target was not propagated to the actuators")
                    for _ in range(cfg.decimation // steps_per_call):
                        scene.write_data_to_sim()
                        sim.step(render=False)
                        scene.update(cfg.sim.dt * steps_per_call)

            joint_errors = np.stack(joint_errors)
            joint_speed = np.stack(joint_speed)
            tip_errors = np.stack(tip_errors)
            final_joint = joint_errors[-1]
            final_tip = tip_errors[-1]
            final_five_finger_max = final_tip.max(axis=-1)
            within_10mm = (tip_errors < 10.0).all(axis=-1)
            final_dwell_2 = (within_10mm[-2] & within_10mm[-1]) if control_steps >= 1 else within_10mm[-1]
            tail_steps = max(2, round(0.2 / control_dt))
            tail_max_speed = joint_speed[-tail_steps:].max(axis=(0, 2))
            settled = tail_max_speed < 0.05
            strict_tip = final_five_finger_max < 10.0
            target_displacement = max_tip_displacement_mm.detach().cpu().numpy() * 1000
            summary = {
                "split": args.split,
                "seed": args.seed,
                "motions": task.bank.motion_count,
                "starts_per_motion": args.starts_per_motion,
                "samples": task.num_envs,
                "hold_seconds": args.hold_seconds,
                "control_dt_s": control_dt,
                "physics_dt_s": cfg.sim.dt,
                "control_steps": control_steps,
                "physics_steps_per_control": cfg.decimation,
                "robot_usd": cfg.scene.robot.spawn.usd_path,
                "drive_config": "corrected_usd" if args.corrected_drive else "original_usd",
                "drive_profile": args.drive_profile,
                "hand_kp_scale": args.hand_kp_scale,
                "hand_kd_scale": args.hand_kd_scale,
                "effective_hand_kp_scale_source": (2.0 if args.drive_profile == "hand_x2" else 1.0)
                * args.hand_kp_scale,
                "effective_hand_kd_scale_source": (2.0 if args.drive_profile == "hand_x2" else 1.0)
                * args.hand_kd_scale,
                "gravity_m_s2": list(actual_gravity),
                "contacts_enabled": not args.disable_hand_contacts,
                "target_frame_stride": args.goal_offset,
                "max_target_tip_displacement_mm": statistics(target_displacement),
                "fk_goal_max_finger_error_mm": statistics(fk_goal_tip_error_mm.max(axis=-1)),
                "fk_start_max_finger_error_mm": statistics(fk_start_tip_error_mm.max(axis=-1)),
                "final_joint_abs_error_rad": statistics(final_joint.flatten()),
                "final_joint_abs_speed_rad_s": statistics(joint_speed[-1].flatten()),
                "tail_max_joint_speed_rad_s_per_sample": statistics(tail_max_speed),
                "settled_speed_threshold_rad_s": 0.05,
                "settled_samples": int(settled.sum()),
                "settled_and_all_five_within_10mm": int((settled & strict_tip).sum()),
                "settled_but_finger_error_over_10mm": int((settled & ~strict_tip).sum()),
                "unsettled_samples": int((~settled).sum()),
                "final_max_joint_abs_error_rad_per_sample": statistics(final_joint.max(axis=-1)),
                "final_five_finger_max_error_mm": statistics(final_five_finger_max),
                "final_finger_error_mm": {
                    name: statistics(final_tip[:, index]) for index, name in enumerate(FINGER_NAMES)
                },
                "final_all_five_within_10mm": int(within_10mm[-1].sum()),
                "final_two_steps_all_five_within_10mm": int(final_dwell_2.sum()),
                "ever_all_five_within_10mm": int(within_10mm.any(axis=0).sum()),
                "worst_five_finger_error_any_time_mm": float(tip_errors.max()),
                "worst_joint_abs_error_any_time_rad": float(joint_errors.max()),
                "max_joint_limit_violation_any_time_rad": max(max_limit_violations),
                "active_contact_count_mean": float(np.mean(active_contact_counts)),
                "active_contact_count_max": max(active_contact_counts),
                "samples_with_torque_saturation": int((torque_saturation_steps > 0).sum()),
                "max_torque_saturation_steps_per_sample": int(torque_saturation_steps.max()),
                "peak_applied_torque_to_limit_ratio": float(peak_applied_torque_ratio.max()),
            }
            output_dir = args.output_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            np.savez_compressed(
                output_dir / "errors.npz",
                joint_abs_error_rad=joint_errors,
                joint_abs_speed_rad_s=joint_speed,
                finger_error_mm=tip_errors,
                tail_max_joint_speed_rad_s=tail_max_speed,
                torque_saturation_steps=torque_saturation_steps,
                peak_applied_torque_ratio=peak_applied_torque_ratio,
                fk_goal_finger_error_mm=fk_goal_tip_error_mm,
                fk_start_finger_error_mm=fk_start_tip_error_mm,
                motion_ids=motion_ids.detach().cpu().numpy(),
                start_frames=start_frames.detach().cpu().numpy(),
                goal_frames=goal_frames.detach().cpu().numpy(),
            )
            with (output_dir / "timeseries.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "time_s",
                        "joint_mae_rad",
                        "joint_max_rad",
                        "joint_speed_max_rad_s",
                        "five_finger_max_mean_mm",
                        "five_finger_max_p95_mm",
                        "five_finger_max_mm",
                        "all_five_within_10mm",
                        "active_contact_count",
                        "max_joint_limit_violation_rad",
                    ]
                )
                for step in range(control_steps + 1):
                    five_max = tip_errors[step].max(axis=-1)
                    writer.writerow(
                        [
                            step * control_dt,
                            joint_errors[step].mean(),
                            joint_errors[step].max(),
                            joint_speed[step].max(),
                            five_max.mean(),
                            np.percentile(five_max, 95),
                            five_max.max(),
                            within_10mm[step].sum(),
                            active_contact_counts[step],
                            max_limit_violations[step],
                        ]
                    )
            with (output_dir / "per_sample.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "motion",
                        "start_frame",
                        "goal_frame",
                        "max_target_tip_displacement_mm",
                        "fk_goal_max_finger_error_mm",
                        "final_joint_mae_rad",
                        "final_joint_max_rad",
                        "tail_max_joint_speed_rad_s",
                        "torque_saturation_steps",
                        "peak_applied_torque_ratio",
                        "final_five_finger_max_mm",
                        *[f"{name}_mm" for name in FINGER_NAMES],
                    ]
                )
                ids = motion_ids.detach().cpu().tolist()
                starts = start_frames.detach().cpu().tolist()
                goals = goal_frames.detach().cpu().tolist()
                for index, motion_id in enumerate(ids):
                    writer.writerow(
                        [
                            task.bank.records[motion_id]["source"],
                            starts[index],
                            goals[index],
                            target_displacement[index],
                            fk_goal_tip_error_mm[index].max(),
                            final_joint[index].mean(),
                            final_joint[index].max(),
                            tail_max_speed[index],
                            torque_saturation_steps[index],
                            peak_applied_torque_ratio[index],
                            final_five_finger_max[index],
                            *final_tip[index],
                        ]
                    )
            print(json.dumps(summary, indent=2))
        finally:
            raw_env.close()


if __name__ == "__main__":
    main()
