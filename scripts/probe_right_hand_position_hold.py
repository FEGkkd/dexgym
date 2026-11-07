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
    parser.add_argument("--starts-per-motion", type=int, default=2)
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    parser.add_argument("--goal-offset", type=int, default=4)
    parser.add_argument("--disable-hand-contacts", action="store_true")
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hand_only_position_hold"))
    args = parser.parse_args()
    if args.starts_per_motion < 1 or args.hold_seconds <= 0 or args.goal_offset < 0:
        parser.error("--starts-per-motion and --hold-seconds must be positive; --goal-offset must be nonnegative")

    load_external_tasks()
    cfg = RightHandTrackingDirectCfg()
    cfg.split = args.split
    cfg.seed = args.seed
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    if args.disable_hand_contacts:
        cfg.scene.robot = make_robot_cfg(hand_contacts=False)
    # Count motions before scene creation so every motion is covered equally.
    manifest = json.loads(Path(cfg.library_path).read_text())
    records = [record for record in manifest["motions"] if record["split"] == args.split]
    cfg.scene.num_envs = len(records) * args.starts_per_motion
    control_dt = cfg.sim.dt * cfg.decimation
    control_steps = round(args.hold_seconds / control_dt)
    if abs(control_steps * control_dt - args.hold_seconds) > 1e-8:
        parser.error("--hold-seconds must be an integer number of 30-Hz control steps")

    with launch_simulation(cfg):
        raw_env = gym.make(RIGHT_HAND_ID, cfg=cfg)
        try:
            task = raw_env.unwrapped
            raw_env.reset(seed=args.seed)
            generator = torch.Generator(device=task.device).manual_seed(args.seed)
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
            tip_errors = []
            max_tip_displacement_mm = (tip_goal - tip_start).norm(dim=-1).max(dim=-1).values
            with torch.inference_mode():
                for step in range(control_steps + 1):
                    q_actual = robot.data.joint_pos.torch[:, task.right_ids]
                    tips_actual = task._current_tips()
                    joint_errors.append((q_actual - q_goal).abs().detach().cpu().numpy().copy())
                    tip_errors.append((tips_actual - tip_goal).norm(dim=-1).detach().cpu().numpy().copy() * 1000)
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
            tip_errors = np.stack(tip_errors)
            final_joint = joint_errors[-1]
            final_tip = tip_errors[-1]
            final_five_finger_max = final_tip.max(axis=-1)
            within_10mm = (tip_errors < 10.0).all(axis=-1)
            final_dwell_2 = (within_10mm[-2] & within_10mm[-1]) if control_steps >= 1 else within_10mm[-1]
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
                "contacts_enabled": not args.disable_hand_contacts,
                "target_frame_stride": args.goal_offset,
                "max_target_tip_displacement_mm": statistics(target_displacement),
                "fk_goal_max_finger_error_mm": statistics(fk_goal_tip_error_mm.max(axis=-1)),
                "fk_start_max_finger_error_mm": statistics(fk_start_tip_error_mm.max(axis=-1)),
                "final_joint_abs_error_rad": statistics(final_joint.flatten()),
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
            }
            output_dir = args.output_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            np.savez_compressed(
                output_dir / "errors.npz",
                joint_abs_error_rad=joint_errors,
                finger_error_mm=tip_errors,
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
                        "five_finger_max_mean_mm",
                        "five_finger_max_p95_mm",
                        "five_finger_max_mm",
                        "all_five_within_10mm",
                    ]
                )
                for step in range(control_steps + 1):
                    five_max = tip_errors[step].max(axis=-1)
                    writer.writerow(
                        [
                            step * control_dt,
                            joint_errors[step].mean(),
                            joint_errors[step].max(),
                            five_max.mean(),
                            np.percentile(five_max, 95),
                            five_max.max(),
                            within_10mm[step].sum(),
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
                            final_five_finger_max[index],
                            *final_tip[index],
                        ]
                    )
            print(json.dumps(summary, indent=2))
        finally:
            raw_env.close()


if __name__ == "__main__":
    main()
