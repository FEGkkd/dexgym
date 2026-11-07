"""Sweep corrected Newton hand-drive timestep and collision settings."""

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch
from dexgym.assets.marvin_wuji.joint_mapping import inspect_joint_contract
from dexgym.assets.marvin_wuji.robot_cfg import resolve_source_mjcf
from dexgym.tasks import RIGHT_HAND_ID
from dexgym.tasks.direct.right_hand_tracking.drive_experiment_cfg import make_drive_experiment_cfg
from isaaclab.app import launch_simulation
from isaaclab.cli import load_external_tasks

FOCUS = "r_index_finger_mcp_flex"
HOLD_SECONDS = 1.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt-denominator", type=int, choices=(120, 240, 500), required=True)
    parser.add_argument("--no-hand-contacts", action="store_true")
    parser.add_argument("--profile", choices=("source", "hand_x2"), default="source")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dt = 1.0 / args.dt_denominator
    steps = round(HOLD_SECONDS / dt)
    if abs(steps * dt - HOLD_SECONDS) > 1e-10:
        raise ValueError("Physics dt must divide the 1.5 s hold exactly")

    load_external_tasks()
    cfg = make_drive_experiment_cfg(
        physics_dt=dt, num_envs=1, hand_contacts=not args.no_hand_contacts, profile=args.profile
    )
    cfg.decimation = 1  # The fixed command is unchanged; sample every physics step.
    cfg.seed = 42
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp

    model = mujoco.MjModel.from_xml_path(str(resolve_source_mjcf()))
    contract = inspect_joint_contract(model)
    with launch_simulation(cfg):
        raw_env = gym.make(RIGHT_HAND_ID, cfg=cfg)
        try:
            task = raw_env.unwrapped
            raw_env.reset(seed=42)
            robot, scene, sim = task.robot, task.scene, task.sim
            focus_id = robot.joint_names.index(FOCUS)
            start = task.home_q[None].clone()
            target = start.clone()
            target[:, focus_id] += 0.1
            limits = robot.data.joint_pos_limits.torch[0].detach().cpu().numpy().copy()
            if not limits[focus_id, 0] <= float(target[0, focus_id]) <= limits[focus_id, 1]:
                raise ValueError("Index MCP target is outside the joint limits")
            source_kp = np.asarray(
                [model.actuator_gainprm[contract.ctrl_by_name[name], 0] for name in robot.joint_names]
            )
            source_kd = np.asarray(
                [-model.actuator_biasprm[contract.ctrl_by_name[name], 2] for name in robot.joint_names]
            )
            source_limit = np.asarray(
                [model.actuator_forcerange[contract.ctrl_by_name[name], 1] for name in robot.joint_names]
            )
            if args.profile == "hand_x2":
                hand_ids = [index for index, name in enumerate(robot.joint_names) if name.startswith("r_")]
                source_kp[hand_ids] *= 2.0
                source_kd[hand_ids] *= 2.0
            kp = robot.data.joint_stiffness.torch[0].detach().cpu().numpy().copy()
            kd = robot.data.joint_damping.torch[0].detach().cpu().numpy().copy()
            effort_limit = robot.data.joint_effort_limits.torch[0].detach().cpu().numpy().copy()
            gain_error = max(np.max(np.abs(kp - source_kp)), np.max(np.abs(kd - source_kd)))
            limit_error = np.max(np.abs(effort_limit - source_limit))
            if gain_error > 1e-4 or limit_error > 1e-5:
                raise AssertionError(
                    f"Corrected runtime drive differs from source: gain={gain_error}, limit={limit_error}"
                )

            def write_pose(q: torch.Tensor) -> None:
                robot.write_joint_position_to_sim_index(position=q)
                robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
                scene.write_data_to_sim()
                sim.forward()
                scene.update(0.0)

            write_pose(target)
            goal_tips = task._current_tips()[0].detach().cpu().numpy().copy()
            write_pose(start)
            solver = sim.physics_manager._solver
            rows = []
            with torch.inference_mode():
                for step in range(steps + 1):
                    q = robot.data.joint_pos.torch[0].detach().cpu().numpy().copy()
                    qd = robot.data.joint_vel.torch[0].detach().cpu().numpy().copy()
                    computed = robot.actuators.computed_effort.torch[0].detach().cpu().numpy().copy()
                    applied = robot.actuators.applied_effort.torch[0].detach().cpu().numpy().copy()
                    tips = task._current_tips()[0].detach().cpu().numpy().copy()
                    contacts = int(solver.mjw_data.nacon.numpy()[0])
                    rows.append(
                        {
                            "time_s": step * dt,
                            "command_rad": float(target[0, focus_id]),
                            "q_rad": float(q[focus_id]),
                            "qd_rad_s": float(qd[focus_id]),
                            "computed_drive_nm": float(computed[focus_id]),
                            "applied_drive_nm": float(applied[focus_id]),
                            "effort_limit_nm": float(effort_limit[focus_id]),
                            "at_effort_limit": int(abs(applied[focus_id]) >= 0.99 * effort_limit[focus_id]),
                            "max_right_hand_speed_rad_s": float(np.abs(qd[task.right_ids]).max()),
                            "max_joint_limit_violation_rad": float(
                                np.maximum(np.maximum(limits[:, 0] - q, q - limits[:, 1]), 0).max()
                            ),
                            "five_finger_max_error_mm": float(np.linalg.norm(tips - goal_tips, axis=-1).max() * 1000),
                            "active_contacts": contacts,
                        }
                    )
                    if step == steps:
                        break
                    robot.actuators.target_command.set_position_index(value=target, full_data=True)
                    if not torch.allclose(robot.actuators.target_command.position.torch, target, atol=1e-6):
                        raise AssertionError("Newton position command did not propagate")
                    scene.write_data_to_sim()
                    sim.step(render=False)
                    scene.update(dt)

            tail = rows[-max(2, round(0.2 / dt)) :]
            after = rows[1:]
            summary = {
                "physics_dt_s": dt,
                "steps": steps,
                "hold_seconds": HOLD_SECONDS,
                "hand_contacts_enabled": not args.no_hand_contacts,
                "robot_usd": cfg.scene.robot.spawn.usd_path,
                "drive_profile": args.profile,
                "source_gain_max_abs_error": float(gain_error),
                "source_effort_limit_max_abs_error_nm": float(limit_error),
                "focus_kp_nm_rad": float(kp[focus_id]),
                "focus_kd_nm_s_rad": float(kd[focus_id]),
                "focus_effort_limit_nm": float(effort_limit[focus_id]),
                "final_joint_abs_error_rad": abs(rows[-1]["q_rad"] - rows[-1]["command_rad"]),
                "final_five_finger_max_error_mm": rows[-1]["five_finger_max_error_mm"],
                "tail_focus_q_span_rad": max(row["q_rad"] for row in tail) - min(row["q_rad"] for row in tail),
                "tail_focus_max_abs_speed_rad_s": max(abs(row["qd_rad_s"]) for row in tail),
                "tail_right_hand_max_abs_speed_rad_s": max(row["max_right_hand_speed_rad_s"] for row in tail),
                "focus_peak_abs_computed_drive_nm": max(abs(row["computed_drive_nm"]) for row in after),
                "focus_peak_abs_applied_drive_nm": max(abs(row["applied_drive_nm"]) for row in after),
                "focus_at_effort_limit_steps": sum(row["at_effort_limit"] for row in after),
                "max_joint_limit_violation_rad": max(row["max_joint_limit_violation_rad"] for row in rows),
                "active_contacts_mean": float(np.mean([row["active_contacts"] for row in rows])),
                "active_contacts_max": max(row["active_contacts"] for row in rows),
            }
            output_dir = args.output_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "response.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary, indent=2))
        finally:
            raw_env.close()


if __name__ == "__main__":
    main()
