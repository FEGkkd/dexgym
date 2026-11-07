"""Compare one fixed index MCP position step in Newton and source MuJoCo."""

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
from dexgym.tasks.direct.right_hand_tracking.env_cfg import RightHandTrackingDirectCfg
from isaaclab.app import launch_simulation
from isaaclab.cli import load_external_tasks

FOCUS = "r_index_finger_mcp_flex"
TIP_SITES = tuple(f"r_{finger}_tip" for finger in ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    parser.add_argument("--step-rad", type=float, default=0.1)
    parser.add_argument("--match-source-gains", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/right_index_mcp_backends"))
    args = parser.parse_args()
    if args.hold_seconds <= 0 or args.step_rad <= 0:
        parser.error("--hold-seconds and --step-rad must be positive")

    load_external_tasks()
    cfg = RightHandTrackingDirectCfg()
    cfg.scene.num_envs = 1
    cfg.seed = 42
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    control_dt = cfg.sim.dt * cfg.decimation
    control_steps = round(args.hold_seconds / control_dt)
    if abs(control_steps * control_dt - args.hold_seconds) > 1e-8:
        parser.error("--hold-seconds must be an integer number of 30-Hz control steps")

    source_mjcf = resolve_source_mjcf()
    model = mujoco.MjModel.from_xml_path(str(source_mjcf))
    contract = inspect_joint_contract(model)
    model.opt.timestep = cfg.sim.dt
    model.opt.gravity[:] = cfg.sim.gravity
    data = mujoco.MjData(model)
    tip_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in TIP_SITES]
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wrist")
    if min(tip_ids) < 0 or wrist_id < 0:
        raise ValueError("Source MJCF lacks the expected wrist or fingertip sites")

    def mujoco_tips_in_wrist() -> np.ndarray:
        wrist_position = data.xpos[wrist_id].copy()
        wrist_rotation = data.xmat[wrist_id].reshape(3, 3).copy()
        return (data.site_xpos[tip_ids] - wrist_position) @ wrist_rotation

    with launch_simulation(cfg):
        raw_env = gym.make(RIGHT_HAND_ID, cfg=cfg)
        try:
            task = raw_env.unwrapped
            raw_env.reset()
            robot, scene, sim = task.robot, task.scene, task.sim
            focus_newton = robot.joint_names.index(FOCUS)
            focus_mujoco = contract.qpos_by_name[FOCUS]
            focus_ctrl = contract.ctrl_by_name[FOCUS]
            q_start = task.home_q[None].clone()
            q_target = q_start.clone()
            q_target[:, focus_newton] += args.step_rad
            lower, upper = robot.data.joint_pos_limits.torch[0, focus_newton].tolist()
            if not lower <= float(q_target[0, focus_newton]) <= upper:
                raise ValueError(f"Requested step exceeds {FOCUS} limits [{lower}, {upper}]")
            q_start_mj = np.empty(model.nq)
            q_target_mj = np.empty(model.nq)
            for name in contract.names:
                q_start_mj[contract.qpos_by_name[name]] = float(q_start[0, robot.joint_names.index(name)])
                q_target_mj[contract.qpos_by_name[name]] = float(q_target[0, robot.joint_names.index(name)])

            def write_newton_pose(q: torch.Tensor) -> None:
                robot.write_joint_position_to_sim_index(position=q)
                robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
                scene.write_data_to_sim()
                sim.forward()
                scene.update(0.0)

            write_newton_pose(q_target)
            newton_goal_tips = task._current_tips()[0].detach().cpu().numpy().copy()
            write_newton_pose(q_start)
            mujoco.mj_resetData(model, data)
            data.qpos[:] = q_target_mj
            data.qvel[:] = 0.0
            # MuJoCo ctrl order is not joint order; use the validated map.
            for name in contract.names:
                data.ctrl[contract.ctrl_by_name[name]] = q_target_mj[contract.qpos_by_name[name]]
            mujoco.mj_forward(model, data)
            mujoco_goal_tips = mujoco_tips_in_wrist()
            data.qpos[:] = q_start_mj
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            kp_newton = robot.data.joint_stiffness.torch[0].detach().cpu().numpy().copy()
            kd_newton = robot.data.joint_damping.torch[0].detach().cpu().numpy().copy()
            limit_newton = robot.data.joint_effort_limits.torch[0].detach().cpu().numpy().copy()
            kp_mujoco = np.asarray(
                [model.actuator_gainprm[contract.ctrl_by_name[name], 0] for name in robot.joint_names]
            )
            kd_mujoco = np.asarray(
                [-model.actuator_biasprm[contract.ctrl_by_name[name], 2] for name in robot.joint_names]
            )
            limit_mujoco = np.asarray(
                [model.actuator_forcerange[contract.ctrl_by_name[name], 1] for name in robot.joint_names]
            )
            imported_kp = kp_newton.copy()
            imported_kd = kd_newton.copy()
            if args.match_source_gains:
                robot.write_joint_stiffness_to_sim_index(
                    stiffness=torch.as_tensor(kp_mujoco, dtype=torch.float32, device=task.device)[None]
                )
                robot.write_joint_damping_to_sim_index(
                    damping=torch.as_tensor(kd_mujoco, dtype=torch.float32, device=task.device)[None]
                )
                kp_newton = robot.data.joint_stiffness.torch[0].detach().cpu().numpy().copy()
                kd_newton = robot.data.joint_damping.torch[0].detach().cpu().numpy().copy()
                if not np.allclose(kp_newton, kp_mujoco, rtol=1e-5, atol=1e-5) or not np.allclose(
                    kd_newton, kd_mujoco, rtol=1e-5, atol=1e-5
                ):
                    raise AssertionError("Newton did not accept source MuJoCo joint gains")
                write_newton_pose(q_start)
            parity = {
                "matched_source_gains_in_newton": args.match_source_gains,
                "imported_all_joint_kp_max_abs_difference": float(np.max(np.abs(imported_kp - kp_mujoco))),
                "imported_all_joint_kd_max_abs_difference": float(np.max(np.abs(imported_kd - kd_mujoco))),
                "all_joint_kp_max_abs_difference": float(np.max(np.abs(kp_newton - kp_mujoco))),
                "all_joint_kd_max_abs_difference": float(np.max(np.abs(kd_newton - kd_mujoco))),
                "all_joint_effort_limit_max_abs_difference": float(np.max(np.abs(limit_newton - limit_mujoco))),
                "focus_newton_kp": float(kp_newton[focus_newton]),
                "focus_source_mujoco_kp": float(kp_mujoco[focus_newton]),
                "focus_newton_kd": float(kd_newton[focus_newton]),
                "focus_source_mujoco_kd": float(kd_mujoco[focus_newton]),
                "focus_newton_effort_limit_nm": float(limit_newton[focus_newton]),
                "focus_source_mujoco_effort_limit_nm": float(limit_mujoco[focus_newton]),
            }

            records = []
            steps_per_call = cfg.decimation if task._physics_handles_decimation else 1
            with torch.inference_mode():
                for step in range(control_steps + 1):
                    nq = float(robot.data.joint_pos.torch[0, focus_newton])
                    nv = float(robot.data.joint_vel.torch[0, focus_newton])
                    nc = float(robot.actuators.computed_effort.torch[0, focus_newton])
                    na = float(robot.actuators.applied_effort.torch[0, focus_newton])
                    mq = float(data.qpos[focus_mujoco])
                    mv = float(data.qvel[focus_mujoco])
                    ma = float(data.actuator_force[focus_ctrl])
                    me = float(data.qfrc_actuator[focus_mujoco])
                    ntip = task._current_tips()[0].detach().cpu().numpy()
                    mtip = mujoco_tips_in_wrist()
                    records.append(
                        {
                            "time_s": step * control_dt,
                            "target_rad": float(q_target[0, focus_newton]),
                            "newton_q_rad": nq,
                            "newton_qd_rad_s": nv,
                            "newton_computed_drive_nm": nc,
                            "newton_applied_drive_nm": na,
                            "newton_effort_limit_nm": float(limit_newton[focus_newton]),
                            "newton_drive_at_limit": abs(na) >= 0.99 * limit_newton[focus_newton],
                            "newton_five_finger_max_error_mm": float(
                                np.linalg.norm(ntip - newton_goal_tips, axis=-1).max() * 1000
                            ),
                            "mujoco_q_rad": mq,
                            "mujoco_qd_rad_s": mv,
                            "mujoco_actuator_force_nm": ma,
                            "mujoco_joint_drive_nm": me,
                            "mujoco_effort_limit_nm": float(limit_mujoco[focus_newton]),
                            "mujoco_drive_at_limit": abs(ma) >= 0.99 * limit_mujoco[focus_newton],
                            "mujoco_five_finger_max_error_mm": float(
                                np.linalg.norm(mtip - mujoco_goal_tips, axis=-1).max() * 1000
                            ),
                        }
                    )
                    if step == control_steps:
                        break
                    robot.actuators.target_command.set_position_index(value=q_target, full_data=True)
                    if not torch.allclose(robot.actuators.target_command.position.torch, q_target, atol=1e-6):
                        raise AssertionError("Newton actuator target was not propagated")
                    for _ in range(cfg.decimation // steps_per_call):
                        scene.write_data_to_sim()
                        sim.step(render=False)
                        scene.update(cfg.sim.dt * steps_per_call)
                    for _ in range(cfg.decimation):
                        mujoco.mj_step(model, data)

            output_dir = args.output_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "response.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
            after = records[1:]
            tail = records[-10:]
            summary = {
                "focus_joint": FOCUS,
                "source_mjcf": str(source_mjcf),
                "newton_usd": cfg.scene.robot.spawn.usd_path,
                "initial_rad": float(q_start[0, focus_newton]),
                "target_rad": float(q_target[0, focus_newton]),
                "hold_seconds": args.hold_seconds,
                "physics_dt_s": cfg.sim.dt,
                "control_dt_s": control_dt,
                "gravity_m_s2": list(cfg.sim.gravity),
                "newton_mujoco_target_fingertip_fk_max_difference_mm": float(
                    np.linalg.norm(newton_goal_tips - mujoco_goal_tips, axis=-1).max() * 1000
                ),
                "drive_parity": parity,
                "newton_final_q_rad": records[-1]["newton_q_rad"],
                "mujoco_final_q_rad": records[-1]["mujoco_q_rad"],
                "newton_final_abs_joint_error_rad": abs(records[-1]["newton_q_rad"] - records[-1]["target_rad"]),
                "mujoco_final_abs_joint_error_rad": abs(records[-1]["mujoco_q_rad"] - records[-1]["target_rad"]),
                "newton_final_five_finger_max_error_mm": records[-1]["newton_five_finger_max_error_mm"],
                "mujoco_final_five_finger_max_error_mm": records[-1]["mujoco_five_finger_max_error_mm"],
                "newton_max_abs_velocity_rad_s": max(abs(row["newton_qd_rad_s"]) for row in after),
                "mujoco_max_abs_velocity_rad_s": max(abs(row["mujoco_qd_rad_s"]) for row in after),
                "newton_final_velocity_rad_s": records[-1]["newton_qd_rad_s"],
                "mujoco_final_velocity_rad_s": records[-1]["mujoco_qd_rad_s"],
                "newton_tail_q_span_rad": max(row["newton_q_rad"] for row in tail)
                - min(row["newton_q_rad"] for row in tail),
                "mujoco_tail_q_span_rad": max(row["mujoco_q_rad"] for row in tail)
                - min(row["mujoco_q_rad"] for row in tail),
                "newton_tail_max_abs_velocity_rad_s": max(abs(row["newton_qd_rad_s"]) for row in tail),
                "mujoco_tail_max_abs_velocity_rad_s": max(abs(row["mujoco_qd_rad_s"]) for row in tail),
                "newton_peak_abs_applied_drive_nm": max(abs(row["newton_applied_drive_nm"]) for row in after),
                "mujoco_peak_abs_drive_nm": max(abs(row["mujoco_actuator_force_nm"]) for row in after),
                "newton_at_limit_steps": int(sum(row["newton_drive_at_limit"] for row in after)),
                "mujoco_at_limit_steps": int(sum(row["mujoco_drive_at_limit"] for row in after)),
                "control_steps": control_steps,
            }
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary, indent=2))
        finally:
            raw_env.close()


if __name__ == "__main__":
    main()
