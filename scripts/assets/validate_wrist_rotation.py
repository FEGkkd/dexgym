"""Damped-Jacobian fixed-TCP wrist trajectory and MuJoCo/Newton execution audit."""

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from dexgym.assets.marvin_wuji.frames import angular_distance_deg, wxyz_to_xyzw
from dexgym.assets.marvin_wuji.joint_mapping import inspect_joint_contract, qpos_to_newton
from dexgym.assets.marvin_wuji.robot_cfg import make_robot_cfg, resolve_source_mjcf
from isaaclab.app import launch_simulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from scipy.spatial.transform import Rotation

DT = 0.002


def solve_pose(model, data, side, q_seed, position_goal, rotation_goal) -> tuple[np.ndarray, int]:
    """Solve one feasible arm-only waypoint while preserving all hand joints."""
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"Joint{i}_{side}") for i in range(1, 8)]
    q_ids = [int(model.jnt_qposadr[jid]) for jid in joint_ids]
    v_ids = [int(model.jnt_dofadr[jid]) for jid in joint_ids]
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"TCP_Link_{side}")
    data.qpos[:] = q_seed
    data.qvel[:] = 0
    for iteration in range(150):
        mujoco.mj_forward(model, data)
        position_error = position_goal - data.xpos[tcp_id]
        current = Rotation.from_quat(wxyz_to_xyzw(data.xquat[tcp_id]))
        rotation_error = (rotation_goal * current.inv()).as_rotvec()
        if np.linalg.norm(position_error) < 2e-5 and np.linalg.norm(rotation_error) < 1e-4:
            return data.qpos.copy(), iteration
        jac_pos = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jac_pos, jac_rot, tcp_id)
        jacobian = np.vstack((jac_pos[:, v_ids], jac_rot[:, v_ids]))
        residual = np.r_[position_error, rotation_error]
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-4 * np.eye(6), residual)
        data.qpos[q_ids] = np.clip(
            data.qpos[q_ids] + np.clip(delta, -0.03, 0.03),
            model.jnt_range[joint_ids, 0] + 0.01,
            model.jnt_range[joint_ids, 1] - 0.01,
        )
    raise ValueError(f"IK did not converge for {side}: position={position_error}, rotation={rotation_error}")


def build_trajectory(model, home, side, steps):
    data = mujoco.MjData(model)
    data.qpos[:] = home
    mujoco.mj_forward(model, data)
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"TCP_Link_{side}")
    position_goal = data.xpos[tcp_id].copy()
    initial_rotation = Rotation.from_quat(wxyz_to_xyzw(data.xquat[tcp_id]))
    signed_angle = 0.25 if side == "L" else -0.25
    waypoints = [home.copy()]
    iterations = []
    for fraction in np.linspace(0.01, 1.0, 100):
        goal = Rotation.from_rotvec([0.0, 0.0, signed_angle * fraction]) * initial_rotation
        solved, count = solve_pose(model, data, side, waypoints[-1], position_goal, goal)
        waypoints.append(solved)
        iterations.append(count)
    command = []
    for step in range(steps):
        progress = min(100.0, 100.0 * step / max(1.0, steps * 0.6))
        low = int(progress)
        high = min(100, low + 1)
        command.append(waypoints[low] * (1.0 - (progress - low)) + waypoints[high] * (progress - low))
    return np.stack(command), position_goal, initial_rotation, signed_angle, max(iterations)


def run_side(model, contract, home, side, steps, robot, scene, sim):
    commands, position_goal, initial_rotation, signed_angle, max_ik_iterations = build_trajectory(
        model, home, side, steps
    )
    data = mujoco.MjData(model)
    data.qpos[:] = home
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    target_data = mujoco.MjData(model)
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"TCP_Link_{side}")
    newton_tcp_id = robot.body_names.index(f"TCP_Link_{side}")
    origin = scene.env_origins[0].detach().cpu().numpy()
    initial_q = torch.as_tensor(
        qpos_to_newton(home, contract, robot.joint_names), dtype=torch.float32, device=robot.device
    )[None, :]
    robot.write_joint_position_to_sim_index(position=initial_q)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(initial_q))
    scene.reset()
    sim.forward()
    scene.update(0.0)
    max_kinematic_position_mm = 0.0
    max_kinematic_orientation_deg = 0.0
    max_jump = 0.0
    max_limit_violation_rad = 0.0
    joint_limits = np.array([contract.limits_by_name[name] for name in robot.joint_names])
    previous_q = initial_q[0].detach().cpu().numpy().copy()
    records = []
    for step, q_command in enumerate(commands):
        target_data.qpos[:] = q_command
        mujoco.mj_forward(model, target_data)
        progress = min(1.0, step / max(1.0, steps * 0.6))
        desired_rotation = Rotation.from_rotvec([0.0, 0.0, signed_angle * progress]) * initial_rotation
        kinematic_position_mm = 1000.0 * np.linalg.norm(target_data.xpos[tcp_id] - position_goal)
        kinematic_orientation_deg = float(
            np.rad2deg(
                np.linalg.norm(
                    (desired_rotation * Rotation.from_quat(wxyz_to_xyzw(target_data.xquat[tcp_id])).inv()).as_rotvec()
                )
            )
        )
        max_kinematic_position_mm = max(max_kinematic_position_mm, kinematic_position_mm)
        max_kinematic_orientation_deg = max(max_kinematic_orientation_deg, kinematic_orientation_deg)
        newton_target = torch.as_tensor(
            qpos_to_newton(q_command, contract, robot.joint_names), dtype=torch.float32, device=robot.device
        )[None, :]
        robot.actuators.target_command.set_position_index(value=newton_target, full_data=True)
        for name in contract.names:
            data.ctrl[contract.ctrl_by_name[name]] = q_command[contract.qpos_by_name[name]]
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)
        mujoco.mj_step(model, data)
        q_newton = robot.data.joint_pos.torch[0].detach().cpu().numpy()
        v_newton = robot.data.joint_vel.torch[0].detach().cpu().numpy()
        if not np.isfinite(q_newton).all() or not np.isfinite(v_newton).all():
            raise AssertionError(f"Non-finite Newton state in {side} wrist trajectory")
        max_limit_violation_rad = max(
            max_limit_violation_rad,
            float(np.max(np.maximum(joint_limits[:, 0] - q_newton, q_newton - joint_limits[:, 1]))),
        )
        max_jump = max(max_jump, float(np.max(np.abs(q_newton - previous_q))))
        previous_q = q_newton.copy()
        newton_pose = robot.data.body_pose_w.torch[0, newton_tcp_id].detach().cpu().numpy().copy()
        newton_pose[:3] -= origin
        mujoco_pose = np.r_[data.xpos[tcp_id], wxyz_to_xyzw(data.xquat[tcp_id])]
        goal_quat = desired_rotation.as_quat()
        records.append(
            {
                "side": side,
                "step": step,
                "time_s": round((step + 1) * DT, 6),
                "kinematic_target_position_error_mm": kinematic_position_mm,
                "kinematic_target_orientation_error_deg": kinematic_orientation_deg,
                "newton_execution_position_error_mm": float(1000 * np.linalg.norm(newton_pose[:3] - position_goal)),
                "mujoco_execution_position_error_mm": float(1000 * np.linalg.norm(mujoco_pose[:3] - position_goal)),
                "newton_execution_orientation_error_deg": float(angular_distance_deg(newton_pose[3:], goal_quat)),
                "mujoco_execution_orientation_error_deg": float(angular_distance_deg(mujoco_pose[3:], goal_quat)),
                "backend_tcp_position_difference_mm": float(1000 * np.linalg.norm(newton_pose[:3] - mujoco_pose[:3])),
                "backend_tcp_orientation_difference_deg": float(angular_distance_deg(newton_pose[3:], mujoco_pose[3:])),
            }
        )
    if (
        max_kinematic_position_mm >= 1.0
        or max_kinematic_orientation_deg >= 0.1
        or max_jump >= 0.5
        or max_limit_violation_rad > 1e-4
    ):
        raise AssertionError("Wrist command path violates fixed-TCP, orientation, continuity, or joint limits")
    result = {
        "side": side,
        "rotation_command_deg": float(np.rad2deg(signed_angle)),
        "max_ik_iterations": max_ik_iterations,
        "max_kinematic_position_error_mm": max_kinematic_position_mm,
        "max_kinematic_orientation_error_deg": max_kinematic_orientation_deg,
        "max_joint_jump_rad_per_step": max_jump,
        "final_execution": {key: value for key, value in records[-1].items() if "execution" in key or "backend" in key},
        "max_joint_limit_violation_rad": max_limit_violation_rad,
        "joint_limits_respected": max_limit_violation_rad <= 1e-4,
        "pass_kinematic": True,
    }
    return result, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf")
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--output", type=Path, default=Path("artifacts/marvin_wuji/wrist_rotation.json"))
    parser.add_argument("--csv", type=Path, default=Path("artifacts/marvin_wuji/wrist_rotation.csv"))
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(resolve_source_mjcf(args.mjcf)))
    contract = inspect_joint_contract(model)
    home = model.key_qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")].copy()

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = make_robot_cfg(args.usd)

    sim_cfg = SimulationCfg(
        dt=DT,
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=1000, nconmax=500), use_cuda_graph=False),
    )
    with launch_simulation(sim_cfg):
        sim = SimulationContext(sim_cfg)
        scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=3.0))
        sim.reset()
        robot = scene["robot"]
        results = []
        rows = []
        for side in ("L", "R"):
            result, side_rows = run_side(model, contract, home, side, args.steps, robot, scene, sim)
            results.append(result)
            rows.extend(side_rows)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"sides": results, "csv": str(args.csv)}, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
