"""Bounded MuJoCo/Newton position-drive response and independent-control checks."""

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from dexgym.assets.marvin_wuji.joint_mapping import inspect_joint_contract, qpos_to_newton
from dexgym.assets.marvin_wuji.robot_cfg import make_robot_cfg, resolve_source_mjcf
from isaaclab.app import launch_simulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

DT = 0.002


def scenarios() -> list[tuple[str, tuple[str, ...], str, float]]:
    return [
        ("static_hold", ("Joint1_L",), "hold", 0.0),
        ("left_arm_step", ("Joint1_L",), "step", 0.1),
        ("right_arm_step", ("Joint1_R",), "step", -0.1),
        ("both_arms_step", ("Joint1_L", "Joint1_R"), "step", 0.1),
        ("left_hand_step", ("l_thumb_cmc_flex",), "step", 0.2),
        ("right_hand_step", ("r_thumb_cmc_flex",), "step", 0.2),
        ("left_wrist_j5", ("Joint5_L",), "step", 0.15),
        ("left_wrist_j6", ("Joint6_L",), "step", 0.15),
        ("left_wrist_j7", ("Joint7_L",), "step", 0.15),
        ("right_wrist_j5", ("Joint5_R",), "step", 0.15),
        ("right_wrist_j6", ("Joint6_R",), "step", 0.15),
        ("right_wrist_j7", ("Joint7_R",), "step", 0.15),
        ("left_wrist_combo", ("Joint5_L", "Joint6_L", "Joint7_L"), "step", 0.08),
        ("right_wrist_combo", ("Joint5_R", "Joint6_R", "Joint7_R"), "step", 0.08),
        ("left_arm_ramp", ("Joint2_L",), "ramp", 0.1),
        ("right_wrist_sine", ("Joint6_R",), "sine", 0.1),
    ]


def target_fraction(mode: str, step: int, steps: int) -> float:
    if mode == "hold":
        return 0.0
    if mode == "step":
        return float(step >= 20)
    if mode == "ramp":
        return min(1.0, step / max(1, steps - 1))
    if mode == "sine":
        return float(np.sin(2.0 * np.pi * step / 125.0))
    raise ValueError(mode)


def reset_home(robot, scene, sim, home: np.ndarray, contract) -> torch.Tensor:
    mapped = qpos_to_newton(home, contract, robot.joint_names)
    q = torch.as_tensor(mapped, dtype=torch.float32, device=robot.device).repeat(robot.num_instances, 1)
    robot.write_joint_position_to_sim_index(position=q)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
    scene.reset()
    sim.forward()
    scene.update(0.0)
    return q


def run_dynamics(model, contract, robot, scene, sim, home, steps: int, csv_path: Path) -> list[dict]:
    data = mujoco.MjData(model)
    joint_index = {name: robot.joint_names.index(name) for name in contract.names}
    records: list[dict] = []
    summaries: list[dict] = []
    for label, controlled, mode, amplitude in scenarios():
        q_home = reset_home(robot, scene, sim, home, contract)
        mujoco.mj_resetData(model, data)
        data.qpos[:] = home
        data.qvel[:] = 0
        for name in contract.names:
            data.ctrl[contract.ctrl_by_name[name]] = home[contract.qpos_by_name[name]]
        mujoco.mj_forward(model, data)
        newton_positions = []
        mujoco_positions = []
        targets = []
        max_jump = 0.0
        previous = q_home[0].detach().cpu().numpy().copy()
        max_all_joint_error = 0.0
        max_limit_violation = 0.0
        joint_limits = np.array([contract.limits_by_name[name] for name in robot.joint_names])
        for step in range(steps):
            target = q_home.clone()
            fraction = target_fraction(mode, step, steps)
            for name in controlled:
                lower, upper = contract.limits_by_name[name]
                value = np.clip(home[contract.qpos_by_name[name]] + amplitude * fraction, lower, upper)
                target[:, joint_index[name]] = float(value)
                data.ctrl[contract.ctrl_by_name[name]] = float(value)
            robot.actuators.target_command.set_position_index(value=target, full_data=True)
            actual_target = robot.actuators.target_command.position.torch
            if not torch.allclose(actual_target, target, atol=1e-6):
                raise AssertionError(f"Unexpected target propagation in {label}")
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(DT)
            mujoco.mj_step(model, data)
            q_newton = robot.data.joint_pos.torch[0].detach().cpu().numpy()
            v_newton = robot.data.joint_vel.torch[0].detach().cpu().numpy()
            if not np.isfinite(q_newton).all() or not np.isfinite(v_newton).all():
                raise AssertionError(f"Non-finite Newton joint state in {label} at step {step}")
            max_limit_violation = max(
                max_limit_violation,
                float(np.max(np.maximum(joint_limits[:, 0] - q_newton, q_newton - joint_limits[:, 1]))),
            )
            max_jump = max(max_jump, float(np.max(np.abs(q_newton - previous))))
            previous = q_newton.copy()
            mapped_mujoco = qpos_to_newton(data.qpos, contract, robot.joint_names)
            max_all_joint_error = max(max_all_joint_error, float(np.max(np.abs(q_newton - mapped_mujoco))))
            focus = controlled[0]
            index = joint_index[focus]
            nq = float(q_newton[index])
            mq = float(data.qpos[contract.qpos_by_name[focus]])
            desired = float(target[0, index])
            newton_positions.append(nq)
            mujoco_positions.append(mq)
            targets.append(desired)
            records.append(
                {
                    "scenario": label,
                    "step": step,
                    "time_s": round((step + 1) * DT, 6),
                    "joint": focus,
                    "target_rad": desired,
                    "newton_pos_rad": nq,
                    "mujoco_pos_rad": mq,
                    "newton_vel_rad_s": float(v_newton[index]),
                    "mujoco_vel_rad_s": float(data.qvel[contract.qpos_by_name[focus]]),
                }
            )
        final_newton = robot.data.joint_pos.torch[0].detach().cpu().numpy()
        final_mujoco = qpos_to_newton(data.qpos, contract, robot.joint_names)
        focus = controlled[0]
        idx = joint_index[focus]
        signed_step = np.sign(amplitude) or 1.0
        overshoot = (
            max(0.0, float(np.max(signed_step * (np.asarray(newton_positions[20:]) - np.asarray(targets[20:])))))
            if mode == "step"
            else None
        )
        settling_time_s = None
        if mode == "step":
            errors = np.abs(np.asarray(newton_positions) - np.asarray(targets))
            for settled_at in range(20, len(errors)):
                if np.all(errors[settled_at:] <= 0.02):
                    settling_time_s = round((settled_at + 1) * DT, 6)
                    break
        summaries.append(
            {
                "scenario": label,
                "controlled_joints": list(controlled),
                "mode": mode,
                "newton_final_position_rad": float(final_newton[idx]),
                "mujoco_final_position_rad": float(final_mujoco[idx]),
                "final_position_difference_rad": float(abs(final_newton[idx] - final_mujoco[idx])),
                "newton_tail_tracking_error_rad": float(
                    np.mean(np.abs(np.asarray(newton_positions[-25:]) - np.asarray(targets[-25:])))
                ),
                "mujoco_tail_tracking_error_rad": float(
                    np.mean(np.abs(np.asarray(mujoco_positions[-25:]) - np.asarray(targets[-25:])))
                ),
                "newton_step_overshoot_rad": overshoot,
                "newton_settling_time_s_at_0_02_rad": settling_time_s,
                "max_joint_jump_rad_per_step": max_jump,
                "max_joint_limit_violation_rad": max_limit_violation,
                "max_all_joint_backend_difference_rad": max_all_joint_error,
            }
        )
        if max_jump > 0.5:
            raise AssertionError(f"Abnormal joint jump in {label}: {max_jump} rad/step")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return summaries


def run_smoke(model, contract, robot, scene, sim, home: np.ndarray, steps: int, resets: int) -> dict:
    joint_index = {name: robot.joint_names.index(name) for name in contract.names}
    probes = ["Joint1_L", "Joint1_R", "l_thumb_cmc_flex", "r_thumb_cmc_flex", "Joint6_L", "Joint6_R"]
    max_jump = 0.0
    final_motion = []
    for repeat in range(resets):
        start = reset_home(robot, scene, sim, home, contract)
        target = start.clone()
        for env in range(robot.num_instances):
            name = probes[env % len(probes)]
            lower, upper = contract.limits_by_name[name]
            idx = joint_index[name]
            target[env, idx] = float(np.clip(float(start[env, idx]) + 0.12, lower, upper))
        robot.actuators.target_command.set_position_index(value=target, full_data=True)
        if not torch.allclose(robot.actuators.target_command.position.torch, target, atol=1e-6):
            raise AssertionError("54-channel target buffer did not preserve independent environment commands")
        previous = start.detach().cpu().numpy().copy()
        for _ in range(steps):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(DT)
            current = robot.data.joint_pos.torch.detach().cpu().numpy()
            velocity = robot.data.joint_vel.torch.detach().cpu().numpy()
            if not np.isfinite(current).all() or not np.isfinite(velocity).all():
                raise AssertionError("Non-finite Newton state during multi-environment smoke")
            max_jump = max(max_jump, float(np.max(np.abs(current - previous))))
            previous = current.copy()
        deltas = np.abs(previous - start.detach().cpu().numpy())
        final_motion.append(
            float(np.mean([deltas[env, joint_index[probes[env % len(probes)]]] for env in range(robot.num_instances)]))
        )
    if max_jump > 0.5 or min(final_motion) < 0.001:
        raise AssertionError(f"Multi-environment motion failure: max_jump={max_jump}, motion={final_motion}")
    return {
        "num_envs": robot.num_instances,
        "joints_per_env": robot.num_joints,
        "resets": resets,
        "steps_per_reset": steps,
        "probed_joint_names": probes,
        "mean_commanded_joint_motion_rad": final_motion,
        "max_joint_jump_rad_per_step": max_jump,
        "finite": True,
        "pass": True,
    }


def run_all_channels(contract, robot, scene, sim, home: np.ndarray, steps: int) -> dict:
    """Drive every joint with opposed commands in paired environments."""
    if robot.num_instances < 2 or robot.num_instances % 2:
        raise ValueError("All-channel test requires an even number of environments >= 2")
    names = list(contract.names)
    per_batch = robot.num_instances // 2
    responses = []
    max_jump = 0.0
    for start in range(0, len(names), per_batch):
        selected = names[start : start + per_batch]
        initial = reset_home(robot, scene, sim, home, contract)
        target = initial.clone()
        for pair, name in enumerate(selected):
            index = robot.joint_names.index(name)
            lower, upper = contract.limits_by_name[name]
            value = float(initial[2 * pair, index])
            target[2 * pair, index] = float(np.clip(value + 0.1, lower, upper))
            target[2 * pair + 1, index] = float(np.clip(value - 0.1, lower, upper))
        robot.actuators.target_command.set_position_index(value=target, full_data=True)
        if not torch.allclose(robot.actuators.target_command.position.torch, target, atol=1e-6):
            raise AssertionError("A target leaked to a different joint or environment")
        previous = initial.detach().cpu().numpy().copy()
        for _ in range(steps):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(DT)
            current = robot.data.joint_pos.torch.detach().cpu().numpy()
            velocity = robot.data.joint_vel.torch.detach().cpu().numpy()
            if not np.isfinite(current).all() or not np.isfinite(velocity).all():
                raise AssertionError("Non-finite state in all-channel test")
            max_jump = max(max_jump, float(np.max(np.abs(current - previous))))
            previous = current.copy()
        for pair, name in enumerate(selected):
            index = robot.joint_names.index(name)
            separation = float(previous[2 * pair, index] - previous[2 * pair + 1, index])
            responses.append({"joint": name, "positive_minus_negative_rad": separation})
    failures = [entry for entry in responses if entry["positive_minus_negative_rad"] < 0.005]
    return {
        "num_envs": robot.num_instances,
        "joint_count": len(responses),
        "steps_per_batch": steps,
        "min_opposed_response_rad": min(entry["positive_minus_negative_rad"] for entry in responses),
        "max_joint_jump_rad_per_step": max_jump,
        "failed_joints": failures,
        "responses": responses,
        "pass": not failures and max_jump < 0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf")
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--mode", choices=("dynamics", "smoke", "channels"), required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--resets", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path, default=Path("artifacts/marvin_wuji/dynamics.csv"))
    args = parser.parse_args()
    mjcf = resolve_source_mjcf(args.mjcf)
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    contract = inspect_joint_contract(model)
    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    home = model.key_qpos[home_key].copy()

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = make_robot_cfg(args.usd)

    sim_cfg = SimulationCfg(
        dt=DT,
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=1000, nconmax=500), use_cuda_graph=False),
    )
    with launch_simulation(sim_cfg):
        sim = SimulationContext(sim_cfg)
        scene = InteractiveScene(SceneCfg(num_envs=args.num_envs, env_spacing=3.0))
        sim.reset()
        robot = scene["robot"]
        if robot.num_joints != 54 or set(robot.joint_names) != set(contract.names):
            raise ValueError("Newton articulation does not match the 54 DoF source contract")
        if args.mode == "smoke":
            result = run_smoke(model, contract, robot, scene, sim, home, args.steps, args.resets)
        elif args.mode == "channels":
            result = run_all_channels(contract, robot, scene, sim, home, args.steps)
        else:
            if args.num_envs != 1:
                parser.error("Dynamics comparison currently uses one environment")
            result = {"scenarios": run_dynamics(model, contract, robot, scene, sim, home, args.steps, args.csv)}
            result["pass"] = True
            result["csv"] = str(args.csv)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
