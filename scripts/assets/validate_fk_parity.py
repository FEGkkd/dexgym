"""Compare 1,000+ named body poses from MuJoCo and Newton MJWarp forward kinematics."""

import argparse
import hashlib
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


def sample_poses(model: mujoco.MjModel, count: int, seed: int = 42) -> tuple[np.ndarray, list[str], int]:
    """Named keyframes, wrist/boundary probes, and uniform legal joint samples."""
    contract = inspect_joint_contract(model)
    poses: list[np.ndarray] = []
    labels: list[str] = []
    for name in ("hardware_zero", "home", "measured_j5_zero", "forward_pose"):
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key < 0:
            raise ValueError(f"Missing source keyframe: {name}")
        poses.append(model.key_qpos[key].copy())
        labels.append(name)
    home = poses[1]
    for side in ("L", "R"):
        for joint in (5, 6, 7):
            name = f"Joint{joint}_{side}"
            low, high = contract.limits_by_name[name]
            idx = contract.qpos_by_name[name]
            for direction in (-1, 1):
                sample = home.copy()
                sample[idx] = np.clip(sample[idx] + direction * 0.3, low + 1e-4, high - 1e-4)
                poses.append(sample)
                labels.append(f"{name}_{direction:+d}")
    for side in ("L", "R"):
        for joint in range(1, 8):
            name = f"Joint{joint}_{side}"
            low, high = contract.limits_by_name[name]
            for fraction in (0.025, 0.975):
                sample = home.copy()
                sample[contract.qpos_by_name[name]] = low + fraction * (high - low)
                poses.append(sample)
                labels.append(f"{name}_boundary_{fraction}")
    rng = np.random.default_rng(seed)
    lower = np.empty(54)
    upper = np.empty(54)
    for name in contract.names:
        lower[contract.qpos_by_name[name]], upper[contract.qpos_by_name[name]] = contract.limits_by_name[name]
    rejected = 0
    while len(poses) < count:
        sample = lower + rng.uniform(0.05, 0.95, 54) * (upper - lower)
        if np.any(sample < lower) or np.any(sample > upper):
            rejected += 1
            continue
        poses.append(sample)
        labels.append("random")
    result = np.stack(poses[:count])
    return result, labels[:count], rejected


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
    }


def validate(mjcf: Path, usd: Path, samples: int, num_envs: int) -> dict:
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    contract = inspect_joint_contract(model)
    poses, labels, rejected = sample_poses(model, samples)
    source_body_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid): bid for bid in range(1, model.nbody)}

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = make_robot_cfg(usd)

    sim_cfg = SimulationCfg(
        dt=0.002,
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=1000, nconmax=500), use_cuda_graph=False),
    )
    tcp_pos: list[float] = []
    tcp_ang: list[float] = []
    all_pos: list[float] = []
    all_ang: list[float] = []
    failures: list[dict] = []
    with launch_simulation(sim_cfg):
        sim = SimulationContext(sim_cfg)
        scene = InteractiveScene(SceneCfg(num_envs=num_envs, env_spacing=3.0))
        sim.reset()
        robot = scene["robot"]
        if robot.num_joints != 54 or set(robot.joint_names) != set(contract.names):
            raise ValueError("Newton articulation lost or renamed one of the 54 joints")
        body_ids = [source_body_names[name] for name in robot.body_names]
        tcp_ids = [robot.body_names.index(name) for name in ("TCP_Link_L", "TCP_Link_R")]
        origins = scene.env_origins.detach().cpu().numpy()
        for start in range(0, samples, num_envs):
            count = min(num_envs, samples - start)
            batch = np.repeat(poses[1][None, :], num_envs, axis=0)
            batch[:count] = poses[start : start + count]
            mapped = qpos_to_newton(batch, contract, robot.joint_names)
            position = torch.as_tensor(mapped, dtype=torch.float32, device=robot.device)
            robot.write_joint_position_to_sim_index(position=position)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(position))
            sim.forward()
            scene.update(0.0)
            actual = robot.data.body_pose_w.torch[:count].detach().cpu().numpy()
            actual[:, :, :3] -= origins[:count, None, :]
            for env in range(count):
                index = start + env
                data.qpos[:] = poses[index]
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                expected_pos = data.xpos[body_ids]
                expected_quat = wxyz_to_xyzw(data.xquat[body_ids])
                position_error_mm = np.linalg.norm(actual[env, :, :3] - expected_pos, axis=1) * 1000.0
                angle_error_deg = angular_distance_deg(actual[env, :, 3:7], expected_quat)
                all_pos.extend(position_error_mm.tolist())
                all_ang.extend(angle_error_deg.tolist())
                tcp_pos.extend(position_error_mm[tcp_ids].tolist())
                tcp_ang.extend(angle_error_deg[tcp_ids].tolist())
                for side, body_index in zip(("left", "right"), tcp_ids, strict=True):
                    pos_error = float(position_error_mm[body_index])
                    ang_error = float(angle_error_deg[body_index])
                    if pos_error >= 1.0 or ang_error >= 0.1:
                        failures.append(
                            {
                                "sample": index,
                                "label": labels[index],
                                "side": side,
                                "position_error_mm": pos_error,
                                "orientation_error_deg": ang_error,
                            }
                        )
    return {
        "source_mjcf_sha256": hashlib.sha256(mjcf.read_bytes()).hexdigest(),
        "samples": samples,
        "parallel_environments": num_envs,
        "body_count": len(body_ids),
        "rejected_samples": rejected,
        "tcp_position_error_mm": summarize(np.array(tcp_pos)),
        "tcp_orientation_error_deg": summarize(np.array(tcp_ang)),
        "all_body_position_error_mm": summarize(np.array(all_pos)),
        "all_body_orientation_error_deg": summarize(np.array(all_ang)),
        "thresholds": {"tcp_position_mm": 1.0, "tcp_orientation_deg": 0.1},
        "failed_tcp_comparisons": len(failures),
        "failures": failures[:100],
        "pass": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf")
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 1000 or args.num_envs < 1:
        parser.error("FK acceptance requires at least 1,000 samples and positive --num-envs")
    result = validate(resolve_source_mjcf(args.mjcf), args.usd.resolve(), args.samples, args.num_envs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "failures"}, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
