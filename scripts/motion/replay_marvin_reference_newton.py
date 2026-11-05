"""Step one real Marvin 54 DoF reference in Isaac Lab 3.0 Newton MJWarp."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from dexgym.assets.marvin_wuji.controller import apply_newton_arm_tracking_profile
from dexgym.assets.marvin_wuji.robot_cfg import make_robot_cfg
from isaaclab.app import launch_simulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from scipy.spatial.transform import Rotation

DT = 0.002
SOURCE_MJCF = Path("/home/ubuntu/dexgym/source/dexgym/assets/marvin_wuji/source_model/dual_arm_marvin.xml")


def stats(x):
    return {"mean": float(np.mean(x)), "p95": float(np.percentile(x, 95)), "max": float(np.max(x))}


def site_from_body(body_pose: np.ndarray, site_pos: np.ndarray, site_quat_xyzw: np.ndarray):
    """Apply the pinned MJCF TCP_Link→tcp fixed transform to Newton body pose."""
    r = Rotation.from_quat(body_pose[..., 3:7].reshape(-1, 4))
    offset = np.broadcast_to(site_pos, body_pose[..., :3].shape).reshape(-1, 3).copy()
    pos = body_pose[..., :3] + r.apply(offset).reshape(body_pose[..., :3].shape)
    quat = (r * Rotation.from_quat(site_quat_xyzw)).as_quat().reshape(body_pose[..., 3:7].shape)
    return pos, quat


def run(reference: Path, usd: Path, output: Path, *, num_envs: int = 1, max_frames: int | None = None):
    ref = np.load(reference, allow_pickle=False)
    time = ref["time"][:max_frames]
    qref = ref["q_full"][:max_frames]
    if len(time) < 2 or num_envs < 1 or not np.allclose(np.diff(time), 0.02, atol=1e-10):
        raise ValueError("Need exact 50 Hz reference and positive num_envs")
    model = mujoco.MjModel.from_xml_path(str(SOURCE_MJCF))
    model_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(54)]
    if tuple(model_names) != tuple(ref["joint_names_full"]):
        raise ValueError("Reference qpos name mismatch")
    qadr = {name: int(model.jnt_qposadr[i]) for i, name in enumerate(model_names)}
    arm_names = tuple(ref["joint_names_arm"])
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"tcp_{s}") for s in ("L", "R")]
    body_names = [f"TCP_Link_{s}" for s in ("L", "R")]
    for sid, name in zip(site_ids, body_names):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if model.site_bodyid[sid] != bid:
            raise ValueError("TCP site is not on expected body")
    site_pos = np.stack([model.site_pos[sid] for sid in site_ids])
    site_quat = np.stack([model.site_quat[sid][[1, 2, 3, 0]] for sid in site_ids])

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = make_robot_cfg(usd)

    sim_cfg = SimulationCfg(
        dt=DT, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=1000, nconmax=500), use_cuda_graph=False)
    )
    n = len(time)
    q_actual = np.zeros((n, num_envs, 54), dtype=np.float32)
    v_actual = np.zeros_like(q_actual)
    torque = np.zeros_like(q_actual)
    tcp_pos = np.zeros((n, num_envs, 2, 3), dtype=np.float32)
    tcp_quat = np.zeros((n, num_envs, 2, 4), dtype=np.float32)
    contacts = np.zeros(n, dtype=np.int32)
    source_actual_contact_frames = 0
    source_data = mujoco.MjData(model)
    parity = []
    sat_count = 0
    effort_samples = 0
    physics_steps = 0
    with launch_simulation(sim_cfg):
        sim = SimulationContext(sim_cfg)
        scene = InteractiveScene(SceneCfg(num_envs=num_envs, env_spacing=3.0))
        sim.reset()
        robot = scene["robot"]
        if robot.num_joints != 54 or set(robot.joint_names) != set(model_names):
            raise ValueError("Newton 54 joint contract mismatch")
        order = [qadr[name] for name in robot.joint_names]
        inverse = [robot.joint_names.index(name) for name in model_names]
        arm_index = [robot.joint_names.index(name) for name in arm_names]
        joint_limits = np.array(
            [model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in robot.joint_names]
        )
        tcp_body_ids = [robot.body_names.index(name) for name in body_names]
        origins = scene.env_origins.detach().cpu().numpy()
        solver = sim.physics_manager._solver

        def write_q(frame):
            q = torch.as_tensor(qref[frame, order], dtype=torch.float32, device=robot.device)[None, :].repeat(
                num_envs, 1
            )
            robot.write_joint_position_to_sim_index(position=q)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
            sim.forward()
            scene.update(0.0)

        def tcp_now():
            bodies = robot.data.body_pose_w.torch[:, tcp_body_ids].detach().cpu().numpy().copy()
            bodies[..., :3] -= origins[:, None, :]
            p = np.zeros((num_envs, 2, 3), dtype=np.float32)
            q = np.zeros((num_envs, 2, 4), dtype=np.float32)
            for side in range(2):
                p[:, side], q[:, side] = site_from_body(bodies[:, side], site_pos[side], site_quat[side])
            return p, q

        for frame in sorted({0, min(20, n - 1), min(60, n - 1), n // 2, n - 1}):
            write_q(frame)
            p, q = tcp_now()
            pos_error = np.linalg.norm(p[0] - ref["tcp_pos"][frame], axis=-1) * 1000
            rot_error = np.rad2deg(
                (Rotation.from_quat(q[0]) * Rotation.from_quat(ref["tcp_quat_xyzw"][frame]).inv()).magnitude()
            )
            parity.append({"frame": frame, "pos_mm": pos_error.tolist(), "rot_deg": rot_error.tolist()})
        write_q(0)
        profile = apply_newton_arm_tracking_profile(robot, damping_scale_source=0.5)

        def record(frame):
            nonlocal source_actual_contact_frames
            q = robot.data.joint_pos.torch.detach().cpu().numpy()
            v = robot.data.joint_vel.torch.detach().cpu().numpy()
            q_actual[frame] = q[:, inverse]
            v_actual[frame] = v[:, inverse]
            torque[frame] = robot.data.applied_torque.torch.detach().cpu().numpy()[:, inverse]
            tcp_pos[frame], tcp_quat[frame] = tcp_now()
            for env in range(num_envs):
                source_data.qpos[:] = q_actual[frame, env]
                mujoco.mj_forward(model, source_data)
                if source_data.ncon:
                    source_actual_contact_frames += 1

        record(0)
        illegal_substep_states = 0
        source_step_contacts = 0
        for frame in range(1, n):
            for sub in range(10):
                alpha = (sub + 1) / 10
                target = (1 - alpha) * qref[frame - 1, order] + alpha * qref[frame, order]
                command = torch.as_tensor(target, dtype=torch.float32, device=robot.device)[None, :].repeat(num_envs, 1)
                robot.actuators.target_command.set_position_index(value=command, full_data=True)
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(DT)
                physics_steps += 1
                contacts[frame] += int(solver.mjw_data.nacon.numpy()[0])
                step_q = robot.data.joint_pos.torch.detach().cpu().numpy()
                illegal_substep_states += int(
                    np.count_nonzero((step_q < joint_limits[:, 0] - 1e-4) | (step_q > joint_limits[:, 1] + 1e-4))
                )
                for env in range(num_envs):
                    source_data.qpos[:] = step_q[env, inverse]
                    mujoco.mj_forward(model, source_data)
                    source_step_contacts += int(source_data.ncon > 0)
                applied = robot.data.applied_torque.torch[:, arm_index].detach().cpu().numpy()
                effort = robot.data.joint_effort_limits.torch[:, arm_index].detach().cpu().numpy()
                sat_count += int(np.count_nonzero(abs(applied) >= effort - 1e-3))
                effort_samples += applied.size
            record(frame)
            if frame % 250 == 0:
                print(f"Newton {num_envs} env: frame {frame}/{n}", flush=True)
    target_pos = ref["tcp_pos"][:n, None]
    target_quat = ref["tcp_quat_xyzw"][:n, None]
    p_error = np.linalg.norm(tcp_pos - target_pos, axis=-1) * 1000
    r_error = np.rad2deg(
        (
            Rotation.from_quat(tcp_quat.reshape(-1, 4))
            * Rotation.from_quat(np.broadcast_to(target_quat, tcp_quat.shape).reshape(-1, 4)).inv()
        ).magnitude()
    ).reshape(n, num_envs, 2)
    q_arm = q_actual[:, :, [qadr[name] for name in arm_names]]
    v_arm = v_actual[:, :, [qadr[name] for name in arm_names]]
    qerr = q_arm - ref["q_arm"][:n, None]
    verr = v_arm - ref["qd_arm"][:n, None]
    acc = np.gradient(v_arm, 0.02, axis=0)
    limit_violation = 0
    for i, name in enumerate(model_names):
        low, high = model.jnt_range[i]
        limit_violation += int(
            np.count_nonzero((q_actual[:, :, qadr[name]] < low - 1e-4) | (q_actual[:, :, qadr[name]] > high + 1e-4))
        )
    report = {
        "backend": "Newton MJWarp",
        "reference": str(reference),
        "reference_q_full_sha256": hashlib.sha256(qref.tobytes()).hexdigest(),
        "source_mjcf_sha256": hashlib.sha256(SOURCE_MJCF.read_bytes()).hexdigest(),
        "usd": str(usd),
        "usd_sha256": hashlib.sha256(usd.read_bytes()).hexdigest(),
        "num_envs": num_envs,
        "frames": n,
        "duration_s": float(time[-1]),
        "complete": n == len(ref["time"]),
        "physics_timestep_s": DT,
        "control_timestep_s": 0.02,
        "physics_steps": physics_steps,
        "profile": profile,
        "hand_contacts_enabled": False,
        "body_to_tcp_site_pos_m": site_pos.tolist(),
        "body_to_tcp_site_quat_xyzw": site_quat.tolist(),
        "fk_parity_probes": parity,
        "per_side": {},
        "newton_contact_total": int(contacts.sum()),
        "source_geometry_contact_frame_env_count": source_actual_contact_frames,
        "source_geometry_contact_physics_step_env_count": source_step_contacts,
        "illegal_joint_state_count": illegal_substep_states,
        "illegal_joint_state_50hz_count": limit_violation,
        "torque_saturation_rate": float(sat_count / effort_samples),
        "max_actual_joint_speed_rad_s": float(np.max(abs(v_arm))),
        "max_actual_joint_accel_rad_s2": float(np.max(abs(acc))),
        "state_finite": bool(np.isfinite(q_actual).all() and np.isfinite(v_actual).all()),
    }
    for side, name in enumerate(("left", "right")):
        sl = slice(side * 7, (side + 1) * 7)
        report["per_side"][name] = {
            "tcp_pos_mm": stats(p_error[:, :, side]),
            "tcp_rot_deg": stats(r_error[:, :, side]),
            "joint_rmse_rad": float(np.sqrt(np.mean(qerr[:, :, sl] ** 2))),
            "joint_velocity_rmse_rad_s": float(np.sqrt(np.mean(verr[:, :, sl] ** 2))),
            "max_actual_joint_speed_rad_s": float(np.max(abs(v_arm[:, :, sl]))),
            "max_actual_joint_accel_rad_s2": float(np.max(abs(acc[:, :, sl]))),
        }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "newton_pd_states.npz",
        time=time,
        q=q_actual,
        qd=v_actual,
        tcp_pos=tcp_pos,
        tcp_quat_xyzw=tcp_quat,
        applied_torque=torque,
        contact_count=contacts,
        tcp_position_error_mm=p_error,
        tcp_orientation_error_deg=r_error,
    )
    (output / "newton_pd_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path)
    ap.add_argument(
        "--usd",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "source/dexgym/assets/marvin_wuji/usd/dual_arm_marvin_no_hand_contacts.usda",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ubuntu/dexgym_data/marvin_motion_library_v1/physics_replays/newton_1env"),
    )
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--max-frames", type=int)
    args = ap.parse_args()
    print(
        json.dumps(
            run(args.reference, args.usd, args.output, num_envs=args.num_envs, max_frames=args.max_frames), indent=2
        )
    )


if __name__ == "__main__":
    main()
