"""Create a clearly labeled smooth two-arm test reference from Marvin FK."""

import hashlib
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_MJCF = Path("/home/ubuntu/dexgym/source/dexgym/assets/marvin_wuji/source_model/dual_arm_marvin.xml")


def generate_procedural_bimanual_reference(path: str | Path, *, frames: int = 201,
                                           mjcf: str | Path = DEFAULT_MJCF) -> dict:
    if frames < 33:
        raise ValueError("Procedural tracking reference needs at least 33 frames")
    path = Path(path)
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    names = tuple(f"Joint{j}_{side}" for side in ("L", "R") for j in range(1, 8))
    ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names])
    if len(ids) != 14 or ids.min() < 0 or model.nq != 54:
        raise ValueError("Expected pinned 54DoF Marvin arm joint order")
    home = model.key_qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")].copy()
    phase = np.linspace(0, 1, frames, dtype=np.float64)
    envelope = np.sin(np.pi * phase) ** 2
    amplitudes = np.array([0.12, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03,
                           -0.12, 0.10, -0.08, 0.07, -0.05, 0.04, -0.03])
    offsets = np.stack([amplitudes[j] * envelope * np.sin(2 * np.pi * phase + 0.18 * j)
                        for j in range(14)], axis=-1)
    q_arm = home[ids][None] + offsets
    q_full = np.repeat(home[None], frames, axis=0)
    q_full[:, ids] = q_arm
    if np.any(q_arm < model.jnt_range[ids, 0] + 0.005) or np.any(q_arm > model.jnt_range[ids, 1] - 0.005):
        raise ValueError("Procedural arm reference violates the safe joint envelope")
    link_names = (("Link3_L", "Link4_L", "tcp_L"), ("Link3_R", "Link4_R", "tcp_R"))
    link_pos = np.zeros((frames, 2, 3, 3), np.float64)
    link_quat = np.zeros((frames, 2, 3, 4), np.float64)
    for t in range(frames):
        data.qpos[:] = q_full[t]
        mujoco.mj_forward(model, data)
        if data.ncon:
            raise ValueError(f"Procedural reference collides in source MuJoCo at frame {t}")
        for side, names_side in enumerate(link_names):
            for index, name in enumerate(names_side):
                if name.startswith("tcp_"):
                    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                    position, matrix = data.site_xpos[site], data.site_xmat[site]
                else:
                    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                    position, matrix = data.xpos[body], data.xmat[body]
                link_pos[t, side, index] = position
                quaternion = Rotation.from_matrix(matrix.reshape(3, 3)).as_quat()
                if t and np.dot(quaternion, link_quat[t - 1, side, index]) < 0:
                    quaternion = -quaternion
                link_quat[t, side, index] = quaternion
    dt = 0.02
    qd_arm = np.gradient(q_arm, dt, axis=0, edge_order=2)
    qdd_arm = np.gradient(qd_arm, dt, axis=0, edge_order=2)
    link_linear = np.gradient(link_pos, dt, axis=0, edge_order=2)
    link_angular = np.zeros((frames, 2, 3, 3), np.float64)
    rotation = Rotation.from_quat(link_quat.reshape(-1, 4)).as_matrix().reshape(frames, 2, 3, 3, 3)
    for t in range(1, frames - 1):
        relative = rotation[t + 1] @ np.swapaxes(rotation[t - 1], -1, -2)
        link_angular[t] = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(2, 3, 3) / (2 * dt)
    link_angular[0] = link_angular[1]
    link_angular[-1] = link_angular[-2]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, motion_id="procedural_bimanual_reach_v1", provenance="procedural_test",
        fps=50, time=np.arange(frames, dtype=np.float64) * dt,
        joint_names_full=np.array([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                                   for i in range(model.njnt)]),
        joint_names_arm=np.array(names), link_names=np.array(link_names),
        q_arm=q_arm.astype(np.float32), qd_arm=qd_arm.astype(np.float32),
        qdd_arm=qdd_arm.astype(np.float32), q_full=q_full.astype(np.float32),
        link_pos=link_pos.astype(np.float32), link_quat_xyzw=link_quat.astype(np.float32),
        link_linear_vel=link_linear.astype(np.float32), link_angular_vel=link_angular.astype(np.float32),
        tcp_pos=link_pos[:, :, 2].astype(np.float32), tcp_quat_xyzw=link_quat[:, :, 2].astype(np.float32),
        tcp_linear_vel=link_linear[:, :, 2].astype(np.float32),
        tcp_angular_vel=link_angular[:, :, 2].astype(np.float32),
    )
    return {"provenance": "procedural_test", "motion_id": "procedural_bimanual_reach_v1",
            "frames": frames, "fps": 50, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "path": str(path)}
