"""Differentiable FK for the SAME standalone Hand21 asset used by PPO.

MuJoCo only supplies the official transforms and a test oracle. Real training
continues in Isaac Lab Newton using the original USD and actuator settings.
"""
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dexgym.algorithms.supermimic.differentiable_fk import _axis_rotation
from dexgym.assets.wuji_hand21 import OFFICIAL_MJCF
from dexgym.data.wuji_retarget import ROBOT_RIGHT_JOINT_NAMES, TIP_SITE_NAMES


class Hand21Kinematics:
    def __init__(self, *, device='cpu', dtype=torch.float32, path: Path = OFFICIAL_MJCF):
        self.model = m = mujoco.MjModel.from_xml_path(str(path))
        if m.nq != 20 or m.nv != 20 or m.njnt != 20:
            raise ValueError('Expected official standalone twenty-joint hand')
        self.joint_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in ROBOT_RIGHT_JOINT_NAMES]
        self.tip_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in TIP_SITE_NAMES]
        self.wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'r_wrist')
        fingers = ('thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky')
        self.level1 = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'r_{n}_proximal') for n in fingers]
        self.level2 = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'r_{n}_middle') for n in fingers]
        if min(self.joint_ids + self.tip_ids + self.level1 + self.level2 + [self.wrist_id]) < 0:
            raise ValueError('Official hand kinematic frames are missing')
        self.joint_column = {joint: col for col, joint in enumerate(self.joint_ids)}
        self.qpos_indices = m.jnt_qposadr[self.joint_ids].copy()
        def tensor(values):
            return torch.as_tensor(np.asarray(values).copy(), device=device, dtype=dtype)
        self.positions = tensor(m.body_pos)
        self.rotations = tensor(Rotation.from_quat(m.body_quat[:, [1, 2, 3, 0]]).as_matrix())
        self.anchors, self.axes = tensor(m.jnt_pos), tensor(m.jnt_axis)
        self.site_offsets = tensor(m.site_pos)
        self.q_reference = tensor(m.qpos0[self.qpos_indices])
        self.limits = tensor(m.jnt_range[self.joint_ids])
        self.body_joints = []
        for body in range(m.nbody):
            first = int(m.body_jntadr[body])
            self.body_joints.append(tuple(range(first, first + int(m.body_jntnum[body]))))
        if any(m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in self.joint_ids):
            raise ValueError('Hand FK expects hinge joints')

    def landmarks(self, q: torch.Tensor) -> torch.Tensor:
        """Return [batch, tip/lvl1/lvl2, finger, XYZ] in the wrist frame."""
        if q.ndim != 2 or q.shape[1] != 20:
            raise ValueError('Expected q[N,20]')
        if q.device != self.positions.device or q.dtype != self.positions.dtype:
            raise ValueError('FK dtype/device must match its configured tensors')
        positions = [q.new_zeros((len(q), 3))]
        rotations = [torch.eye(3, dtype=q.dtype, device=q.device).expand(len(q), 3, 3)]
        for body in range(1, self.model.nbody):
            parent = int(self.model.body_parentid[body])
            p = positions[parent] + rotations[parent] @ self.positions[body]
            r = rotations[parent] @ self.rotations[body]
            for joint in self.body_joints[body]:
                column = self.joint_column[joint]
                anchor = self.anchors[joint]
                p = p + r @ anchor
                r = r @ _axis_rotation(self.axes[joint], q[:, column] - self.q_reference[column])
                p = p - r @ anchor
            positions.append(p)
            rotations.append(r)
        tips = []
        for site in self.tip_ids:
            body = int(self.model.site_bodyid[site])
            tips.append(positions[body] + rotations[body] @ self.site_offsets[site])
        points = torch.stack((torch.stack(tips, 1),
                              torch.stack([positions[i] for i in self.level1], 1),
                              torch.stack([positions[i] for i in self.level2], 1)), 1)
        wrist_rotation = rotations[self.wrist_id]
        return torch.einsum('bji,bkfj->bkfi', wrist_rotation,
                            points - positions[self.wrist_id][:, None, None])

    def __call__(self, q: torch.Tensor) -> torch.Tensor:
        return self.landmarks(q)[:, 0]
