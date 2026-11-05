"""Torch FK from pinned MuJoCo body/joint/site transforms, with arm gradients."""

from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def _axis_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation for a batch of hinge angles."""
    axis = axis / axis.norm()
    x, y, z = axis.unbind()
    skew = torch.stack((torch.stack((x * 0, -z, y)), torch.stack((z, x * 0, -x)), torch.stack((-y, x, x * 0))))
    eye = torch.eye(3, device=angle.device, dtype=angle.dtype)
    outer = axis[:, None] * axis[None, :]
    c, s = angle.cos()[..., None, None], angle.sin()[..., None, None]
    return c * eye + (1 - c) * outer + s * skew


class DifferentiableMarvinFK:
    """Return [batch, side, tracked link, xyz/rotation], side order L then R."""

    def __init__(self, model_path: str | Path, device: str | torch.device = "cpu", dtype=torch.float32):
        model = mujoco.MjModel.from_xml_path(str(model_path))
        self.device = torch.device(device)
        self.dtype = dtype
        self.names = tuple(f"Joint{j}_{side}" for side in ("L", "R") for j in range(1, 8))
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.names]
        if min(joint_ids) < 0 or model.nq != 54:
            raise ValueError("Expected pinned 54DoF Marvin arm joints")
        self.arm_id = {jid: i for i, jid in enumerate(joint_ids)}
        self.model = model
        self.targets = []
        for side in ("L", "R"):
            items = []
            for name in (f"Link3_{side}", f"Link4_{side}", f"tcp_{side}"):
                is_site = name.startswith("tcp_")
                obj = mujoco.mjtObj.mjOBJ_SITE if is_site else mujoco.mjtObj.mjOBJ_BODY
                obj_id = mujoco.mj_name2id(model, obj, name)
                body_id = int(model.site_bodyid[obj_id]) if is_site else obj_id
                chain = []
                while body_id != 0:
                    chain.append(body_id)
                    body_id = int(model.body_parentid[body_id])
                chain.reverse()
                items.append((chain, obj_id if is_site else None))
            self.targets.append(items)
        self.body_pos = self._tensor(model.body_pos)
        self.body_rot = self._tensor(Rotation.from_quat(model.body_quat[:, [1, 2, 3, 0]]).as_matrix())
        self.joint_pos = self._tensor(model.jnt_pos)
        self.joint_axis = self._tensor(model.jnt_axis)
        self.site_pos = self._tensor(model.site_pos)
        self.site_rot = self._tensor(Rotation.from_quat(model.site_quat[:, [1, 2, 3, 0]]).as_matrix())

    def _tensor(self, value):
        return torch.as_tensor(np.asarray(value).copy(), device=self.device, dtype=self.dtype)

    def _one(self, q: torch.Tensor, chain: list[int], site: int | None):
        batch = q.shape[0]
        p = torch.zeros((batch, 3), device=q.device, dtype=q.dtype)
        rot = torch.eye(3, device=q.device, dtype=q.dtype).expand(batch, 3, 3)
        for body in chain:
            p = p + (rot @ self.body_pos[body])
            rot = rot @ self.body_rot[body]
            first = int(self.model.body_jntadr[body])
            number = int(self.model.body_jntnum[body])
            for joint in range(first, first + number):
                if joint not in self.arm_id:
                    raise ValueError(f"Unexpected non-arm joint before tracked link: {joint}")
                position = self.joint_pos[joint]
                p = p + rot @ position
                rot = rot @ _axis_rotation(self.joint_axis[joint], q[:, self.arm_id[joint]])
                p = p - rot @ position
        if site is not None:
            p = p + rot @ self.site_pos[site]
            rot = rot @ self.site_rot[site]
        return p, rot

    def __call__(self, q_arm: torch.Tensor):
        if q_arm.ndim != 2 or q_arm.shape[-1] != 14:
            raise ValueError("Expected [batch,14] arm q")
        positions, rotations = [], []
        for side in self.targets:
            side_p, side_r = [], []
            for chain, site in side:
                p, r = self._one(q_arm, chain, site)
                side_p.append(p)
                side_r.append(r)
            positions.append(torch.stack(side_p, dim=1))
            rotations.append(torch.stack(side_r, dim=1))
        return torch.stack(positions, dim=1), torch.stack(rotations, dim=1)
