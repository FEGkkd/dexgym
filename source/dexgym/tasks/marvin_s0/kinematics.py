"""Vectorized Marvin TCP FK from the pinned MuJoCo body/joint tree.

The source model supplies constant transforms once. All sampled goals are
computed on the selected torch device; q_goal is never returned to the actor.
MuJoCo constants are WXYZ, while all runtime quaternions are XYZW.
"""

import mujoco
import torch

from dexgym.assets.marvin_wuji.controller import ARM_JOINT_NAMES


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    av, aw = a[..., :3], a[..., 3:]
    bv, bw = b[..., :3], b[..., 3:]
    av, bv = torch.broadcast_tensors(av, bv)
    return torch.cat((aw * bv + bw * av + torch.cross(av, bv, dim=-1), aw * bw - (av * bv).sum(-1, keepdim=True)), -1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = q[..., :3]
    t = 2.0 * torch.cross(qv, v.expand_as(qv), dim=-1)
    return v + q[..., 3:] * t + torch.cross(qv, t, dim=-1)


def orientation_error(target_xyzw: torch.Tensor, current_xyzw: torch.Tensor) -> torch.Tensor:
    """SO(3) logarithm of target * inverse(current), in world coordinates."""
    inverse = torch.cat((-current_xyzw[..., :3], current_xyzw[..., 3:]), dim=-1)
    q = quat_multiply(target_xyzw, inverse)
    q = torch.where(q[..., 3:] < 0.0, -q, q)
    norm = torch.linalg.vector_norm(q[..., :3], dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(norm, q[..., 3:].clamp_min(0.0))
    return torch.where(norm > 1e-8, q[..., :3] * angle / norm.clamp_min(1e-8), 2.0 * q[..., :3])


class MarvinArmFK:
    """Exact batched FK of the two seven-joint TCP chains in the pinned MJCF."""

    names = ARM_JOINT_NAMES

    def __init__(self, model: mujoco.MjModel, device: str | torch.device):
        self.device = torch.device(device)
        self.chains = []
        for side in ("L", "R"):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"TCP_Link_{side}")
            if body < 0:
                raise ValueError(f"Missing TCP_Link_{side}")
            ids = []
            while body > 0:
                ids.append(int(body))
                body = int(model.body_parentid[body])
            chain = []
            for body in reversed(ids):
                joint_ids = [j for j in range(model.njnt) if int(model.jnt_bodyid[j]) == body]
                if len(joint_ids) > 1:
                    raise ValueError("Multi-joint body is unsupported in Marvin arm FK")
                joint_id = joint_ids[0] if joint_ids else None
                joint_name = (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) if joint_id is not None else None
                )
                if joint_name and joint_name not in self.names:
                    raise ValueError(f"Unexpected joint in TCP chain: {joint_name}")
                chain.append(
                    (
                        torch.as_tensor(model.body_pos[body].copy(), dtype=torch.float64, device=self.device),
                        torch.as_tensor(
                            model.body_quat[body, [1, 2, 3, 0]].copy(), dtype=torch.float64, device=self.device
                        ),
                        self.names.index(joint_name) if joint_name else None,
                        torch.as_tensor(model.jnt_pos[joint_id].copy(), dtype=torch.float64, device=self.device)
                        if joint_id is not None
                        else None,
                        torch.as_tensor(model.jnt_axis[joint_id].copy(), dtype=torch.float64, device=self.device)
                        if joint_id is not None
                        else None,
                        float(model.qpos0[model.jnt_qposadr[joint_id]]) if joint_id is not None else 0.0,
                    )
                )
            if sum(segment[2] is not None for segment in chain) != 7:
                raise ValueError(f"Expected seven arm joints in {side} TCP chain")
            self.chains.append(chain)

    def __call__(self, q_arm: torch.Tensor) -> torch.Tensor:
        if q_arm.ndim != 2 or q_arm.shape[-1] != 14:
            raise ValueError("q_arm must have shape [batch, 14]")
        result = []
        for chain in self.chains:
            p = torch.zeros((q_arm.shape[0], 3), dtype=q_arm.dtype, device=q_arm.device)
            r = torch.zeros((q_arm.shape[0], 4), dtype=q_arm.dtype, device=q_arm.device)
            r[:, 3] = 1.0
            for body_pos, body_quat, joint_index, pivot, axis, reference in chain:
                offset = body_pos.to(dtype=q_arm.dtype)
                rotation = body_quat.to(dtype=q_arm.dtype)
                p = p + quat_rotate(r, offset)
                r = quat_multiply(r, rotation)
                if joint_index is not None:
                    point = pivot.to(dtype=q_arm.dtype)
                    direction = axis.to(dtype=q_arm.dtype)
                    angle = 0.5 * (q_arm[:, joint_index] - reference)
                    delta = torch.cat(
                        (direction.expand(q_arm.shape[0], -1) * torch.sin(angle[:, None]), torch.cos(angle[:, None])),
                        -1,
                    )
                    p = p + quat_rotate(r, point)
                    r = quat_multiply(r, delta)
                    p = p - quat_rotate(r, point)
            result.append(torch.cat((p, r), dim=-1))
        return torch.stack(result, dim=1)
