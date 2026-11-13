"""Differentiable four-hinge fingertip positions in the measured wrist frame."""

import mujoco
import numpy as np
import torch
from torch import nn

from .robot import FINGER_SITES, FIRMWARE_FINGERS


class WristLocalFingertips(nn.Module):
    def __init__(self, model, names):
        super().__init__()
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        indices, axes, anchors, points = [], [], [], []
        for side in "lr":
            wrist = model.site(f"tcp_{side.upper()}").id
            p = data.site_xpos[wrist]
            r = data.site_xmat[wrist].reshape(3, 3)
            joints = [[int(model.actuator(f"{side}_{finger}{joint}").trnid[0]) for joint in range(4)]
                      for finger in FIRMWARE_FINGERS]
            sites = [model.site(f"{side}_{tip}").id for tip in FINGER_SITES]
            for tip, chain in zip(sites, joints, strict=True):
                body, actual = int(model.site_bodyid[tip]), []
                while body != int(model.site_bodyid[wrist]) and body:
                    count = int(model.body_jntnum[body])
                    first = int(model.body_jntadr[body])
                    actual.extend(reversed(range(first, first + count)))
                    body = int(model.body_parentid[body])
                if (not body or actual[::-1] != chain
                        or any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in chain)
                        or np.any(model.qpos0[model.jnt_qposadr[chain]])):
                    raise ValueError("Expected four zero-reference finger hinges below the wrist TCP")
            indices.append([[names.index(model.joint(j).name) for j in finger] for finger in joints])
            axes.append(data.xaxis[joints] @ r)
            anchors.append((data.xanchor[joints] - p) @ r)
            points.append((data.site_xpos[sites] - p) @ r)
        self.register_buffer("indices", torch.as_tensor(indices, dtype=torch.long))
        self.register_buffer("axes", torch.as_tensor(np.array(axes), dtype=torch.float32))
        self.register_buffer("anchors", torch.as_tensor(np.array(anchors), dtype=torch.float32))
        self.register_buffer("points", torch.as_tensor(np.array(points), dtype=torch.float32))

    def forward(self, q):
        angle = q[..., self.indices].unsqueeze(-1)
        sine, cosine = angle.sin(), angle.cos()
        p = self.points.expand(*q.shape[:-1], -1, -1, -1)
        # Apply each hinge to its descendants, starting from the distal joint.
        for joint in (3, 2, 1, 0):
            axis, anchor = self.axes[:, :, joint].expand_as(p), self.anchors[:, :, joint]
            v = p - anchor
            c, s = cosine[..., joint, :], sine[..., joint, :]
            p = anchor + c * v + s * torch.cross(axis, v, dim=-1) + (1. - c) * (axis * v).sum(-1, keepdim=True) * axis
        return p
