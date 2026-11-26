"""Differentiable geometry for the P02 model's signed wrist and palm frames."""

from typing import NamedTuple

import torch
from torch import Tensor, nn

from ...p02.model import PALMS, WRISTS
from ....hand2_cube_supermimic.rotations import mm, mv, quat_to_mat, so3_log
from ....wholebody_s0_1.kinematics import RobotKinematics


class Geometry(NamedTuple):
    wrist: Tensor  # B x 2 x 3 in W
    rotation: Tensor  # B x 2 x 3 x 3, mapping U to W
    local: Tensor  # B x 2 x 5 x 4 x 3 in P
    tips: Tensor  # B x 2 x 5 x 3 in W

    @property
    def pinch(self):
        tip = self.local[..., -1, :]
        return tip[:, :, 1:] - tip[:, :, :1]

    @property
    def cross(self):
        return self.tips[:, 1, None, :, :] - self.tips[:, 0, :, None, :]

    @property
    def relative(self):
        inv = self.rotation[:, 0].transpose(-1, -2)
        return mv(inv, self.wrist[:, 1] - self.wrist[:, 0]), mm(inv, self.rotation[:, 1])


class GeometryVelocity(NamedTuple):
    wrist_linear: Tensor  # B x 2 x 3 in W
    wrist_angular: Tensor  # B x 2 x 3 in W
    tips: Tensor  # B x 2 x 5 x 3 in W


def difference(current: Geometry, previous: Geometry, dt: float) -> GeometryVelocity:
    return GeometryVelocity(
        (current.wrist - previous.wrist) / dt,
        so3_log(mm(current.rotation, previous.rotation.transpose(-1, -2))) / dt,
        (current.tips - previous.tips) / dt,
    )


class ResidualKinematics(nn.Module):
    """Evaluate canonical [L arm, R arm, L hand, R hand] joint arrays."""

    def __init__(self, robot):
        super().__init__()
        model = robot.model
        self.tree = RobotKinematics(model, robot.names, proximal=True)
        if tuple(WRISTS) != ("tcp_L", "tcp_R"):
            raise ValueError("Unexpected signed wrist sites")
        if tuple(PALMS) != ("l_wrist", "r_wrist"):
            raise ValueError("Unexpected signed palm bodies")
        if any(
            model.site(finger[-1]).id != wanted
            for finger, wanted in zip(
                (finger for side in robot.landmark_names for finger in side),
                robot.tip_ids,
                strict=True,
            )
        ):
            raise ValueError("Finger landmarks must align with the signed P02 order")

    def forward(self, q: Tensor) -> Geometry:
        tree = self.tree
        pos, rot = tree.body_frames(q)
        site = pos[:, tree.site_bodies] + mv(rot[:, tree.site_bodies], tree.site_offset)
        wrist = site[:, :2]
        rotation = mm(rot[:, tree.site_bodies[:2]], tree.wrist_rotation)
        tips = site[:, 2:].reshape(-1, 2, 5, 3)
        anchors = pos[:, tree.anchor_bodies] + mv(rot[:, tree.anchor_bodies], tree.anchor_offsets)
        points = torch.cat((anchors.reshape(-1, 2, 5, 3, 3), tips.unsqueeze(-2)), dim=-2)
        palms = pos[:, tree.wrist_bodies]
        rotations = rot[:, tree.wrist_bodies].transpose(-1, -2)
        local = mv(rotations[:, :, None, None], points - palms[:, :, None, None])
        return Geometry(wrist, rotation, local, tips)


def quaternion_xyzw_to_matrix(q: Tensor) -> Tensor:
    return quat_to_mat(q[..., [3, 0, 1, 2]])


def pose9(p: Tensor, rotation: Tensor) -> Tensor:
    return torch.cat((p, rotation[..., :2].transpose(-1, -2).reshape(*p.shape[:-1], 6)), dim=-1)


def error6(p: Tensor, rotation: Tensor, target: Tensor, target_rotation: Tensor) -> Tensor:
    return torch.cat((target - p, so3_log(mm(rotation.transpose(-1, -2), target_rotation))), dim=-1)
