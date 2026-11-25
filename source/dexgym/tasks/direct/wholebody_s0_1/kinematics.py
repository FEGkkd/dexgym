"""Robot TCP, finger landmarks and elbow geometry in one batched FK."""

from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from ..hand2_cube_supermimic.kinematics import HandKinematics
from ..hand2_cube_supermimic.rotations import axis_angle_to_mat, mm, mv, quat_to_mat
from ..wholebody_s0.robot import FINGER_SITES, FIRMWARE_FINGERS
from .wrist_calibration import TCP_SITES, WRIST_BODIES


MAPPED_FOUR_POINT = "mapped_bone_directions_four_point_v1"
REFERENCE_HAND_FK = "reference_hand_fk_v1"
MIXED_HAND_GOALS = "reference_fk_and_mapped_clip_targets_v1"
MAPPED_HAND = "mapped_hand_v1"


def finger_anchor_ids(model, *, proximal=False):
    indices = (0, 2, 3) if proximal else (2, 3)
    return [int(model.actuator(f"{side}_{finger}{index}").trnid[0])
            for side in "lr" for finger in FIRMWARE_FINGERS for index in indices]


class Pose(NamedTuple):
    wrist: torch.Tensor
    rotation: torch.Tensor
    hand: torch.Tensor
    relative_position: torch.Tensor
    relative_rotation: torch.Tensor


class Geometry(NamedTuple):
    pose: Pose
    swivel: torch.Tensor
    ray: torch.Tensor
    upper_arm: torch.Tensor


def relative_pose(wrist, rotation):
    inverse_left = rotation[..., 0, :, :].transpose(-1, -2)
    return mv(inverse_left, wrist[..., 1, :] - wrist[..., 0, :]), mm(inverse_left, rotation[..., 1, :, :])


def make_pose(wrist, rotation, hand):
    return Pose(wrist, rotation, hand, *relative_pose(wrist, rotation))


class RightWristKinematics(nn.Module):
    """Serial wrist FK for comparing arm targets during training."""

    def __init__(self, model, names, physical_wrist=False, *, side="R"):
        super().__init__()
        site = model.site(f"tcp_{side}")
        bodies = []
        body = model.body(f"{side.lower()}_wrist").id if physical_wrist else int(site.bodyid[0])
        while body:
            bodies.append(body)
            body = int(model.body_parentid[body])
        bodies.reverse()
        joints = [int(model.body_jntadr[body]) for body in bodies]
        if any(model.body_jntnum[body] > 1 for body in bodies):
            raise ValueError("Expected a serial arm with at most one hinge per link")
        moving = torch.tensor([j >= 0 for j in joints])
        ids = [names.index(model.joint(j).name) if j >= 0 else 0 for j in joints]
        self.register_buffer("joint", torch.tensor(ids))
        self.register_buffer("moving", moving)
        self.register_buffer("rotation", quat_to_mat(torch.tensor(model.body_quat[bodies], dtype=torch.float32)))
        self.register_buffer("axis", torch.tensor(model.jnt_axis[np.maximum(joints, 0)], dtype=torch.float32))
        self.register_buffer("anchor", torch.tensor(model.jnt_pos[np.maximum(joints, 0)], dtype=torch.float32))
        self.register_buffer("offset", torch.tensor(model.body_pos[bodies], dtype=torch.float32))
        self.register_buffer("site_pos", torch.zeros(3) if physical_wrist else torch.tensor(site.pos, dtype=torch.float32))
        self.register_buffer("site_rot", torch.eye(3) if physical_wrist else
                             quat_to_mat(torch.tensor(site.quat, dtype=torch.float32)))

    def forward(self, q):
        hinge = axis_angle_to_mat(self.axis, q[:, self.joint] * self.moving)
        r = mm(self.rotation, hinge)
        p = self.offset + mv(self.rotation, self.anchor - mv(hinge, self.anchor))
        while p.shape[1] > 1:
            n = 2 * (p.shape[1] // 2)
            parent = r[:, :n:2]
            p = torch.cat((p[:, :n:2] + mv(parent, p[:, 1:n:2]), p[:, n:]), dim=1)
            r = torch.cat((mm(parent, r[:, 1:n:2]), r[:, n:]), dim=1)
        return p[:, 0] + mv(r[:, 0], self.site_pos), mm(r[:, 0], self.site_rot)


class LinkLevel(nn.Module):
    def __init__(self, buffers):
        super().__init__()
        for name, value in buffers.items():
            self.register_buffer(name, value)

    def forward(self, q, positions, rotations):
        parent_p, parent_r = positions[:, self.parent], rotations[:, self.parent]
        hinge = axis_angle_to_mat(self.axis, q[:, self.joint] * self.moving)
        p = parent_p + mv(parent_r, self.off + mv(self.rot, self.anchor - mv(hinge, self.anchor)))
        r = mm(mm(parent_r, self.rot), hinge)
        return torch.cat((positions, p), dim=1), torch.cat((rotations, r), dim=1)


class RobotKinematics(nn.Module):
    def __init__(self, model, names, physical_wrist=False, *, proximal=False):
        super().__init__()
        self.physical_wrist = physical_wrist
        if np.any(model.qpos0):
            raise ValueError("Expected zero-reference MJCF hinges")
        wrists = [model.site(name).id for name in TCP_SITES]
        tips = [model.site(f"{side}_{finger}").id for side in "lr" for finger in FINGER_SITES]
        sites = wrists + tips
        tree = HandKinematics(model, range(model.nbody), [model.joint(name).id for name in names], sites, "cpu")
        self.levels = nn.ModuleList(LinkLevel(value) for value in tree.levels)
        joint = finger_anchor_ids(model, proximal=proximal)
        self.anchor_count = 3 if proximal else 2
        self.register_buffer("site_bodies", torch.tensor([tree.restore[i] for i in tree.tip_bodies]))
        self.register_buffer("site_offset", tree.tip_offset)
        self.register_buffer("wrist_rotation", quat_to_mat(torch.tensor(model.site_quat[wrists], dtype=torch.float32)))
        self.register_buffer("wrist_bodies",
                             torch.tensor([tree.restore[model.body(name).id] for name in WRIST_BODIES]))
        self.register_buffer("anchor_bodies", torch.tensor([tree.restore[i] for i in model.jnt_bodyid[joint]]))
        self.register_buffer("anchor_offsets", torch.tensor(model.jnt_pos[joint], dtype=torch.float32))
        # Link2 and Link4 lie on either side of the arm's swivel redundancy.
        for name, link in (("shoulders", "Link2"), ("elbows", "Link4")):
            self.register_buffer(name, torch.tensor([tree.restore[model.body(f"{link}_{side}").id] for side in "LR"]))
        self.register_buffer("identity", torch.eye(3)[None, None])
        arm_joints = np.asarray([[model.joint(f"Joint{i}_{side}").id for i in range(1, 8)] for side in "LR"])
        arm_bodies = [[tree.restore[model.jnt_bodyid[j]] for j in arm] for arm in arm_joints]
        self.register_buffer("arm_joint_bodies", torch.tensor(arm_bodies))
        self.register_buffer("arm_joint_axes", torch.tensor(model.jnt_axis[arm_joints], dtype=torch.float32))
        self.register_buffer("arm_joint_anchors", torch.tensor(model.jnt_pos[arm_joints], dtype=torch.float32))

    def body_frames(self, q):
        p = q.new_zeros(q.shape[0], 1, 3)
        r = self.identity.expand(q.shape[0], -1, -1, -1)
        for level in self.levels:
            p, r = level(q, p, r)
        return p, r

    def wrist_jacobian(self, q):
        p, r = self.body_frames(q)
        if self.physical_wrist:
            wrist = p[:, self.wrist_bodies]
        else:
            bodies = self.site_bodies[:2]
            wrist = p[:, bodies] + mv(r[:, bodies], self.site_offset[:2])
        rotation = r[:, self.arm_joint_bodies]
        origin = p[:, self.arm_joint_bodies] + mv(rotation, self.arm_joint_anchors)
        axis = mv(rotation, self.arm_joint_axes)
        linear = torch.linalg.cross(axis, wrist[:, :, None] - origin, dim=-1)
        return torch.cat((linear, axis), -1).transpose(-1, -2)

    def forward(self, q):
        p, r = self.body_frames(q)
        site = p[:, self.site_bodies] + mv(r[:, self.site_bodies], self.site_offset)
        if self.physical_wrist:
            wrist, rotation = p[:, self.wrist_bodies], r[:, self.wrist_bodies]
        else:
            wrist, rotation = site[:, :2], mm(r[:, self.site_bodies[:2]], self.wrist_rotation)
        tip = site[:, 2:].reshape(-1, 2, 5, 3)
        anchors = p[:, self.anchor_bodies] + mv(r[:, self.anchor_bodies], self.anchor_offsets)
        hand = torch.cat((anchors.reshape(-1, 2, 5, self.anchor_count, 3), tip.unsqueeze(-2)), dim=-2)
        hand = mv(rotation.transpose(-1, -2)[:, :, None, None], hand - wrist[:, :, None, None])
        ray = wrist - p[:, self.shoulders]
        ray = torch.nn.functional.normalize(ray, dim=-1, eps=1e-6)
        upper = p[:, self.elbows] - p[:, self.shoulders]
        swivel = upper - (upper * ray).sum(-1, keepdim=True) * ray
        return Geometry(make_pose(wrist, rotation, hand), swivel, ray, upper)
