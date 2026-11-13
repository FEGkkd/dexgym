"""Build human wrist and hand goals independently of the reference joint angles."""

import json
from pathlib import Path
from typing import NamedTuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import SIDES
from .bag import manus_keypoint_order
from .robot import FINGER_SITES, FIRMWARE_FINGERS

INTENT_DIM = 48
GOAL_VERSION = "tracker_tcp_manus_bone_directions_v1"


class IntentTargets(NamedTuple):
    wrist: np.ndarray
    rotation: np.ndarray
    local: np.ndarray


def unit(vectors):
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(~np.isfinite(lengths) | (lengths <= 1e-6)):
        raise ValueError("Degenerate human-intent bone segment or palm axis")
    return vectors / lengths


def palm_basis(wrist, knuckles):
    forward = unit(knuckles.mean(axis=-2) - wrist)
    across = knuckles[..., 0, :] - knuckles[..., -1, :]
    thumbward = unit(across - (across * forward).sum(-1, keepdims=True) * forward)
    return np.stack((forward, thumbward, np.cross(forward, thumbward)), axis=-1)


class HumanIntentGeometry:
    """Transfer MANUS bone directions onto fixed Wuji roots and bone lengths."""

    def __init__(self, model):
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        roots, lengths, bases, mount_position, mount_rotation = [], [], [], [], []
        for side in "lr":
            palm = data.body(f"{side}_wrist")
            tcp = data.site(f"tcp_{side.upper()}")
            palm_rotation, tcp_rotation = palm.xmat.reshape(3, 3), tcp.xmat.reshape(3, 3)
            fingers = []
            for finger, tip in zip(FIRMWARE_FINGERS, FINGER_SITES, strict=True):
                # Joint 0 and 1 share the proximal segment. Use joint 0 for its
                # root and joint 2, joint 3 and the tip for the three endpoints.
                anchors = [data.joint(int(model.actuator(f"{side}_{finger}{i}").trnid[0])).xanchor
                           for i in (0, 2, 3)]
                points = np.vstack((*anchors, data.site(f"{side}_{tip}").xpos))
                fingers.append((points - palm.xpos) @ palm_rotation)
            fingers = np.stack(fingers)
            root = fingers[:, 0]
            roots.append(root)
            lengths.append(np.linalg.norm(np.diff(fingers, axis=1), axis=-1))
            bases.append(palm_basis(np.zeros(3), root[1:]))
            mount_position.append(tcp_rotation.T @ (palm.xpos - tcp.xpos))
            mount_rotation.append(tcp_rotation.T @ palm_rotation)
        self.roots, self.lengths, self.bases = map(np.stack, (roots, lengths, bases))
        self.tcp_to_palm_position = np.stack(mount_position)
        self.tcp_to_palm_rotation = np.stack(mount_rotation)

    def targets(self, clip, orders):
        local = []
        for i, (side, order) in enumerate(zip(SIDES, orders, strict=True)):
            points = clip[f"manus_{side}_pose"][:, order, :3].astype(float)
            wrist = points[:, 0]
            chains = points[:, 1:].reshape(-1, 5, 4, 3)
            knuckles = chains[:, 1:, 0]
            basis = palm_basis(wrist, knuckles)
            transform = self.bases[i] @ basis.transpose(0, 2, 1)
            directions = np.einsum("tij,tfkj->tfki", transform, unit(np.diff(chains, axis=2)))
            local.append(self.roots[i, :, None] + np.cumsum(directions * self.lengths[i, :, :, None], axis=2))
        pose = clip["target_tcp_world"]
        rotation = Rotation.from_quat(pose[..., 3:].reshape(-1, 4)).as_matrix().reshape(-1, 2, 3, 3)
        return IntentTargets(pose[..., :3], rotation, np.stack(local, axis=1))

    def world_tips(self, targets):
        in_tcp = (np.einsum("sij,tsfj->tsfi", self.tcp_to_palm_rotation, targets.local[..., -1, :])
                  + self.tcp_to_palm_position[:, None])
        return np.einsum("tsij,tsfj->tsfi", targets.rotation, in_tcp) + targets.wrist[:, :, None]


def source_orders(library, entry):
    root = library.root
    roots = [root.parent, root.parent.parent]
    daily = library.manifest.get("source_daily_manifests", {}).get(entry.get("recording_date"))
    if daily:
        roots.append(Path(daily["path"]).parent.parent)
    paths = [folder / "episodes" / entry["source_episode"] / "quality.json" for folder in roots]
    quality_path = next((path for path in paths if path.is_file()), None)
    if quality_path is None:
        raise FileNotFoundError(f"Missing recorded MANUS topology for {entry['source_episode']}")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if quality["source_manifest_sha256"] != entry["source_manifest_sha256"]:
        raise ValueError(f"MANUS topology is from a different recording: {quality_path}")
    return tuple(manus_keypoint_order(quality["streams"][f"manus_{side}"]["topology"]) for side in SIDES)
