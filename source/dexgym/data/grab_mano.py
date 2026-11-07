"""Reconstruct wrist-relative right-hand GRAB landmarks from MANO poses.

The local MANO pickle contains Chumpy wrappers around shape directions. This
converter only needs its numeric skeleton, skinning, and pose-blend arrays;
the compatibility unpickler never evaluates Chumpy code.
"""

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TIP_VERTEX_IDS = (745, 317, 444, 556, 673)  # thumb, index, middle, ring, pinky in MANO_RIGHT
FINGER_JOINT_IDS = ((13, 14, 15), (1, 2, 3), (4, 5, 6), (10, 11, 12), (7, 8, 9))


class _UnusedChumpy:
    """Accept serialized Chumpy objects that are not needed for landmark FK."""

    def __setstate__(self, state):
        self.state = state


class _ManoUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy."):
            return _UnusedChumpy
        return super().find_class(module, name)


@dataclass(frozen=True)
class ManoModel:
    regressor: np.ndarray  # 16 x 778
    parents: np.ndarray  # 16
    weights: np.ndarray  # 778 x 16
    posedirs: np.ndarray  # 778 x 3 x 135


def load_mano_model(path: str | Path) -> ManoModel:
    """Load the numeric MANO fields from the trusted, local model file."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Please import `csc_matrix`", category=DeprecationWarning)
        with Path(path).open("rb") as stream:
            raw = _ManoUnpickler(stream, encoding="latin1").load()
    regressor = np.asarray(raw["J_regressor"].toarray(), dtype=np.float64)
    weights = np.asarray(raw["weights"], dtype=np.float64)
    posedirs = np.asarray(raw["posedirs"], dtype=np.float64)
    tree = np.asarray(raw["kintree_table"])
    if regressor.shape != (16, 778) or weights.shape != (778, 16) or posedirs.shape != (778, 3, 135):
        raise ValueError("Unexpected MANO_RIGHT numeric model shape")
    if tree.shape != (2, 16) or not np.array_equal(tree[1], np.arange(16)):
        raise ValueError("Unexpected MANO_RIGHT skeleton order")
    parents = tree[0].astype(np.int64)
    parents[0] = -1
    if any(parents[index] >= index for index in range(1, 16)):
        raise ValueError("MANO joints must be ordered parent before child")
    return ManoModel(regressor, parents, weights, posedirs)


def load_subject_template(path: str | Path) -> np.ndarray:
    """Read the 778 XYZ vertices from a GRAB subject's binary PLY hand template."""
    with Path(path).open("rb") as stream:
        lines = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PLY header has no end_header")
            lines.append(line.strip())
            if line.strip() == b"end_header":
                break
        if lines[:2] != [b"ply", b"format binary_little_endian 1.0"]:
            raise ValueError("Expected binary little-endian PLY")
        if b"element vertex 778" not in lines or lines[lines.index(b"element vertex 778") + 1 :][:3] != [
            b"property float x", b"property float y", b"property float z"
        ]:
            raise ValueError("Expected 778 XYZ float vertices in MANO order")
        vertices = np.frombuffer(stream.read(778 * 3 * 4), dtype="<f4").reshape(778, 3).astype(np.float64)
    if not np.isfinite(vertices).all():
        raise ValueError("Subject hand template contains nonfinite vertices")
    return vertices


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Vectorized Rodrigues formula, including the zero-angle limit."""
    angle = np.linalg.norm(rotvec, axis=-1)
    axis = np.divide(rotvec, angle[..., None], out=np.zeros_like(rotvec), where=angle[..., None] > 1e-12)
    x, y, z = np.moveaxis(axis, -1, 0)
    zero = np.zeros_like(x)
    skew = np.stack(
        (np.stack((zero, -z, y), -1), np.stack((z, zero, -x), -1), np.stack((-y, x, zero), -1)), -2
    )
    identity = np.eye(3, dtype=np.float64)
    return identity + np.sin(angle)[..., None, None] * skew + (1 - np.cos(angle))[..., None, None] * (skew @ skew)


def mano_keypoints(fullpose: np.ndarray, template: np.ndarray, model: ManoModel) -> np.ndarray:
    """Return `[T,21,3]` MediaPipe-order landmarks relative to the MANO wrist.

    `fullpose` is GRAB's right-hand 45D local axis-angle pose. Global hand
    orientation and translation are intentionally excluded: the RL task fixes
    the robot wrist and tracks only finger articulation.
    """
    pose = np.asarray(fullpose, dtype=np.float64)
    vertices = np.asarray(template, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 45 or vertices.shape != (778, 3):
        raise ValueError("Expected fullpose[T,45] and template[778,3]")
    if not np.isfinite(pose).all() or not np.isfinite(vertices).all():
        raise ValueError("MANO input contains nonfinite values")
    count = len(pose)
    if count == 0:
        return np.empty((0, 21, 3), dtype=np.float32)
    joint_rest = model.regressor @ vertices
    rotation = np.broadcast_to(np.eye(3), (count, 16, 3, 3)).copy()
    rotation[:, 1:] = _rotvec_to_matrix(pose.reshape(count, 15, 3))
    pose_feature = (rotation[:, 1:] - np.eye(3)).reshape(count, 135)
    tip_ids = np.asarray(TIP_VERTEX_IDS)
    tip_posed = vertices[tip_ids][None] + np.einsum(
        "vci,ti->tvc", model.posedirs[tip_ids], pose_feature, optimize=True
    )

    global_tf = np.zeros((count, 16, 4, 4), dtype=np.float64)
    global_tf[:, :, 3, 3] = 1.0
    for joint in range(16):
        parent = int(model.parents[joint])
        local = np.zeros((count, 4, 4), dtype=np.float64)
        local[:, :3, :3] = rotation[:, joint]
        local[:, 3, 3] = 1.0
        local[:, :3, 3] = joint_rest[joint] if parent < 0 else joint_rest[joint] - joint_rest[parent]
        global_tf[:, joint] = local if parent < 0 else global_tf[:, parent] @ local
    joints = global_tf[:, :, :3, 3]
    transforms_translation = joints - np.einsum("tjkl,jl->tjk", global_tf[:, :, :3, :3], joint_rest)
    tip_weights = model.weights[tip_ids]
    tips = np.einsum("vj,tjkl,tvl->tvk", tip_weights, global_tf[:, :, :3, :3], tip_posed, optimize=True)
    tips += np.einsum("vj,tjk->tvk", tip_weights, transforms_translation, optimize=True)

    points = np.empty((count, 21, 3), dtype=np.float64)
    points[:, 0] = joints[:, 0]
    for finger, chain in enumerate(FINGER_JOINT_IDS):
        start = 1 + finger * 4
        points[:, start : start + 3] = joints[:, chain]
        points[:, start + 3] = tips[:, finger]
    points -= joints[:, 0:1]
    return points.astype(np.float32)
