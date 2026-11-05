"""BVH hierarchy parser, declared-order Euler FK, and SO(3) resampling."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class Joint:
    name: str
    parent: int
    offset: np.ndarray
    channels: tuple[str, ...]
    channel_start: int


@dataclass(frozen=True)
class BVHMotion:
    joints: tuple[Joint, ...]
    frames: np.ndarray
    frame_time: float
    source: Path


@dataclass(frozen=True)
class HumanFK:
    time: np.ndarray
    joints: tuple[Joint, ...]
    position: np.ndarray
    rotation: np.ndarray
    quat_xyzw: np.ndarray
    local_rotation: np.ndarray
    local_translation: np.ndarray
    scale_m_per_unit: float
    removed_t_pose: bool


def parse_bvh(path: str | Path) -> BVHMotion:
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    joints: list[Joint] = []
    channel_count = 0
    stack: list[int] = []
    pending: int | None = None
    end_site = False
    motion_line = None
    for line_number, raw in enumerate(lines):
        line = raw.strip()
        if line == "MOTION":
            motion_line = line_number
            break
        if not line or line == "HIERARCHY":
            continue
        if line.startswith(("ROOT ", "JOINT ")):
            name = line.split(maxsplit=1)[1]
            pending = len(joints)
            joints.append(Joint(name, stack[-1] if stack else -1, np.zeros(3), (), channel_count))
            end_site = False
        elif line == "End Site":
            pending = len(joints)
            joints.append(Joint(f"{joints[stack[-1]].name}__EndSite", stack[-1], np.zeros(3), (), channel_count))
            end_site = True
        elif line == "{":
            if pending is None:
                raise ValueError(f"Unexpected opening brace at line {line_number + 1}")
            stack.append(pending)
            pending = None
        elif line == "}":
            if not stack:
                raise ValueError(f"Unexpected closing brace at line {line_number + 1}")
            stack.pop()
        elif line.startswith("OFFSET "):
            if not stack:
                raise ValueError("OFFSET outside joint")
            index = stack[-1]
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Invalid OFFSET at line {line_number + 1}")
            joint = joints[index]
            joints[index] = Joint(
                joint.name, joint.parent, np.array([float(x) for x in parts[1:]]), joint.channels, joint.channel_start
            )
        elif line.startswith("CHANNELS "):
            if not stack or end_site:
                raise ValueError("CHANNELS outside joint")
            parts = line.split()
            count = int(parts[1])
            channels = tuple(parts[2:])
            if len(channels) != count or any(
                c not in {f"{axis}{kind}" for axis in "XYZ" for kind in ("position", "rotation")} for c in channels
            ):
                raise ValueError(f"Invalid CHANNELS at line {line_number + 1}")
            index = stack[-1]
            joint = joints[index]
            joints[index] = Joint(joint.name, joint.parent, joint.offset, channels, channel_count)
            channel_count += count
    if motion_line is None or stack or not joints:
        raise ValueError("Incomplete BVH hierarchy or missing MOTION")
    header = [line.strip() for line in lines[motion_line + 1 : motion_line + 3]]
    frames_match = re.fullmatch(r"Frames:\s*(\d+)", header[0])
    time_match = re.fullmatch(r"Frame Time:\s*([0-9.eE+-]+)", header[1])
    if not frames_match or not time_match:
        raise ValueError("Invalid BVH motion header")
    frame_count, frame_time = int(frames_match.group(1)), float(time_match.group(1))
    values = np.fromstring(" ".join(lines[motion_line + 3 :]), sep=" ", dtype=np.float64)
    if frame_count < 1 or frame_time <= 0 or values.size != frame_count * channel_count:
        raise ValueError(
            f"BVH data shape mismatch: frames={frame_count}, channels={channel_count}, values={values.size}"
        )
    frames = values.reshape(frame_count, channel_count)
    if not np.isfinite(frames).all():
        raise ValueError("Non-finite BVH channels")
    return BVHMotion(tuple(joints), frames, frame_time, path)


def _local_rotations(motion: BVHMotion, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count, joint_count = len(frames), len(motion.joints)
    local = np.broadcast_to(np.eye(3), (count, joint_count, 3, 3)).copy()
    local_translation = np.zeros((count, joint_count, 3))
    for index, joint in enumerate(motion.joints):
        channels = {name: frames[:, joint.channel_start + k] for k, name in enumerate(joint.channels)}
        order = "".join(name[0] for name in joint.channels if name.endswith("rotation"))
        if order:
            angles = np.stack([channels[f"{axis}rotation"] for axis in order], axis=-1)
            local[:, index] = Rotation.from_euler(order, angles, degrees=True).as_matrix()
        for axis_index, axis in enumerate("XYZ"):
            if f"{axis}position" in channels:
                local_translation[:, index, axis_index] = channels[f"{axis}position"]
    return local, local_translation


def _fk(
    joints: tuple[Joint, ...], local: np.ndarray, translation: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    count, joint_count = local.shape[:2]
    pos = np.zeros((count, joint_count, 3))
    rot = np.zeros((count, joint_count, 3, 3))
    for index, joint in enumerate(joints):
        displacement = (joint.offset + translation[:, index]) * scale
        if joint.parent < 0:
            pos[:, index] = displacement
            rot[:, index] = local[:, index]
        else:
            parent = joint.parent
            pos[:, index] = pos[:, parent] + np.einsum("tij,tj->ti", rot[:, parent], displacement)
            rot[:, index] = rot[:, parent] @ local[:, index]
    return pos, rot


def _continuous_quat(rotation: np.ndarray) -> np.ndarray:
    quat = Rotation.from_matrix(rotation.reshape(-1, 3, 3)).as_quat().reshape(*rotation.shape[:2], 4)
    for t in range(1, len(quat)):
        flip = np.sum(quat[t] * quat[t - 1], axis=-1) < 0
        quat[t, flip] *= -1
    return quat


def human_fk(motion: BVHMotion, *, scale_m_per_unit: float, remove_t_pose: bool) -> HumanFK:
    if scale_m_per_unit <= 0:
        raise ValueError("Scale must be positive and explicit")
    if remove_t_pose and len(motion.frames) < 2:
        raise ValueError("Cannot remove T-pose from a one-frame motion")
    frames = motion.frames[1:] if remove_t_pose else motion.frames
    local, translation = _local_rotations(motion, frames)
    pos, rot = _fk(motion.joints, local, translation, scale_m_per_unit)
    time = np.arange(len(frames), dtype=np.float64) * motion.frame_time
    return HumanFK(
        time, motion.joints, pos, rot, _continuous_quat(rot), local, translation, scale_m_per_unit, remove_t_pose
    )


def resample_motion(fk: HumanFK, *, sample_rate_hz: float = 50) -> HumanFK:
    if sample_rate_hz <= 0:
        raise ValueError("Sample rate must be positive")
    if len(fk.time) == 1:
        return fk
    duration = float(fk.time[-1])
    count = max(2, round(duration * sample_rate_hz) + 1)
    target_time = np.linspace(0, duration, count)
    local = np.zeros((count, len(fk.joints), 3, 3))
    translation = np.zeros((count, len(fk.joints), 3))
    for index, joint in enumerate(fk.joints):
        local[:, index] = Slerp(fk.time, Rotation.from_matrix(fk.local_rotation[:, index]))(target_time).as_matrix()
        for axis in range(3):
            translation[:, index, axis] = np.interp(target_time, fk.time, fk.local_translation[:, index, axis])
    pos, rot = _fk(fk.joints, local, translation, fk.scale_m_per_unit)
    return HumanFK(
        target_time,
        fk.joints,
        pos,
        rot,
        _continuous_quat(rot),
        local,
        translation,
        fk.scale_m_per_unit,
        fk.removed_t_pose,
    )
