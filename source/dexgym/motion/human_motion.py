"""HumanUpperBodyMotion v1 arrays from resampled human FK."""

import numpy as np
from scipy.spatial.transform import Rotation

from .bvh import HumanFK

SIDES = ("Left", "Right")
POSE_NODES = {
    "shoulder_pose": "Shoulder",
    "upper_arm_pose": "Arm",
    "elbow_pose": "ForeArm",
    "forearm_pose": "ForeArm",
    "hand_pose": "Hand",
}


def _angular_velocity(rotations: np.ndarray, time: np.ndarray) -> np.ndarray:
    if len(time) < 2:
        return np.zeros((len(time), 3))
    left = Rotation.from_matrix(rotations[:-1])
    right = Rotation.from_matrix(rotations[1:])
    step = (right * left.inv()).as_rotvec() / np.diff(time)[:, None]
    velocity = np.empty((len(time), 3))
    velocity[0], velocity[-1] = step[0], step[-1]
    if len(time) > 2:
        velocity[1:-1] = (step[:-1] + step[1:]) * 0.5
    return velocity


def build_upper_body(fk: HumanFK, audit: dict) -> dict[str, np.ndarray]:
    """Build 50 Hz diagnostic or accepted arrays; acceptance remains in metadata."""
    lookup = {joint.name: i for i, joint in enumerate(fk.joints)}
    for side in SIDES:
        for suffix in dict.fromkeys(POSE_NODES.values()):
            if side + suffix not in lookup:
                raise ValueError(f"Missing required node {side + suffix}")
    data: dict[str, np.ndarray] = {"time": fk.time.copy()}
    for field, suffix in POSE_NODES.items():
        data[field] = np.stack(
            [
                np.concatenate((fk.position[:, lookup[side + suffix]], fk.quat_xyzw[:, lookup[side + suffix]]), axis=-1)
                for side in SIDES
            ],
            axis=1,
        )
    data["hand_linear_velocity"] = (
        np.gradient(data["hand_pose"][..., :3], fk.time, axis=0) if len(fk.time) > 1 else np.zeros((1, 2, 3))
    )
    data["hand_angular_velocity"] = np.stack(
        [_angular_velocity(fk.rotation[:, lookup[side + "Hand"]], fk.time) for side in SIDES], axis=1
    )
    valid = [
        bool(
            audit.get("full_6d_accepted", False)
            and audit.get("hands", {}).get(side.lower(), {}).get("orientation_reconstruction_valid", False)
        )
        for side in SIDES
    ]
    data["hand_orientation_valid"] = np.broadcast_to(np.asarray(valid, dtype=bool), (len(fk.time), 2)).copy()
    return data
