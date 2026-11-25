"""Calibrated task-space targets and recorded teleoperation home."""

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from ..wholebody_s0.intent import HumanIntentGeometry
from . import DOF
from .wrist_calibration import WristCalibration


def joint_names(model):
    if (model.nq, model.nv, model.nu) != (DOF,) * 3:
        raise ValueError("Expected a fixed 54-joint position-controlled robot")
    return [model.joint(int(i)).name for i in np.argsort(model.jnt_qposadr)]


def actuator_ids(model, names):
    return [int(np.flatnonzero(model.actuator_trnid[:, 0] == model.joint(name).id).item()) for name in names]


def command_bounds(model, names):
    joint = np.stack([model.joint(name).range for name in names])
    control = model.actuator_ctrlrange[actuator_ids(model, names)]
    return np.maximum(joint[:, 0], control[:, 0]), np.minimum(joint[:, 1], control[:, 1])


def recorded_home(model, arm_path, hand_path):
    params = yaml.safe_load(Path(arm_path).read_text(encoding="utf-8"))
    params = next(iter(params.values()))["ros__parameters"]
    hands = yaml.safe_load(Path(hand_path).read_text(encoding="utf-8"))
    if (hands["schema"], hands["units"], hands["source"]) != (1, "rad", "actual_disabled_hand_feedback"):
        raise ValueError("Expected the recorded Wuji hand Home file")
    values = {}
    for side, suffix in (("left", "L"), ("right", "R")):
        arm = params[f"home_{side}_deg"]
        if len(arm) != 7:
            raise ValueError(f"Expected seven Home joints for the {side} arm")
        values.update({f"Joint{i}_{suffix}": value for i, value in enumerate(np.deg2rad(arm), 1)})
        hand = hands["hands"][side]
        if len(hand["joint_names"]) != 20:
            raise ValueError(f"Expected twenty recorded Home joints for the {side} hand")
        for name, position in zip(hand["joint_names"], hand["positions"], strict=True):
            joint = model.joint(int(model.actuator(name).trnid[0])).name
            if not joint.startswith(side[0] + "_"):
                raise ValueError(f"Hand Home actuator is assigned to the wrong side: {name}")
            values[joint] = position
    names = joint_names(model)
    home = np.asarray([values[name] for name in names], dtype=np.float64)
    lo, hi = command_bounds(model, names)
    if not np.isfinite(home).all() or np.any((home < lo) | (home > hi)):
        raise ValueError("Recorded Home is outside the model command limits")
    return home


class TargetMapper:
    """Use the same named MANUS landmarks for recorded clips and live goal frames."""

    def __init__(self, model, *, physical_wrist=False):
        self.geometry = HumanIntentGeometry(model)
        self.calibration = None
        if physical_wrist:
            quat = Rotation.from_matrix(self.geometry.tcp_to_palm_rotation).as_quat()
            self.calibration = WristCalibration(self.geometry.tcp_to_palm_position, quat)

    def from_clip(self, clip, manus_orders):
        targets = self.geometry.targets(clip, manus_orders)
        if self.calibration is not None:
            wrist, rotation = self.calibration.compose(targets.wrist, targets.rotation)
            return wrist, rotation, targets.local
        local = np.einsum("sij,tsfkj->tsfki", self.geometry.tcp_to_palm_rotation, targets.local)
        local += self.geometry.tcp_to_palm_position[None, :, None, None, :]
        return targets.wrist, targets.rotation, local

    def from_frame(self, tcp_pose_xyzw, left_manus_pose, right_manus_pose, manus_orders):
        clip = {
            "target_tcp_world": np.asarray(tcp_pose_xyzw)[None],
            "manus_left_pose": np.asarray(left_manus_pose)[None],
            "manus_right_pose": np.asarray(right_manus_pose)[None],
        }
        return tuple(array[0] for array in self.from_clip(clip, manus_orders))


def native_state(model, q, qd=None):
    data = mujoco.MjData(model)
    data.qpos[:] = q
    if qd is not None:
        data.qvel[:] = qd
    mujoco.mj_forward(model, data)
    return data
