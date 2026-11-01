"""Offline geometry gate for Reset samples used with a rigid mimic screw.

The physics joint cannot represent an arbitrary cap pose: its axial travel
must equal the pitch multiplied by its relative spin. This check uses the
bottle's *local* axis and does not require Isaac Sim or a GPU.
"""

from __future__ import annotations

import math
from typing import Sequence

from .config import BottleConfig


def _normalized_xyzw(quaternion: Sequence[float]) -> tuple[float, float, float, float]:
    if len(quaternion) != 4 or not all(math.isfinite(float(x)) for x in quaternion):
        raise ValueError("quaternion must contain four finite xyzw values")
    norm = math.sqrt(sum(float(x) ** 2 for x in quaternion))
    if norm < 1e-8:
        raise ValueError("quaternion has zero norm")
    return tuple(float(x) / norm for x in quaternion)


def mimic_reset_geometry(bottle_pose_xyzw: Sequence[float],
                         cap_pose_xyzw: Sequence[float], theta_rad: float,
                         config: BottleConfig) -> dict[str, float]:
    """Measure cap alignment and signed-pitch error before a physics Reset.

    Poses are world XYZ followed by XYZW quaternion. ``theta_rad`` is the
    candidate's continuous relative thread angle. The result is expressed
    along the current bottle-local +Z axis, never the fixed world Z axis.
    """
    if (len(bottle_pose_xyzw) != 7 or len(cap_pose_xyzw) != 7
            or not all(math.isfinite(float(x)) for x in (*bottle_pose_xyzw,
                                                           *cap_pose_xyzw, theta_rad))):
        raise ValueError("poses and thread angle must be finite xyz+xyzw")
    bx, by, bz, bw = _normalized_xyzw(bottle_pose_xyzw[3:])
    cx, cy, cz, cw = _normalized_xyzw(cap_pose_xyzw[3:])
    axis = (2 * (bx * bz + bw * by), 2 * (by * bz - bw * bx),
            1 - 2 * (bx * bx + by * by))
    delta = tuple(float(c) - float(b) for b, c in zip(bottle_pose_xyzw[:3],
                                                     cap_pose_xyzw[:3]))
    absolute_axial = sum(a * d for a, d in zip(axis, delta))
    axial = absolute_axial - config.cap_initial_z
    target = config.thread_sign * config.thread_pitch * float(theta_rad) / (2 * math.pi)
    lateral = math.sqrt(sum((d - absolute_axial * a) ** 2
                            for a, d in zip(axis, delta)))
    # Relative orientation in bottle coordinates; report twist discrepancy
    # independently of axial position and swing misalignment.
    inverse_bottle = (-bx, -by, -bz, bw)
    ax, ay, az, aw = inverse_bottle
    rel_z = aw * cz + ax * cy - ay * cx + az * cw
    rel_w = aw * cw - ax * cx - ay * cy - az * cz
    measured_twist = 2 * math.atan2(rel_z, rel_w)
    twist_error = math.atan2(math.sin(measured_twist - theta_rad),
                             math.cos(measured_twist - theta_rad))
    return {"axial_m": axial, "target_axial_m": target,
            "pitch_error_m": axial - target, "lateral_m": lateral,
            "measured_twist_rad": measured_twist,
            "twist_error_rad": twist_error}
