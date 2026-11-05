"""Separate BVH channel availability, FK orientation, and capture confidence."""

import numpy as np
from scipy.spatial.transform import Rotation

from .bvh import BVHMotion, HumanFK


def _longest_frozen_run(speed: np.ndarray, threshold_deg_s: float = 0.1) -> int:
    longest = run = 0
    for value in speed:
        run = run + 1 if value < threshold_deg_s else 0
        longest = max(longest, run)
    return longest


def audit_hand_orientation(
    motion: BVHMotion,
    fk: HumanFK,
    *,
    hand_names: tuple[str, ...] = ("LeftHand", "RightHand"),
    active_threshold_deg: float = 0.1,
    jump_threshold_deg: float = 45.0,
    source_full_orientation_validated: bool = False,
) -> dict:
    """Audit actual wrist channels; world SO(3) from FK is not evidence of observed palm SO(3)."""
    by_name = {joint.name: (index, joint) for index, joint in enumerate(motion.joints)}
    result = {
        "hands": {},
        "full_6d_accepted": False,
        "source_full_orientation_validated": source_full_orientation_validated,
    }
    accepted = []
    for name in hand_names:
        side = "left" if name.lower().startswith("left") else "right"
        if name not in by_name:
            result["hands"][side] = {
                "orientation_available": False,
                "orientation_reconstruction_valid": False,
                "orientation_source": "missing_node",
                "orientation_confidence": "none",
                "rotation_quality": "missing_node",
                "full_6d_accepted": False,
                "orientation_limitations": ["Hand/wrist node absent"],
            }
            accepted.append(False)
            continue
        index, joint = by_name[name]
        channels = tuple(c for c in joint.channels if c.endswith("rotation"))
        starts = motion.frames[1:] if fk.removed_t_pose else motion.frames
        ranges = {}
        for offset, channel in enumerate(joint.channels):
            if channel.endswith("rotation"):
                column = starts[:, joint.channel_start + offset]
                ranges[channel] = float(np.ptp(column))
        active = [channel[0] for channel, spread in ranges.items() if spread >= active_threshold_deg]
        available = len(channels) == 3 and set(c[0] for c in channels) == set("XYZ")
        quat = fk.quat_xyzw[:, index]
        finite = bool(np.isfinite(quat).all() and np.isfinite(fk.local_rotation[:, index]).all())
        norm_error = float(np.max(np.abs(np.linalg.norm(quat, axis=-1) - 1))) if finite else float("inf")
        parent = joint.parent
        hierarchy_error_deg = float("inf")
        if parent >= 0 and finite:
            composed = fk.rotation[:, parent] @ fk.local_rotation[:, index]
            hierarchy_error_deg = float(
                np.rad2deg(
                    np.max(Rotation.from_matrix(np.swapaxes(composed, -1, -2) @ fk.rotation[:, index]).magnitude())
                )
            )
        local_rot = Rotation.from_matrix(fk.local_rotation[:, index])
        increments = (local_rot[:-1].inv() * local_rot[1:]).magnitude() if len(fk.time) > 1 else np.array([])
        dt = np.diff(fk.time)
        speed = np.rad2deg(increments / dt) if len(dt) else np.array([])
        max_step = float(np.rad2deg(increments.max())) if len(increments) else 0.0
        reconstruction_valid = available and finite and norm_error < 1e-5 and hierarchy_error_deg < 1e-3
        jump_count = int(np.count_nonzero(np.rad2deg(increments) > jump_threshold_deg))
        high_speed = bool(len(speed) and (np.percentile(speed, 95) > 720 or speed.max() > 1800))
        limitations = []
        if not source_full_orientation_validated:
            limitations.append(
                "BVH conversion and joint calibration do not prove independently measured palm orientation"
            )
        if len(active) < 3:
            limitations.append(f"Only {len(active)} of 3 local wrist rotation axes vary in this clip")
        if high_speed:
            limitations.append(
                "Local wrist angular speed exceeds conservative diagnostic thresholds (P95 720 deg/s or max 1800 deg/s)"
            )
        if jump_count:
            limitations.append(f"{jump_count} unrepaired local SO(3) jumps exceed {jump_threshold_deg} deg per sample")
        hand_accepted = (
            reconstruction_valid and source_full_orientation_validated and jump_count == 0 and not high_speed
        )
        confidence = "high" if hand_accepted else ("low" if reconstruction_valid else "none")
        quality = (
            "missing_or_invalid"
            if not reconstruction_valid
            else (
                "jumping"
                if jump_count
                else (
                    "implausible_angular_speed" if high_speed else ("limited_axis" if len(active) < 3 else "continuous")
                )
            )
        )
        hand = {
            "node": name,
            "parent_node": motion.joints[parent].name if parent >= 0 else None,
            "rotation_channels": list(channels),
            "channel_dynamic_range_deg": ranges,
            "active_local_rotation_axes": active,
            "orientation_available": available,
            "orientation_reconstruction_valid": reconstruction_valid,
            "orientation_source": "bvh_reconstructed",
            "orientation_confidence": confidence,
            "full_6d_accepted": hand_accepted,
            "rotation_quality": quality,
            "orientation_limitations": limitations,
            "world_quaternion_max_norm_error": norm_error,
            "fk_hierarchy_max_rotation_error_deg": hierarchy_error_deg,
            "local_rotation_max_step_deg": max_step,
            "local_rotation_speed_p95_deg_s": float(np.percentile(speed, 95)) if len(speed) else 0.0,
            "local_rotation_speed_max_deg_s": float(speed.max()) if len(speed) else 0.0,
            "frozen_run_s": float(_longest_frozen_run(speed) * (np.median(dt) if len(dt) else 0.0)),
            "jump_count": jump_count,
            "high_speed_flag": high_speed,
        }
        result["hands"][side] = hand
        accepted.append(hand_accepted)
    result["full_6d_accepted"] = len(accepted) == 2 and all(accepted)
    return result
