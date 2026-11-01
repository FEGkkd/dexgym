"""Offline right-hand fingertip fitting on the official Beta 1 MJCF."""

from dataclasses import replace
import math

import mujoco
import numpy as np


TRIPOD = ("thumb", "index_finger", "middle_finger")


def fit_right_tripod(screen, sample, *, site_radial_gap_targets_m,
                     max_iterations: int = 5):
    """Fit only physical finger joint state, with no runtime cap manipulation.

    The desired site gap is a geometry proposal. Full collision, native USD,
    same-solver reach and zero-action probes must still admit the result.
    """
    if (len(site_radial_gap_targets_m) != 3 or max_iterations < 1
            or any(not math.isfinite(x) or not 0 <= x <= .012
                   for x in site_radial_gap_targets_m)):
        raise ValueError("Tripod gap targets must be finite and nonnegative")
    model = screen.model
    names = sample.metadata["right_joint_order"]
    indices = [i for i, name in enumerate(names)
               if any(digit in name for digit in TRIPOD)]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names[i])
                 for i in indices]
    if len(indices) != 12 or any(i < 0 for i in joint_ids):
        raise ValueError("Official right tripod joint mapping changed")
    columns = [int(model.jnt_dofadr[j]) for j in joint_ids]
    original = np.asarray(sample.right.joint_position, dtype=float)
    q = original.copy()
    targets = np.asarray(site_radial_gap_targets_m)
    for _ in range(max_iterations):
        current = replace(sample, right=replace(
            sample.right, joint_position=tuple(q), joint_target=tuple(q)))
        screen.check(current)
        jacobian = []
        gaps = []
        for digit in TRIPOD:
            site_id = screen.right_tip_sites[digit]
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros_like(jacp)
            mujoco.mj_jacSite(model, screen.data, jacp, jacr, site_id)
            position = (np.asarray(screen.data.site_xpos[site_id])
                        - np.asarray(sample.cap.root_pose[:3]))
            radius = np.linalg.norm(position[:2])
            radial = position[:2] / max(radius, 1e-9)
            jacobian.append(radial @ jacp[:2, columns])
            gaps.append(radius - screen.bottle.cap_outer_radius)
        matrix = np.asarray(jacobian)
        residual = -np.maximum(np.asarray(gaps) - targets, 0)
        if np.max(np.abs(residual)) < 1e-5:
            break
        update = matrix.T @ np.linalg.solve(
            matrix @ matrix.T + np.eye(3) * 1e-5, residual)
        update = np.clip(update, -.10, .10)
        for index, joint_id, dq in zip(indices, joint_ids, update):
            lower, upper = model.jnt_range[joint_id]
            q[index] = np.clip(q[index] + dq, lower + 1e-4, upper - 1e-4)
    fitted = replace(sample, right=replace(
        sample.right, joint_position=tuple(q), joint_target=tuple(q)),
        metadata={**sample.metadata,
                  "mjcf_tripod_fit_target_site_gaps_m": list(site_radial_gap_targets_m),
                  "mjcf_tripod_fit_max_joint_delta_rad": float(np.max(np.abs(q-original)))})
    return fitted
