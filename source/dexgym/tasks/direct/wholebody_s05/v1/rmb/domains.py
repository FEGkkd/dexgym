"""Independent static witnesses and contiguous frozen reference domains."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .mapping import packed_geometry


THRESHOLDS_M = np.array([0.003, 0.005, 0.010, 0.015])
RESIDUAL_RAD = np.deg2rad(np.repeat((8.0, 5.0), (14, 40)))
# Kinematics solve acts on the ten global tips. Function and path checks
# below, rather than the least-squares objective, determine witness membership.
DAMPING = 0.07


def radial_envelope(robot):
    model = robot.model
    robot.set(np.zeros(54))
    bases, radii = [], []
    for site_id in robot.tip_ids:
        body = int(model.site(site_id).bodyid[0])
        joints = []
        while body:
            n = int(model.body_jntnum[body])
            if n > 1:
                raise ValueError("Triangle-inequality bound requires a serial tree of single hinges")
            if n:
                joints.append(robot.data.joint(int(model.body_jntadr[body])).xanchor.copy())
            body = int(model.body_parentid[body])
        if not joints:
            raise ValueError("Fingertip has no signed chain")
        points = np.stack([robot.data.site(site_id).xpos.copy(), *joints])
        bases.append(points[-1])
        radii.append(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    return np.array(bases).reshape(2, 5, 3), np.array(radii).reshape(2, 5)


def radial_lower_bound(mapped_tip, envelope):
    anchors, length = envelope
    return np.maximum(0.0, np.linalg.norm(mapped_tip - anchors, axis=-1) - length - 1e-9)


def active_pinch_error(geometric, target, active):
    distance = np.linalg.norm(geometric["pinch_xyz"] - target["pinch_xyz"], axis=-1)
    return float(np.max(distance[active], initial=0))


def nondegenerate(target, nominal, current, active):
    tests = [
        (np.linalg.norm(current["local_landmark_xyz"] - target["local_landmark_xyz"], axis=-1),
         np.linalg.norm(nominal["local_landmark_xyz"] - target["local_landmark_xyz"], axis=-1), "p95"),
        (np.linalg.norm(current["cross_xyz"] - target["cross_xyz"], axis=-1),
         np.linalg.norm(nominal["cross_xyz"] - target["cross_xyz"], axis=-1), "max"),
        (np.linalg.norm(current["relative_wrist_xyz"] - target["relative_wrist_xyz"]),
         np.linalg.norm(nominal["relative_wrist_xyz"] - target["relative_wrist_xyz"]), "max"),
    ]
    for candidate, base, statistic in tests:
        reduce = (lambda a: np.percentile(a, 95)) if statistic == "p95" else np.max
        if reduce(candidate) > reduce(base) + 0.002:
            return False
    if active_pinch_error(current, target, active) > active_pinch_error(nominal, target, active) + 0.002:
        return False
    goal = Rotation.from_quat(target["relative_wrist_xyzw"])
    baseline = (goal.inv() * Rotation.from_quat(nominal["relative_wrist_xyzw"])).magnitude()
    updated = (goal.inv() * Rotation.from_quat(current["relative_wrist_xyzw"])).magnitude()
    return updated <= baseline + np.deg2rad(1.0)


def realized(robot, q):
    g = robot.geometry(q)
    return packed_geometry(g.wrists, g.rotations, g.local, g.landmarks)


@dataclass(frozen=True)
class Witness:
    q: np.ndarray
    geometry: dict
    nominal_error_m: float
    error_m: float
    improved: bool


def residual_witness(robot, reference, target, active, steps, swept_proof):
    nominal = realized(robot, reference)
    wanted = target["world_tip_xyz"]
    base_error = float(np.linalg.norm(nominal["world_tip_xyz"] - wanted, axis=-1).max())
    need = max(0.002, 0.10 * base_error)
    best = Witness(reference.copy(), nominal, base_error, base_error, False)
    x = reference.copy()
    names = [finger[-1] for side in robot.landmark_names for finger in side]
    lower = np.maximum(robot.bounds[:, 0], reference - RESIDUAL_RAD)
    upper = np.minimum(robot.bounds[:, 1], reference + RESIDUAL_RAD)
    regularizer = DAMPING**2 * np.eye(30)
    for _ in range(steps):
        tips = robot.geometry(x).tips
        jac = np.concatenate([robot.point_jac("site", name)[0] for name in names], axis=0)
        delta = jac.T @ np.linalg.solve(jac @ jac.T + regularizer, (wanted - tips).ravel())
        trial = np.clip(x + np.clip(delta, -0.04, 0.04), lower, upper)
        if np.max(np.abs(trial - x)) < 1e-7:
            break
        x = trial
        mapped = realized(robot, trial)
        error = float(np.linalg.norm(mapped["world_tip_xyz"] - wanted, axis=-1).max())
        if error >= best.error_m or not nondegenerate(target, nominal, mapped, active):
            continue
        # The accepted straight configuration path always starts at the
        # nominal command. Continuous Servo qualification remains separate.
        if not robot.legality(trial)["feasible"] or not swept_proof.permits(reference, trial):
            continue
        best = Witness(trial.copy(), mapped, base_error, error, error <= base_error - need)
    return best


def contiguous_segments(eligible, source_segments, generations):
    """Return half-open, contiguous same-source windows from a frozen partition."""
    mask = np.asarray(eligible, dtype=bool)
    if not len(mask):
        return []
    cut = np.r_[True, (source_segments[1:] != source_segments[:-1]) | (generations[1:] != generations[:-1])]
    windows = []
    start = None
    for tick, valid in enumerate(mask & (source_segments >= 0)):
        if start is not None and (not valid or cut[tick]):
            windows.append((start, tick))
            start = None
        if valid and start is None:
            start = tick
    if start is not None:
        windows.append((start, len(mask)))
    return windows


def threshold_hard(features, thresholds, segments):
    mask = np.zeros(len(features), dtype=bool)
    for start, stop in segments:
        crossing = np.isfinite(features[start:stop]) & (features[start:stop] > thresholds)
        window = np.zeros(stop - start, dtype=bool)
        for column in range(features.shape[1]):
            for first, last in contiguous_segments(crossing[:, column], np.zeros(len(crossing)), np.zeros(len(crossing))):
                if last - first >= 5:
                    window[max(0, first - 10):min(len(window), last + 10)] = True
        for first, last in contiguous_segments(window, np.zeros(len(window)), np.zeros(len(window))):
            if last - first >= 25:
                mask[start + first:start + last] = True
    return mask
