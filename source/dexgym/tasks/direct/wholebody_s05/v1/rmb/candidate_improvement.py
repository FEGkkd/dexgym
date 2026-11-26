"""Check bounded static residual witnesses for diagnostic nominal-reference candidates."""

import numpy as np
from scipy.spatial.transform import Rotation

from .domains import RESIDUAL_RAD, active_pinch_error, nondegenerate, realized, residual_witness
from .schema import GEOMETRY_SHAPES


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def validated_target(geometry, pinch_active):
    if set(geometry) != GEOMETRY_SHAPES.keys():
        raise ValueError("Expected all named independent mapped-geometry fields")
    target = {name: finite_array(geometry[name], shape, name) for name, shape in GEOMETRY_SHAPES.items()}
    for name in ("wrist_xyzw", "relative_wrist_xyzw"):
        if not np.allclose(np.linalg.norm(target[name], axis=-1), 1., rtol=0, atol=1e-5):
            raise ValueError(f"{name} must contain unit XYZW quaternions")
    active = np.asarray(pinch_active)
    if active.shape != (2, 4) or active.dtype != np.bool_:
        raise ValueError("pinch_active must be the fixed boolean mask with shape (2, 4)")
    return target, active


def functional_errors(geometry, target, active):
    local = np.linalg.norm(geometry["local_landmark_xyz"] - target["local_landmark_xyz"], axis=-1)
    cross = np.linalg.norm(geometry["cross_xyz"] - target["cross_xyz"], axis=-1)
    relative = np.linalg.norm(geometry["relative_wrist_xyz"] - target["relative_wrist_xyz"])
    wrist_rotation = Rotation.from_quat(target["relative_wrist_xyzw"]).inv() * Rotation.from_quat(
        geometry["relative_wrist_xyzw"])
    return {
        "p95_local_m": float(np.percentile(local, 95)),
        "max_active_pinch_m": active_pinch_error(geometry, target, active) if active.any() else None,
        "max_cross_m": float(cross.max()),
        "relative_wrist_translation_m": float(relative),
        "relative_wrist_rotation_deg": float(np.rad2deg(wrist_rotation.magnitude())),
    }


def safety_result(robot, q):
    checked = dict(robot.legality(q))
    checked["feasible"] = bool(checked["feasible"])
    return checked


def evaluate_candidate(robot, q_nominal, mapped_geometry, pinch_active, *, sweep, search_steps=8, witness_q=None):
    """Use the frozen static criterion on one candidate row and one independent mapped target.

    The caller owns verified HMB lineage, the nominal active-state mask and any
    previous-command path. This evidence is standstill FK and swept collision
    geometry from the same signed MuJoCo model. It excludes Servo, dynamic,
    hardware and phase qualification.
    """
    q_nominal = finite_array(q_nominal, (54,), "q_nominal")
    target, active = validated_target(mapped_geometry, pinch_active)
    if witness_q is None and (type(search_steps) is not int or not 4 <= search_steps <= 12):
        raise ValueError("The bounded static search allows 4–12 witness iterations")
    trial = finite_array(witness_q, (54,), "witness_q") if witness_q is not None else None
    nominal_safety = safety_result(robot, q_nominal)
    nominal_pose_proven = nominal_safety["feasible"] and bool(sweep.permits(q_nominal, q_nominal))
    result = {
        "scope": "STATIC_MUJOCO_FK_AND_SWEPT_COLLISION",
        "status": "UNKNOWN",
        "reason": None,
        "servo_rollout_executed": False,
        "dynamic_qualification": False,
        "phase_qualification": False,
        "nominal_endpoint_safety": nominal_safety,
        "nominal_pose_proven": nominal_pose_proven,
        "search_steps_requested": search_steps if witness_q is None else None,
        "search_executed": False,
        "active_pinch_pairs": int(active.sum()),
        "witness_q_rad": None,
    }
    if not nominal_safety["feasible"]:
        return dict(result, status="NOMINAL_NOT_LEGAL", reason="NOMINAL_ENDPOINT_ILLEGAL")
    if not nominal_pose_proven:
        return dict(result, reason="NOMINAL_STATIC_SAFETY_NOT_PROVEN")

    nominal = realized(robot, q_nominal)
    goal = target["world_tip_xyz"]
    base_error = float(np.linalg.norm(nominal["world_tip_xyz"] - goal, axis=-1).max())
    need = max(.002, .10 * base_error)
    result.update(nominal_max_ten_tip_error_m=base_error, required_improvement_m=need,
                  nominal_function_errors=functional_errors(nominal, target, active))
    if trial is None:
        result["search_executed"] = True
        trial = residual_witness(robot, q_nominal, target, active, search_steps, sweep).q

    delta = np.abs(trial - q_nominal)
    result.update(max_arm_residual_deg=float(np.rad2deg(delta[:14].max())),
                  max_hand_residual_deg=float(np.rad2deg(delta[14:].max())))
    in_envelope = np.all((trial >= q_nominal - RESIDUAL_RAD) & (trial <= q_nominal + RESIDUAL_RAD))
    result["within_residual_envelope"] = bool(in_envelope)
    if not in_envelope:
        return dict(result, reason="RESIDUAL_ENVELOPE_EXCEEDED")

    trial_safety = safety_result(robot, trial)
    result["trial_endpoint_safety"] = trial_safety
    if not trial_safety["feasible"]:
        return dict(result, reason="WITNESS_ENDPOINT_ILLEGAL")
    result["swept_path_proven"] = bool(sweep.permits(q_nominal, trial))
    if not result["swept_path_proven"]:
        return dict(result, reason="WITNESS_PATH_NOT_PROVEN")

    geometry = realized(robot, trial)
    observed = float(np.linalg.norm(geometry["world_tip_xyz"] - goal, axis=-1).max())
    preserved = bool(nondegenerate(target, nominal, geometry, active))
    result.update(
        trial_max_ten_tip_error_m=observed, improvement_m=base_error - observed,
        trial_function_errors=functional_errors(geometry, target, active), functional_nondegenerate=preserved,
    )
    if not preserved:
        return dict(result, reason="FUNCTION_GEOMETRY_DEGENERATED")
    if observed > base_error - need:
        return dict(result, reason="IMPROVEMENT_BELOW_FROZEN_MINIMUM")
    return dict(result, status="WITNESSED", witness_q_rad=trial.tolist())
