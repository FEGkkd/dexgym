"""Paired Native MuJoCo Servo evidence for one preassigned static DCR-008 witness."""

import numpy as np
import torch

from ..task import PHYSICS_DT, SUBSTEPS
from ..task.geometry import Geometry, quaternion_xyzw_to_matrix
from ..task.physics import ServoProfile, signature
from ..task.smoke import make_task, target_frame
from .candidate_improvement import evaluate_candidate, finite_array, functional_errors, validated_target
from .domains import RESIDUAL_RAD, realized
from .schema import TICK_NS


TICKS = 100
SCORE_TICKS = 25
SUBSTEP_NS = round(PHYSICS_DT * 1_000_000_000)
COMMAND_TOL_RAD = 2e-6
FUNCTION_INCREASE_LIMITS = {
    "p95_local_m": .002,
    "max_active_pinch_m": .002,
    "max_cross_m": .002,
    "relative_wrist_translation_m": .002,
    "relative_wrist_rotation_deg": 1.,
}


def scorer(rows_b0, rows_b54):
    def p95(rows):
        errors = [row["max_ten_tip_error_m"] for row in rows[-SCORE_TICKS:]]
        if len(errors) != SCORE_TICKS or any(value is None or not np.isfinite(value) for value in errors):
            return None
        return float(np.percentile(errors, 95))

    base, trial = p95(rows_b0), p95(rows_b54)
    need = max(.002, .10 * base) if base is not None else None
    status = {
        "b0_p95_m": base, "b54_witness_p95_m": trial, "required_improvement_m": need,
        "improvement_m": None, "score_timing": "SAME_COMMAND_TICK_POST_CONTROL",
        "functional_nondegenerate": None, "functions": {},
        "numeric_gate_met": False,
    }
    if (base is None or trial is None or len(rows_b0) != TICKS or len(rows_b54) != TICKS
            or any(row["status"] != "EXECUTED" for rows in (rows_b0, rows_b54) for row in rows)):
        return status
    paired = list(zip(rows_b0[-SCORE_TICKS:], rows_b54[-SCORE_TICKS:], strict=True))
    if any(a["command_tick_ns"] != b["command_tick_ns"] for a, b in paired):
        raise ValueError("Paired witness ticks are not on the same time axis")
    for name, limit in FUNCTION_INCREASE_LIMITS.items():
        errors = [(a["function_errors"][name], b["function_errors"][name]) for a, b in paired]
        applicable = errors[0][0] is not None
        if any((x is not None) != applicable or (y is not None) != applicable for x, y in errors):
            raise ValueError("A fixed functional mask must apply equally to all paired ticks")
        finite = not applicable or all(np.isfinite(x) and np.isfinite(y) for x, y in errors)
        max_increase = max(y - x for x, y in errors) if applicable and finite else None
        status["functions"][name] = {
            "applicable": applicable, "max_same_tick_increase": max_increase, "allowed_increase": limit,
            "within_limit": finite and max_increase <= limit if applicable else None,
        }
    preserved = all(row["within_limit"] is not False for row in status["functions"].values())
    settled = all(row["issued_endpoint_error_rad"] is not None
                  and row["issued_endpoint_error_rad"] <= COMMAND_TOL_RAD for pair in paired for row in pair)
    status.update(improvement_m=base - trial, functional_nondegenerate=preserved, issued_endpoint_reached=settled,
                  numeric_gate_met=preserved and settled and trial <= base - need)
    return status


def trace_row(tick, source_tick_ns):
    command_tick_ns = source_tick_ns + tick * TICK_NS
    return {
        "tick": tick, "command_tick_ns": command_tick_ns, "feedback_tick_ns": None,
        "status": "NOT_RUN", "executed_substeps": 0, "events": None, "max_ten_tip_error_m": None,
    }


def physical_feedback(task):
    robot = task.guard.robot
    return np.r_[task.physics.data[0].qpos[robot.q_indices], task.physics.data[0].qvel[robot.v_indices]].copy()


def execute_lane(task, nominal, endpoint, geometry, target, active, action, source_tick_ns):
    rows = [trace_row(tick, source_tick_ns) for tick in range(TICKS)]
    ids = torch.arange(1)
    try:
        task.reset(ids, nominal, torch.zeros_like(nominal))
    except ValueError as error:
        return {"failure": "INITIAL_RESET_REJECTED", "detail": str(error), "actual_transitions": 0,
                "actual_substeps": 0, "rows": rows}

    frame = target_frame(task, nominal, 0)._replace(geometry=geometry, pinch_active=torch.tensor(active[None]))
    failure = None
    for row in rows:
        if failure is not None:
            row["status"] = "NOT_RUN_AFTER_FAILURE"
            continue
        task.observe(frame._replace(tick_ns=torch.tensor([row["command_tick_ns"]])))
        transition = task.step(action)
        substeps, event = int(transition.substeps[0]), int(transition.events[0])
        state = physical_feedback(task)
        issued = transition.c[0].numpy()
        contact_depth = transition.contact_depth_m[0].item()
        physical_time = task.physics.data[0].time
        state_finite = bool(np.isfinite(state).all())
        issued_finite = bool(np.isfinite(issued).all())
        row.update(
            executed_substeps=substeps, events=event, fallback=bool(transition.fallback[0]),
            terminated=bool(transition.terminated[0]), hard_stop=bool(transition.safety_failure[0]),
            issued_endpoint_error_rad=float(np.max(np.abs(issued - endpoint))) if issued_finite else None,
            q_issued_rad=issued.tolist() if issued_finite else None,
            q_actual_rad=state[:54].tolist() if state_finite else None,
            qd_actual_rad_s=state[54:].tolist() if state_finite else None,
            feedback_tick_ns=row["command_tick_ns"] + substeps * SUBSTEP_NS if substeps else None,
            native_time_s=float(physical_time) if np.isfinite(physical_time) else None,
            contact_depth_m=contact_depth if np.isfinite(contact_depth) else None,
        )
        if (event or bool(transition.fallback[0]) or not bool(transition.valid[0])
                or substeps != SUBSTEPS or not state_finite or not issued_finite
                or not np.isfinite(contact_depth) or contact_depth > 0 or not np.isfinite(physical_time)):
            failure = "PHYSICAL_HARD_STOP" if row["hard_stop"] else "INCOMPLETE_OR_REJECTED_TRANSITION"
            row.update(status="FAILED", failure=failure)
            continue
        actual = realized(task.guard.robot, state[:54])
        error = np.linalg.norm(actual["world_tip_xyz"] - target["world_tip_xyz"], axis=-1).max()
        row.update(status="EXECUTED", max_ten_tip_error_m=float(error),
                   function_errors=functional_errors(actual, target, active))
    return {
        "failure": failure, "rows": rows,
        "actual_transitions": sum(row["executed_substeps"] > 0 for row in rows),
        "actual_substeps": sum(row["executed_substeps"] for row in rows),
    }


class PairedStaticServo:
    """Reuse two identical-profile native tasks across caller-selected diagnostic points."""

    def __init__(self, robot, *, expected_model_signature_sha256=None):
        self.robot = robot
        profile = ServoProfile()
        self.model = signature(robot, profile)
        if expected_model_signature_sha256 is not None and self.model["sha256"] != expected_model_signature_sha256:
            raise ValueError("The supplied signed model/Servo identity differs from the physical runtime")
        self.signature_prebound = expected_model_signature_sha256 is not None
        self.b0, _, _ = make_task(robot, 1, profile, "B0", "cpu")
        self.b54, _, _ = make_task(robot, 1, profile, "B54", "cpu")

    @torch.inference_mode()
    def evaluate(self, q_nominal, q_witness, mapped_geometry, pinch_active, *, source_tick_ns=0):
        if type(source_tick_ns) is not int or not 0 <= source_tick_ns <= 2**63 - 1 - TICKS * TICK_NS:
            raise ValueError("Expected a bounded nonnegative source timestamp in integer nanoseconds")
        q_nominal = finite_array(q_nominal, (54,), "q_nominal")
        q_witness = finite_array(q_witness, (54,), "q_witness")
        target, active = validated_target(mapped_geometry, pinch_active)
        static = evaluate_candidate(self.robot, q_nominal, target, active, sweep=self.b0.guard.sweep,
                                    witness_q=q_witness)
        result = {
            "scope": "DCR008_STATIC_NATIVE_MUJOCO_SERVO_DIAGNOSTIC",
            "model_signature_sha256": self.model["sha256"], "model_signature_prebound": self.signature_prebound,
            "source_tick_ns": source_tick_ns, "planned_physical_transitions": 2 * TICKS,
            "actual_physical_transitions": 0, "actual_physics_substeps": 0, "scoring_ticks_per_lane": SCORE_TICKS,
            "controller_hz": 50, "physics_hz": 500, "oracle_action_constant": True,
            "dynamic_qualification": False, "phase_qualification": False,
            "initial_q_rad": q_nominal.tolist(), "initial_qd_rad_s": [0.] * 54,
            "requested_witness_q_rad": q_witness.tolist(), "kinematic_static_witness": static,
            "servo_rollout_executed": False, "status": "INVALID_STATIC_WITNESS", "lanes": {},
        }
        if static["status"] != "WITNESSED":
            return dict(result, reason=static["reason"])

        def batch(values):
            return torch.tensor(values[None], dtype=torch.float32)

        geometry = Geometry(batch(target["wrist_xyz"]), quaternion_xyzw_to_matrix(batch(target["wrist_xyzw"])),
                            batch(target["local_landmark_xyz"]), batch(target["world_tip_xyz"]))
        nominal = batch(q_nominal)
        # Admission checked the exact radian envelope before fp32 normalization.
        residual = batch(np.clip((q_witness - q_nominal) / RESIDUAL_RAD, -1., 1.))
        lanes = {
            "B0": execute_lane(self.b0, nominal, q_nominal, geometry, target, active, torch.zeros_like(nominal),
                               source_tick_ns),
            "B54_ORACLE_WITNESS": execute_lane(self.b54, nominal, q_witness, geometry, target, active, residual,
                                               source_tick_ns),
        }
        actual = sum(lane["actual_transitions"] for lane in lanes.values())
        resulting_score = scorer(lanes["B0"]["rows"], lanes["B54_ORACLE_WITNESS"]["rows"])
        return dict(result, lanes=lanes, actual_physical_transitions=actual,
                    actual_physics_substeps=sum(lane["actual_substeps"] for lane in lanes.values()),
                    servo_rollout_executed=actual > 0, score=resulting_score,
                    status="PHYSICAL_STATIC_WITNESSED" if resulting_score["numeric_gate_met"] else "NO_PHYSICAL_WITNESS")


def evaluate_servo_witness(robot, q_nominal, q_witness, mapped_geometry, pinch_active, *,
                           expected_model_signature_sha256=None, source_tick_ns=0):
    return PairedStaticServo(robot, expected_model_signature_sha256=expected_model_signature_sha256).evaluate(
        q_nominal, q_witness, mapped_geometry, pinch_active, source_tick_ns=source_tick_ns)
