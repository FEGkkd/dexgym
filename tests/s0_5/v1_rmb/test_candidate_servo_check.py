"""Native Servo witness evidence and negative controls on paired scoring."""

from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel
from dexgym.tasks.direct.wholebody_s05.v1.rmb.candidate_servo_check import (
    FUNCTION_INCREASE_LIMITS,
    TICKS,
    PairedStaticServo,
    scorer,
)
from dexgym.tasks.direct.wholebody_s05.v1.rmb.domains import realized
from dexgym.tasks.direct.wholebody_s05.v1.rmb.schema import TICK_NS
from dexgym.tasks.direct.wholebody_s05.v1.task.physics import ServoProfile, signature


GENERATED = Path(__file__).resolve().parents[3] / "source/dexgym/assets/marvin_wuji/hand2_2/generated"
ACTIVE = np.ones((2, 4), dtype=bool)


@pytest.fixture(scope="module")
def servo(tmp_path_factory):
    assets = GENERATED
    if sys.platform == "win32":
        assets = tmp_path_factory.mktemp("s05-paired-servo") / "generated"
        shutil.copytree(GENERATED, assets, ignore=shutil.ignore_patterns("usd"))
    robot = RobotModel(assets / "dual_arm_marvin.xml")
    return PairedStaticServo(robot, expected_model_signature_sha256=signature(robot, ServoProfile())["sha256"])


def test_actual_signed_servo_pair_runs_full_identical_time_axis(servo):
    q = servo.robot.home.copy()
    q[0] += .035
    start = 1_000_000_000
    evidence = servo.evaluate(servo.robot.home, q, realized(servo.robot, q), ACTIVE, source_tick_ns=start)
    assert evidence["status"] == "PHYSICAL_STATIC_WITNESSED"
    assert evidence["model_signature_prebound"] and evidence["servo_rollout_executed"]
    assert not evidence["dynamic_qualification"] and not evidence["phase_qualification"]
    assert evidence["actual_physical_transitions"] == evidence["planned_physical_transitions"] == 2 * TICKS
    assert evidence["actual_physics_substeps"] == 20 * TICKS
    for lane in evidence["lanes"].values():
        assert lane["failure"] is None
        for i, row in enumerate(lane["rows"]):
            assert row["status"] == "EXECUTED" and row["events"] == 0
            assert row["command_tick_ns"] == start + i * TICK_NS
            assert row["feedback_tick_ns"] == start + (i + 1) * TICK_NS
    score = evidence["score"]
    assert score["functional_nondegenerate"] and score["issued_endpoint_reached"]
    assert score["b54_witness_p95_m"] < score["b0_p95_m"] - score["required_improvement_m"]


def test_bad_static_witness_runs_no_physical_transitions(servo):
    base = servo.robot.home.copy()
    q = base.copy()
    q[0] += np.deg2rad(8.01)
    evidence = servo.evaluate(base, q, realized(servo.robot, base), ACTIVE)
    assert evidence["status"] == "INVALID_STATIC_WITNESS"
    assert evidence["reason"] == "RESIDUAL_ENVELOPE_EXCEEDED"
    assert evidence["actual_physical_transitions"] == 0 and not evidence["servo_rollout_executed"]


def paired_rows(distance):
    return [{
        "status": "EXECUTED", "command_tick_ns": i * TICK_NS, "max_ten_tip_error_m": distance,
        "function_errors": dict.fromkeys(FUNCTION_INCREASE_LIMITS, 0.), "issued_endpoint_error_rad": 0.,
    } for i in range(TICKS)]


def test_rejects_incomplete_tail_and_nonconverged_command():
    b0, b54 = paired_rows(.020), paired_rows(.002)
    assert scorer(b0, b54)["numeric_gate_met"]
    b54[-1]["issued_endpoint_error_rad"] = .001
    assert not scorer(b0, b54)["numeric_gate_met"]
    b54[-1].update(status="NOT_RUN_AFTER_FAILURE", max_ten_tip_error_m=None)
    result = scorer(b0, b54)
    assert not result["numeric_gate_met"] and result["b54_witness_p95_m"] is None


def test_rejects_functional_degradation_and_asymmetric_timestamps():
    b0, b54 = paired_rows(.020), paired_rows(.002)
    b54[-1]["function_errors"]["max_active_pinch_m"] = .0021
    result = scorer(b0, b54)
    assert result["improvement_m"] > result["required_improvement_m"]
    assert not result["functional_nondegenerate"] and not result["numeric_gate_met"]
    b54[-1]["command_tick_ns"] += 1
    with pytest.raises(ValueError, match="same time axis"):
        scorer(b0, b54)
