"""Native static residual-witness diagnostics with independent synthetic targets."""

from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel
from dexgym.tasks.direct.wholebody_s05.v1.rmb.candidate_improvement import evaluate_candidate
from dexgym.tasks.direct.wholebody_s05.v1.rmb.domains import realized
from dexgym.tasks.direct.wholebody_s05.v1.task.native_sweep import NativeSweep


GENERATED = Path(__file__).resolve().parents[3] / "source/dexgym/assets/marvin_wuji/hand2_2/generated"
ACTIVE = np.ones((2, 4), dtype=bool)


@pytest.fixture(scope="module")
def robot(tmp_path_factory):
    assets = GENERATED
    if sys.platform == "win32":
        assets = tmp_path_factory.mktemp("s05-candidate") / "generated"
        shutil.copytree(GENERATED, assets, ignore=shutil.ignore_patterns("usd"))
    return RobotModel(assets / "dual_arm_marvin.xml")


def fixture(robot):
    q = robot.home.copy()
    q[0] += .035
    return q, realized(robot, q), NativeSweep(robot)


def test_native_static_search_witness_preserves_function(robot):
    _, target, sweep = fixture(robot)
    result = evaluate_candidate(robot, robot.home, target, ACTIVE, sweep=sweep)
    assert result["status"] == "WITNESSED"
    assert result["required_improvement_m"] == pytest.approx(.10 * result["nominal_max_ten_tip_error_m"])
    assert result["improvement_m"] >= result["required_improvement_m"]
    assert result["witness_q_rad"] is not None
    assert result["functional_nondegenerate"] and result["swept_path_proven"]
    assert not result["dynamic_qualification"] and not result["servo_rollout_executed"]


@pytest.mark.parametrize(("axis", "degrees"), [(0, 8.001), (14, 5.001)])
def test_rejects_any_axis_outside_frozen_residual_envelope(robot, axis, degrees):
    _, target, sweep = fixture(robot)
    trial = robot.home.copy()
    trial[axis] += np.deg2rad(degrees)
    result = evaluate_candidate(robot, robot.home, target, ACTIVE, sweep=sweep, witness_q=trial)
    assert result["status"] == "UNKNOWN"
    assert result["reason"] == "RESIDUAL_ENVELOPE_EXCEEDED"
    assert result["witness_q_rad"] is None


def test_improving_legal_witness_rejected_when_function_geometry_degrades(robot):
    trial, target, sweep = fixture(robot)
    trial[[15, 19]] += .08
    result = evaluate_candidate(robot, robot.home, target, ACTIVE, sweep=sweep, witness_q=trial)
    assert result["within_residual_envelope"] and result["swept_path_proven"]
    assert result["improvement_m"] > result["required_improvement_m"]
    assert result["trial_function_errors"]["p95_local_m"] > .002
    assert result["trial_function_errors"]["max_active_pinch_m"] > .002
    assert result["reason"] == "FUNCTION_GEOMETRY_DEGENERATED"
    assert result["status"] == "UNKNOWN" and result["witness_q_rad"] is None


def test_requires_independent_sweep_and_legal_endpoint(robot):
    trial, target, _ = fixture(robot)
    no_sweep = SimpleNamespace(permits=lambda start, end: np.array_equal(start, end))
    result = evaluate_candidate(robot, robot.home, target, ACTIVE, sweep=no_sweep, witness_q=trial)
    assert result["nominal_pose_proven"] and result["trial_endpoint_safety"]["feasible"]
    assert result["reason"] == "WITNESS_PATH_NOT_PROVEN"
    assert result["witness_q_rad"] is None

    axis = robot.names.index("l_index_finger_mcp_abd")
    start, colliding = robot.home.copy(), robot.home.copy()
    start[axis] += .20
    colliding[axis] += .275
    result = evaluate_candidate(robot, start, target, ACTIVE, sweep=NativeSweep(robot), witness_q=colliding)
    assert result["within_residual_envelope"]
    assert result["reason"] == "WITNESS_ENDPOINT_ILLEGAL"
    assert result["trial_endpoint_safety"]["penetration_m"] < 0


def test_bounded_search_without_improvement_remains_unknown(robot):
    target = realized(robot, robot.home)
    result = evaluate_candidate(robot, robot.home, target, ACTIVE, sweep=NativeSweep(robot))
    assert result["required_improvement_m"] == .002
    assert result["status"] == "UNKNOWN" and result["reason"] == "IMPROVEMENT_BELOW_FROZEN_MINIMUM"
    assert result["witness_q_rad"] is None
