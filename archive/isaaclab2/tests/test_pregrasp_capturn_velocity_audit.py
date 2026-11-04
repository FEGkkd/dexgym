"""Numerical regression for the 500 Hz Mimic velocity-layer audit."""

import importlib.util
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[1] / "scripts/environments/audit_phase1b_velocity_layers.py"
_SPEC = importlib.util.spec_from_file_location("phase1b_velocity_audit", _PATH)
_AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIT)


def _trace():
    rows = []
    for i, (spin, pose_spin, sliding, force) in enumerate((
            (0.0, None, False, 0.0),
            (2.0, 1.0, False, 2.0),
            (4.0, 3.0, True, 6.0))):
        rows.append({
            "substep": i, "reset_sample_index": 7,
            "physx_screw_spin_rad_s": spin,
            "relative_body_spin_projected_rad_s": spin,
            "pose_differenced_spin_rad_s": pose_spin,
            "joint_differenced_spin_rad_s": pose_spin,
            "pose_relative_opening_rad": (0.0, 0.002, 0.008)[i],
            "drive_sliding": sliding,
            "right_cap_normal_force_n": force,
        })
    return {"physx_hz": 500, "episode_success": True, "rows": rows}


def test_velocity_layers_return_numeric_errors_and_drive_modes():
    result = _AUDIT.audit(_trace())
    assert result["pose_opening_rad"] == pytest.approx(0.008)
    assert result["screw_vs_body"]["mae_rad_s"] == 0
    assert result["screw_vs_pose_difference"]["mae_rad_s"] == 1
    assert result["screw_vs_pose_difference"]["integrated_bias_rad"] == pytest.approx(0.004)
    assert result["joint_difference_vs_pose_difference"]["mae_rad_s"] == 0
    assert result["by_drive_mode"]["sticking"]["substeps"] == 1
    assert result["by_drive_mode"]["sliding"]["mean_right_cap_force_n"] == 6


def test_velocity_layers_reject_nonfinite_screw_speed():
    trace = _trace()
    trace["rows"][2]["physx_screw_spin_rad_s"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        _AUDIT.audit(trace)
