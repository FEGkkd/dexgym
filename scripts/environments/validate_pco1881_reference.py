"""GPU fixture for the sourced PCO1881 example; not a real-bottle calibration."""

from __future__ import annotations

import json
import math

from _bimanual_validation import run_fixture
from dexteroushands_lab.assets.bottle import BottleConfig


def main() -> None:
    cfg = BottleConfig.pco1881_reference()
    common = (
        "--bottle_profile", "pco1881_reference", "--bottle_center_z", "0.095",
        "--fixture_hold_bottle", "--steps", "2400",
    )
    cases = {
        "below_breakaway": ("--test_torque", "0.6"),
        "open": ("--test_torque", "1.2"),
        "tighten": ("--test_torque", "-1.2"),
        "tilted_two_envs": ("--test_torque", "1.2", "--num_envs", "2",
                            "--bottle_tilt_deg", "30", "--bottle_center_z", "0.11"),
    }
    results = {}
    for name, args in cases.items():
        result = run_fixture(*common, *args)
        results[name] = result
        print("PCO1881_CASE " + json.dumps({
            "name": name, "torque_nm": result["test_torque_nm"],
            "states": result["thread"],
            "max_constraint_error_m": result["max_constraint_error_m"],
            "max_cap_speed_m_s": result["max_cap_speed_m_s"],
        }, sort_keys=True), flush=True)
    low, opened, tightened, tilted = (results[name] for name in cases)
    theta = lambda result, i=0: result["thread"][i]["theta_rad"]
    if not (abs(theta(low)) < 0.05 and theta(opened) > 0.2
            and theta(tightened) < -0.2 and theta(tilted) > 0.2
            and abs(theta(tilted, 1)) < 0.05):
        raise AssertionError("PCO1881 static breakaway, signed motion or environment isolation failed")
    for name, result in results.items():
        if result["physics_dt_s"] != 0.0001 or result["status"] != "ran_without_nonfinite_state":
            raise AssertionError(f"Wrong physics step or nonfinite state in {name}")
        if result["max_constraint_error_m"] > 0.0002:
            raise AssertionError(f"Screw constraint drift in {name}")
        if not math.isfinite(result["max_cap_speed_m_s"]) or result["max_cap_speed_m_s"] > 5:
            raise AssertionError(f"Unbounded cap speed in {name}")
        for state in result["thread"]:
            if abs(state["constraint_error_m"]) > 0.0001:
                raise AssertionError(f"Final pitch residual in {name}")
    if not (opened["thread"][0]["axial_m"] - low["thread"][0]["axial_m"] > 0.0001
            and tightened["thread"][0]["axial_m"] - low["thread"][0]["axial_m"] < -0.0001):
        raise AssertionError("Axial motion does not follow signed pitch")
    if cfg.breakaway_torque != 0.9:
        raise AssertionError("Reference opening torque changed without updating this fixture")
    print("PCO1881_VALIDATION_PASS " + json.dumps({
        "low_theta_rad": theta(low), "open_theta_rad": theta(opened),
        "tighten_theta_rad": theta(tightened),
        "tilted_theta_rad": theta(tilted),
        "independent_env_theta_rad": theta(tilted, 1),
        "nominal_breakaway_nm": cfg.breakaway_torque,
        "physics_dt_s": low["physics_dt_s"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
