"""GPU fixture: a fully unscrewed cap releases and can fall under a small probe force.

The lateral probe is an external test force, not a learned robot action.
"""

from __future__ import annotations

import json
import math

from _bimanual_validation import run_fixture
from dexteroushands_lab.assets.bottle import BottleConfig


def main():
    result = run_fixture(
        "--num_envs", "2", "--steps", "900", "--bottle_center_z", "0.1",
        "--test_torque", "0.04", "--allow_detachment",
        "--detach_probe_force", "0.05",
    )
    released, untouched = result["thread"]
    step = result["detachment_steps"][0]
    if step is None or not 400 <= step <= 700:
        raise AssertionError(f"Cap did not release at the thread end: {step}")
    if result["detachment_steps"][1] is not None:
        raise AssertionError("Unforced second environment detached")
    if result["joint_enabled"] != [False, True]:
        raise AssertionError("PhysX joint enabled flags did not change independently")
    if not released["detached"] or untouched["detached"]:
        raise AssertionError("Thread coupler states did not separate independently")
    if released["lateral_error_m"] < 0.1 or untouched["lateral_error_m"] > 0.01:
        raise AssertionError("Released cap did not separate from the bottle")
    if result["cap_final_pose"][0][2] > BottleConfig().cap_height:
        raise AssertionError("Released cap did not fall to the ground")
    if (not math.isfinite(result["max_cap_speed_m_s"])
            or result["max_cap_speed_m_s"] > BottleConfig().max_body_linear_velocity + 0.1):
        raise AssertionError("Released cap velocity exceeded the configured safety bound")
    if result["max_constraint_error_m"] > 0.0004:
        raise AssertionError("Screw constraint drift before release exceeded tolerance")
    print("DETACHMENT_VALIDATION_PASS " + json.dumps({
        "detachment_step": step,
        "released_lateral_m": released["lateral_error_m"],
        "untouched_lateral_m": untouched["lateral_error_m"],
        "released_cap_center_z_m": result["cap_final_pose"][0][2],
        "joint_enabled": result["joint_enabled"],
        "max_cap_speed_m_s": result["max_cap_speed_m_s"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
