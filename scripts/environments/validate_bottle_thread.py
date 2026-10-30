"""Deterministic GPU checks of finite breakaway and bottle-local screw motion.

Run with the Isaac Sim Python and WUJI_DESCRIPTION_ROOT pinned to v2026.9.11.
No RL task or policy is used. Results include the applied scripted test torque.
"""

from __future__ import annotations

import argparse
import json
import math

from _bimanual_validation import run_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("beta1", "beta2"), default="beta1")
    args = parser.parse_args()
    common = ("--variant", args.variant, "--steps", "120")
    cases = {
        "rest": ("--num_envs", "1", "--test_torque", "0"),
        "below_breakaway": ("--num_envs", "1", "--test_torque", "0.01"),
        "unscrew": ("--num_envs", "1", "--test_torque", "0.04"),
        "reverse": ("--num_envs", "1", "--test_torque", "-0.04"),
        "tilted_two_envs": ("--num_envs", "2", "--bottle_tilt_deg", "30", "--test_torque", "0.04"),
    }
    results = {}
    for name, arguments in cases.items():
        result = run_fixture(*common, *arguments)
        results[name] = result
        print("THREAD_CASE " + json.dumps({
            "case": name, "torque_nm": result["test_torque_nm"],
            "thread": result["thread"],
            "max_constraint_error_m": result["max_constraint_error_m"],
            "max_cap_speed_m_s": result["max_cap_speed_m_s"],
            "bottle_final_velocity": result["bottle_final_velocity"],
        }, sort_keys=True), flush=True)
    rest, low, forward, reverse, tilt = (results[name] for name in cases)
    def theta(result, env=0):
        return result["thread"][env]["theta_rad"]
    if abs(theta(rest)) > 0.06 or abs(theta(low)) > 0.06:
        raise AssertionError("Below-breakaway rotation exceeded the fixture tolerance")
    if theta(forward) < 0.03 or theta(reverse) > -0.03:
        raise AssertionError("Above-breakaway torque did not produce signed progress")
    if theta(tilt) < 0.03 or abs(theta(tilt, 1)) > 0.06:
        raise AssertionError("Tilted local-axis or second-environment isolation failed")
    for name, result in results.items():
        if result["left_joint_count"] != 20 or result["right_joint_count"] != 20:
            raise AssertionError(f"Hand joint count failed in {name}")
        if result["max_constraint_error_m"] > 0.002:
            raise AssertionError(f"Thread constraint drift in {name}")
        if not math.isfinite(result["max_cap_speed_m_s"]) or result["max_cap_speed_m_s"] > 5:
            raise AssertionError(f"Unbounded cap speed in {name}")
        for state in result["thread"]:
            if abs(state["axial_m"] - state["expected_axial_m"]) > 0.002:
                raise AssertionError(f"Pitch coupling failed in {name}")
    if abs(forward["bottle_final_velocity"][0][5]) < 0.01:
        raise AssertionError("Free bottle did not react to the cap torque")
    print("THREAD_VALIDATION_PASS " + json.dumps({
        "variant": args.variant,
        "forward_theta_rad": theta(forward),
        "reverse_theta_rad": theta(reverse),
        "tilted_theta_rad": theta(tilt),
        "low_torque_theta_rad": theta(low),
        "independent_env_theta_rad": theta(tilt, 1),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
