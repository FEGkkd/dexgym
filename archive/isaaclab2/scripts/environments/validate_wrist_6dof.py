"""GPU checks that both floating hands respond independently on all six axes."""

from __future__ import annotations

import argparse
import json
import math

from _bimanual_validation import run_fixture


def _angle(a, b):
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return 2 * math.acos(min(1.0, dot))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("beta1", "beta2"), default="beta1")
    args = parser.parse_args()
    common = ("--variant", args.variant, "--steps", "120", "--num_envs", "1")
    baseline = run_fixture(*common)
    axes = (("x", ("0.01", "0", "0"), ("0.08", "0", "0")),
            ("y", ("0", "0.01", "0"), ("0", "0.08", "0")),
            ("z", ("0", "0", "0.01"), ("0", "0", "0.08")))
    measurements = {}
    for axis, translation, rotation in axes:
        result = run_fixture(*common, "--left_xyz", *translation,
                             "--right_rot", *rotation)
        index = "xyz".index(axis)
        left_delta = result["left_final_pose"][0][index] - baseline["left_final_pose"][0][index]
        right_angle = _angle(result["right_final_pose"][0][3:7],
                             baseline["right_final_pose"][0][3:7])
        left_orientation_change = _angle(result["left_final_pose"][0][3:7],
                                         baseline["left_final_pose"][0][3:7])
        measurements[axis] = {
            "left_translation_m": left_delta, "right_rotation_rad": right_angle,
            "left_orientation_change_rad": left_orientation_change,
            "max_wrist_speed_m_s": result["max_wrist_speed_m_s"],
            "max_wrist_angular_speed_rad_s": result["max_wrist_angular_speed_rad_s"],
            "max_wrist_force_n": result["max_wrist_force_n"],
            "max_wrist_torque_nm": result["max_wrist_torque_nm"],
        }
        print("WRIST_CASE " + json.dumps({"axis": axis, **measurements[axis]}, sort_keys=True), flush=True)
        if left_delta < 0.002 or right_angle < 0.005:
            raise AssertionError(f"Independent six-axis response failed for {axis}")
        if result["max_wrist_force_n"] > 40.01 or result["max_wrist_torque_nm"] > 2.01:
            raise AssertionError(f"Wrist wrench limit failed for {axis}")
        if not math.isfinite(result["max_wrist_speed_m_s"]) or result["max_wrist_speed_m_s"] > 5:
            raise AssertionError(f"Unbounded wrist speed for {axis}")
        if result["max_wrist_angular_speed_rad_s"] > 6.5:
            raise AssertionError(f"Wrist angular speed limit failed for {axis}")
    print("WRIST_VALIDATION_PASS " + json.dumps({"variant": args.variant, "axes": measurements}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
