"""GPU check that all 20 official finger position drives respond on both sides."""

from __future__ import annotations

import argparse
import json

from _bimanual_validation import run_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("beta1", "beta2"), default="beta1")
    args = parser.parse_args()
    common = ("--variant", args.variant, "--steps", "120")
    baseline = run_fixture(*common)
    measurements = {}
    for side in ("left", "right"):
        command = ("--left_fingers" if side == "left" else "--right_fingers")
        result = run_fixture(*common, command, *("0.5" for _ in range(20)))
        names = result[f"{side}_joint_names"]
        positions = result[f"{side}_final_joints"][0]
        base_positions = baseline[f"{side}_final_joints"][0]
        displacement = {name: positions[i] - base_positions[i] for i, name in enumerate(names)}
        other = "right" if side == "left" else "left"
        other_deviation = max(abs(a - b) for a, b in zip(
            result[f"{other}_final_joints"][0], baseline[f"{other}_final_joints"][0]
        ))
        measurements[side] = {
            "min_commanded_joint_displacement_rad": min(displacement.values()),
            "max_other_hand_joint_deviation_rad": other_deviation,
            "joint_displacement_rad": displacement,
        }
        if len(displacement) != 20 or min(displacement.values()) < 0.05:
            raise AssertionError(f"A {side} finger drive failed to respond: {measurements[side]}")
        if other_deviation > 0.02:
            raise AssertionError(f"{side} command changed the other hand: {other_deviation}")
    print("HAND2_FINGER_VALIDATION_PASS " + json.dumps({
        "variant": args.variant, "sides": measurements,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
