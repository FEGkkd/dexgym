"""Verify left and right official hand contact near the index finger independently."""

from __future__ import annotations

import argparse
import json

from _bimanual_validation import run_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("beta1", "beta2"), default="beta1")
    args = parser.parse_args()
    measurements = {}
    for side in ("left", "right"):
        result = run_fixture("--variant", args.variant, "--steps", "30",
                             "--contact_probe_side", side)
        active = result[f"max_{side}_hand_contact_force_n"]
        other = result[f"max_{'right' if side == 'left' else 'left'}_hand_contact_force_n"]
        measurements[side] = {"active_force_n": active, "other_force_n": other}
        if active <= 0.01 or other > 0.01:
            raise AssertionError(f"Independent {side} contact sensing failed: {measurements[side]}")
        if result["max_constraint_error_m"] > 0.005:
            raise AssertionError(f"Contact caused thread constraint drift on {side}")
    print("HAND2_CONTACT_VALIDATION_PASS " + json.dumps({
        "variant": args.variant, "contacts": measurements,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
