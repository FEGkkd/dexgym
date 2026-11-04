"""Compare 500 Hz Mimic joint, rigid-body, and pose-derived angular speeds."""

import argparse
import json
import math
from pathlib import Path


def finite_series(rows, key, start=0):
    values = [row[key] for row in rows[start:]]
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise ValueError(f"{key} contains a missing or nonfinite value")
    return values


def errors(a, b, dt):
    if not a or len(a) != len(b):
        raise ValueError("Angular-speed series must have equal nonzero lengths")
    delta = [x - y for x, y in zip(a, b)]
    return {
        "mae_rad_s": sum(abs(x) for x in delta) / len(delta),
        "max_abs_rad_s": max(abs(x) for x in delta),
        "integrated_bias_rad": sum(delta) * dt,
    }


def audit(trace):
    rows = trace["rows"]
    if len(rows) < 2 or trace.get("physx_hz") != 500:
        raise ValueError("Expected at least two substeps of a 500 Hz trace")
    if any(row["substep"] != i for i, row in enumerate(rows)):
        raise ValueError("Substep indices must be consecutive")
    reset_rows = {row["reset_sample_index"] for row in rows}
    if len(reset_rows) != 1:
        raise ValueError("Trace spans more than one Reset sample")
    dt = 1 / trace["physx_hz"]
    screw = finite_series(rows, "physx_screw_spin_rad_s")
    body = finite_series(rows, "relative_body_spin_projected_rad_s")
    pose = finite_series(rows, "pose_differenced_spin_rad_s", 1)
    joint = finite_series(rows, "joint_differenced_spin_rad_s", 1)
    result = {
        "episode_success": bool(trace["episode_success"]),
        "reset_sample_index": next(iter(reset_rows)),
        "substeps": len(rows),
        "pose_opening_rad": rows[-1]["pose_relative_opening_rad"] - rows[0]["pose_relative_opening_rad"],
        "screw_velocity_integral_rad": sum(screw[1:]) * dt,
        "body_velocity_integral_rad": sum(body[1:]) * dt,
        "screw_vs_body": errors(screw, body, dt),
        "screw_vs_pose_difference": errors(screw[1:], pose, dt),
        "body_vs_pose_difference": errors(body[1:], pose, dt),
        "joint_difference_vs_pose_difference": errors(joint, pose, dt),
        "previous_screw_vs_pose_difference": errors(screw[:-1], pose, dt),
        "next_screw_vs_pose_difference": errors(screw[2:], pose[:-1], dt),
        "trapezoid_screw_vs_pose_difference": errors(
            [(screw[i - 1] + screw[i]) / 2 for i in range(1, len(rows))], pose, dt),
    }
    if all("drive_sliding" in row for row in rows):
        result["by_drive_mode"] = {}
        for label, sliding in (("sticking", False), ("sliding", True)):
            selected = [row for row in rows[1:] if row["drive_sliding"] is sliding]
            if selected:
                result["by_drive_mode"][label] = {
                    "substeps": len(selected),
                    "screw_vs_pose_difference": errors(
                        [row["physx_screw_spin_rad_s"] for row in selected],
                        [row["pose_differenced_spin_rad_s"] for row in selected], dt),
                    "mean_right_cap_force_n": sum(
                        row["right_cap_normal_force_n"] for row in selected) / len(selected),
                }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(json.loads(args.trace.read_text()))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
