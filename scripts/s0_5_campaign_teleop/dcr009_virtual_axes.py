#!/usr/bin/env python3
"""Check the declared OpenVR-world to MuJoCo scene axes on virtual fixtures."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import mujoco
import numpy as np


APPLICABILITY = "VIRTUAL_SCENE_COORDINATE_CONVENTION_ONLY"
TOLERANCE = 1e-10
# Historical software convention in teleop/openwbi_validation_20261120/
# ros_inputs.py:22,68–81. This is a virtual coordinate declaration. It does
# not measure a heading, an installation, or attachment across real captures.
UP_H, UP_W = (0, 1, 0), (0, 0, 1)
HORIZONTAL_H, HORIZONTAL_W = (1, 0, 0), (-1, 0, 0)

# Positions and directions are independently enumerated as paired coordinates
# in a signed virtual standard. Expected W values do not call declared_rotation.
FIXTURES = (
    ("left_skew", (-.12, .31, -.04), (.12, -.04, .31),
     ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
     ((-1, 0, 0), (0, 1, 0), (0, 0, -1))),
    ("right_skew", (.09, -.15, .28), (-.09, .28, -.15),
     ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
     ((0, 0, 1), (1, 0, 0), (0, 1, 0))),
    ("third_axis", (0, 0, .29), (0, .29, 0),
     ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
     ((0, -1, 0), (0, 0, 1), (-1, 0, 0))),
)


def basis(horizontal, up):
    return np.column_stack((horizontal, up, np.cross(horizontal, up)))


def declared_rotation():
    return basis(HORIZONTAL_W, UP_W) @ basis(HORIZONTAL_H, UP_H).T


def _numbers(values):
    return " ".join(str(float(value)) for value in values)


def fixture_results(rotation, *, swapped=False):
    rotation = np.asarray(rotation, dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=TOLERANCE)
            or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=TOLERANCE)):
        raise ValueError("The virtual scene transform must be proper SO(3)")
    sites = []
    for name, _, target, _, axes in FIXTURES:
        sites.append(f'<site name="{name}" pos="{_numbers(target)}" xyaxes="{_numbers(axes[0] + axes[1])}"/>')
    xml = "<mujoco><worldbody>" + "".join(sites) + "</worldbody></mujoco>"
    model = mujoco.MjModel.from_xml_string(xml)
    state = mujoco.MjData(model)
    mujoco.mj_kinematics(model, state)
    row_results = []
    inputs = [FIXTURES[1], FIXTURES[0], FIXTURES[2]] if swapped else FIXTURES
    for expected, source in zip(FIXTURES, inputs):
        site = state.site(expected[0])
        h_position, h_axes = source[1], np.asarray(source[3]).T
        expected_axes = np.asarray(expected[4]).T
        if not np.allclose(expected_axes, site.xmat.reshape(3, 3), rtol=0, atol=TOLERANCE):
            raise ValueError("The declared third W axis differs from the MuJoCo site basis")
        position = float(np.linalg.norm(rotation @ h_position - site.xpos))
        direction = float(np.max(np.linalg.norm(rotation @ h_axes - expected_axes, axis=0)))
        row_results.append({"fixture": expected[0], "input_fixture": source[0],
                            "position_error_m": position, "max_axis_vector_error": direction,
                            "pass": max(position, direction) <= TOLERANCE})
    return row_results


def evaluate():
    rotation = declared_rotation()
    valid = fixture_results(rotation)
    wrong_inverse = fixture_results(rotation @ np.diag((-1.0, 1.0, -1.0)))
    wrong_sides = fixture_results(rotation, swapped=True)
    try:
        fixture_results(rotation @ np.diag((1.0, 1.0, -1.0)))
    except ValueError:
        reflected_rejected = True
    else:
        reflected_rejected = False
    controls = {
        "declared_basis_positive": all(row["pass"] for row in valid),
        "reverse_horizontal_rejected": any(not row["pass"] for row in wrong_inverse),
        "left_right_swap_rejected": any(not row["pass"] for row in wrong_sides),
        "reflection_rejected": reflected_rejected,
    }
    return {
        "scope": APPLICABILITY, "status": "PASS" if all(controls.values()) else "FAIL",
        "source_convention": {"up_H": UP_H, "up_W": UP_W,
                              "horizontal_H": HORIZONTAL_H, "horizontal_W": HORIZONTAL_W,
                              "H_frame_literal": "openvr_world"},
        "virtual_world_from_tracker_rotation": rotation.tolist(), "holdouts": valid, "controls": controls,
        "target_lineage": "Fixed virtual direction and point fixtures. No HMB, M1, IK, or motion fit.",
        "capture_heading_qualified": False, "physical_calibration_qualified": False,
        "tracker_to_anatomical_palm_qualified": False,
    }


def support_on_model(receipt_path):
    """Confirm the measured virtual model axes underlying the declaration."""
    with receipt_path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["scope"] != "EXACT_MODEL_FK_AND_OPTIONAL_NATIVE_TCP_ONLY":
        raise ValueError("Expected the exact DCR-009 model FK receipt")
    gravity = np.asarray(receipt["gravity_live_m_s2"], dtype=float)
    left = np.asarray(receipt["simulator_bridges"]["l"]["world_from_base"], dtype=float)
    right = np.asarray(receipt["simulator_bridges"]["r"]["world_from_base"], dtype=float)
    rightward = right[:3, 3] - left[:3, 3]
    model_up = -gravity / np.linalg.norm(gravity)
    model_rightward = rightward / np.linalg.norm(rightward)
    return {
        "model_frame_receipt_sha256": digest,
        "live_model_sha256": receipt["model_identity"]["live"]["xml_sha256"],
        "virtual_up_W": model_up.tolist(),
        "virtual_rightward_W": model_rightward.tolist(),
        "virtual_forward_W": np.cross(model_up, model_rightward).tolist(),
        "rightward_standard": "Named robot left-base origin toward the right-base origin",
        "forward_standard": "Up cross rightward under the co-oriented virtual operator standard",
        "pass": bool(np.allclose(model_up, UP_W, rtol=0, atol=TOLERANCE)
                     and np.allclose(model_rightward, HORIZONTAL_W, rtol=0, atol=TOLERANCE)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-frame-receipt", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = evaluate()
    if args.model_frame_receipt:
        model_support = support_on_model(args.model_frame_receipt)
        results["exact_model_support"] = model_support
        if not model_support["pass"]:
            results["status"] = "FAIL"
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(results, indent=2, allow_nan=False) + "\n")
    return 0 if results["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
