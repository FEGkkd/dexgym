#!/usr/bin/env python3
"""Audit named geometry of two exact MuJoCo models and optional Marvin FK."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.p02.model import (
    FINGERS, SIDES, RobotModel, asset_files, digest,
)


LIVE_SHA = "be69f387f5bc25c8011a992a88ac2c2449c2c0419e09318b740fcdcd1402871d"
SIGNED_SHA = "939b3651e2426e596eed482e70a68c2382720df1fd55b124aa5f0893b640e92d"
POSITION_M = 0.0001
ROTATION_RAD = 0.001


def pose(position, axes):
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(axes).reshape(3, 3)
    transform[:3, 3] = position
    return transform


def body(robot, name):
    value = robot.data.body(name)
    return pose(value.xpos, value.xmat)


def pose_error(one, two):
    translation = np.linalg.norm(one[:3, 3] - two[:3, 3])
    rotation = Rotation.from_matrix(one[:3, :3].T @ two[:3, :3]).magnitude()
    return [float(translation), float(rotation)]


def palm_basis(geometry, side):
    """Model-only palm frame, with origin at the named Wuji wrist body."""
    root = geometry.local[side, 1:, 0]
    distal = root.mean(axis=0)
    distal /= np.linalg.norm(distal)
    thumbward = root[0] - root[-1]
    thumbward -= distal * np.dot(distal, thumbward)
    thumbward /= np.linalg.norm(thumbward)
    return np.column_stack((distal, thumbward, np.cross(distal, thumbward)))


def model_identity(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"Model identity mismatch at {path}: {actual} != {expected}")
    return {"xml_sha256": actual, "dependencies": asset_files(path)}


def coordinate_table(live, signed):
    if live.names != signed.names:
        raise ValueError("The ordered named 54-axis coordinate tables differ")
    result = []
    for index, name in enumerate(live.names):
        a, b = live.joints[index], signed.joints[index]
        drive_a, drive_b = live.actuators[index], signed.actuators[index]
        parent_a = live.model.body(int(a.bodyid[0])).name
        parent_b = signed.model.body(int(b.bodyid[0])).name
        same = (parent_a == parent_b and drive_a.name == drive_b.name
                and np.allclose(a.axis, b.axis, rtol=0, atol=1e-12)
                and np.allclose(a.pos, b.pos, rtol=0, atol=1e-12)
                and np.isclose(live.model.qpos0[live.q_indices[index]],
                               signed.model.qpos0[signed.q_indices[index]], rtol=0, atol=1e-12)
                and np.allclose(drive_a.gear, drive_b.gear, rtol=0, atol=1e-12))
        result.append({
            "joint": name, "live_parent_body": parent_a, "signed_parent_body": parent_b,
            "live_actuator": drive_a.name, "signed_actuator": drive_b.name,
            "live_axis": a.axis.tolist(), "signed_axis": b.axis.tolist(),
            "live_zero_rad": float(live.model.qpos0[live.q_indices[index]]),
            "signed_zero_rad": float(signed.model.qpos0[signed.q_indices[index]]),
            "same_local_coordinates": bool(same),
            "live_bounds_rad": live.bounds[index].tolist(),
            "signed_bounds_rad": signed.bounds[index].tolist(),
        })
    return result


def common_probes(live, signed):
    low = np.maximum(live.bounds[:, 0], signed.bounds[:, 0])
    high = np.minimum(live.bounds[:, 1], signed.bounds[:, 1])
    if np.any(low >= high) or np.any(live.home < low - 1e-10) or np.any(live.home > high + 1e-10):
        raise ValueError("Live Home is outside the common bounded FK domain")
    rng = np.random.default_rng(20261128)
    # Each paired model receives exactly the same values. These are static FK
    # probes; some may fail the collision or J6/J7 rules and none is commanded.
    def bounded(q):
        return np.clip(q, low, high)

    values = [("home", live.home), ("signed_home", bounded(signed.home))]
    for index, name in enumerate(live.names):
        for sign in (-1, 1):
            q = live.home.copy()
            q[index] += sign * 0.04
            q = bounded(q)
            if abs(q[index] - live.home[index]) > 1e-8:
                values.append((f"{name}:{sign:+d}", q))
    for index, delta in enumerate(rng.uniform(-0.15, 0.15, (24, 54))):
        values.append((f"mixed_near_home:{index}", bounded(live.home + delta)))
    for index, q in enumerate(rng.uniform(low, high, (24, 54))):
        values.append((f"mixed_common_range:{index}", q))
    return values


def metric(samples):
    values = np.asarray(samples)
    return {"p95": np.quantile(values, 0.95, axis=0).tolist(),
            "max": values.max(axis=0).tolist()}


def audit(live_path, signed_path, native=None, *, live_sha=LIVE_SHA, signed_sha=SIGNED_SHA):
    identity = {"live": model_identity(live_path, live_sha),
                "signed": model_identity(signed_path, signed_sha)}
    live, signed = RobotModel(live_path), RobotModel(signed_path)
    coordinates = coordinate_table(live, signed)
    probes = common_probes(live, signed)
    side_results = {}
    bridges = {}
    for side, side_code in enumerate(SIDES):
        arm = slice(side * 7, (side + 1) * 7)
        base = f"Base_{side_code.upper()}"
        palm = f"{side_code}_wrist"
        live_home = live.geometry(live.home)
        model_palm_from_virtual_anatomical = pose(np.zeros(3), palm_basis(live_home, side))
        world_from_base = body(live, base)
        world_from_palm = body(live, palm)
        world_from_scoring = pose(live_home.wrists[side], live_home.rotations[side])
        palm_from_scoring = np.linalg.solve(world_from_palm, world_from_scoring)
        palm_from_native = None
        if native is not None:
            palm_from_native = np.linalg.solve(world_from_palm, world_from_base @ native.fk(side, live.home[arm]))
        bridges[side_code] = {
            "world_from_base": world_from_base.tolist(),
            "model_palm_from_scoring_tcp": palm_from_scoring.tolist(),
            "model_palm_from_native_tcp_home_candidate":
                palm_from_native.tolist() if palm_from_native is not None else None,
            "model_palm_from_virtual_anatomical": model_palm_from_virtual_anatomical.tolist(),
            "virtual_anatomical_from_scoring_tcp":
                np.linalg.solve(model_palm_from_virtual_anatomical, palm_from_scoring).tolist(),
            "virtual_anatomical_from_native_tcp_home_candidate":
                np.linalg.solve(model_palm_from_virtual_anatomical, palm_from_native).tolist()
                if palm_from_native is not None else None,
            "model_only_palm_facing_axis_in_virtual_anatomical": [0, 0, 1 if side_code == "l" else -1],
        }
        fields = {name: [] for name in ("base", "wrist_scoring_tcp", "model_palm", "live_palm_scoring_closure",
                                       "native_tcp_held_out", "world_landmark_euclidean",
                                       "palm_local_landmark_euclidean", "virtual_anatomical_flexion_invariance")}
        held_out_native_names = []
        for label, q in probes:
            a, b = live.geometry(q), signed.geometry(q)
            live_palm = body(live, palm)
            live_tcp = pose(a.wrists[side], a.rotations[side])
            fields["base"].append(pose_error(body(live, base), body(signed, base)))
            fields["wrist_scoring_tcp"].append(pose_error(live_tcp, pose(b.wrists[side], b.rotations[side])))
            fields["model_palm"].append(pose_error(live_palm, body(signed, palm)))
            fields["live_palm_scoring_closure"].append(pose_error(live_palm @ palm_from_scoring, live_tcp))
            fields["world_landmark_euclidean"].append(np.linalg.norm(a.landmarks[side] - b.landmarks[side], axis=-1))
            fields["palm_local_landmark_euclidean"].append(np.linalg.norm(a.local[side] - b.local[side], axis=-1))
            fields["virtual_anatomical_flexion_invariance"].append(
                pose_error(model_palm_from_virtual_anatomical, pose(np.zeros(3), palm_basis(a, side)))
            )
            # Native bridge comparison uses only independently displaced nearby
            # arm states and excludes Home, duplicate arm samples and all fingers.
            active_arm_label = label.startswith(tuple(name + ":" for name in live.names[arm]))
            if palm_from_native is not None and active_arm_label:
                native_tcp = world_from_base @ native.fk(side, q[arm])
                fields["native_tcp_held_out"].append(pose_error(live_palm @ palm_from_native, native_tcp))
                held_out_native_names.append(label)
        statistics = {key: metric(samples) for key, samples in fields.items() if samples}
        equivalent = all(np.all(np.asarray(statistics[name]["max"]) <= [POSITION_M, ROTATION_RAD])
                         for name in ("base", "wrist_scoring_tcp", "model_palm", "live_palm_scoring_closure"))
        native_qualified = (len(held_out_native_names) == 14
                            and np.all(np.asarray(statistics["native_tcp_held_out"]["max"])
                                       <= [0.0015, np.deg2rad(0.5)]))
        side_results[side_code] = {
            "metrics": statistics,
            "same_base_palm_scoring_tcp_on_probes": bool(equivalent),
            "native_tcp_home_bridge_passed_14_axis_perturbations": bool(native_qualified),
            "native_tcp_held_out_names": held_out_native_names,
            "same_finger_geometry_on_probes": bool(max(
                np.max(statistics["world_landmark_euclidean"]["max"]),
                np.max(statistics["palm_local_landmark_euclidean"]["max"])
            ) <= POSITION_M),
        }
    return {
        "scope": "EXACT_MODEL_FK_AND_OPTIONAL_NATIVE_TCP_ONLY",
        "model_identity": identity,
        "named_axes": coordinates,
        "named_axes_locally_identical": all(item["same_local_coordinates"] for item in coordinates),
        "home_max_joint_difference_rad": float(np.max(np.abs(live.home - signed.home))),
        "gravity_live_m_s2": live.model.opt.gravity.tolist(),
        "gravity_signed_m_s2": signed.model.opt.gravity.tolist(),
        "probe_count": len(probes), "probe_names": [label for label, _ in probes],
        "pose_fields_units": ["m", "rad"], "finger_fields_units": "m",
        "finger_order": list(FINGERS), "landmark_order": ["root", "third_axis", "fourth_axis", "tip"],
        "virtual_anatomical_frame": {
            "origin": "Named live Wuji wrist body origin",
            "x": "Distal direction from body origin to mean of four non-thumb named MCP anchors",
            "y": "Index-from-pinky MCP direction after orthogonalization to x",
            "z": "Cross product of x and y",
            "normal": "Positive z on the left; negative z on the right",
            "scope": "Robot-model coordinate convention only. Human marker attachment is unmeasured.",
        },
        "sides": side_results, "simulator_bridges": bridges,
        "scope_exclusions": ["physical_calibration", "tracker_attachment", "manus_attachment",
                             "dynamics", "safety", "untested_pose_equivalence"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", required=True, type=Path)
    parser.add_argument("--signed", required=True, type=Path)
    parser.add_argument("--marvin-bridge", type=Path)
    parser.add_argument("--marvin-config", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if bool(args.marvin_bridge) != bool(args.marvin_config):
        parser.error("Provide both the Marvin bridge and configuration")
    native = None
    try:
        if args.marvin_bridge:
            from dexgym.tasks.direct.wholebody_s05.v1.rmb.nominal import Marvin

            native = Marvin(args.marvin_bridge, args.marvin_config)
        receipt = audit(args.live, args.signed, native)
        if native is not None:
            receipt["native_identity"] = {"bridge_sha256": digest(args.marvin_bridge),
                                          "config_sha256": digest(args.marvin_config)}
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    finally:
        if native is not None:
            native.close()


if __name__ == "__main__":
    main()
