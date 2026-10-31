"""Inspect an actual captured Beta 1 left-hand pose against the 70 mm bottle.

This offline MJCF geometry diagnostic uses the measured 20 joint positions,
not the position-drive targets. USD/PhysX hold and Reset replay remain separate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from analyze_grasp_reach import model_for


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--output", type=Path)
args = parser.parse_args()


def _local_point(point, bottle_pose):
    q = bottle_pose[3:7]
    x, y, z, w = q
    inverse = (-x, -y, -z, w)
    delta = tuple(float(a)-float(b) for a, b in zip(point, bottle_pose[:3]))
    ax, ay, az = inverse[:3]
    cross = (ay*delta[2]-az*delta[1], az*delta[0]-ax*delta[2],
             ax*delta[1]-ay*delta[0])
    cross2 = (ay*cross[2]-az*cross[1], az*cross[0]-ax*cross[2],
              ax*cross[1]-ay*cross[0])
    return tuple(delta[i] + 2*(w*cross[i]+cross2[i]) for i in range(3))


def inspect(sample):
    if sample.get("validated"):
        raise ValueError("This diagnostic expects a captured, unvalidated candidate")
    radius = sample["metadata"]["bottle_parameters"]["bottle_radius"]
    if not math.isclose(radius, .035, abs_tol=1e-12):
        raise ValueError("The left.png diagnostic is pinned to a 70 mm bottle")
    model = model_for(radius)
    data = mujoco.MjData(model)
    hand_pose = sample["left"]["root_pose"]
    data.qpos[:7] = [*hand_pose[:3], hand_pose[6], *hand_pose[3:6]]
    names = sample["metadata"]["left_joint_order"]
    if len(names) != 20 or len(sample["left"]["joint_position"]) != 20:
        raise ValueError("A complete canonical 20 joint state is required")
    joint_position = dict(zip(names, sample["left"]["joint_position"]))
    for index in range(1, model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        data.qpos[model.jnt_qposadr[index]] = joint_position[name]
    bottle_pose = sample["bottle"]["root_pose"]
    bottle_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "diagnostic_bottle")
    model.body_pos[bottle_body] = bottle_pose[:3]
    model.body_quat[bottle_body] = [bottle_pose[6], *bottle_pose[3:6]]
    mujoco.mj_forward(model, data)
    cylinder = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "diagnostic_cylinder")
    bodies = {}
    all_distances = []
    for geom in range(model.ngeom):
        if geom == cylinder or not (model.geom_contype[geom] and
                                    model.geom_conaffinity[geom]):
            continue
        body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom]))
        if not body.startswith("l_"):
            continue
        points = np.zeros(6)
        gap = float(mujoco.mj_geomDistance(
            model, data, geom, cylinder, .5, points))
        local = _local_point(points[3:6], bottle_pose)
        record = {"body": body, "gap_m": gap,
                  "bottle_local_contact_m": list(local),
                  "contact_angle_rad": math.atan2(local[1], local[0]),
                  "contact_axial_m": local[2]}
        all_distances.append(gap)
        prior = bodies.get(body)
        if prior is None or gap < prior["gap_m"]:
            bodies[body] = record
    digits = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
    nearest = {}
    for digit in digits:
        matches = [item for body, item in bodies.items() if f"l_{digit}_" in body]
        if not matches:
            raise RuntimeError(f"No official collision geometry for {digit}")
        nearest[digit] = min(matches, key=lambda item: item["gap_m"])
    sites = {}
    for digit in digits:
        name = f"l_{digit}_tip"
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        local = _local_point(data.site_xpos[site], bottle_pose)
        sites[digit] = {"site": name, "bottle_local_position_m": list(local),
                        "radial_distance_m": math.hypot(local[0], local[1]),
                        "axial_m": local[2]}
    four = digits[1:]
    angles = [nearest[digit]["contact_angle_rad"] for digit in four]
    mean = (sum(math.cos(a) for a in angles), sum(math.sin(a) for a in angles))
    mean_angle = math.atan2(mean[1], mean[0])
    angle_gap = lambda a, b: abs(math.atan2(math.sin(a-b), math.cos(a-b)))
    spread = max(angle_gap(angle, mean_angle) for angle in angles)
    opposition = angle_gap(nearest["thumb"]["contact_angle_rad"], mean_angle)
    ordered = all(sites[a]["axial_m"] > sites[b]["axial_m"] + .01
                  for a, b in zip(four, four[1:]))
    sidewall = all(-.025 <= item["contact_axial_m"] <= .065
                   for item in nearest.values())
    distal = all(nearest[digit]["body"].endswith("_distal") for digit in four)
    near_surface = all(-.002 <= item["gap_m"] <= .003
                       for item in nearest.values())
    palm_gap = bodies["l_wrist"]["gap_m"]
    geometry_qualified = (ordered and sidewall and distal and near_surface
                          and min(all_distances) >= -.002
                          and -.002 <= palm_gap <= .012
                          and spread <= math.radians(35)
                          and opposition >= math.radians(130))
    return {"status": "official_MJCF_captured_state_geometry_only",
            "preset_id": sample["metadata"].get("preset_id"),
            "bottle_radius_m": radius, "geometry_qualified": geometry_qualified,
            "four_finger_axial_order": ordered,
            "four_finger_distal_colliders_nearest": distal,
            "all_five_on_body_sidewall_band": sidewall,
            "all_five_near_surface": near_surface,
            "four_finger_half_arc_rad": spread,
            "thumb_opposition_rad": opposition,
            "palm_gap_m": palm_gap,
            "minimum_hand_bottle_distance_m": min(all_distances),
            "nearest_collision_surfaces": nearest,
            "fingertip_sites": sites,
            "physx_grasp_validated": False}


sample = json.loads(args.candidate.read_text(encoding="utf-8"))
result = inspect(sample)
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print("CAPTURED_POWER_GEOMETRY " + json.dumps({
    "geometry_qualified": result["geometry_qualified"],
    "four_finger_half_arc_deg": math.degrees(result["four_finger_half_arc_rad"]),
    "thumb_opposition_deg": math.degrees(result["thumb_opposition_rad"]),
    "palm_gap_mm": 1000*result["palm_gap_m"],
    "physx_grasp_validated": False,
}), flush=True)
