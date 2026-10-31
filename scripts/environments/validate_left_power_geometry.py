"""Check a Beta 1 cylindrical-grasp proposal against official MJCF geometry.

This is an offline proposal check. The official Isaac USD, PhysX contact,
gravity-on hold and independent Reset replay decide physical acceptance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from analyze_grasp_reach import model_for


DIGITS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")


def inspect(proposal: dict, *, radius_m: float = .035, steps: int = 21) -> dict:
    if radius_m <= 0:
        raise ValueError("Bottle radius must be positive")
    if steps < 2:
        raise ValueError("The closing path needs at least two samples")
    model = model_for(radius_m)
    data = mujoco.MjData(model)
    bottle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                               "diagnostic_cylinder")
    geoms = [i for i in range(model.ngeom) if i != bottle and
             model.geom_contype[i] and model.geom_conaffinity[i]]
    body = {i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[i])) for i in geoms}
    groups = {name: [i for i in geoms if name in body[i]] for name in DIGITS}
    cross = [(a, b) for a in geoms for b in geoms if a < b and any(
        a in groups[left] and b in groups[right] or
        b in groups[left] and a in groups[right]
        for ix, left in enumerate(DIGITS) for right in DIGITS[ix+1:])]
    joints = [(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i),
               model.jnt_qposadr[i], model.jnt_range[i])
              for i in range(1, model.njnt)]
    start = np.array([proposal["pregrasp_joints_rad"][name]
                      for name, _, _ in joints])
    goal = np.array([proposal["joints_rad"][name] for name, _, _ in joints])
    for values in (start, goal):
        for value, (name, _, bounds) in zip(values, joints):
            if not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"Joint target outside official range: {name}")
    q = proposal["wrist_quaternion_wxyz"]
    data.qpos[:7] = [*proposal["wrist_position_w_m"], *q]
    path = []
    final_contacts = {}
    for fraction in np.linspace(0., 1., steps):
        data.qpos[[address for _, address, _ in joints]] = (
            (1-fraction)*start + fraction*goal)
        mujoco.mj_forward(model, data)
        cross_gap = min(float(mujoco.mj_geomDistance(
            model, data, a, b, .5, None)) for a, b in cross)
        bottle_gap = min(float(mujoco.mj_geomDistance(
            model, data, geom, bottle, .5, None)) for geom in geoms)
        path.append({"fraction": float(fraction),
                     "minimum_cross_finger_gap_m": cross_gap,
                     "minimum_bottle_gap_m": bottle_gap})
        if fraction == 1.:
            for digit, ids in groups.items():
                nearest = None
                for geom in ids:
                    points = np.zeros(6)
                    gap = float(mujoco.mj_geomDistance(
                        model, data, geom, bottle, .5, points))
                    if nearest is None or gap < nearest["distance_m"]:
                        nearest = {"distance_m": gap, "body": body[geom],
                                   "angle_rad": float(math.atan2(points[4], points[3])),
                                   "axial_m": float(points[5] - .5)}
                final_contacts[digit] = nearest
    angles = np.array([final_contacts[name]["angle_rad"] for name in DIGITS])
    vectors = np.column_stack((np.cos(angles), np.sin(angles)))
    four_mean = np.mean(vectors[1:], axis=0)
    four_mean /= np.linalg.norm(four_mean)
    spread = max(math.acos(float(np.clip(np.dot(v, four_mean), -1., 1.)))
                 for v in vectors[1:])
    opposition = math.acos(float(np.clip(np.dot(vectors[0], four_mean), -1., 1.)))
    geometry_qualified = (
        path[0]["minimum_bottle_gap_m"] >= .002 and
        min(row["minimum_cross_finger_gap_m"] for row in path) >= .001 and
        spread <= math.radians(70) and opposition >= math.radians(130) and
        all(-.025 <= item["axial_m"] <= .065 and
            -.003 <= item["distance_m"] <= .003
            for item in final_contacts.values()))
    return {"status": "official_MJCF_geometry_only_not_PhysX_acceptance",
            "proposal_index": proposal["index"],
            "geometry_qualified": geometry_qualified,
            "bottle_radius_m": radius_m,
            "four_finger_half_arc_rad": spread,
            "thumb_opposition_rad": opposition,
            "final_digit_proximity": final_contacts,
            "closing_path": path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = json.loads(args.proposal.read_text())
    result = inspect(source["best"][0], radius_m=float(source["radius_m"]))
    print(json.dumps({key: value for key, value in result.items()
                      if key != "closing_path"}, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
