"""Offline right Beta 1 cap reach proposals; native USD/PhysX is the acceptance test.

The proxy cylinders approximate the engineering bottle, neck and cap. The
cap is 17 mm in radius and 25 mm high; the bottle remains the 35 mm target.
No result from this script is a physical grasp validation.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def _mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw)


def _model():
    root = Path(os.environ["WUJI_DESCRIPTION_ROOT"]) / "hand2/hand2_beta1/body"
    tree = ET.parse(root / "mjcf/right.xml")
    xml = tree.getroot()
    xml.find("compiler").set("meshdir", str(root / "meshes/right"))
    xml.find("./worldbody/body").insert(0, ET.Element("freejoint", name="reach_root"))
    world = xml.find("worldbody")
    for name, radius, half_height, z in (("proxy_bottle", .035, .1, .5),
                                          ("proxy_neck", .013, .02, .62),
                                          ("proxy_cap", .017, .0125, .6455)):
        body = ET.SubElement(world, "body", name=name, pos=f"0 0 {z}")
        ET.SubElement(body, "geom", name=name, type="cylinder",
                      size=f"{radius} {half_height}")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "right_reach.xml"
        tree.write(path)
        return mujoco.MjModel.from_xml_path(str(path))


def scan(samples: int, seed: int):
    model = _model()
    data = mujoco.MjData(model)
    rng = random.Random(seed)
    ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
           for name in ("proxy_bottle", "proxy_neck", "proxy_cap")}
    colliders = [i for i in range(model.ngeom) if i not in ids.values()
                 and model.geom_contype[i] and model.geom_conaffinity[i]]
    def body_name(geom):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                int(model.geom_bodyid[geom]))
    self_pairs = [(a, b) for k, a in enumerate(colliders)
                  for b in colliders[k+1:]
                  if body_name(a).split("_")[1] != body_name(b).split("_")[1]
                  and "wrist" not in (body_name(a).split("_")[1],
                                      body_name(b).split("_")[1])]
    joints = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i):
              (model.jnt_qposadr[i], model.jnt_range[i])
              for i in range(1, model.njnt)}
    best = []
    for index in range(samples):
        x = rng.uniform(.06, .18)
        y = rng.uniform(-.07, .07)
        z = rng.uniform(.60, .72)
        roll, pitch, yaw = (rng.uniform(-1.2, 1.2) for _ in range(3))
        q = (math.sqrt(.5), 0, math.sqrt(.5), 0)
        for axis, angle in (((1, 0, 0), roll), ((0, 1, 0), pitch),
                            ((0, 0, 1), yaw)):
            twist = (math.cos(angle/2), *(a*math.sin(angle/2) for a in axis))
            q = _mul(twist, q)
        if abs(sum(value*value for value in q) - 1.0) > 1e-9:
            raise RuntimeError("Right reach scan generated a non-unit quaternion")
        data.qpos[:7] = [x, y, z, *q]
        for name, (address, bounds) in joints.items():
            lower, upper = map(float, bounds)
            if name.endswith("_abd"):
                value = rng.uniform(max(lower, -.3), min(upper, .65))
            else:
                value = rng.uniform(max(lower, 0), min(upper, 1.4))
            data.qpos[address] = value
        mujoco.mj_forward(model, data)
        distance = {"cap": {}, "bottle": {}}
        sidewall = {}
        for collider in colliders:
            body = body_name(collider)
            for key, proxy in (("cap", ids["proxy_cap"]),
                               ("bottle", ids["proxy_bottle"]),
                               ("bottle", ids["proxy_neck"])):
                nearest = np.zeros(6)
                value = float(mujoco.mj_geomDistance(
                    model, data, collider, proxy, .2, nearest))
                distance[key][body] = min(value, distance[key].get(body, math.inf))
                if (key == "cap" and .6375 <= nearest[5] <= .6535
                        and math.hypot(nearest[3], nearest[4]) > .016):
                    if body not in sidewall or value < sidewall[body][0]:
                        sidewall[body] = (value, nearest[3:6].tolist())
        digits = {}
        for body, value in distance["cap"].items():
            digit = body.split("_")[1]
            digits[digit] = min(digits.get(digit, math.inf), value)
        side_digits = {}
        for body, item in sidewall.items():
            digit = body.split("_")[1]
            if digit not in side_digits or item[0] < side_digits[digit][0]:
                side_digits[digit] = item
        side_thumb = side_digits.get("thumb", (0.1, None))
        side_others = [(name, pair) for name, pair in side_digits.items()
                       if name not in ("thumb", "wrist")]
        side_other_name, side_other = min(
            side_others, key=lambda pair: pair[1][0],
            default=(None, (0.1, None)))
        cap_depth = min(distance["cap"].values())
        bottle_depth = min(distance["bottle"].values())
        # Rank contact candidates, but never label them physically accepted.
        score = (max(0, side_thumb[0]-.003) + max(0, side_other[0]-.003)
                 + 8*max(0, -.001-cap_depth)
                 + 8*max(0, .003-bottle_depth))
        separation = None
        if side_thumb[1] is not None and side_other[1] is not None:
            a = math.atan2(side_thumb[1][1], side_thumb[1][0])
            b = math.atan2(side_other[1][1], side_other[1][0])
            separation = abs(math.atan2(math.sin(a-b), math.cos(a-b)))
            score += .01*max(0, math.pi/2 - separation)
        if score < .03:
            self_depth = min(float(mujoco.mj_geomDistance(
                model, data, a, b, .2, None)) for a, b in self_pairs)
            score += 8*max(0, .002-self_depth)
        else:
            self_depth = None
        if len(best) < 30 or score < best[-1][0]:
            item = {"index": index, "score": score,
                    "wrist_position_w_m": [x, y, z],
                    "wrist_quaternion_wxyz": q,
                    "joints_rad": {name: float(data.qpos[address])
                                   for name, (address, _) in joints.items()},
                    "digit_cap_distances_m": digits,
                    "minimum_cap_distance_m": cap_depth,
                    "minimum_bottle_distance_m": bottle_depth,
                    "minimum_self_distance_m": self_depth,
                    "digit_sidewall_distances_m": {
                        name: pair[0] for name, pair in side_digits.items()},
                    "digit_sidewall_points_w_m": {
                        name: pair[1] for name, pair in side_digits.items()},
                    "sidewall_opposing_digit": side_other_name,
                    "sidewall_angular_separation_rad": separation,
                    "body_cap_distances_m": distance["cap"],
                    "body_bottle_distances_m": distance["bottle"]}
            best.append((score, item))
            best.sort(key=lambda pair: pair[0])
            best = best[:30]
    return {"source": "official_beta1_right_MJCF_proxy_geometry_only",
            "official_source_commit": "c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95",
            "right_mjcf_sha256": sha256((Path(os.environ["WUJI_DESCRIPTION_ROOT"])
                / "hand2/hand2_beta1/body/mjcf/right.xml").read_bytes()).hexdigest(),
            "seed": seed, "samples": samples,
            "cap_radius_m": .017, "cap_height_m": .025,
            "bottle_radius_m": .035, "best": [item for _, item in best]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = scan(args.samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for item in result["best"][:5]:
        print("RIGHT_CAP_REACH " + json.dumps({key: item[key] for key in
              ("index", "score", "wrist_position_w_m", "digit_cap_distances_m",
               "minimum_cap_distance_m", "minimum_bottle_distance_m")}))


if __name__ == "__main__":
    main()
