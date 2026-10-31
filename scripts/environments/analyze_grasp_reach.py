"""Offline official-MJCF fingertip/convex-collision reach scan for a cylinder.

This is a *geometric proposal* tool. The formal asset is the official USD;
every candidate still needs native PhysX clearance and dynamic validation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def model_for(radius: float):
    root = Path(os.environ["WUJI_DESCRIPTION_ROOT"])
    body_dir = root / "hand2/hand2_beta1/body"
    xml = ET.parse(body_dir / "mjcf/left.xml")
    tree = xml.getroot()
    tree.find("compiler").set("meshdir", str(body_dir / "meshes/left"))
    tree.find("./worldbody/body").insert(0, ET.Element("freejoint", name="diagnostic_root"))
    bottle = ET.SubElement(tree.find("worldbody"), "body", name="diagnostic_bottle",
                           pos="0 0 0.5")
    ET.SubElement(bottle, "geom", name="diagnostic_cylinder", type="cylinder",
                  size=f"{radius} 0.1")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "reach.xml"
        xml.write(path)
        return mujoco.MjModel.from_xml_path(str(path))


def _mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw)


def scan(radius: float, count: int, seed: int):
    rng = random.Random(seed)
    model = model_for(radius)
    data = mujoco.MjData(model)
    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "diagnostic_cylinder")
    colliders = [i for i in range(model.ngeom) if i != bottle_id
                 and model.geom_contype[i] and model.geom_conaffinity[i]]
    joints = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i):
              model.jnt_qposadr[i] for i in range(1, model.njnt)}
    ranked = []
    for index in range(count):
        x = rng.uniform(-0.19, -0.11)
        y = rng.uniform(-0.025, 0.045)
        z = rng.uniform(0.43, 0.53)
        yaw = rng.uniform(-0.55, 0.55)
        base = (math.sqrt(.5), 0, -math.sqrt(.5), 0)
        q = _mul((math.cos(yaw/2), 0, 0, math.sin(yaw/2)), base)
        data.qpos[:7] = [x, y, z, *q]
        data.qpos[7:] = 0
        # The four finger MCP/PIP/DIP commands and thumb opposition are
        # sampled independently; no shared scalar grip coefficient.
        for digit in ("index_finger", "middle_finger", "ring_finger", "pinky"):
            data.qpos[joints[f"l_{digit}_mcp_flex"]] = rng.uniform(0.1, 0.75)
            data.qpos[joints[f"l_{digit}_mcp_abd"]] = rng.uniform(-0.3, 0.3)
            data.qpos[joints[f"l_{digit}_pip"]] = rng.uniform(0.35, 1.5)
            data.qpos[joints[f"l_{digit}_dip"]] = rng.uniform(0.1, 0.9)
        for name, bounds in (("cmc_flex", (-.25, .95)), ("cmc_abd", (-.25, .69)),
                             ("mcp", (.1, 1.2)), ("ip", (.1, 1.1))):
            data.qpos[joints[f"l_thumb_{name}"]] = rng.uniform(*bounds)
        mujoco.mj_forward(model, data)
        distances = {}
        for geom_id in colliders:
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                    int(model.geom_bodyid[geom_id]))
            distances[body] = float(mujoco.mj_geomDistance(
                model, data, geom_id, bottle_id, 0.5, None))
        tips = {}
        for i in range(model.nsite):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            tips[name] = data.site_xpos[i].tolist()
        depth = min(distances.values())
        contact_digits = [name for name, dist in distances.items()
                          if -0.002 <= dist <= 0.003 and
                          ("thumb" in name or "index_finger" in name or
                           "middle_finger" in name or "ring_finger" in name)]
        thumb = any("thumb" in name for name in contact_digits)
        others = len({name.split("_")[1] for name in contact_digits if "thumb" not in name})
        score = (int(thumb and others >= 2), -max(0, -depth - .002),
                 len(contact_digits), -abs(depth))
        result = {"index": index, "wrist_position_w_m": [x, y, z],
                  "wrist_quaternion_wxyz": q,
                  "joints_rad": {name: float(data.qpos[adr]) for name, adr in joints.items()},
                  "minimum_mjcf_convex_distance_m": depth,
                  "body_distances_m": distances, "tip_sites_w_m": tips,
                  "contact_digits": contact_digits, "score": score}
        ranked.append(result)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return {"radius_m": radius, "samples": count, "seed": seed,
            "source": "official_beta1_mjcf_geometric_proposal_only",
            "best": ranked[:20],
            "nominal_clear_candidates": sum(
                int(x["score"][0] and x["minimum_mjcf_convex_distance_m"] >= -.002)
                for x in ranked),
            "contact_digit_candidates_including_deep_penetration": sum(
                int(any("thumb" in name for name in x["contact_digits"]) and
                    len({name.split("_")[1] for name in x["contact_digits"]
                         if "thumb" not in name}) >= 2)
                for x in ranked)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radii", type=float, nargs="+", default=[.035, .03, .025])
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [scan(radius, args.samples, args.seed) for radius in args.radii]
    for result in results:
        best = result["best"][0]
        print("REACH_SCAN " + json.dumps({"radius_m": result["radius_m"],
              "clear_candidates": result["nominal_clear_candidates"],
              "best_min_distance_m": best["minimum_mjcf_convex_distance_m"],
              "best_contact_digits": best["contact_digits"]}))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
