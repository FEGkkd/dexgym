"""Screen randomized anatomical pregrasps with the pinned official MJCF meshes.

This is an offline proposal filter. It uses the official collision meshes and
the engineering bottle/cap cylinders, but the native Isaac USD collision
geometry must also pass `validate_pregrasp_reset_physx.py` before admission.
No one-second hold is run or required.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from dexteroushands_lab.assets.bottle import BottleConfig
from dexteroushands_lab.assets.wuji_hand2_beta import load_hand2_spec
from dexteroushands_lab.scenes.pregrasp_reset import PregraspResetGenerator, validate_pregrasp_state


DIGITS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")


def combined_model(source_root: Path, bottle: BottleConfig):
    """Two official MJCFs sharing one world, with conservative cylinder proxies."""
    body_dir = source_root / "hand2/hand2_beta1/body"
    tree = ET.parse(body_dir / "mjcf/left.xml")
    root = tree.getroot()
    other = ET.parse(body_dir / "mjcf/right.xml").getroot()
    root.find("compiler").set("meshdir", str(body_dir / "meshes/left"))
    for side, xml in (("left", root), ("right", other)):
        mesh_dir = body_dir / "meshes" / side
        for mesh in xml.findall("./asset/mesh"):
            mesh.set("file", str(mesh_dir / mesh.get("file")))
    root.find("./worldbody/body").insert(0, ET.Element("freejoint", name="l_pregrasp_root"))
    right_body = other.find("./worldbody/body")
    right_body.insert(0, ET.Element("freejoint", name="r_pregrasp_root"))
    root.find("asset").extend(list(other.find("asset")))
    root.find("worldbody").append(right_body)
    for tag in ("contact", "actuator"):
        dst, src = root.find(tag), other.find(tag)
        if dst is not None and src is not None:
            dst.extend(list(src))
    world = root.find("worldbody")
    for name, radius, half_height, z in (
        ("proxy_bottle", bottle.bottle_radius, bottle.bottle_height/2, .5),
        ("proxy_neck", bottle.neck_outer_radius + bottle.thread_height,
         bottle.neck_height/2, .5 + bottle.bottle_height/2 + bottle.neck_height/2),
        ("proxy_cap", bottle.cap_outer_radius, bottle.cap_height/2,
         .5 + bottle.cap_initial_z),
    ):
        body = ET.SubElement(world, "body", name=name, pos=f"0 0 {z}")
        ET.SubElement(body, "geom", name=name, type="cylinder",
                      size=f"{radius} {half_height}")
    # MuJoCo needs a path for relative include resolution, though mesh paths
    # are absolute. The model is used only to reject geometric proposals.
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "combined_pregrasp.xml"
        tree.write(path)
        return mujoco.MjModel.from_xml_path(str(path))


class MjcfGeometryScreen:
    def __init__(self, model, bottle: BottleConfig):
        self.model = model
        self.data = mujoco.MjData(model)
        self.bottle = bottle
        self.ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                    for name in ("proxy_bottle", "proxy_neck", "proxy_cap")}
        self.colliders = {side: [] for side in ("l", "r")}
        self.names = {}
        for index in range(model.ngeom):
            if not model.geom_contype[index] or not model.geom_conaffinity[index]:
                continue
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                    int(model.geom_bodyid[index]))
            self.names[index] = body
            for side in self.colliders:
                if body.startswith(side + "_"):
                    self.colliders[side].append(index)
        self.joints = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i):
                       int(model.jnt_qposadr[i]) for i in range(model.njnt)}
        self.self_pairs = {}
        for side, geoms in self.colliders.items():
            self.self_pairs[side] = [(a, b) for ix, a in enumerate(geoms)
                                     for b in geoms[ix+1:]
                                     if self._digit(a) != self._digit(b)
                                     and "wrist" not in (self._digit(a), self._digit(b))]

    def _digit(self, geom):
        name = self.names[geom]
        for digit in DIGITS:
            if name.startswith(("l_" + digit, "r_" + digit)):
                return digit
        return "wrist"

    def _distance(self, geom_a, geom_b):
        points = np.zeros(6)
        gap = float(mujoco.mj_geomDistance(
            self.model, self.data, geom_a, geom_b, .20, points))
        return gap, points

    def check(self, sample) -> dict:
        data, model = self.data, self.model
        if sample.bottle.root_pose != (0, 0, .5, 0, 0, 0, 1):
            raise ValueError("MJCF proxy screen requires canonical upright bottle pose")
        for side, hand in (("l", sample.left), ("r", sample.right)):
            pose = hand.root_pose
            address = self.joints[f"{side}_pregrasp_root"]
            data.qpos[address:address+7] = (*pose[:3], pose[6], *pose[3:6])
            names = sample.metadata[f"{'left' if side == 'l' else 'right'}_joint_order"]
            for name, value in zip(names, hand.joint_position):
                data.qpos[self.joints[name]] = value
        mujoco.mj_forward(model, data)
        by_object = {}
        per_digit = {}
        for side, object_name in (("l", "proxy_bottle"), ("r", "proxy_cap")):
            per_digit[side] = {}
            for digit in DIGITS:
                candidates = []
                sidewall = []
                for geom in self.colliders[side]:
                    if self._digit(geom) != digit:
                        continue
                    gap, near = self._distance(geom, self.ids[object_name])
                    candidates.append((gap, self.names[geom], near))
                    if (side == "r" and
                            abs(near[5] - sample.cap.root_pose[2]) <
                            self.bottle.cap_height/2 - .002 and
                            math.hypot(near[3] - sample.cap.root_pose[0],
                                       near[4] - sample.cap.root_pose[1]) >
                            self.bottle.cap_outer_radius - .001):
                        sidewall.append((gap, self.names[geom], near))
                gap, body, near = min(candidates, key=lambda item: item[0])
                per_digit[side][digit] = {
                    "gap_m": gap, "body": body,
                    "object_point_w_m": near[3:6].tolist(),
                    "azimuth_rad": math.atan2(near[4], near[3]),
                }
                if side == "r" and sidewall:
                    side_gap, side_body, side_near = min(sidewall, key=lambda item: item[0])
                    per_digit[side][digit]["sidewall_gap_m"] = side_gap
                    per_digit[side][digit]["sidewall_body"] = side_body
                    per_digit[side][digit]["sidewall_point_w_m"] = side_near[3:6].tolist()
        for side in self.colliders:
            for obj in self.ids:
                nearest = min((self._distance(g, self.ids[obj])[0], self.names[g])
                              for g in self.colliders[side])
                by_object[f"{side}_{obj}"] = {"minimum_gap_m": nearest[0],
                                               "nearest_body": nearest[1]}
        self_gap = {side: min(self._distance(a, b)[0]
                              for a, b in self.self_pairs[side])
                    for side in self.colliders}
        hand_gap = min(self._distance(a, b)[0] for a in self.colliders["l"]
                       for b in self.colliders["r"])
        left = per_digit["l"]
        long_angles = [left[d]["azimuth_rad"] for d in DIGITS[1:]]
        mean_angle = math.atan2(sum(math.sin(a) for a in long_angles),
                                sum(math.cos(a) for a in long_angles))
        wrap = lambda a: abs(math.atan2(math.sin(a), math.cos(a)))
        left_arc = max(wrap(a - mean_angle) for a in long_angles)
        left_opposition = wrap(left["thumb"]["azimuth_rad"] - mean_angle)
        right = per_digit["r"]
        index_middle_angle = math.atan2(
            math.sin(right["index_finger"]["azimuth_rad"]) +
            math.sin(right["middle_finger"]["azimuth_rad"]),
            math.cos(right["index_finger"]["azimuth_rad"]) +
            math.cos(right["middle_finger"]["azimuth_rad"]),
        )
        right_opposition = wrap(right["thumb"]["azimuth_rad"] - index_middle_angle)
        qualified = (
            all(.0005 <= left[d]["gap_m"] <= .012 for d in DIGITS)
            and left_arc <= math.radians(75)
            and left_opposition >= math.radians(125)
            and all(abs(left[d]["object_point_w_m"][2] - sample.bottle.root_pose[2])
                    <= .075 for d in DIGITS)
            and all(.0005 <= right[d]["gap_m"]
                    and .0005 <= right[d].get("sidewall_gap_m", math.inf) <= .009
                    for d in DIGITS[:3])
            and right_opposition >= math.radians(90)
            and all(right[d]["gap_m"] >= .01 for d in DIGITS[3:])
            and by_object["l_proxy_cap"]["minimum_gap_m"] >= .003
            and by_object["r_proxy_bottle"]["minimum_gap_m"] >= .003
            and by_object["r_proxy_neck"]["minimum_gap_m"] >= .002
            and all(gap >= .001 for gap in self_gap.values())
            and hand_gap >= .003
        )
        return {
            "qualified": qualified,
            "backend": "official_beta1_MJCF_mesh_and_engineering_cylinder_proxy",
            "per_digit": per_digit,
            "per_object": by_object,
            "minimum_cross_digit_self_gap_m": self_gap,
            "minimum_hand_hand_gap_m": hand_gap,
            "left_four_finger_half_arc_rad": left_arc,
            "left_thumb_opposition_rad": left_opposition,
            "right_thumb_tripod_opposition_rad": right_opposition,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--max_accepted", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1 or args.max_accepted < 1:
        raise ValueError("Positive --seeds and --max_accepted are required")
    root = Path(os.environ["WUJI_DESCRIPTION_ROOT"])
    bottle = BottleConfig()
    specs = (load_hand2_spec(root, side="left"), load_hand2_spec(root, side="right"))
    generator = PregraspResetGenerator(bottle, *specs)
    screen = MjcfGeometryScreen(combined_model(root, bottle), bottle)
    accepted, rejected = [], {}
    for seed in range(args.start_seed, args.start_seed + args.seeds):
        try:
            sample = generator.draw(seed=seed)
            validate_pregrasp_state(sample, bottle, *specs)
            result = screen.check(sample)
        except ValueError as exc:
            rejected[str(exc)] = rejected.get(str(exc), 0) + 1
            continue
        if result["qualified"]:
            accepted.append({"sample": asdict(sample), "mjcf_geometry": result})
            if len(accepted) >= args.max_accepted:
                break
        else:
            rejected["geometry"] = rejected.get("geometry", 0) + 1
    payload = {"status": "MJCF_SCREEN_ONLY_NOT_NATIVE_USD_QUALIFIED",
               "attempted": sum(rejected.values()) + len(accepted),
               "accepted": accepted, "rejected": rejected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"attempted": payload["attempted"],
                      "mjcf_screen_pass": len(accepted), "rejected": rejected}))


if __name__ == "__main__":
    main()
