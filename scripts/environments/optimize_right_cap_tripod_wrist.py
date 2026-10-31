"""Offline right wrist and index search for separated cap sidewall contacts.

The thumb and middle joints stay at a known two-digit PhysX hold target;
the wrist SE(3) and index joints are optimized together. Official Beta 1 MJCF
collision geometry is a proposal source only. A result cannot enter a Reset
bank until the native USD/PhysX gradual-close, unsupported hold and fresh
Reset replay pass independently.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from analyze_right_cap_reach import _model, _mul


INDEX_JOINTS = (
    "r_index_finger_mcp_flex", "r_index_finger_mcp_abd",
    "r_index_finger_pip", "r_index_finger_dip",
)
OPPOSING_JOINTS = (
    "r_thumb_cmc_flex", "r_thumb_cmc_abd", "r_thumb_mcp",
    "r_middle_finger_mcp_flex", "r_middle_finger_mcp_abd",
    "r_middle_finger_pip",
)
TARGET_AZIMUTH_DEG = {"thumb": -100.0, "index": 20.0, "middle": 95.0}


def max_cyclic_gap_deg(angles) -> float:
    ordered = sorted(float(angle) % 360.0 for angle in angles)
    if len(ordered) < 2:
        raise ValueError("At least two contact sectors are required")
    return max(b - a for a, b in zip(ordered, ordered[1:] + [ordered[0] + 360]))


def world_delta_quaternion(rotvec, base_quaternion_wxyz):
    angle = math.sqrt(sum(float(x)**2 for x in rotvec))
    if angle < 1e-12:
        return tuple(base_quaternion_wxyz)
    twist = (math.cos(angle/2),
             *(math.sin(angle/2) * float(x)/angle for x in rotvec))
    return _mul(twist, base_quaternion_wxyz)


def optimize(proposal: dict, *, seed: int, maxiter: int,
             adjust_opposing_digits: bool = False):
    model = _model()
    data = mujoco.MjData(model)
    base_position = np.array(proposal["wrist_position_w_m"], dtype=float)
    base_quaternion = tuple(proposal["wrist_quaternion_wxyz"])
    joint_ids = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): i
                 for i in range(1, model.njnt)}
    for name, value in proposal["joints_rad"].items():
        data.qpos[model.jnt_qposadr[joint_ids[name]]] = value
    index_ids = [joint_ids[name] for name in INDEX_JOINTS]
    index_addresses = [model.jnt_qposadr[i] for i in index_ids]
    opposing_ids = [joint_ids[name] for name in OPPOSING_JOINTS] if adjust_opposing_digits else []
    opposing_addresses = [model.jnt_qposadr[i] for i in opposing_ids]
    bounds = [(-.025, .025), (-.025, .025), (-.020, .020),
              (-.65, .65), (-.65, .65), (-.65, .65)]
    bounds += [tuple(model.jnt_range[i]) for i in index_ids]
    for name, joint_id in zip(OPPOSING_JOINTS, opposing_ids):
        center = proposal["joints_rad"][name]
        radius = .5 if name.endswith("_pip") else .35
        lower, upper = model.jnt_range[joint_id]
        bounds.append((max(lower, center-radius), min(upper, center+radius)))
    proxy = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
             for name in ("proxy_cap", "proxy_bottle", "proxy_neck")}
    def body(geom):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                int(model.geom_bodyid[geom]))
    colliders = [i for i in range(model.ngeom)
                 if i not in proxy.values() and model.geom_contype[i]
                 and model.geom_conaffinity[i]]
    digits = {digit: [i for i in colliders if body(i).startswith(
        "r_thumb" if digit == "thumb" else f"r_{digit}_finger")]
        for digit in TARGET_AZIMUTH_DEG}
    if any(not geoms for geoms in digits.values()):
        raise RuntimeError("Official right hand collision geoms are incomplete")
    bottle_geoms = (proxy["proxy_bottle"], proxy["proxy_neck"])
    self_pairs = [(a, b) for k, a in enumerate(colliders)
                  for b in colliders[k+1:]
                  if body(a).split("_")[1] != body(b).split("_")[1]
                  and "wrist" not in (body(a).split("_")[1],
                                      body(b).split("_")[1])]

    def state(values):
        data.qpos[:3] = base_position + values[:3]
        data.qpos[3:7] = world_delta_quaternion(values[3:6], base_quaternion)
        data.qpos[index_addresses] = values[6:10]
        if opposing_addresses:
            data.qpos[opposing_addresses] = values[10:]
        mujoco.mj_forward(model, data)
        contact = {}
        for digit, geoms in digits.items():
            options = []
            target = math.radians(TARGET_AZIMUTH_DEG[digit])
            for geom in geoms:
                nearest = np.zeros(6)
                separation = float(mujoco.mj_geomDistance(
                    model, data, geom, proxy["proxy_cap"], .2, nearest))
                point = nearest[3:].copy()
                azimuth = math.atan2(point[1], point[0])
                angle_error = abs(math.atan2(math.sin(azimuth-target),
                                             math.cos(azimuth-target)))
                axial_error = max(0.0, point[2]-.6535, .6375-point[2])
                cost = (abs(separation-.001) + 5*axial_error
                        + .017*angle_error)
                options.append((cost, separation, point, body(geom)))
            _, separation, point, collider = min(options, key=lambda item: item[0])
            contact[digit] = {"distance_m": separation,
                              "cap_point_w_m": point.tolist(),
                              "azimuth_deg": math.degrees(math.atan2(point[1], point[0])),
                              "body": collider}
        cap_depth = min(float(mujoco.mj_geomDistance(
            model, data, i, proxy["proxy_cap"], .2, None))
            for i in colliders)
        bottle_depth = min(float(mujoco.mj_geomDistance(
            model, data, i, j, .2, None))
            for i in colliders for j in bottle_geoms)
        self_depth = min(float(mujoco.mj_geomDistance(
            model, data, a, b, .2, None)) for a, b in self_pairs)
        gap = max_cyclic_gap_deg([item["azimuth_deg"] for item in contact.values()])
        return contact, cap_depth, bottle_depth, self_depth, gap

    evaluations = 0
    def objective(values):
        nonlocal evaluations
        evaluations += 1
        contact, cap_depth, bottle_depth, self_depth, gap = state(values)
        score = 0.0
        for digit, item in contact.items():
            target = math.radians(TARGET_AZIMUTH_DEG[digit])
            angle = math.radians(item["azimuth_deg"])
            delta = abs(math.atan2(math.sin(angle-target), math.cos(angle-target)))
            z = item["cap_point_w_m"][2]
            score += (abs(item["distance_m"]-.001)
                      + 5*max(0.0, z-.6535, .6375-z)
                      + .017*delta)
        score += (8*max(0.0, -.001-cap_depth)
                  + 8*max(0.0, .003-bottle_depth)
                  + 8*max(0.0, .002-self_depth)
                  + .017*math.radians(max(0.0, gap-170.0))
                  + .02*float(np.linalg.norm(values[:3]))
                  + .001*float(np.linalg.norm(values[3:6])))
        return score

    result = differential_evolution(
        objective, bounds, seed=seed, maxiter=maxiter, popsize=7,
        polish=True, workers=1,
    )
    contact, cap_depth, bottle_depth, self_depth, gap = state(result.x)
    output = {
        "index": proposal["index"],
        "wrist_position_w_m": (base_position+result.x[:3]).tolist(),
        "wrist_quaternion_wxyz": list(world_delta_quaternion(
            result.x[3:6], base_quaternion)),
        "joints_rad": {**proposal["joints_rad"],
                       **dict(zip(INDEX_JOINTS, map(float, result.x[6:10]))),
                       **dict(zip(OPPOSING_JOINTS, map(float, result.x[10:])))},
        "tripod_wrist_search": {
            "backend": "official_beta1_MJCF_proxy_geometry_only",
            "seed": seed, "maxiter": maxiter, "evaluations": evaluations,
            "objective_m": float(result.fun),
            "wrist_translation_delta_m": result.x[:3].tolist(),
            "wrist_world_rotation_vector_rad": result.x[3:6].tolist(),
            "adjusted_opposing_digits": adjust_opposing_digits,
            "opposing_joint_names": list(OPPOSING_JOINTS if adjust_opposing_digits else ()),
            "contact": contact,
            "max_cyclic_sector_gap_deg": gap,
            "minimum_cap_clearance_m": cap_depth,
            "minimum_bottle_clearance_m": bottle_depth,
            "minimum_cross_digit_clearance_m": self_depth,
            "geometry_qualified": bool(
                gap <= 170 and cap_depth >= -.001 and bottle_depth >= .003
                and self_depth >= .002 and all(
                    abs(item["distance_m"]-.001) <= .002
                    and .6375 <= item["cap_point_w_m"][2] <= .6535
                    for item in contact.values())),
        },
    }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--proposal_index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--adjust_opposing_digits", action="store_true",
                        help="Also vary bounded thumb and middle-finger targets")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.proposal.read_text())
    matches = [item for item in source["best"] if item["index"] == args.proposal_index]
    if len(matches) != 1:
        raise ValueError("Proposal index absent or ambiguous")
    candidate = optimize(matches[0], seed=args.seed, maxiter=args.maxiter,
                         adjust_opposing_digits=args.adjust_opposing_digits)
    output = {"source": "official_beta1_MJCF_tripod_wrist_geometry_only",
              "official_source_commit": source["official_source_commit"],
              "right_mjcf_sha256": source["right_mjcf_sha256"],
              "cap_radius_m": source["cap_radius_m"],
              "cap_height_m": source["cap_height_m"],
              "bottle_radius_m": source["bottle_radius_m"],
              "best": [candidate]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print("RIGHT_CAP_TRIPOD_WRIST_SEARCH " + json.dumps(
        candidate["tripod_wrist_search"]), flush=True)


if __name__ == "__main__":
    main()
