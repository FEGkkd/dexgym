"""Collision-aware official-MJCF proposal search for left thumb opposition.

Optimizes wrist 6DoF and independent thumb/index/middle/ring flexion with
signed convex-mesh distances. This only proposes candidates; USD/PhysX is
the acceptance backend. No RL, MuJoCo dynamics, or pose teleport is used.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from analyze_grasp_reach import model_for


JOINTS = [
    "l_thumb_cmc_flex", "l_thumb_cmc_abd", "l_thumb_mcp", "l_thumb_ip",
    "l_index_finger_mcp_flex", "l_index_finger_mcp_abd",
    "l_index_finger_pip", "l_index_finger_dip",
    "l_middle_finger_mcp_flex", "l_middle_finger_mcp_abd",
    "l_middle_finger_pip", "l_middle_finger_dip",
    "l_ring_finger_mcp_flex", "l_ring_finger_mcp_abd",
    "l_ring_finger_pip", "l_ring_finger_dip",
]


def optimize(radius: float, starts: int, seed: int):
    model = model_for(radius)
    data = mujoco.MjData(model)
    bottle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "diagnostic_cylinder")
    geom_by_digit = {}
    all_geom = []
    for i in range(model.ngeom):
        if i == bottle or not model.geom_contype[i] or not model.geom_conaffinity[i]:
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i]))
        all_geom.append((body, i))
        digit = next((name for name in ("thumb", "index_finger", "middle_finger",
                                        "ring_finger", "pinky") if name in body), "wrist")
        geom_by_digit.setdefault(digit, []).append(i)
    joint_addr = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i):
                  int(model.jnt_qposadr[i]) for i in range(1, model.njnt)}
    lower = np.array([-.20, -.06, .42, -.5, -.5, -.7,
                      -.8, -1.4, -.8, -.8,
                      -.2, -.5, -.2, -.2,
                      -.2, -.5, -.2, -.2,
                      -.2, -.5, -.2, -.2], dtype=float)
    upper = np.array([-.09, .07, .54, .5, .5, .7,
                      1.2, .69, 1.5, 1.5,
                      1.5, .5, 2.0, 1.5,
                      1.5, .5, 2.0, 1.5,
                      1.5, .5, 2.0, 1.5], dtype=float)
    base = Rotation.from_quat((0, -math.sqrt(.5), 0, math.sqrt(.5)))

    def evaluate(values, details=False):
        q = (Rotation.from_rotvec(values[3:6]) * base).as_quat()
        data.qpos[:7] = [*values[:3], q[3], q[0], q[1], q[2]]
        data.qpos[7:] = 0
        for name, value in zip(JOINTS, values[6:]):
            data.qpos[joint_addr[name]] = value
        mujoco.mj_forward(model, data)
        distances = {}
        contact_normals = {}
        all_dist = []
        for digit, ids in geom_by_digit.items():
            best = (float("inf"), None)
            for geom_id in ids:
                points = np.zeros(6)
                dist = float(mujoco.mj_geomDistance(
                    model, data, geom_id, bottle, .5, points))
                all_dist.append(dist)
                if dist < best[0]:
                    best = (dist, points)
            distances[digit] = best[0]
            if best[1] is not None:
                radial = best[1][3:5]
                norm = np.linalg.norm(radial)
                contact_normals[digit] = radial / norm if norm > 1e-8 else np.zeros(2)
        tip_sites = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i):
                     data.site_xpos[i].copy() for i in range(model.nsite)}
        if details:
            sites = {name: value.tolist() for name, value in tip_sites.items()}
            return distances, float(min(all_dist)), contact_normals, sites
        residual = []
        # Any deep intersection is disallowed, including the palm and pinky.
        residual.extend(max(0.0, -d-.001)/.002 for d in all_dist)
        for digit in ("thumb", "index_finger", "middle_finger", "ring_finger"):
            residual.append((distances[digit]-.001)/.005)
            site = tip_sites[f"l_{digit}_tip"]
            radial_distance = float(np.linalg.norm(site[:2]))
            residual.append((radial_distance-radius)/.004)
            residual.append(max(0.0, abs(float(site[2]-.5))-.075)/.01)
        # Require a thumb normal that opposes the long fingers on the cylinder.
        thumb_site = tip_sites["l_thumb_tip"][:2]
        thumb = thumb_site/max(1e-8, float(np.linalg.norm(thumb_site)))
        for digit in ("index_finger", "middle_finger", "ring_finger"):
            site = tip_sites[f"l_{digit}_tip"][:2]
            radial = site/max(1e-8, float(np.linalg.norm(site)))
            residual.append(max(0.0, float(np.dot(thumb, radial))+.25)*2)
        residual.extend((values[3:6]/.8).tolist())
        residual.append((values[0]+.15)/.05)
        return np.asarray(residual)

    rng = random.Random(seed)
    results = []
    for attempt in range(starts):
        initial = np.array([rng.uniform(-.18, -.12), rng.uniform(-.01, .04),
                            rng.uniform(.45, .51), *[rng.uniform(-.25,.25) for _ in range(3)],
                            rng.uniform(-.2,.7), rng.uniform(-.4,.65),
                            rng.uniform(.2,1.0), rng.uniform(.2,.9),
                            rng.uniform(0,.8),rng.uniform(-.2,.2),rng.uniform(.3,1.4),rng.uniform(.1,.8),
                            rng.uniform(0,.8),rng.uniform(-.2,.2),rng.uniform(.3,1.4),rng.uniform(.1,.8),
                            rng.uniform(0,.8),rng.uniform(-.2,.2),rng.uniform(.3,1.4),rng.uniform(.1,.8)])
        fit = least_squares(evaluate, initial, bounds=(lower, upper),
                            max_nfev=180, xtol=1e-5, ftol=1e-5, gtol=1e-5)
        distances, minimum, normals, sites = evaluate(fit.x, details=True)
        q = (Rotation.from_rotvec(fit.x[3:6]) * base).as_quat()
        joint_values = {name: float(value) for name, value in zip(JOINTS, fit.x[6:])}
        for i in range(1, model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_values.setdefault(name, 0.0)
        results.append({"index": attempt,
                        "objective_norm": float(np.linalg.norm(evaluate(fit.x))),
                        "wrist_position_w_m": fit.x[:3].tolist(),
                        "wrist_quaternion_wxyz": [q[3], q[0], q[1], q[2]],
                        "joints_rad": joint_values,
                        "minimum_mjcf_convex_distance_m": minimum,
                        "digit_distances_m": distances,
                        "digit_radial_normals_xy": {k:v.tolist() for k,v in normals.items()},
                        "tip_sites_w_m": sites,
                        "optimizer_status": int(fit.status),
                        "optimizer_nfev": int(fit.nfev)})
    results.sort(key=lambda x: x["objective_norm"])
    return {"radius_m": radius, "source": "official_beta1_mjcf_collision_optimization_proposals_only",
            "seed": seed, "starts": starts, "best": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, default=.035)
    parser.add_argument("--starts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = optimize(args.radius, args.starts, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for item in result["best"]:
        print("OPTIMIZED_REACH " + json.dumps({"index": item["index"],
              "objective_norm": item["objective_norm"],
              "minimum_distance_m": item["minimum_mjcf_convex_distance_m"],
              "digit_distances_m": item["digit_distances_m"]}))


if __name__ == "__main__":
    main()
