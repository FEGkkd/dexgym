"""Prepare collision-clear MJCF finger preshapes for a pinned reach proposal.

This is an offline geometric proposal stage only. Official USD + PhysX must
still verify clearance and contact; several MJCF-positive proposals failed
that check in Phase 0C.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from analyze_grasp_reach import model_for


def prepare(source: dict, clearance_m: float = .005,
            self_clearance_m: float = .003):
    model = model_for(float(source["radius_m"]))
    data = mujoco.MjData(model)
    bottle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                               "diagnostic_cylinder")
    colliders = [i for i in range(model.ngeom) if i != bottle
                 and model.geom_contype[i] and model.geom_conaffinity[i]]
    def digit(geom_id):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[geom_id]))
        return body.split("_")[1] if "finger" in body or "thumb" in body else "wrist"
    # Cross-digit pairs are the missing constraint in the original preshape
    # fitter. The official MJCF has one convex collider per link; sibling
    # fingers can overlap even when every hand-bottle distance is positive.
    self_pairs = [(a, b) for i, a in enumerate(colliders)
                  for b in colliders[i+1:]
                  if digit(a) != digit(b) and "wrist" not in (digit(a), digit(b))]
    joint_ids = range(1, model.njnt)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in joint_ids]
    addresses = np.array([model.jnt_qposadr[i] for i in joint_ids])
    lower = np.array([model.jnt_range[i, 0] for i in joint_ids]) + 1e-5
    upper = np.array([model.jnt_range[i, 1] for i in joint_ids]) - 1e-5
    output = {**source, "preshape_clearance_target_m": clearance_m,
              "preshape_self_clearance_target_m": self_clearance_m,
              "preshape_backend": "official_beta1_MJCF_geometry_only"}
    output["best"] = []
    for proposal in source["best"]:
        item = dict(proposal)
        quaternion = item["wrist_quaternion_wxyz"]
        data.qpos[:7] = [*item["wrist_position_w_m"], *quaternion]
        goal = np.array([item["joints_rad"][name] for name in names])

        def distances(joints):
            data.qpos[addresses] = joints
            mujoco.mj_forward(model, data)
            bottle_distances = np.array([mujoco.mj_geomDistance(
                model, data, i, bottle, .5, np.zeros(6)) for i in colliders])
            self_distances = np.array([mujoco.mj_geomDistance(
                model, data, i, j, .5, np.zeros(6)) for i, j in self_pairs])
            return bottle_distances, self_distances

        def residual(joints):
            bottle_distances, self_distances = distances(joints)
            return np.r_[np.maximum(0, clearance_m - bottle_distances)/.002,
                         np.maximum(0, self_clearance_m - self_distances)/.002,
                         .1*(joints - goal)]

        fitted = least_squares(residual, np.clip(goal, lower, upper),
                               bounds=(lower, upper), max_nfev=160)
        bottle_distances, self_distances = distances(fitted.x)
        item["pregrasp_joints_rad"] = {
            name: float(value) for name, value in zip(names, fitted.x)}
        item["pregrasp_tip_sites_w_m"] = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i):
            data.site_xpos[i].tolist() for i in range(model.nsite)}
        item["preshape_minimum_mjcf_clearance_m"] = float(min(bottle_distances))
        item["preshape_minimum_mjcf_self_clearance_m"] = float(min(self_distances))
        item["preshape_max_joint_change_rad"] = float(max(abs(fitted.x-goal)))
        item["preshape_optimizer_evaluations"] = int(fitted.nfev)
        output["best"].append(item)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reach_scan", type=Path, required=True)
    parser.add_argument("--clearance_m", type=float, default=.005)
    parser.add_argument("--self_clearance_m", type=float, default=.003)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.clearance_m <= 0:
        parser.error("--clearance_m must be positive")
    if args.self_clearance_m <= 0:
        parser.error("--self_clearance_m must be positive")
    result = prepare(json.loads(args.reach_scan.read_text()),
                     args.clearance_m, args.self_clearance_m)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for item in result["best"]:
        print("MJCF_PRESHAPE " + json.dumps({
            "proposal": item["index"],
            "clearance_m": item["preshape_minimum_mjcf_clearance_m"],
            "self_clearance_m": item["preshape_minimum_mjcf_self_clearance_m"],
            "max_joint_change_rad": item["preshape_max_joint_change_rad"],
        }))


if __name__ == "__main__":
    main()
