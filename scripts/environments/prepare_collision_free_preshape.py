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


def prepare(source: dict, clearance_m: float = .005):
    model = model_for(float(source["radius_m"]))
    data = mujoco.MjData(model)
    bottle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                               "diagnostic_cylinder")
    colliders = [i for i in range(model.ngeom) if i != bottle
                 and model.geom_contype[i] and model.geom_conaffinity[i]]
    joint_ids = range(1, model.njnt)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in joint_ids]
    addresses = np.array([model.jnt_qposadr[i] for i in joint_ids])
    lower = np.array([model.jnt_range[i, 0] for i in joint_ids]) + 1e-5
    upper = np.array([model.jnt_range[i, 1] for i in joint_ids]) - 1e-5
    output = {**source, "preshape_clearance_target_m": clearance_m,
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
            return np.array([mujoco.mj_geomDistance(
                model, data, i, bottle, .5, np.zeros(6)) for i in colliders])

        def residual(joints):
            return np.r_[np.maximum(0, clearance_m - distances(joints))/.002,
                         .1*(joints - goal)]

        fitted = least_squares(residual, np.clip(goal, lower, upper),
                               bounds=(lower, upper), max_nfev=160)
        item["pregrasp_joints_rad"] = {
            name: float(value) for name, value in zip(names, fitted.x)}
        item["preshape_minimum_mjcf_clearance_m"] = float(min(distances(fitted.x)))
        item["preshape_max_joint_change_rad"] = float(max(abs(fitted.x-goal)))
        item["preshape_optimizer_evaluations"] = int(fitted.nfev)
        output["best"].append(item)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reach_scan", type=Path, required=True)
    parser.add_argument("--clearance_m", type=float, default=.005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.clearance_m <= 0:
        parser.error("--clearance_m must be positive")
    result = prepare(json.loads(args.reach_scan.read_text()), args.clearance_m)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for item in result["best"]:
        print("MJCF_PRESHAPE " + json.dumps({
            "proposal": item["index"],
            "clearance_m": item["preshape_minimum_mjcf_clearance_m"],
            "max_joint_change_rad": item["preshape_max_joint_change_rad"],
        }))


if __name__ == "__main__":
    main()
