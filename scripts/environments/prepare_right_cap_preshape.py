"""Fit a right-hand cap preshape clear of cap, bottle and other fingers.

This uses official Beta 1 MJCF geometry as a proposal only. The official USD
and native PhysX must independently check the result before it is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from analyze_right_cap_reach import _model


def fit(source, *, cap_clearance=.004, bottle_clearance=.004,
        self_clearance=.002):
    model = _model()
    data = mujoco.MjData(model)
    proxy = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
             for name in ("proxy_bottle", "proxy_neck", "proxy_cap")}
    colliders = [i for i in range(model.ngeom) if i not in proxy.values()
                 and model.geom_contype[i] and model.geom_conaffinity[i]]
    def body(geom):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                int(model.geom_bodyid[geom]))
    self_pairs = [(a, b) for k, a in enumerate(colliders)
                  for b in colliders[k+1:]
                  if body(a).split("_")[1] != body(b).split("_")[1]
                  and "wrist" not in (body(a).split("_")[1],
                                      body(b).split("_")[1])]
    joint_ids = list(range(1, model.njnt))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in joint_ids]
    addresses = np.array([model.jnt_qposadr[i] for i in joint_ids])
    lower = np.array([model.jnt_range[i, 0] for i in joint_ids]) + 1e-5
    upper = np.array([model.jnt_range[i, 1] for i in joint_ids]) - 1e-5
    output = {**source, "preshape_backend": "official_beta1_right_MJCF_geometry_only",
              "clearance_targets_m": {"cap": cap_clearance,
                                      "bottle": bottle_clearance,
                                      "cross_digit": self_clearance},
              "best": []}
    for item in source["best"]:
        item = dict(item)
        data.qpos[:7] = [*item["wrist_position_w_m"],
                         *item["wrist_quaternion_wxyz"]]
        goal = np.array([item["joints_rad"][name] for name in names])
        def measures(joints):
            data.qpos[addresses] = joints
            mujoco.mj_forward(model, data)
            cap = np.array([mujoco.mj_geomDistance(
                model, data, geom, proxy["proxy_cap"], .3, None)
                for geom in colliders])
            bottle = np.array([min(mujoco.mj_geomDistance(
                model, data, geom, proxy[name], .3, None)
                for name in ("proxy_bottle", "proxy_neck"))
                for geom in colliders])
            own = np.array([mujoco.mj_geomDistance(model, data, a, b, .3, None)
                            for a, b in self_pairs])
            return cap, bottle, own
        def residual(joints):
            cap, bottle, own = measures(joints)
            return np.r_[np.maximum(0, cap_clearance-cap)/.002,
                         np.maximum(0, bottle_clearance-bottle)/.002,
                         np.maximum(0, self_clearance-own)/.002,
                         .1*(joints-goal)]
        fitted = least_squares(residual, np.clip(goal, lower, upper),
                               bounds=(lower, upper), max_nfev=160)
        cap, bottle, own = measures(fitted.x)
        item["pregrasp_joints_rad"] = {name: float(value)
                                       for name, value in zip(names, fitted.x)}
        item["preshape_minimum_cap_clearance_m"] = float(min(cap))
        item["preshape_minimum_bottle_clearance_m"] = float(min(bottle))
        item["preshape_minimum_self_clearance_m"] = float(min(own))
        item["preshape_max_joint_change_rad"] = float(max(abs(fitted.x-goal)))
        item["preshape_optimizer_evaluations"] = int(fitted.nfev)
        output["best"].append(item)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reach_scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = fit(json.loads(args.reach_scan.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for item in output["best"][:5]:
        print("RIGHT_CAP_PRESHAPE " + json.dumps({key: item[key] for key in
              ("index", "preshape_minimum_cap_clearance_m",
               "preshape_minimum_bottle_clearance_m",
               "preshape_minimum_self_clearance_m",
               "preshape_max_joint_change_rad")}))


if __name__ == "__main__":
    main()
