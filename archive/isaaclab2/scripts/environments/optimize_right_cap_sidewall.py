"""Refine a right-hand MJCF proposal toward a cap sidewall pinch.

Only middle-finger joints are changed. This geometric proposal must still pass
official USD/PhysX penetration, force, unsupported hold and Reset replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from analyze_right_cap_reach import _model


def refine(proposal):
    model = _model()
    data = mujoco.MjData(model)
    data.qpos[:7] = [*proposal["wrist_position_w_m"],
                     *proposal["wrist_quaternion_wxyz"]]
    joints = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): i
              for i in range(1, model.njnt)}
    for name, value in proposal["joints_rad"].items():
        data.qpos[model.jnt_qposadr[joints[name]]] = value
    variable_names = tuple(f"r_middle_finger_{part}" for part in
                           ("mcp_flex", "mcp_abd", "pip", "dip"))
    addresses = np.array([model.jnt_qposadr[joints[n]] for n in variable_names])
    original = data.qpos[addresses].copy()
    lower = np.array([model.jnt_range[joints[n], 0] for n in variable_names]) + 1e-5
    upper = np.array([model.jnt_range[joints[n], 1] for n in variable_names]) - 1e-5
    names = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
             for name in ("proxy_cap", "proxy_bottle", "proxy_neck")}
    colliders = [i for i in range(model.ngeom) if i not in names.values()
                 and model.geom_contype[i] and model.geom_conaffinity[i]]
    def body(geom):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                int(model.geom_bodyid[geom]))
    middle_distal = next(i for i in colliders if body(i) == "r_middle_finger_distal")
    self_pairs = [(a,b) for k,a in enumerate(colliders)
                  for b in colliders[k+1:]
                  if body(a).split("_")[1] != body(b).split("_")[1]
                  and "wrist" not in (body(a).split("_")[1],
                                      body(b).split("_")[1])]
    def measure(values):
        data.qpos[addresses] = values
        mujoco.mj_forward(model, data)
        point = np.zeros(6)
        distal = float(mujoco.mj_geomDistance(
            model, data, middle_distal, names["proxy_cap"], .2, point))
        cap = np.array([mujoco.mj_geomDistance(
            model, data, i, names["proxy_cap"], .2, None)
            for i in colliders])
        bottle = np.array([min(mujoco.mj_geomDistance(
            model, data, i, names[n], .2, None)
            for n in ("proxy_bottle", "proxy_neck")) for i in colliders])
        own = np.array([mujoco.mj_geomDistance(model, data, a, b, .2, None)
                        for a,b in self_pairs])
        return distal, point[3:].copy(), cap, bottle, own
    def residual(values):
        distal, point, cap, bottle, own = measure(values)
        return np.r_[(distal-.0005)/.002,
                     max(0, point[2]-.6535)/.002,
                     max(0, .6375-point[2])/.002,
                     np.maximum(0, -.001-cap)/.002,
                     np.maximum(0, .002-bottle)/.002,
                     np.maximum(0, .001-own)/.002,
                     .1*(values-original)]
    fitted = least_squares(residual, np.clip(original, lower, upper),
                           bounds=(lower, upper), max_nfev=250)
    distal, point, cap, bottle, own = measure(fitted.x)
    output = dict(proposal)
    output["joints_rad"] = {**proposal["joints_rad"],
                            **{name: float(value) for name,value in
                               zip(variable_names, fitted.x)}}
    output["sidewall_refinement"] = {
        "backend": "official_beta1_MJCF_proxy_geometry_only",
        "middle_distal_distance_m": distal,
        "middle_distal_nearest_cap_point_w_m": point.tolist(),
        "minimum_cap_distance_m": float(min(cap)),
        "minimum_bottle_neck_distance_m": float(min(bottle)),
        "minimum_cross_digit_distance_m": float(min(own)),
        "middle_joint_delta_rad": {name: float(a-b) for name,a,b in
                                    zip(variable_names, fitted.x, original)},
        "optimizer_evaluations": int(fitted.nfev),
    }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reach_scan", type=Path, required=True)
    parser.add_argument("--proposal_index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.reach_scan.read_text())
    matching = [item for item in source["best"] if item["index"] == args.proposal_index]
    if len(matching) != 1:
        raise ValueError("Proposal absent or ambiguous")
    result = refine(matching[0])
    output = {**source, "best": [result]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print("RIGHT_CAP_SIDEWALL_REFINEMENT " + json.dumps(result["sidewall_refinement"]))


if __name__ == "__main__":
    main()
