"""Finite official-MJCF contact-point optimization for the 70 mm bottle.

This addresses Phase 0C's measured one-sided contact. Closest *collision*
points, not fingertip sites, define radial opposition. Output is only a
geometric proposal; the USD/PhysX slow close and one-second release remain
mandatory. No MuJoCo runtime is used by the Isaac scene.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from analyze_grasp_reach import model_for


def optimize(source, max_nfev, opposition_cosine=-.35, opposition_weight=3.0,
             self_clearance_m=.003, target_contact_m=.0005,
             wrist_x_m=None, orientation_limit_rad=.7,
             max_goal_overlap_m=.001, contact_axial_margin_m=.02):
    radius = float(source["radius_m"])
    model = model_for(radius)
    data = mujoco.MjData(model)
    bottle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                               "diagnostic_cylinder")
    geom_ids = [i for i in range(model.ngeom) if i != bottle
                and model.geom_contype[i] and model.geom_conaffinity[i]]
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                    int(model.geom_bodyid[i])) for i in geom_ids]
    def finger_group(body):
        return next((name for name in ("thumb", "index_finger", "middle_finger",
                                       "ring_finger", "pinky") if name in body), None)
    self_pairs = [(a, b) for i, (body_a, a) in enumerate(zip(body_names, geom_ids))
                  for body_b, b in zip(body_names[i+1:], geom_ids[i+1:])
                  if finger_group(body_a) is not None and finger_group(body_b) is not None
                  and finger_group(body_a) != finger_group(body_b)]
    joint_ids = list(range(1, model.njnt))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in joint_ids]
    addresses = np.array([model.jnt_qposadr[i] for i in joint_ids])
    lower_joint = np.array([model.jnt_range[i, 0] for i in joint_ids]) + 1e-5
    upper_joint = np.array([model.jnt_range[i, 1] for i in joint_ids]) - 1e-5
    base = Rotation.from_quat((0, -np.sqrt(.5), 0, np.sqrt(.5)))
    lower = np.r_[[-.2, -.08, .42,
                   -orientation_limit_rad, -orientation_limit_rad,
                   -orientation_limit_rad], lower_joint]
    upper = np.r_[[-.085, .08, .55,
                   orientation_limit_rad, orientation_limit_rad,
                   orientation_limit_rad], upper_joint]
    if wrist_x_m is not None:
        lower[0] = max(lower[0], wrist_x_m - .001)
        upper[0] = min(upper[0], wrist_x_m + .001)
        if lower[0] >= upper[0]:
            raise ValueError("Wrist X interval is outside the reach-search workspace")
    digits = ("thumb", "index_finger", "middle_finger", "ring_finger")
    records = []
    for proposal in source["best"]:
        q = proposal["wrist_quaternion_wxyz"]
        rotation = Rotation.from_quat((q[1], q[2], q[3], q[0]))
        seed = np.r_[proposal["wrist_position_w_m"],
                     (rotation*base.inv()).as_rotvec(),
                     [proposal["joints_rad"][name] for name in names]]
        seed = np.clip(seed, lower + 1e-6, upper - 1e-6)

        def evaluate(values, detail=False):
            rot = Rotation.from_rotvec(values[3:6])*base
            qxyzw = rot.as_quat()
            data.qpos[:7] = [*values[:3], qxyzw[3], *qxyzw[:3]]
            data.qpos[addresses] = values[6:]
            mujoco.mj_forward(model, data)
            distances = []
            closest = {digit: (float("inf"), None, None) for digit in digits}
            for body, geom in zip(body_names, geom_ids):
                points = np.zeros(6)
                dist = float(mujoco.mj_geomDistance(model, data, geom,
                                                    bottle, .5, points))
                distances.append(dist)
                for digit in digits:
                    if digit in body and dist < closest[digit][0]:
                        closest[digit] = (dist, points[3:6].copy(), body)
            self_distances = [float(mujoco.mj_geomDistance(
                model, data, a, b, .5, np.zeros(6))) for a, b in self_pairs]
            radial = {}
            for digit in digits:
                point = closest[digit][1]
                if point is None:
                    raise RuntimeError(f"No official collision point for {digit}")
                xy = point[:2]
                radial[digit] = xy/max(1e-9, np.linalg.norm(xy))
            if detail:
                sites = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i):
                         data.site_xpos[i].tolist() for i in range(model.nsite)}
                return distances, self_distances, closest, radial, rot, sites
            # A position-drive target may lie slightly behind the object's
            # surface to generate a bounded contact preload. This is never
            # used as an initial joint state; PhysX must enforce clearance.
            residual = [max(0.0, -max_goal_overlap_m-dist)/.002
                        for dist in distances]
            residual.extend(max(0.0, self_clearance_m-dist)/.002
                            for dist in self_distances)
            residual.extend((closest[digit][0]-target_contact_m)/.005
                            for digit in digits)
            # The diagnostic cylinder is 0.2 m tall and centered at z=.5.
            # A tip touching its top rim can appear radially opposed yet
            # immediately lose contact as the free bottle falls.
            residual.extend(max(0.0, abs(closest[digit][1][2]-.5)
                                -(.1-contact_axial_margin_m))/.005
                            for digit in digits)
            thumb = radial["thumb"]
            residual.extend(opposition_weight*max(
                0.0, float(np.dot(thumb, radial[digit]))-opposition_cosine)
                            for digit in digits[1:])
            residual.extend(.03*(values[6:]-seed[6:]))
            residual.extend(.1*(values[:3]-seed[:3])/.05)
            residual.extend(.1*(values[3:6]-seed[3:6]))
            return np.asarray(residual)

        fit = least_squares(evaluate, seed, bounds=(lower, upper),
                            max_nfev=max_nfev, ftol=1e-4, xtol=1e-4,
                            gtol=1e-4)
        distances, self_distances, closest, radial, rotation, sites = evaluate(
            fit.x, detail=True)
        qxyzw = rotation.as_quat()
        item = {**proposal,
                "wrist_position_w_m": fit.x[:3].tolist(),
                "wrist_quaternion_wxyz": [qxyzw[3], *qxyzw[:3]],
                "joints_rad": {name: float(value) for name, value in
                               zip(names, fit.x[6:])},
                "tip_sites_w_m": sites,
                "minimum_mjcf_convex_distance_m": float(min(distances)),
                "minimum_mjcf_self_clearance_m": float(min(self_distances)),
                "collision_contact_points_w_m": {
                    digit: closest[digit][1].tolist() for digit in digits},
                "collision_contact_axial_heights_m": {
                    digit: float(closest[digit][1][2]-.5) for digit in digits},
                "collision_contact_bodies": {
                    digit: closest[digit][2] for digit in digits},
                "collision_contact_distances_m": {
                    digit: closest[digit][0] for digit in digits},
                "collision_radial_directions_xy": {
                    digit: radial[digit].tolist() for digit in digits},
                "thumb_opposition_dot": {
                    digit: float(np.dot(radial["thumb"], radial[digit]))
                    for digit in digits[1:]},
                "antipodal_objective_norm": float(np.linalg.norm(evaluate(fit.x))),
                "optimizer_nfev": int(fit.nfev),
                "optimizer_status": int(fit.status)}
        item.pop("pregrasp_joints_rad", None)
        records.append(item)
    records.sort(key=lambda item: item["antipodal_objective_norm"])
    return {"radius_m": radius, "source": "official_beta1_MJCF_contact_point_proposals_only",
            "seed_source": source.get("seed"), "starts": len(records),
            "max_nfev": max_nfev,
            "opposition_cosine_target": opposition_cosine,
            "opposition_weight": opposition_weight,
            "self_clearance_target_m": self_clearance_m,
            "target_contact_m": target_contact_m,
            "wrist_x_target_m": wrist_x_m,
            "orientation_limit_rad": orientation_limit_rad,
            "max_goal_overlap_m": max_goal_overlap_m,
            "contact_axial_margin_m": contact_axial_margin_m,
            "best": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reach_scan", type=Path, required=True)
    parser.add_argument("--max_nfev", type=int, default=120)
    parser.add_argument("--opposition_cosine", type=float, default=-.35)
    parser.add_argument("--opposition_weight", type=float, default=3.0)
    parser.add_argument("--self_clearance_m", type=float, default=.003)
    parser.add_argument("--target_contact_m", type=float, default=.0005,
                        help="Desired MJCF goal distance; slightly negative preloads finger drives")
    parser.add_argument("--wrist_x_m", type=float,
                        help="Constrain world wrist X to a 2 mm diagnostic interval")
    parser.add_argument("--orientation_limit_rad", type=float, default=.7,
                        help="Per-axis wrist rotation-vector search bound")
    parser.add_argument("--max_goal_overlap_m", type=float, default=.001,
                        help="Virtual overlap allowed only for a finger position target")
    parser.add_argument("--contact_axial_margin_m", type=float, default=.02,
                        help="Keep nominal digit contacts off cylinder top/bottom rims")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (not -1 <= args.opposition_cosine < 1 or args.opposition_weight <= 0
            or args.self_clearance_m <= 0):
        parser.error("Opposition target/weight must be physically bounded and positive")
    if not 0 < args.orientation_limit_rad <= 3.14:
        parser.error("--orientation_limit_rad must be in (0, 3.14]")
    if not 0 <= args.max_goal_overlap_m <= .01:
        parser.error("--max_goal_overlap_m must be in [0, 0.01] m")
    if not 0 < args.contact_axial_margin_m < .1:
        parser.error("--contact_axial_margin_m must be in (0, 0.1) m")
    result = optimize(json.loads(args.reach_scan.read_text()), args.max_nfev,
                      args.opposition_cosine, args.opposition_weight,
                      args.self_clearance_m, args.target_contact_m,
                      args.wrist_x_m, args.orientation_limit_rad,
                      args.max_goal_overlap_m, args.contact_axial_margin_m)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for item in result["best"]:
        print("ANTIPODAL_MJCF_PROPOSAL " + json.dumps({
            "index": item["index"],
            "objective_norm": item["antipodal_objective_norm"],
            "minimum_distance_m": item["minimum_mjcf_convex_distance_m"],
            "minimum_self_clearance_m": item["minimum_mjcf_self_clearance_m"],
            "thumb_opposition_dot": item["thumb_opposition_dot"],
            "contact_distances_m": item["collision_contact_distances_m"],
        }), flush=True)


if __name__ == "__main__":
    main()
