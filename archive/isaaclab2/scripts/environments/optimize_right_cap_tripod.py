"""Refine a right cap proposal so the index joins thumb and middle at the sidewall.

This is an offline official Beta 1 MJCF geometry search. It is not an Isaac
USD/PhysX grasp validation and must not be admitted to the stable Reset bank
without an unsupported hold and independent Reset replay.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from analyze_right_cap_reach import _model


_INDEX_JOINTS = (
    "r_index_finger_mcp_flex", "r_index_finger_mcp_abd",
    "r_index_finger_pip", "r_index_finger_dip",
)


def refine_index(proposal: dict, *, seed: int = 0, maxiter: int = 35,
                 target_azimuth_deg: float | None = None) -> dict:
    """Move only the index finger; keep the proven thumb/middle wrist proposal.

    Cap and bottle proxies match the Phase 0E engineering bottle in metres.
    The nearest cap point must lie on the cylindrical sidewall, not the lid.
    """
    model = _model()
    data = mujoco.MjData(model)
    data.qpos[:7] = [*proposal["wrist_position_w_m"],
                     *proposal["wrist_quaternion_wxyz"]]
    joints = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): i
              for i in range(1, model.njnt)}
    for name, value in proposal["joints_rad"].items():
        data.qpos[model.jnt_qposadr[joints[name]]] = value
    index_ids = [joints[name] for name in _INDEX_JOINTS]
    addresses = [model.jnt_qposadr[i] for i in index_ids]
    bounds = [tuple(model.jnt_range[i]) for i in index_ids]
    cap = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "proxy_cap")
    bottle = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
              for name in ("proxy_bottle", "proxy_neck")]
    def body(geom: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                int(model.geom_bodyid[geom]))
    colliders = [i for i in range(model.ngeom)
                 if model.geom_contype[i] and model.geom_conaffinity[i]
                 and i not in (cap, *bottle)]
    index_geoms = [i for i in colliders if body(i).startswith("r_index_finger")]
    other_geoms = [i for i in colliders if body(i).startswith("r_")
                   and not body(i).startswith(("r_index_finger", "r_wrist"))]
    if not index_geoms or not other_geoms:
        raise RuntimeError("Official MJCF finger collision geometry is missing")

    def azimuth_cost(point):
        if target_azimuth_deg is None:
            return 0.0
        angle = math.atan2(point[1], point[0])
        target = math.radians(target_azimuth_deg)
        return .017 * abs(math.atan2(math.sin(angle-target),
                                    math.cos(angle-target)))

    def measure(values):
        data.qpos[addresses] = values
        mujoco.mj_forward(model, data)
        cap_points = []
        for geom in index_geoms:
            nearest = np.zeros(6)
            distance = float(mujoco.mj_geomDistance(
                model, data, geom, cap, .2, nearest))
            cap_points.append((distance, nearest[3:].copy(), body(geom)))
        # A lid point is penalized separately; retain the candidate nearest
        # to the sidewall even when another index link is nearer to the lid.
        sidewall = min(cap_points, key=lambda item: item[0]
                       + 5 * max(0, item[1][2] - .6535)
                       + 5 * max(0, .6375 - item[1][2])
                       + azimuth_cost(item[1]))
        bottle_clearance = min(float(mujoco.mj_geomDistance(
            model, data, i, j, .2, None))
            for i in index_geoms for j in bottle)
        other_clearance = min(float(mujoco.mj_geomDistance(
            model, data, i, j, .2, None))
            for i in index_geoms for j in other_geoms)
        return sidewall, bottle_clearance, other_clearance

    evaluations = 0
    def objective(values):
        nonlocal evaluations
        evaluations += 1
        sidewall, bottle_clearance, other_clearance = measure(values)
        distance, point, _ = sidewall
        return (abs(distance - .001)
                + 5 * max(0, point[2] - .6535)
                + 5 * max(0, .6375 - point[2])
                + azimuth_cost(point)
                + 8 * max(0, .003 - bottle_clearance)
                + 8 * max(0, .002 - other_clearance))

    fitted = differential_evolution(objective, bounds, seed=seed,
                                    maxiter=maxiter, popsize=6, polish=True,
                                    workers=1)
    sidewall, bottle_clearance, other_clearance = measure(fitted.x)
    distance, point, body_name = sidewall
    output = {**proposal,
              "joints_rad": {**proposal["joints_rad"],
                             **dict(zip(_INDEX_JOINTS, map(float, fitted.x)))}}
    # The input's reach, sidewall and preshape measurements describe the old
    # index configuration. Drop them so no downstream tool mistakes stale
    # proxy distances for measurements of this new candidate.
    for key in ("score", "digit_cap_distances_m", "minimum_cap_distance_m",
                "minimum_bottle_distance_m", "minimum_self_distance_m",
                "digit_sidewall_distances_m", "digit_sidewall_points_w_m",
                "sidewall_opposing_digit", "sidewall_angular_separation_rad",
                "body_cap_distances_m", "body_bottle_distances_m",
                "sidewall_refinement", "pregrasp_joints_rad",
                "preshape_minimum_cap_clearance_m",
                "preshape_minimum_bottle_clearance_m",
                "preshape_minimum_self_clearance_m",
                "preshape_max_joint_change_rad",
                "preshape_optimizer_evaluations"):
        output.pop(key, None)
    output["index_refinement"] = {
        "backend": "official_beta1_MJCF_proxy_geometry_only",
        "index_sidewall_distance_m": distance,
        "index_nearest_cap_point_w_m": point.tolist(),
        "index_contact_body": body_name,
        "minimum_index_bottle_clearance_m": bottle_clearance,
        "minimum_index_other_digit_clearance_m": other_clearance,
        "objective_m": float(fitted.fun),
        "index_sidewall_azimuth_deg": math.degrees(math.atan2(point[1], point[0])),
        "target_sidewall_azimuth_deg": target_azimuth_deg,
        "optimizer_evaluations": evaluations,
        "seed": seed,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--proposal_index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maxiter", type=int, default=35)
    parser.add_argument("--target_azimuth_deg", type=float, default=None,
                        help="Optional cap-local sidewall sector for the index, in degrees")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.proposal.read_text())
    matches = [item for item in source["best"] if item["index"] == args.proposal_index]
    if len(matches) != 1:
        raise ValueError("Proposal index absent or ambiguous")
    result = {**source, "best": [refine_index(
        matches[0], seed=args.seed, maxiter=args.maxiter,
        target_azimuth_deg=args.target_azimuth_deg)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("RIGHT_CAP_TRIPOD_GEOMETRY " + json.dumps(
        result["best"][0]["index_refinement"]))


if __name__ == "__main__":
    main()
