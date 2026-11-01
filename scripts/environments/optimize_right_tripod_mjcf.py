"""Propose a right cap-sidewall tripod around a measured PhysX state.

This is an offline official-MJCF collision screen. It never labels a proposal
as a valid grasp; the native USD/PhysX scene must independently test it with
free bottle/cap, gravity, normal drives, and a fresh Reset replay.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation

from dexteroushands_lab.assets.bottle import BottleConfig
from dexteroushands_lab.assets.wuji_hand2_beta import load_hand2_spec

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.environments.generate_pregrasp_reset_candidates import (  # noqa: E402
    MjcfGeometryScreen, combined_model,
)


JOINTS = (
    "r_index_finger_mcp_flex", "r_index_finger_mcp_abd",
    "r_index_finger_pip", "r_index_finger_dip",
    "r_middle_finger_mcp_abd", "r_middle_finger_pip",
    "r_thumb_cmc_abd",
)
DIGITS = ("thumb", "index_finger", "middle_finger")


def _wxyz(pose_xyzw):
    return [pose_xyzw[6], *pose_xyzw[3:6]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--cap-pose-trace", type=Path,
                        help="Use the final free cap/bottle pose from a later PhysX springback trace")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--maxiter", type=int, default=60)
    args = parser.parse_args()
    if args.maxiter < 1:
        parser.error("maxiter must be positive")
    trace = json.loads(args.trace.read_text())
    row = trace["rows"][-1]
    cap_row = (json.loads(args.cap_pose_trace.read_text())["rows"][-1]
               if args.cap_pose_trace is not None else row)
    cfg = BottleConfig()
    official = Path(os.environ["WUJI_DESCRIPTION_ROOT"])
    spec = load_hand2_spec(official, side="right")
    screen = MjcfGeometryScreen(combined_model(official, cfg), cfg)
    model, data = screen.model, screen.data
    cap = np.asarray(cap_row["cap_pose_w_xyzw"], dtype=float)
    bottle = np.asarray(cap_row["bottle_pose_w_xyzw"], dtype=float)
    wrist = np.asarray(row["right_wrist_pose_w_xyzw"], dtype=float)
    q0 = trace["final_right_joint_position"]
    for name, pose in (("proxy_cap", cap), ("proxy_bottle", bottle)):
        model.body(name).pos[:] = pose[:3]
        model.body(name).quat[:] = _wxyz(pose)
    bottle_rot = Rotation.from_quat(bottle[3:])
    neck = model.body("proxy_neck")
    neck.pos[:] = bottle[:3] + bottle_rot.apply(
        [0, 0, (cfg.bottle_height + cfg.neck_height) / 2])
    neck.quat[:] = _wxyz(bottle)
    cap_id = screen.ids["proxy_cap"]
    right_geoms = {digit: [g for g in screen.colliders["r"]
                           if screen._digit(g) == digit] for digit in DIGITS}
    root_address = screen.joints["r_pregrasp_root"]
    cap_inverse = Rotation.from_quat(cap[3:]).inv()
    base_rotation = Rotation.from_quat(wrist[3:])
    limits = {
        "r_index_finger_mcp_flex": (-0.15, 0.15),
        "r_index_finger_mcp_abd": (-0.15, 0.15),
        "r_index_finger_pip": (-0.15, 0.15),
        "r_index_finger_dip": (-0.15, 0.15),
        "r_middle_finger_mcp_abd": (-0.10, 0.10),
        "r_middle_finger_pip": (-0.15, 0.15),
        "r_thumb_cmc_abd": (-0.10, 0.10),
    }
    official_limits = {joint.name: (joint.lower, joint.upper) for joint in spec.joints}
    bounds = ([(-0.005, 0.005)] * 3 + [(-0.12, 0.12)] * 3
              + [(max(limits[name][0], official_limits[name][0] - q0[name]),
                  min(limits[name][1], official_limits[name][1] - q0[name]))
                 for name in JOINTS])

    def evaluate(x, details=False):
        translation, rotvec, joint_delta = x[:3], x[3:6], x[6:]
        pos = wrist[:3] + translation
        quat = (Rotation.from_rotvec(rotvec) * base_rotation).as_quat()
        data.qpos[root_address:root_address + 7] = [*pos, quat[3], *quat[:3]]
        for name, initial in q0.items():
            data.qpos[screen.joints[name]] = initial
        for name, delta in zip(JOINTS, joint_delta):
            data.qpos[screen.joints[name]] = q0[name] + delta
        mujoco.mj_forward(model, data)
        # A shortened proxy demands an interior sidewall patch, away from the
        # bottom and top rims of the 25 mm example cap.
        model.geom_size[cap_id, 1] = cfg.cap_height / 2 - 0.002
        gaps, local_points = {}, {}
        for digit in DIGITS:
            candidates = []
            for geom in right_geoms[digit]:
                gap, near = screen._distance(geom, cap_id)
                candidates.append((gap, geom, near))
            gap, _, near = min(candidates, key=lambda item: item[0])
            gaps[digit] = gap
            local_points[digit] = cap_inverse.apply(near[3:6] - cap[:3])
        model.geom_size[cap_id, 1] = cfg.cap_height / 2
        full_gaps = {digit: min(screen._distance(g, cap_id)[0]
                                for g in right_geoms[digit]) for digit in DIGITS}
        self_gaps = {}
        for first, second in (("thumb", "index_finger"),
                              ("thumb", "middle_finger"),
                              ("index_finger", "middle_finger")):
            self_gaps[f"{first}/{second}"] = min(
                screen._distance(a, b)[0]
                for a in right_geoms[first] for b in right_geoms[second])
        cross_gap = min(screen._distance(g, screen.ids["proxy_bottle"])[0]
                        for geoms in right_geoms.values() for g in geoms)
        loss = 0.0
        for digit in DIGITS:
            loss += ((gaps[digit] - 0.0005) / 0.002) ** 2
            loss += (max(0.0, abs(local_points[digit][2])
                         - (cfg.cap_height / 2 - 0.003)) / 0.001) ** 2
            loss += (max(0.0, -full_gaps[digit] - 0.0003) / 0.001) ** 2
        for gap in self_gaps.values():
            loss += (max(0.0, 0.0005 - gap) / 0.001) ** 2
        loss += (max(0.0, 0.002 - cross_gap) / 0.001) ** 2
        loss += 0.03 * np.sum((translation / 0.005) ** 2)
        loss += 0.03 * np.sum((rotvec / 0.12) ** 2)
        loss += 0.02 * np.sum((joint_delta / 0.15) ** 2)
        if details:
            return {
                "loss": float(loss),
                "wrist_pose_w_xyzw": [*pos.tolist(), *quat.tolist()],
                "right_joint_position": {name: float(q0[name] + joint_delta[JOINTS.index(name)])
                                         if name in JOINTS else float(q0[name]) for name in q0},
                "sidewall_proxy_gap_m": {key: float(value) for key, value in gaps.items()},
                "sidewall_proxy_point_cap_local_m": {
                    key: value.tolist() for key, value in local_points.items()},
                "full_cap_gap_m": {key: float(value) for key, value in full_gaps.items()},
                "cross_digit_gap_m": {key: float(value) for key, value in self_gaps.items()},
                "right_bottle_gap_m": float(cross_gap),
            }
        return loss

    zero = np.zeros(len(bounds))
    initial = evaluate(zero, details=True)
    solution = differential_evolution(evaluate, bounds, seed=args.seed,
                                      maxiter=args.maxiter, popsize=5,
                                      polish=True, updating="immediate")
    candidate = evaluate(solution.x, details=True)
    payload = {
        "status": "MJCF_ONLY_REQUIRES_NATIVE_USD_PHYSX_REPLAY",
        "source_trace": str(args.trace), "source_sample_index": trace["sample_index"],
        "cap_pose_trace": str(args.cap_pose_trace) if args.cap_pose_trace else None,
        "official_root": str(official), "official_source_commit": spec.source_commit,
        "right_usd_sha256": spec.usd_sha256, "asset_variant": spec.variant,
        "seed": args.seed,
        "optimizer_iterations": int(solution.nit),
        "optimizer_evaluations": int(solution.nfev),
        "optimizer_success": bool(solution.success),
        "initial": initial, "candidate": candidate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("RIGHT_TRIPOD_MJCF", json.dumps({
        "initial_loss": initial["loss"], "candidate_loss": candidate["loss"],
        "optimizer_success": bool(solution.success),
        "evaluations": int(solution.nfev),
        "candidate_gaps_m": candidate["sidewall_proxy_gap_m"],
        "candidate_self_gaps_m": candidate["cross_digit_gap_m"],
    }))


if __name__ == "__main__":
    main()
