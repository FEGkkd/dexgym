"""Propose distinct natural-left pregrasps for native PGS reachability testing.

These are MJCF-only geometry proposals. The output is deliberately unqualified
for PPO until ``validate_phase1b_pregrasp_pgs.py`` admits each row under the
actual USD, gravity, contact and solver configuration. Existing A/B banks and
their frozen evaluation split are read-only inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from dexteroushands_lab.assets.bottle import BottleConfig
from dexteroushands_lab.scenes.grasp_init import grasp_sample_from_dict
from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.mjcf_tripod_fit import fit_right_tripod
from scripts.environments.generate_pregrasp_reset_candidates import combined_model, MjcfGeometryScreen


DIGITS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
FLEX_JOINTS = ("l_thumb_mcp", "l_index_finger_pip", "l_middle_finger_pip",
               "l_ring_finger_pip", "l_pinky_pip")


def fit_natural_left(screen, natural):
    """Fit five physical finger joints and one radial wrist offset to 1-mm gaps."""
    q0 = np.asarray(natural.left.joint_position, dtype=float)
    names = natural.metadata["left_joint_order"]
    indices = [names.index(name) for name in FLEX_JOINTS]
    limits = [screen.model.jnt_range[mujoco.mj_name2id(
        screen.model, mujoco.mjtObj.mjOBJ_JOINT, names[index])]
        for index in indices]
    wrist = np.asarray(natural.left.root_pose[:3], dtype=float)
    bottle = np.asarray(natural.bottle.root_pose[:3], dtype=float)
    radial = wrist - bottle
    radial[2] = 0.0
    radial /= np.linalg.norm(radial)

    def propose(values):
        position = wrist - values[0] * radial
        pose = (*position.tolist(), *natural.left.root_pose[3:])
        q = q0.copy()
        q[indices] = values[1:]
        hand = replace(natural.left, root_pose=pose, wrist_target_pose=pose,
                       joint_position=tuple(q), joint_target=tuple(q))
        return replace(natural, left=hand)

    def residual(values):
        geometry = screen.check(propose(values))
        gaps = np.array([geometry["per_digit"]["l"][digit]["gap_m"]
                         for digit in DIGITS])
        self_gap = geometry["minimum_cross_digit_self_gap_m"]["l"]
        left_cap_gap = geometry["per_object"]["l_proxy_cap"]["minimum_gap_m"]
        return np.r_[(gaps - .001) * 1000, (values[1:] - q0[indices]) * .5,
                     values[0] * 50, max(0, -self_gap) * 2000,
                     max(0, -left_cap_gap) * 2000]

    lower = np.r_[0., [max(limits[j][0] + 1e-4, q0[index] - .5)
                       for j, index in enumerate(indices)]]
    upper = np.r_[.025, [min(limits[j][1] - 1e-4, q0[index] + .5)
                         for j, index in enumerate(indices)]]
    solution = least_squares(residual, np.r_[0., q0[indices]], bounds=(lower, upper),
                             max_nfev=100, diff_step=1e-3)
    sample = propose(solution.x)
    geometry = screen.check(sample)
    return sample, geometry, {"wrist_inward_m": float(solution.x[0]),
                              "max_finger_joint_change_rad": float(np.max(
                                  np.abs(solution.x[1:] - q0[indices]))),
                              "optimizer_evaluations": solution.nfev}


def fit_thumb_tip(screen, sample, target_gap_m: float):
    """Bring the opposed thumb *tip*, not only its proximal collider, near the bottle."""
    q0 = np.asarray(sample.left.joint_position, dtype=float)
    names = sample.metadata["left_joint_order"]
    indices = [i for i, name in enumerate(names) if name.startswith("l_thumb_")]
    site = mujoco.mj_name2id(screen.model, mujoco.mjtObj.mjOBJ_SITE, "l_thumb_tip")
    limits = np.array([screen.model.jnt_range[mujoco.mj_name2id(
        screen.model, mujoco.mjtObj.mjOBJ_JOINT, names[i])] for i in indices])

    def propose(values):
        q = q0.copy()
        q[indices] = values
        hand = replace(sample.left, joint_position=tuple(q), joint_target=tuple(q))
        return replace(sample, left=hand)

    def residual(values):
        geometry = screen.check(propose(values))
        tip = screen.data.site_xpos[site]
        tip_gap = (np.linalg.norm(tip[:2] - sample.bottle.root_pose[:2])
                   - screen.bottle.bottle_radius)
        proximal_gap = geometry["per_digit"]["l"]["thumb"]["gap_m"]
        self_gap = geometry["minimum_cross_digit_self_gap_m"]["l"]
        return np.r_[(tip_gap - target_gap_m) * 1000,
                     max(0, -proximal_gap - .0005) * 2500,
                     max(0, -self_gap) * 2500, (values - q0[indices]) * .5]

    lower = np.maximum(limits[:, 0] + 1e-4, q0[indices] - .7)
    upper = np.minimum(limits[:, 1] - 1e-4, q0[indices] + .7)
    solution = least_squares(residual, q0[indices], bounds=(lower, upper),
                             max_nfev=150, diff_step=1e-3)
    fitted = propose(solution.x)
    geometry = screen.check(fitted)
    tip = screen.data.site_xpos[site]
    tip_gap = (np.linalg.norm(tip[:2] - sample.bottle.root_pose[:2])
               - screen.bottle.bottle_radius)
    return fitted, geometry, {"thumb_tip_gap_m": float(tip_gap),
                              "max_thumb_joint_change_rad": float(np.max(
                                  np.abs(solution.x - q0[indices]))),
                              "optimizer_evaluations": solution.nfev}


def finger_probe_targets(screen, sample):
    """Create ordinary finger-target feasibility commands, never PPO actions."""
    q = np.asarray(sample.left.joint_position, dtype=float).copy()
    names = sample.metadata["left_joint_order"]
    initial = screen.check(sample)
    for digit, joint_name in zip(DIGITS, FLEX_JOINTS):
        index = names.index(joint_name)
        gap = initial["per_digit"]["l"][digit]["gap_m"]
        test = q.copy()
        test[index] += .01
        perturbed = replace(sample, left=replace(
            sample.left, joint_position=tuple(test), joint_target=tuple(test)))
        derivative = (screen.check(perturbed)["per_digit"]["l"][digit]["gap_m"]
                      - gap) / .01
        if abs(derivative) <= 1e-4:
            continue
        joint_id = mujoco.mj_name2id(screen.model, mujoco.mjtObj.mjOBJ_JOINT,
                                     joint_name)
        lo, hi = screen.model.jnt_range[joint_id]
        q[index] = np.clip(q[index] + np.clip((.0001 - gap) / derivative, -.10, .10),
                           lo + 1e-4, hi - 1e-4)
    right = fit_right_tripod(screen, sample,
                             site_radial_gap_targets_m=(.0001, .0025, .0025))
    return q.tolist(), list(right.right.joint_position)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--additional-left-inward-m", nargs="+", type=float,
                        default=(0.,))
    parser.add_argument("--right-train-limit", type=int, default=12)
    parser.add_argument("--thumb-tip-target-m", type=float,
                        help="Optional fingertip-aware thumb fit; still requires native PGS admission")
    args = parser.parse_args()
    if (args.right_train_limit < 1 or any(not 0 <= x <= .003
        for x in args.additional_left_inward_m)):
        parser.error("Invalid right limit or additional left radial offsets")
    if args.thumb_tip_target_m is not None and not .001 <= args.thumb_tip_target_m <= .004:
        parser.error("Thumb tip target must be 1–4 mm from the bottle sidewall")
    root = Path(__file__).resolve().parents[2]
    natural_bank = json.loads((root / "docs/assets/phase0_pregrasp_reset_bank.json").read_text())
    pregrasp_bank = json.loads((root / "docs/assets/phase1b_pregrasp_bank.json").read_text())
    manifest = json.loads((root / "docs/assets/phase1b_pregrasp_split.json").read_text())
    train_ids = set(manifest["splits"]["A"]["train_local_shape_seeds"])
    right_bank = [grasp_sample_from_dict(row) for row in pregrasp_bank
                  if row["seed"] in train_ids][:args.right_train_limit]
    cfg = BottleConfig()
    screen = MjcfGeometryScreen(combined_model(
        Path(os.environ["WUJI_DESCRIPTION_ROOT"]), cfg), cfg)
    candidates, source_results = [], []
    for natural_row in natural_bank:
        natural = grasp_sample_from_dict(natural_row)
        left, left_geometry, fit = fit_natural_left(screen, natural)
        if args.thumb_tip_target_m is not None:
            left, left_geometry, thumb_fit = fit_thumb_tip(
                screen, left, args.thumb_tip_target_m)
            fit["thumb_fit"] = thumb_fit
        left_gaps = [left_geometry["per_digit"]["l"][d]["gap_m"] for d in DIGITS]
        left_safe = (min(left_gaps) > (-.001 if args.thumb_tip_target_m else -.0005)
                     and max(left_gaps) < .003
                     and left_geometry["minimum_cross_digit_self_gap_m"]["l"] > 0
                     and left_geometry["left_four_finger_half_arc_rad"] <= math.radians(75)
                     and left_geometry["left_thumb_opposition_rad"] >= math.radians(125))
        if args.thumb_tip_target_m is not None:
            left_safe &= abs(fit["thumb_fit"]["thumb_tip_gap_m"]
                             - args.thumb_tip_target_m) <= .001
        source = {"natural_seed": natural.seed, "fit": fit,
                  "left_gap_m": left_gaps, "left_safe": left_safe,
                  "candidate_count": 0}
        if left_safe:
            for right in right_bank:
                for extra_index, inward in enumerate(args.additional_left_inward_m):
                    pos = np.asarray(left.left.root_pose[:3], dtype=float)
                    origin = np.asarray(right.bottle.root_pose[:3], dtype=float)
                    direction = pos - origin
                    direction[2] = 0
                    direction /= np.linalg.norm(direction)
                    updated = pos - inward * direction
                    pose = (*updated.tolist(), *left.left.root_pose[3:])
                    seed = (100000 + natural.seed * 100 + right.seed % 100)
                    if inward:
                        seed = seed * 10 + round(inward * 100000)
                    meta = {k: v for k, v in right.metadata.items()
                            if k not in ("phase1b_pgs_reach_evidence", "geometry_evidence",
                                         "source_probe", "source_replay")}
                    meta.update({"schema": "dexgym-phase1b-natural-left-hybrid-proposal-v1",
                                 "phase1b_reset_level": "C", "local_shape_seed": seed,
                                 "left_natural_source_seed": natural.seed,
                                 "right_natural_source_seed": right.metadata["right_natural_source_seed"],
                                 "left_from_natural_local_family": True,
                                 "left_fit": fit, "additional_left_inward_m": inward,
                                 "geometry_qualified": False, "hold_validated": False})
                    hand = replace(left.left, root_pose=pose, wrist_target_pose=pose)
                    sample = replace(right, left=hand, seed=seed, metadata=meta,
                                     validated=False, validation=None)
                    geometry = screen.check(sample)
                    left_g = [geometry["per_digit"]["l"][d]["gap_m"] for d in DIGITS]
                    right_g = [geometry["per_digit"]["r"][d].get(
                        "sidewall_gap_m", math.inf) for d in DIGITS[:3]]
                    safe = (min(left_g) > -.001 and max(left_g) < .0035
                            and min(right_g) > -.0005 and max(right_g) < .004
                            and min(geometry["minimum_cross_digit_self_gap_m"].values()) > 0
                            and geometry["minimum_hand_hand_gap_m"] > .003
                            and min(geometry["per_object"][key]["minimum_gap_m"]
                                    for key in ("l_proxy_cap", "r_proxy_bottle",
                                                "r_proxy_neck")) > 0)
                    if not safe:
                        continue
                    left_target, right_target = finger_probe_targets(screen, sample)
                    meta.update({"left_probe_joint_target": left_target,
                                 "right_probe_joint_target": right_target,
                                 "geometry_evidence": {"official_mjcf": {
                                     **geometry, "qualified": True,
                                     "phase1b_criterion": "natural_left_local_family_plus_qualified_right_tripod"}}})
                    candidates.append(asdict(replace(sample, metadata=meta)))
                    source["candidate_count"] += 1
        source_results.append(source)
    if len({row["seed"] for row in candidates}) != len(candidates):
        raise ValueError("Duplicate local candidate seeds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(candidates, indent=2, sort_keys=True) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    report = {"status": "MJCF_PROPOSALS_ONLY_NOT_NATIVE_PGS_QUALIFIED",
              "candidate_count": len(candidates),
              "left_safe_source_count": sum(x["left_safe"] for x in source_results),
              "candidate_sha256": hashlib.sha256(payload.encode()).hexdigest(),
              "source_results": source_results,
              "additional_left_inward_m": args.additional_left_inward_m,
              "right_train_limit": args.right_train_limit,
              "thumb_tip_target_m": args.thumb_tip_target_m}
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "source_results"}))


if __name__ == "__main__":
    main()
