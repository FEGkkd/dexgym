#!/usr/bin/env python3
"""Fit independent raw MANUS hand intent to the signed rig at fixed Home wrists."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
from itertools import combinations
import json
from pathlib import Path
from time import monotonic_ns

import mujoco
import numpy as np
from scipy.optimize import least_squares

from dexgym.tasks.direct.wholebody_s05.p02.intent import CHAINS, hand_basis, unit
from dexgym.tasks.direct.wholebody_s05.p02.model import COUPLING_LIMIT
from dexgym.tasks.direct.wholebody_s05.v1.rmb.schema import file_sha, write_json
from summarize_teleop_parity_hand_slices import ROOT, SIDES, from_input, jsonl, margin, shape
from swept_live_guard import EXACT_MODEL_RECEIPT_SHA, SIGNED_SHA, Reason, verified_robot
from swept_live_guard_v2 import SweptEnvelopeV2


PAIRS = np.asarray(list(combinations(range(5), 2)))
NEAR_M = 0.020
PINCH_VECTOR_TOLERANCE_M = 0.005
GEOMETRY_SHA = "5781ed98ec6d4e3ce9cdd6da62be9e6fb1997a09e43f38b87376e10f08cf3511"
WINDOW_SHA = "0931007455bf9aaf178f6260726e037e74cc69a404b9345439c186263c9b822c"
FIELDS = ("tick_ns", "applied_substeps", "raw_manus_local", "raw_manus_local_mask",
          "target_local_landmark_xyz", "target_local_landmark_xyz_mask")
MAX_FIT_EVALUATIONS = 42
PATH_EDGE_MAX_RAD = 0.035
PATH_BUDGET_NS = 2_000_000_000
PROOF_QUERIES = 255
PROOF_DEPTH = 24
PINCH_PROOF_SAMPLE_COUNT = 12


def pair_vectors(points):
    return points[PAIRS[:, 1], -1] - points[PAIRS[:, 0], -1]


def geometric_shape(points):
    return shape(np.asarray(points).tolist())


@dataclass(frozen=True)
class HandIntent:
    raw: np.ndarray
    rotation: np.ndarray
    directions: np.ndarray
    vectors: np.ndarray
    apertures: np.ndarray

    @classmethod
    def from_raw(cls, raw, model_basis):
        points = np.asarray(raw, dtype=np.float64)
        if points.shape != (5, 4, 3) or not np.isfinite(points).all():
            raise ValueError("Expected five finite four-landmark MANUS chains")
        rotation = model_basis @ hand_basis(points[1:, 0]).T
        if abs(np.linalg.det(rotation) - 1) > 1e-10:
            raise ValueError("Expected a proper, unscaled rotation into the model palm frame")
        vectors = pair_vectors(points) @ rotation.T
        return cls(points, rotation, unit(np.diff(points, axis=1)) @ rotation.T,
                   vectors, np.linalg.norm(vectors, axis=-1))

    def evidence(self):
        return {"raw_manus_landmarks_m": self.raw.tolist(), "model_palm_from_raw_rotation": self.rotation.tolist(),
                "rotated_raw_landmarks_m": (self.raw @ self.rotation.T).tolist(),
                "rotated_segment_directions": self.directions.tolist(), "rotated_tip_pair_vectors_m": self.vectors.tolist(),
                "raw_tip_pair_aperture_m": self.apertures.tolist(), "raw_shape": geometric_shape(self.raw)}

    def errors(self, points):
        directions = unit(np.diff(points, axis=1))
        vector = pair_vectors(points)
        apertures = np.linalg.norm(vector, axis=-1)
        angles = np.rad2deg(np.arccos(np.clip(np.sum(directions * self.directions, axis=-1), -1, 1)))
        return {"segment_direction_error_deg": angles.tolist(), "tip_pair_vectors_m": vector.tolist(),
                "tip_pair_aperture_m": apertures.tolist(), "tip_pair_vector_error_m": np.linalg.norm(
                    vector - self.vectors, axis=-1).tolist(),
                "tip_pair_aperture_error_m": abs(apertures - self.apertures).tolist(), "shape": geometric_shape(points)}


class FunctionalFit:
    """Bounded task-space least squares on the signed chain with a fixed full-state Home seed."""

    def __init__(self, robot, guard):
        self.robot = robot
        self.guard = guard
        model = robot.model
        self.home = robot.geometry(robot.home)
        self.basis = np.stack([hand_basis(points[1:, 0]) for points in self.home.local])
        self.names = robot.landmark_names
        self.joints = np.asarray([[[model.joint(name).id for name in finger[:3]] for finger in side]
                                  for side in self.names])
        self.sites = np.asarray(robot.tip_ids).reshape(2, 5)
        self.width = model.nv

    def _points(self, side, indices):
        model, data = self.robot.model, self.robot.data
        joints, sites = self.joints[side], self.sites[side]
        xyz = np.concatenate((data.xanchor[joints], data.site_xpos[sites, None]), axis=1)
        points = (xyz - self.home.palm[side]) @ self.home.palm_rotation[side]
        jac = np.zeros((5, 4, 3, len(indices)))
        jacp = np.zeros((3, self.width))
        v_indices = self.robot.v_indices[indices]
        for finger in range(5):
            for joint in range(1, 3):
                body = int(model.jnt_bodyid[joints[finger, joint]])
                mujoco.mj_jac(model, data, jacp, None, xyz[finger, joint], body)
                jac[finger, joint] = self.home.palm_rotation[side].T @ jacp[:, v_indices]
            mujoco.mj_jacSite(model, data, jacp, None, int(sites[finger]))
            jac[finger, -1] = self.home.palm_rotation[side].T @ jacp[:, v_indices]
        return points, jac

    def _task(self, sides, goals, indices, full):
        errors, derivatives = [], []
        for side in sides:
            goal = goals[side]
            points, jac = self._points(side, indices)
            steps = np.diff(points, axis=1)
            length = np.linalg.norm(steps, axis=-1)
            direction = steps / length[..., None]
            step_jac = np.diff(jac, axis=1)
            projection = np.einsum("fkc,fkcq->fkq", direction, step_jac)
            direction_jac = (step_jac - direction[..., None] * projection[:, :, None]) / length[..., None, None]
            vectors = pair_vectors(points)
            vjac = jac[PAIRS[:, 1], -1] - jac[PAIRS[:, 0], -1]
            apertures = np.linalg.norm(vectors, axis=-1)
            ajac = np.einsum("pc,pcq->pq", vectors / np.maximum(apertures[:, None], 1e-9), vjac)
            near = (PAIRS[:, 0] == 0) & (goal.apertures <= NEAR_M)
            w_vec = np.where(near, 2., .18)
            w_gap = np.where(near, 1., .22)
            errors.extend((.012 * (direction - goal.directions),
                           w_vec[:, None] * (vectors - goal.vectors), w_gap * (apertures - goal.apertures)))
            derivatives.extend((.012 * direction_jac, w_vec[:, None, None] * vjac, w_gap[:, None] * ajac))
        errors.append(.0003 * (full[indices] - self.robot.home[indices]))
        derivatives.append(.0003 * np.eye(len(indices)))
        return errors, derivatives

    def _clearance(self, relevant, indices):
        model, data = self.robot.model, self.robot.data
        a = self.robot.geom_a[relevant]
        b = self.robot.geom_b[relevant]
        desired = self.robot.clearance_m + .0015
        broad = np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b], axis=-1) - self.robot.radius[relevant]
        error = np.zeros(len(relevant))
        jac = np.zeros((len(relevant), len(indices)))
        points = np.zeros(6)
        first, second = np.zeros((3, self.width)), np.zeros((3, self.width))
        for k in np.flatnonzero((broad < desired) | self.robot.has_plane[relevant]):
            gap = mujoco.mj_geomDistance(model, data, int(a[k]), int(b[k]), desired, points)
            if gap >= desired:
                continue
            normal = points[3:] - points[:3]
            if gap < 0:
                normal = -normal
            if np.linalg.norm(normal) < 1e-9:
                normal = data.geom_xpos[b[k]] - data.geom_xpos[a[k]]
            normal /= max(np.linalg.norm(normal), 1e-9)
            mujoco.mj_jac(model, data, first, None, points[:3], int(model.geom_bodyid[a[k]]))
            mujoco.mj_jac(model, data, second, None, points[3:], int(model.geom_bodyid[b[k]]))
            error[k] = 18 * (desired - gap)
            jac[k] = -18 * normal @ (second - first)[:, self.robot.v_indices[indices]]
        return error, jac

    def solve(self, goals, sides, seed, *, clearance=False):
        indices = np.concatenate([np.arange(14 + 20 * side, 34 + 20 * side) for side in sides])
        fixed = self.robot.home.copy()
        low, high = self.robot.bounds[indices].T
        relevant = np.flatnonzero(np.any(self.guard.weights[:, indices] > 0, axis=1)) if clearance else None
        last_x = last = None

        def evaluate(x):
            nonlocal last_x, last
            if last_x is None or not np.array_equal(x, last_x):
                fixed[indices] = x
                self.robot.set(fixed)
                errors, derivatives = self._task(sides, goals, indices, fixed)
                if relevant is not None:
                    envelope, jacobian = self._clearance(relevant, indices)
                    errors.append(envelope)
                    derivatives.append(jacobian)
                last_x = x.copy()
                last = (np.concatenate([x.ravel() for x in errors]),
                        np.concatenate([x.reshape(-1, len(indices)) for x in derivatives]))
            return last

        result = least_squares(lambda x: evaluate(x)[0], np.clip(seed[indices], low, high),
                               jac=lambda x: evaluate(x)[1], bounds=(low, high), x_scale="jac",
                               ftol=1e-5, xtol=1e-5, gtol=1e-7, max_nfev=MAX_FIT_EVALUATIONS)
        fixed[indices] = result.x
        return fixed, {"evaluations": result.nfev, "status": result.status, "objective_m2": float(2 * result.cost),
                       "clearance_penalty_in_search": clearance}


def finite_margin(robot, low, high):
    coupling = np.maximum(robot.coupling * low, robot.coupling * high).sum(axis=1)
    return bool(np.isfinite(low).all() and np.isfinite(high).all() and
                np.all(low >= robot.bounds[:, 0] - 1e-9) and np.all(high <= robot.bounds[:, 1] + 1e-9)
                and np.all(coupling <= COUPLING_LIMIT + 1e-9))


def certify(robot, guard, candidate, *, splits=1):
    """Certify piecewise-linear Home-path boxes with all full-model pairs and limits."""
    endpoint = robot.legality(candidate)
    value = {"status": "UNKNOWN", "endpoint": endpoint, "end_q_rad": candidate.tolist(), "path": None}
    if not endpoint["feasible"] or endpoint["collision_clearance_m"] < robot.clearance_m:
        value["reason"] = "CANDIDATE_ENDPOINT_DENIED"
        return value
    deadline = monotonic_ns() + PATH_BUDGET_NS
    for first_group in (None, "THUMBS_FIRST", "NON_THUMBS_FIRST"):
        paths = [robot.home]
        if first_group is not None:
            thumb = np.r_[14:18, 34:38]
            waypoint = candidate.copy() if first_group == "NON_THUMBS_FIRST" else robot.home.copy()
            waypoint[thumb] = robot.home[thumb] if first_group == "NON_THUMBS_FIRST" else candidate[thumb]
            paths.append(waypoint)
        paths.append(candidate)
        path, worst_gap = [], None
        reason = "PERMIT"
        for start, end in zip(paths[:-1], paths[1:], strict=True):
            edges = max(1, int(np.ceil(np.max(np.abs(end - start)) / (PATH_EDGE_MAX_RAD / splits))))
            samples = np.linspace(start, end, edges + 1)
            for a, b in zip(samples[:-1], samples[1:], strict=True):
                low, high = np.minimum(a, b), np.maximum(a, b)
                if not finite_margin(robot, low, high):
                    reason = "LIMIT_OR_COUPLING"
                    break
                status = guard.prove(low, high, deadline)
                gap = guard.last_metrics["min_sample_gap_m"]
                if gap is not None:
                    worst_gap = gap if worst_gap is None else min(worst_gap, gap)
                if status != Reason.PERMIT:
                    reason = f"{status.name}:{guard.last_metrics['termination']}"
                    break
            if reason != "PERMIT":
                break
            path.append({"end_q_rad": end.tolist(), "axis_aligned_boxes": edges})
        if reason == "PERMIT":
            return {**value, "status": "SIGNED_HOME_PATH_PERMITTED", "reason": reason,
                    "path": {"waypoint_order": first_group or "DIRECT", "segments": path,
                             "minimum_observed_proof_sample_gap_m": worst_gap}}
        value["reason"] = reason
        if monotonic_ns() >= deadline:
            break
    return value


def read_archive(directory, baseline, source, geometry, window_plan):
    source_rows = [row for row in jsonl(source) if row["in_start_window"]]
    start = baseline["window"]["hmb_cutoff_start_ns"]
    stop = baseline["window"]["hmb_cutoff_stop_exclusive_ns"]
    proxies = [proxy for proxy in window_plan["proxy_evidence"] if proxy["status"] == "PLANNED_PROXY_PROBE"]
    near_rows = {(row["partition"], row["tick"], row["time_ns"]): row["right_thumb_to_index_mm"]
                 for row in geometry["raw_near_pinch_rows"]}
    rows, paired, recovered = [], 0, set()
    for part in baseline["partitions"]:
        fields = {}
        for name in FIELDS:
            path = directory / part["directory"] / f"{name}.npy"
            if file_sha(path) != part["arrays"][name]["sha256"]:
                raise ValueError(f"Frozen archive differs at {path}")
            fields[name] = np.load(path, allow_pickle=False)
        for index, cutoff in enumerate(fields["tick_ns"]):
            if not start <= cutoff < stop:
                continue
            row_index = len(rows)
            observed = source_rows[row_index]
            key = part["partition"], index, int(cutoff)
            if key != (observed["partition"], observed["partition_tick"], observed["tick_ns"]):
                raise ValueError(f"Original archive and HMB ledger clocks differ at {key}")
            raw = fields["raw_manus_local"][index]
            raw_mask = fields["raw_manus_local_mask"][index]
            frozen = fields["target_local_landmark_xyz"][index]
            frozen_mask = fields["target_local_landmark_xyz_mask"][index].all(axis=(1, 2))
            is_paired = bool(raw_mask.all() and frozen_mask.all() and fields["applied_substeps"][index] == 10)
            paired += is_paired
            raw_gap = float(np.linalg.norm(raw[1, 1, -1] - raw[1, 0, -1])) if raw_mask[1] else None
            legacy = near_rows.get(key)
            if (is_paired and raw_gap <= NEAR_M) != (legacy is not None):
                raise ValueError(f"Original raw near-pinch membership differs at {key}")
            if legacy is not None:
                if (abs(1000 * raw_gap - legacy["raw"]) > 1e-4 or
                        abs(1000 * np.linalg.norm(frozen[1, 1, -1] - frozen[1, 0, -1]) - legacy["both"]) > 1e-4):
                    raise ValueError(f"Raw or frozen score point differs from archived geometry at {key}")
                recovered.add(key)
            supports = [p for p in proxies if p["within_proxy_first_row"] <= row_index <
                        p["within_proxy_stop_row_exclusive"]]
            observed_shape = from_input(observed) if supports else None
            for proxy in supports:
                value = margin(proxy["gesture"], observed_shape, proxy["side"])
                if value is None or value < 0:
                    raise ValueError(f"Preregistered signed source gesture differs at {key}")
            rows.append({"key": key, "source_index": observed["source_index"], "raw": raw, "raw_mask": raw_mask,
                         "frozen": frozen, "frozen_mask": frozen_mask, "paired": is_paired,
                         "near_pinch": legacy is not None, "proxy": supports, "source_proxy_shape": observed_shape})
    if len(rows) != len(source_rows) or len(rows) != 4804 or paired != 4801 or len(recovered) != len(near_rows) or len(recovered) != 187:
        raise ValueError("This run does not preserve the frozen ROOT05 domain")
    return rows


def endpoints(robot, geometry, goals, sides, proxies):
    scores = {SIDES[side]: goals[side].errors(geometry.local[side]) for side in sides}
    shapes = [scores[side]["shape"] if side in scores else None for side in SIDES]
    return {"score_local_landmarks_m": geometry.local.tolist(), "score_world_landmarks_m": geometry.landmarks.tolist(),
            "home_wrist_xyz_m": geometry.wrists.tolist(), "home_wrist_rotation": geometry.rotations.tolist(),
            "score_tip_sites": [robot.model.site(i).name for i in robot.tip_ids],
            "scores_by_side": scores, "source_proxy_margins": {p["gesture"]: margin(p["gesture"], shapes, p["side"])
                                                               for p in proxies}}


def ledger_row(fit, row, order, witnesses):
    key = row["key"]
    sides = {1} if row["near_pinch"] else set()
    for proxy in row["proxy"]:
        sides.update(range(2) if proxy["side"] == "both" else [SIDES.index(proxy["side"])])
    sides = sorted(sides)
    qualified_sides = sides if row["paired"] else []
    value = {"partition": key[0], "partition_tick": key[1], "tick_ns": key[2], "start_window_row": order,
             "source_index": row["source_index"], "frozen_paired_mask": row["paired"],
             "raw_manus_mask": row["raw_mask"].tolist(), "frozen_target_mask": row["frozen_mask"].tolist(),
             "candidate_mask": [side in qualified_sides for side in range(2)],
             "legacy_raw_near_pinch_cohort": row["near_pinch"],
             "source_proxy_windows": [p["gesture"] for p in row["proxy"]], "source_proxy_shape": row["source_proxy_shape"],
             "HUMAN_FUNCTIONAL_INTENT": None, "FROZEN_ROBOT_MORPHOLOGY": None,
             "ROBOT_FUNCTIONAL_CANDIDATE": None, "KINEMATIC_WITNESS": None}
    if not qualified_sides:
        return value
    robot = fit.robot
    goals = {side: HandIntent.from_raw(row["raw"][side], fit.basis[side]) for side in qualified_sides}
    q, search = fit.solve(goals, qualified_sides, robot.home)
    actual = robot.geometry(q)
    value["HUMAN_FUNCTIONAL_INTENT"] = {SIDES[side]: goal.evidence() for side, goal in goals.items()}
    frozen_scores = {SIDES[side]: goals[side].errors(row["frozen"][side]) for side in qualified_sides}
    frozen_shapes = [frozen_scores[side]["shape"] if side in frozen_scores else None for side in SIDES]
    value["FROZEN_ROBOT_MORPHOLOGY"] = {
        "local_landmarks_m": row["frozen"].tolist(), "scores_by_side": frozen_scores,
        "source_proxy_margins": {p["gesture"]: margin(p["gesture"], frozen_shapes, p["side"]) for p in row["proxy"]}}
    value["ROBOT_FUNCTIONAL_CANDIDATE"] = {**endpoints(robot, actual, goals, qualified_sides, row["proxy"]),
                                           "q_rad": q.tolist(), "search": search}
    value["KINEMATIC_WITNESS"] = {"status": "NOT_SELECTED_FOR_PROOF"}
    if order not in witnesses:
        return value
    proof = certify(robot, fit.guard, q)
    attempts = [{"search": "INDEPENDENT_TASK_SPACE_FIT", "proof": proof}]
    if proof["status"] == "UNKNOWN":
        for seed_name, seed in (("TASK_SPACE_SEED", q), ("SIGNED_HOME_SEED", robot.home)):
            alternative, search = fit.solve(goals, qualified_sides, seed, clearance=True)
            proof = certify(robot, fit.guard, alternative, splits=2)
            attempts.append({"search": seed_name, "optimization": search, "proof": proof})
            if proof["status"] == "SIGNED_HOME_PATH_PERMITTED":
                q = alternative
                break
    value["KINEMATIC_WITNESS"] = {"status": proof["status"], "attempts": attempts}
    if proof["status"] == "SIGNED_HOME_PATH_PERMITTED":
        qualified = endpoints(robot, robot.geometry(q), goals, qualified_sides, row["proxy"])
        right = qualified["scores_by_side"].get("right")
        qualified["near_pinch_score_tolerance_met"] = (
            bool(right["tip_pair_aperture_m"][0] <= NEAR_M and
                 right["tip_pair_vector_error_m"][0] <= PINCH_VECTOR_TOLERANCE_M) if row["near_pinch"] else None)
        value["KINEMATIC_WITNESS"].update(qualified)
    return value


def distribution(values):
    array = np.asarray(values)
    return {"count": int(array.size), "min": float(array.min()), "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)), "max": float(array.max())} if array.size else {"count": 0}


def accumulate(value, counters, samples):
    near = value["legacy_raw_near_pinch_cohort"]
    names = [*value["source_proxy_windows"], *(["legacy_right_raw_near_pinch"] if near else [])]
    candidate = value["ROBOT_FUNCTIONAL_CANDIDATE"]
    witness = value["KINEMATIC_WITNESS"]
    for name in names:
        counters[name]["rows"] += 1
        if candidate is None:
            counters[name]["MASKED"] += 1
            continue
        counters[name][witness["status"]] += 1
        for layer, data in (("frozen", value["FROZEN_ROBOT_MORPHOLOGY"]), ("candidate", candidate), ("witness", witness)):
            for side, metrics in data.get("scores_by_side", {}).items():
                if name == "legacy_right_raw_near_pinch" and side == "left":
                    continue
                for key, numbers in (("thumb_index_aperture_mm", 1000 * metrics["tip_pair_aperture_m"][0]),
                                     ("thumb_index_vector_error_mm", 1000 * metrics["tip_pair_vector_error_m"][0]),
                                     ("median_segment_direction_error_deg", float(np.median(metrics["segment_direction_error_deg"]))),
                                     ("mean_aperture_error_mm", 1000 * float(np.mean(metrics["tip_pair_aperture_error_m"])))):
                    samples[name][layer][side][key].append(numbers)
                human = value["HUMAN_FUNCTIONAL_INTENT"][side]["raw_shape"]
                fitted = metrics["shape"]
                if human is not None and fitted is not None:
                    samples[name][layer][side]["mean_not_thumb_curl_absolute_error"].append(
                        float(np.mean(np.abs(np.asarray(fitted["curl"]) - human["curl"]))))
                    samples[name][layer][side]["outer_digit_spread_ratio_error"].append(
                        abs(fitted["spread_ratio"] - human["spread_ratio"]))
        if near and witness.get("near_pinch_score_tolerance_met"):
            counters[name]["LEGAL_NEAR_PINCH_SCORE_TOLERANCE_MET"] += 1
        for layer, data in (("frozen", value["FROZEN_ROBOT_MORPHOLOGY"]), ("candidate", candidate), ("witness", witness)):
            proxy_margin = data.get("source_proxy_margins", {}).get(name)
            if proxy_margin is not None and proxy_margin >= 0:
                counters[name][f"{layer.upper()}_SATISFIES_SOURCE_PROXY_RULE"] += 1


def main(args):
    baseline_path = args.baseline / "receipt.json"
    baseline = json.loads(baseline_path.read_bytes())
    geometry = json.loads(args.geometry.read_bytes())
    window_plan = json.loads(args.window_plan.read_bytes())
    identities = {name: file_sha(path) for name, path in {"baseline_receipt": baseline_path, "geometry": args.geometry,
                  "window_plan": args.window_plan, "input_ticks": args.input_ticks, "signed_model": args.model,
                  "model_receipt": args.model_receipt, "script": Path(__file__)}.items()}
    if (identities["geometry"] != GEOMETRY_SHA or identities["window_plan"] != WINDOW_SHA
            or identities["model_receipt"] != EXACT_MODEL_RECEIPT_SHA or identities["signed_model"] != SIGNED_SHA
            or identities["input_ticks"] != window_plan["input_sha256"]
            or identities["baseline_receipt"] != geometry["sha256"]["original_execution_receipt"]
            or any(entry["root"] != ROOT for entry in (baseline, geometry, window_plan))
            or any(baseline[key] != geometry[key] for key in ("full", "window"))):
        raise ValueError("Wrong frozen ROOT05 source, scoring archive, or signed model receipt")
    model = json.loads(args.model_receipt.read_bytes())["model_identity"]["signed"]
    robot = verified_robot(args.model, SIGNED_SHA, model)
    guard = SweptEnvelopeV2(robot, max_queries=PROOF_QUERIES, max_depth=PROOF_DEPTH)
    fit = FunctionalFit(robot, guard)
    rows = read_archive(args.baseline, baseline, args.input_ticks, geometry, window_plan)
    near = [i for i, row in enumerate(rows) if row["near_pinch"]]
    witnesses = {near[i] for i in np.linspace(0, len(near) - 1, args.pinch_proof_samples, dtype=int)}
    proxy_centers = {p["gesture"]: (p["within_proxy_first_row"] + p["within_proxy_stop_row_exclusive"] - 1) // 2
                     for p in window_plan["proxy_evidence"] if p["status"] == "PLANNED_PROXY_PROBE"}
    witnesses.update(proxy_centers.values())
    args.out.mkdir()
    ledger = args.out / "rows.jsonl.gz"
    counters = defaultdict(Counter)
    samples = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    with ledger.open("xb") as raw, gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0, compresslevel=3) as stream:
        for order, row in enumerate(rows):
            value = ledger_row(fit, row, order, witnesses)
            accumulate(value, counters, samples)
            stream.write((json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode())
    identities["ledger"] = file_sha(ledger)
    write_json(args.out / "summary.json", {
        "scope": "OFFLINE_SIGNED_MODEL_KINEMATIC_DIAGNOSTIC_ONLY", "root": ROOT,
        "formal_coverage_credit": False, "hardware_authorization": False, "simulated_physics_steps": 0,
        "signed_joint_order": robot.names, "signed_home_q_rad": robot.home.tolist(), "paired_timeline_rows": 4801,
        "frozen_timeline_rows": len(rows), "full": baseline["full"], "window": baseline["window"],
        "finger_order": CHAINS, "landmark_order": ["proximal", "PIP", "DIP", "score_tip"],
        "tip_pair_order": [[CHAINS[a], CHAINS[b]] for a, b in PAIRS],
        "intent_scope": "Signed MANUS source and proper rotation into model palm axes, at unit scale.",
        "candidate_scope": "Independent box-bounded task-space fit from signed Home; exact signed FK score sites.",
        "witness_scope": "Full 54-axis signed-model, constant Home arm/wrist axes, configured 0.5 mm clearance, named pairs, joint and drive bounds, wrist coupling and continuous V2 path boxes. No added contact exceptions.",
        "contact_scope": "Score-site distance and raw MANUS aperture describe geometry. The independent endpoint and swept results describe configured collider clearance. Historical contact and dynamic tracking are unobserved here.",
        "unknown_scope": "The finite search did not find or prove a witness for the requested row. It gives no global infeasibility conclusion.",
        "home_path_max_nominal_segment_rad": PATH_EDGE_MAX_RAD, "max_fit_evaluations": MAX_FIT_EVALUATIONS,
        "proof_query_limit_per_box": PROOF_QUERIES, "proof_depth_limit_per_box": PROOF_DEPTH,
        "near_pinch_score_threshold_m": NEAR_M, "near_pinch_vector_error_threshold_m": PINCH_VECTOR_TOLERANCE_M,
        "selected_witness_window_rows": sorted(witnesses), "source_proxy_center_rows": proxy_centers,
        "insufficient_source_proxy_windows": [p["gesture"] for p in window_plan["proxy_evidence"]
                                               if p["status"] != "PLANNED_PROXY_PROBE"],
        "counts_by_cohort": counters, "statistics_by_cohort": {
            key: {layer: {side: {name: distribution(series) for name, series in scores.items()} for side, scores in sides.items()}
                  for layer, sides in layers.items()} for key, layers in samples.items()}, "sha256": identities})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("baseline", "geometry", "window-plan", "input-ticks", "model", "model-receipt", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--pinch-proof-samples", type=int, default=PINCH_PROOF_SAMPLE_COUNT, choices=range(188),
                        metavar="0..187", help="Evenly spaced legacy near-pinch rows to attempt proving; all are fitted.")
    main(parser.parse_args())
