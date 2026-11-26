"""Read-only comparison of frozen M1 pinch variants on the signed Train HMB."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .mapping import IntentMapper, SMOOTH_BANDS_M, compiled_topologies, pinch_points, smooth_weights
from .schema import TICK_NS, file_sha, write_json


HARD = "M1_HARD_20"
BANDS = {f"M1_SMOOTH_{round(lo * 1000)}_{round(hi * 1000)}": (lo, hi) for lo, hi in SMOOTH_BANDS_M}
LINEAR_METRICS = ("pinch_vector_mm", "pinch_distance_mm", "local_rmse_mm", "cross_mm", "relative_wrist_mm")
FUNCTION_METRICS = (*LINEAR_METRICS, "relative_wrist_deg")


def compare_tick(mapper, wrist, palms, order):
    original = mapper.map(wrist, palms, order)
    before = original["unadjusted"]
    desired = original["desired_pinch"]
    distances = []
    for (root, indices), points in zip(order, palms, strict=True):
        human = points[indices, :3] - points[root, :3]
        distances.append(np.linalg.norm(human[1:, -1] - human[0, -1], axis=-1))
    variants = {HARD: original["hard_m1_audit"]}
    for name, band in BANDS.items():
        local = np.stack([pinch_points(points, wanted, smooth_weights(distance, band))
                          for points, wanted, distance in zip(before["local_landmark_xyz"], desired,
                                                              distances, strict=True)])
        variants[name] = mapper.place("M1", wrist, local)
    if not all(np.isfinite(value).all() for geometry in (before, *variants.values()) for value in geometry.values()):
        raise ValueError("nonfinite geometry in paired M1 mapping")
    if not np.isfinite(desired).all():
        raise ValueError("nonfinite independent pinch target")
    return original["pinch_active"], desired, before, variants


def functional_errors(target, before, desired, active):
    # Desired pinch is the P02 human vector scaled into the robot palm.
    # The non-pinch anchor is the same mapped geometry before correcting pinch.
    pinch = target["pinch_xyz"][active]
    wanted = desired[active]
    delta_local = target["local_landmark_xyz"] - before["local_landmark_xyz"]
    delta_cross = target["cross_xyz"] - before["cross_xyz"]
    rotation = Rotation.from_quat(before["relative_wrist_xyzw"]).inv() * Rotation.from_quat(
        target["relative_wrist_xyzw"])
    return {
        "pinch_vector_mm": 1000 * np.linalg.norm(pinch - wanted, axis=-1).max(),
        "pinch_distance_mm": 1000 * np.abs(np.linalg.norm(pinch, axis=-1) - np.linalg.norm(wanted, axis=-1)).max(),
        "local_rmse_mm": 1000 * np.sqrt(np.square(delta_local).sum(axis=-1).mean()),
        "cross_mm": 1000 * np.linalg.norm(delta_cross, axis=-1).max(),
        "relative_wrist_mm": 1000 * np.linalg.norm(target["relative_wrist_xyz"] - before["relative_wrist_xyz"]),
        "relative_wrist_deg": np.rad2deg(rotation.magnitude()),
    }


def new_samples():
    return {name: {
        "extra_step_mm": [],
        "function": {metric: [] for metric in FUNCTION_METRICS},
        "bone_ticks": 0,
        "bone_max_mm": np.zeros((2, 5, 3)),
    } for name in (HARD, *BANDS)}


def collect_partition(raw, meta, usable, mapper, samples, *, version_fields):
    counts = Counter(planned_ticks=len(usable), hmb_joint_valid_ticks=int(raw["joint_valid"].sum()),
                     eligible_ticks=int(usable.sum()), mapped_ticks=0, paired_mapping_invalid_ticks=0,
                     continuous_pairs=0, crossing_ticks=0,
                     entry_ticks=0, exit_ticks=0, entered_pairs=0, exited_pairs=0, multi_pair_ticks=0,
                     active_function_ticks=0, active_pairs=0)
    compiled = {}
    previous = None
    for i in np.flatnonzero(usable):
        key = tuple(raw["topology_id"][i, :2])
        try:
            if key not in compiled:
                compiled[key], _ = compiled_topologies(meta, key)
            active, desired, before, variants = compare_tick(mapper, raw["pose"][i, 2:4, 0],
                                                            raw["pose"][i, :2], compiled[key])
        except (ValueError, KeyError, IndexError) as error:
            counts["paired_mapping_invalid_ticks"] += 1
            counts[f"paired_mapping_failure_{type(error).__name__}"] += 1
            previous = None
            continue
        before_bone = np.linalg.norm(np.diff(before["local_landmark_xyz"], axis=2), axis=-1)
        extra_steps = {}
        if previous is not None:
            j, old_active, old_before, old_variants = previous
            adjacent = (i == j + 1 and raw["segment_id"][i] == raw["segment_id"][j]
                        and raw["tick_ns"][i] - raw["tick_ns"][j] == TICK_NS
                        and all(np.array_equal(raw[field][i, :4], raw[field][j, :4])
                                for field in version_fields))
            if adjacent:
                counts["continuous_pairs"] += 1
                entered = int((active & ~old_active).sum())
                left = int((old_active & ~active).sum())
                if entered or left:
                    counts.update({"crossing_ticks": 1, "entry_ticks": int(entered > 0),
                                   "exit_ticks": int(left > 0), "entered_pairs": entered, "exited_pairs": left,
                                   "multi_pair_ticks": int(entered + left > 1)})
                    base_step = before["world_tip_xyz"] - old_before["world_tip_xyz"]
                    for name, geometry in variants.items():
                        extra = (geometry["world_tip_xyz"] - old_variants[name]["world_tip_xyz"]) - base_step
                        extra_steps[name] = float(1000 * np.linalg.norm(extra, axis=-1).max())
        counts["mapped_ticks"] += 1
        active_tick = active.any()
        if active_tick:
            counts.update({"active_function_ticks": 1, "active_pairs": int(active.sum())})
        for name, geometry in variants.items():
            row = samples[name]
            if name in extra_steps:
                row["extra_step_mm"].append(extra_steps[name])
            bone = np.linalg.norm(np.diff(geometry["local_landmark_xyz"], axis=2), axis=-1)
            np.maximum(row["bone_max_mm"], 1000 * np.abs(bone - before_bone), out=row["bone_max_mm"])
            row["bone_ticks"] += 1
            if active_tick:
                for metric, value in functional_errors(geometry, before, desired, active).items():
                    row["function"][metric].append(float(value))
        previous = (i, active, before, variants)
    return counts


def percentile(values):
    return float(np.percentile(values, 95)) if values else None


def summarize(samples, active_roots):
    hard = samples[HARD]
    hard_function = {name: percentile(values) for name, values in hard["function"].items()}
    rows = {}
    for name, row in samples.items():
        steps = row["extra_step_mm"]
        p95, maximum = percentile(steps), max(steps, default=None)
        continuity = None if not steps else p95 <= 5 and maximum <= 15
        functional = {}
        for metric, values in row["function"].items():
            baseline, observed = hard_function[metric], percentile(values)
            limit = baseline + max(.05 * baseline, .25) if baseline is not None else None
            functional[metric] = {
                "n": len(values), "roots": active_roots, "hard_p95": baseline, "p95": observed, "limit": limit,
                "within_numeric_limit": observed <= limit if observed is not None and limit is not None else None,
                "adequate_support": len(values) >= 200 and active_roots >= 3,
            }
        bone_delta = row["bone_max_mm"] - hard["bone_max_mm"]
        valid_bones = bool(np.all(bone_delta <= .25)) if row["bone_ticks"] else None
        rows[name] = {
            "band_mm": [round(x * 1000) for x in BANDS[name]] if name in BANDS else None,
            "extra_world_tip_step_mm": {"n": len(steps), "p95": p95, "max": maximum,
                                       "within_numeric_limits": continuity},
            "function_on_hard_active_ticks": functional,
            "bone_ticks": row["bone_ticks"],
            "bone_max_change_mm_by_side_finger_bone": row["bone_max_mm"].tolist(),
            "bone_max_increase_over_hard_mm": float(bone_delta.max()) if row["bone_ticks"] else None,
            "bones_within_numeric_limit": valid_bones,
        }
    return rows


def audit(args):
    from ...hmb_reader import CONFIG_SHA, GEOMETRY, MANIFEST_SHA, PLAN_SHA, VERSION_FIELDS, HMBReader, checked_json
    from ...p02.model import RobotModel, asset_files
    from .audit_nominal_workspace import wrist_scope
    from .build import CAL_SHA, FIT_SHA, HANDOFF_SHA, MODEL_MANIFEST_SHA, MODEL_SHA, REGRESSION_SHA, scope_hashes

    docs = args.docs / "evidence"
    handoff = checked_json(docs / "P01_F/p02_handoff_manifest.json", HANDOFF_SHA)
    if args.products.resolve() != Path(handoff["ubuntu_products"]).resolve():
        raise ValueError("HMB archive does not match the signed handoff")
    if file_sha(args.model) != MODEL_SHA:
        raise ValueError("model XML does not match the signed P02 model")
    p02 = docs / "P02"
    signed_assets = checked_json(p02 / "refinement/model_assets_sha256.json", MODEL_MANIFEST_SHA)
    expected_assets = {name.split("/generated/")[-1]: digest for name, digest in signed_assets.items()}
    asset_root = args.model.resolve().parent
    observed_assets = {Path(name).relative_to(asset_root).as_posix(): digest
                       for name, digest in asset_files(args.model).items()}
    if expected_assets != observed_assets:
        raise ValueError("model assets do not match the signed P02 model")
    model = RobotModel(args.model)
    calibrated = checked_json(p02 / "mapping_contract.json", CAL_SHA)
    fit = checked_json(p02 / "refinement/fitted_candidates.json", FIT_SHA)
    mapper = IntentMapper(model.geometry(model.home), calibrated, fit)
    e1 = docs / "P01_E_E1"
    reader = HMBReader(e1 / "hmb_manifest.json", e1 / "production_plan.json",
                       docs / "P01_E_E0/candidate_config.json", args.products, "Train", GEOMETRY)
    samples, totals = new_samples(), Counter()
    partitions, roots, active_roots = [], set(), set()
    for key in reader.keys():
        raw, meta, refs, _ = reader.read(key)
        root = meta["identity"]["root_episode_id"]
        roots.add(root)
        denied = not scope_hashes(refs).isdisjoint(REGRESSION_SHA)
        counts = collect_partition(raw, meta, wrist_scope(raw, denied), mapper, samples, version_fields=VERSION_FIELDS)
        counts["regression_source_denied_ticks"] = counts["planned_ticks"] if denied else 0
        if counts["active_function_ticks"]:
            active_roots.add(root)
        totals.update(counts)
        partitions.append({"key": key, "root": root, "regression_source_denied": denied, **counts})

    signed = reader.manifest["counts"]["Train"]
    expected = (signed["source_count"], signed["counts"]["partitions"], signed["counts"]["planned_ticks"],
                signed["counts"]["causal.joint_valid"])
    if (len(roots), len(partitions), totals["planned_ticks"], totals["hmb_joint_valid_ticks"]) != expected:
        raise ValueError("paired diagnostic did not account for the full signed Train split")
    return {
        "scope": "TRAIN_M1_MAPPING_CONTINUITY_DIAGNOSTIC", "split": "Train", "performed_ik": False,
        "mapping_selected": False, "gate_authority": False,
        "scorer": {
            "mask": "Fixed hard distance <=20 mm for the active pinch pairs and functional ticks.",
            "pinch_reference": "P02 human desired vector transformed and scaled in the robot palm.",
            "other_reference": "Same unadjusted unit-scale M1 frame, with no pinch correction.",
            "cross_pairs": "All 25 V2 left-major pairs, scored by the worst applicable pair per tick.",
            "paired_geometry_complete": totals["paired_mapping_invalid_ticks"] == 0,
            "insufficient_crossings": not (totals["entry_ticks"] and totals["exit_ticks"]),
            "functional_support_reference": {"ticks": 200, "roots": 3},
        },
        "identity_sha256": {"hmb_manifest": MANIFEST_SHA, "hmb_plan": PLAN_SHA, "hmb_config": CONFIG_SHA,
                            "handoff": HANDOFF_SHA, "fit": FIT_SHA, "calibration": CAL_SHA, "model": MODEL_SHA,
                            "model_assets": MODEL_MANIFEST_SHA, "implementation": file_sha(Path(__file__))},
        "totals": dict(totals), "source_count": len(roots), "partitions": partitions,
        "variants": summarize(samples, len(active_roots)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=Path("docs/s0-5"))
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.out, audit(args))


if __name__ == "__main__":
    main()
