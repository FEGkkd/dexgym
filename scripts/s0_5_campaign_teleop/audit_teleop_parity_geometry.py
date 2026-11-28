#!/usr/bin/env python3
"""Decompose original hand intent and compare the two verified MuJoCo models."""

import argparse
import json
from pathlib import Path

import numpy as np

from audit_grouped_rmb_diagnostic import read_arrays
from audit_root_five_geometry_layers import stats
from dexgym.tasks.direct.wholebody_s05.p02.intent import hand_basis, unit
from dexgym.tasks.direct.wholebody_s05.v1.rmb.schema import file_sha, write_json
from swept_live_guard import verified_robot


def gaps(points):
    return 1000 * np.linalg.norm(points[:, 1:, -1] - points[:, :1, -1], axis=-1)


def reconstruct(home, model_basis, raw):
    geometry = {}
    lengths = np.linalg.norm(np.diff(home.local, axis=2), axis=-1)
    human_basis = np.stack([hand_basis(p[1:, 0]) for p in raw])
    rotation = model_basis @ human_basis.transpose(0, 2, 1)
    in_model_basis = np.einsum("sij,sfkj->sfki", rotation, raw)
    raw_steps = np.diff(in_model_basis, axis=2)
    directions = unit(raw_steps)
    origin = home.local[:, :, :1]
    origin_delta = origin - in_model_basis[:, :, :1]
    geometry["raw"] = gaps(raw)
    geometry["same_bones_model_origins"] = gaps(in_model_basis + origin_delta)
    geometry["model_bones_same_origins"] = gaps(np.concatenate((
        in_model_basis[:, :, :1], in_model_basis[:, :, :1] + np.cumsum(directions * lengths[..., None], axis=2)), axis=2))
    rebuilt = np.concatenate((origin, origin + np.cumsum(directions * lengths[..., None], axis=2)), axis=2)
    geometry["both"] = gaps(rebuilt)
    return geometry, rebuilt


def model_comparison(signed, native, samples):
    if tuple(signed.names) != tuple(native.names):
        raise ValueError("Models differ in named actuator order")
    template = {"same_actuator_aliases": [x.name for x in signed.actuators] == [x.name for x in native.actuators],
                "same_hand_ctrlrange": bool(np.array_equal([x.ctrlrange for x in signed.actuators[14:]],
                                                           [x.ctrlrange for x in native.actuators[14:]])),
                "signed_collision_pairs": len(signed.collision_pairs), "native_collision_pairs": len(native.collision_pairs)}
    for key, field in (("joint_axis", "axis"), ("joint_range", "range")):
        a = np.stack([getattr(j, field) for j in signed.joints])
        b = np.stack([getattr(j, field) for j in native.joints])
        template[key] = {"max_abs_difference": float(abs(a - b).max()),
                         "different_axes": [name for name, differs in zip(signed.names, np.any(a != b, axis=1)) if differs]}
    template["joint_zero"] = {
        "same_qpos0": bool(np.array_equal(signed.model.qpos0[signed.q_indices], native.model.qpos0[native.q_indices])),
        "home_difference_deg_max": float(np.rad2deg(abs(signed.home - native.home)).max()),
    }
    local, wrists, tips = [], [], []
    for q in samples:
        if (np.any(q < np.maximum(signed.bounds[:, 0], native.bounds[:, 0]) - 1e-7)
                or np.any(q > np.minimum(signed.bounds[:, 1], native.bounds[:, 1]) + 1e-7)):
            continue
        g, h = signed.geometry(q), native.geometry(q)
        local.append(1000 * np.linalg.norm(g.local[:, :, -1] - h.local[:, :, -1], axis=-1))
        wrists.append(1000 * np.linalg.norm(g.wrists - h.wrists, axis=-1))
        tips.append(1000 * np.linalg.norm(g.tips - h.tips, axis=-1))
    return {**template, "same_numeric_q_samples_proposed": len(samples), "local_tip_delta_mm": stats(local),
            "wrist_delta_mm": stats(wrists), "world_tip_delta_mm": stats(tips)}


def main(args):
    previous = json.loads((args.baseline / "receipt.json").read_bytes())
    evidence = json.loads(args.model_receipt.read_bytes())["model_identity"]
    signed = verified_robot(args.model, evidence["signed"]["xml_sha256"], evidence["signed"])
    native = verified_robot(args.native_model, evidence["live"]["xml_sha256"], evidence["live"])
    home = signed.geometry(signed.home)
    basis = np.stack([hand_basis(x[1:, 0]) for x in home.local])
    first = previous["window"]["hmb_cutoff_start_ns"]
    stop = previous["window"]["hmb_cutoff_stop_exclusive_ns"]
    modes = ("raw", "same_bones_model_origins", "model_bones_same_origins", "both")
    result = {mode: [] for mode in modes}
    same = 0.
    examples = []
    q_samples = [signed.home]
    states = ("tick_ns", "raw_manus_local", "raw_manus_local_mask", "target_local_landmark_xyz",
              "target_local_landmark_xyz_mask", "applied_substeps", "q_measured")
    for part in previous["partitions"]:
        rows = read_arrays(args.baseline, part, states)
        for i, tick in enumerate(rows["tick_ns"]):
            if (not first <= tick < stop or rows["applied_substeps"][i] != 10
                    or not rows["raw_manus_local_mask"][i].all() or not rows["target_local_landmark_xyz_mask"][i].all()):
                continue
            g, built = reconstruct(home, basis, rows["raw_manus_local"][i])
            same = max(same, float(abs(built - rows["target_local_landmark_xyz"][i]).max()))
            for mode in modes:
                result[mode].append(g[mode])
            if (len(result["raw"]) - 1) % 80 == 0:
                q_samples.append(rows["q_measured"][i])
            if g["raw"][1, 0] <= 20:
                examples.append({"partition": part["partition"], "tick": i, "time_ns": int(tick),
                                 "right_thumb_to_index_mm": {mode: float(g[mode][1, 0]) for mode in modes}})
    if len(result["raw"]) != 4801 or same > 3e-7 or len(examples) != 187:
        raise ValueError("Frozen ROOT05 geometry domain or operation chain differs")
    right_near = {mode: int(np.count_nonzero(np.asarray(rows)[:, 1, 0] <= 20)) for mode, rows in result.items()}
    write_json(args.out, {"scope": "DIAGNOSTIC_ONLY", "root": previous["root"], "paired_rows": 4801,
               "full": previous["full"], "window": previous["window"],
               "max_position_reconstruction_error_m": same, "same_threshold_mm": 20,
               "right_thumb_index_at_most_20mm": right_near,
               "thumb_to_each_other_finger_mm_by_side": {mode: stats(rows) for mode, rows in result.items()},
               "raw_near_pinch_rows": examples, "signed_vs_native_geometry": model_comparison(signed, native, q_samples),
               "model_identities": {name: identity["xml_sha256"] for name, identity in evidence.items()},
               "sha256": {"script": file_sha(Path(__file__)), "model_receipt": file_sha(args.model_receipt),
                           "original_execution_receipt": file_sha(args.baseline / "receipt.json")}})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for item in ("baseline", "model", "native-model", "model-receipt", "out"):
        p.add_argument("--" + item, type=Path, required=True)
    main(p.parse_args())
