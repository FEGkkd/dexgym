#!/usr/bin/env python3
"""Verify the common-yaw root-five diagnostic and measure Home-to-START rotations."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_dcr009_guarded_diagnostic import read, rows
from replay_dcr009_guarded_diagnostic import align_pair_yaw, registered_targets
from replay_home_anchored_pilot import read_frames
from summarize_dcr009_segmented_start import goal_difference, inspect


def actual_home(attempt):
    home, = [row for row in rows(attempt / "evidence/trace.jsonl") if row["kind"] == "home_feedback"]
    return np.asarray(home["anatomical_palm_world"])


def target_angles(home, frames, native):
    home_tcp = home @ frames[5:7]
    world_goal = frames[1:3] @ np.asarray(native)
    change = home_tcp[:, :3, :3].transpose(0, 2, 1) @ world_goal[:, :3, :3]
    return np.degrees(Rotation.from_matrix(change).magnitude())


def compare(unaligned_path, aligned_path, frames_path, anchor_path):
    previous, prev_bins, prev_end, previous_input, prev_anchor, prev_guard = inspect(unaligned_path, "segmented")
    actual, run_bins, run_end, actual_input, run_anchor, run_guard = inspect(aligned_path, "segmented")
    base_out = read(unaligned_path / "result.json")
    yaw_out = read(aligned_path / "result.json")
    anchor = read(anchor_path)
    if (not previous_input or previous_input != actual_input or prev_guard != run_guard
            or base_out.get("align_start_yaw_to_home") or yaw_out.get("align_start_yaw_to_home") is not True
            or any(item["source_cutoff_ns"] != anchor["source_cutoff_ns"] or item["inputs"] != anchor["inputs"]
                   or item["center_offset_world_m"] != [0, 0, 0] for item in (prev_anchor, run_anchor))):
        raise ValueError("Paired yaw audit requires matching source, zero offset and guarded native profile")
    frames = read_frames(frames_path)
    home = actual_home(aligned_path)
    prev_home = actual_home(unaligned_path)
    raw = np.asarray(anchor["registered_wrist_start_h"])
    locked, unaligned, native_before = registered_targets(frames, home, raw)
    expected, aligned, yaw = align_pair_yaw(locked, home, unaligned)
    native_after = np.linalg.inv(frames[1:3]) @ aligned @ frames[3:5] @ frames[5:7]
    for field, matrix in (("shared_transform", expected[0]), ("unaligned_registered_wrist", unaligned),
                          ("start_world_registered_wrist", aligned), ("runtime_native_goal", native_after),
                          ("unaligned_native_goal", native_before)):
        if not np.allclose(run_anchor[field], matrix, atol=1e-7, rtol=0):
            raise ValueError(f"The observed native yaw run did not use the derived {field}")
    if abs(run_anchor["yaw_alignment"]["yaw_deg"] - yaw["yaw_deg"]) > 1e-7:
        raise ValueError("The signed yaw differs from the archived calculation")
    old = np.asarray(prev_anchor["runtime_native_goal"])
    actual_before = target_angles(prev_home, frames, old)
    same_home_before = target_angles(home, frames, native_before)
    after = target_angles(home, frames, native_after)
    relative = lambda pair: np.linalg.inv(pair[0]) @ pair[1]
    hip = home[1, :2, 3] - home[0, :2, 3]
    result_line = aligned[1, :2, 3] - aligned[0, :2, 3]
    heading = np.degrees(np.arctan2(abs(hip[0] * result_line[1] - hip[1] * result_line[0]), hip @ result_line))
    gap = lambda pair: float(np.linalg.norm(pair[1, :3, 3] - pair[0, :3, 3]) * 1000)
    pivot_delta = float(np.linalg.norm(unaligned[:, :3, 3].mean(axis=0) - aligned[:, :3, 3].mean(axis=0)) * 1000)
    relative_delta = float(np.max(abs(relative(unaligned) - relative(aligned))))
    if max(abs(gap(unaligned) - gap(aligned)), relative_delta, heading, pivot_delta) > 1e-8:
        raise ValueError("The symmetric z-axis alignment did not preserve its required invariants")
    same_native_baseline = goal_difference(prev_anchor, {"runtime_native_goal": native_before.tolist()}, "runtime_native_goal")
    cutoff = min(prev_end, run_end)
    common = [index for index in range(1, cutoff + 1) if index in prev_bins and index in run_bins]
    return {
        "scope": "DIAGNOSTIC_ONLY", "root": anchor["root"], "center_offset_mm": 0,
        "formal_coverage_credit": False, "same_two_model_sweep_guard": True, "yaw_alignment": yaw,
        "geometry": {"remaining_heading_mismatch_deg": float(heading), "midpoint_change_mm": pivot_delta,
                     "gap_before_mm": gap(unaligned), "gap_after_mm": gap(aligned),
                     "relative_se3_max_element_delta": relative_delta},
        "home_to_engineering_tcp_rotation_deg": {
            "side_order": ["left", "right"], "unaligned_actual_run": actual_before.tolist(),
            "unaligned_using_aligned_run_home": same_home_before.tolist(), "aligned_actual_run": after.tolist(),
            "yaw_only_reduction_with_same_home": (same_home_before - after).tolist(),
        },
        "original_target_agreement_across_measured_home": same_native_baseline,
        "common_time_support": {"cutoff_s": cutoff * .02, "bins": len(common)},
        "unaligned": previous, "aligned": actual,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("unaligned", "aligned", "frames", "anchor", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.unaligned, args.aligned, args.frames, args.anchor)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
