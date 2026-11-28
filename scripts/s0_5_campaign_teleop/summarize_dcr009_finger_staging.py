#!/usr/bin/env python3
"""Audit the root-five wrist dwell and right-ring motion against native guard receipts."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np

from audit_dcr009_guarded_diagnostic import read, rows
from summarize_dcr009_segmented_start import compare


STEP = re.compile(
    r"S05_DCR009_FINGER_STEP stage=(\d+) side=(left|right) command=(\d+) proof=(\d+) "
    r"published=([01]) reason=(\d+) delta_rad=(\S+)"
)
RIGHT_RING = {46, 47, 48, 49}
STAGES = {
    1: ("right", 3), 2: ("right", 0), 3: ("right", 1), 4: ("right", 2), 5: ("right", 4),
    6: ("left", 0), 7: ("left", 1), 8: ("left", 2), 9: ("left", 3), 10: ("left", 4),
}


def summarize(baseline, staged):
    result = read(staged / "result.json")
    if result.get("wrist_first_fingers") is not True or result.get("align_start_yaw_to_home") is not True:
        raise ValueError("The staged attempt is not labeled as wrist-first and yaw-aligned")
    paired = compare(baseline, staged, "segmented")
    if not paired["original_goal_equality"]["within_tolerance"] or not paired["same_two_model_guard_configuration"]:
        raise ValueError("The paired source, yaw target or unchanged sweep configuration did not verify")
    trace = list(rows(staged / "evidence/trace.jsonl"))
    events = [row for row in trace if "finger" in row["kind"] or "dwell" in row["kind"]]
    guard = {row["sequence"]: row for row in rows(staged / "sweep-trace.jsonl")}
    attempts, other_axes, pre_wrist_motion = [], [], []
    stage_counts = Counter()
    first_arm_fail = next((row for row in guard.values() if row["group"] == 0 and row["reason"] != "PERMIT"), None)
    with (staged / "sim.log").open(encoding="utf-8", errors="replace") as log:
        for line in log:
            m = STEP.search(line)
            if not m:
                continue
            stage, side, command, proof, published, native_reason, max_delta = m.groups()
            stage, proof = int(stage), int(proof)
            stage_counts[f"{stage}/{side}"] += 1
            record = guard.get(proof)
            if not record or "issued_measured_proposed_rad" not in record:
                continue
            issued, measured, proposed = np.asarray(record["issued_measured_proposed_rad"])
            changed = set(np.flatnonzero(abs(proposed - issued) > 1e-12).tolist())
            if stage == 0 and changed:
                pre_wrist_motion.append(proof)
            selection = STAGES.get(stage)
            if selection and selection[0] == side:
                start = (14 if side == "left" else 34) + 4 * selection[1]
                allowed = set(range(start, start + 4))
            else:
                allowed = set()
            if changed - allowed:
                other_axes.append({"proof": proof, "stage": stage, "unexpected_indices": sorted(changed - allowed)})
            if stage == 1 and side == "right" and changed:
                attempts.append({
                    "native_command": int(command), "proof": proof, "published": published == "1",
                    "native_reason": int(native_reason), "server_reason": record["reason"],
                    "axes": sorted(changed), "max_step_rad": float(max_delta),
                    "issued_rad": issued[46:50].tolist(), "measured_rad": measured[46:50].tolist(),
                    "proposed_rad": proposed[46:50].tolist(),
                })
    if other_axes or pre_wrist_motion:
        raise ValueError(f"An isolated step exceeded its allowed axes: {other_axes}, before wrist: {pre_wrist_motion}")
    positive = [row for row in attempts if row["published"]]
    denied = [row for row in attempts if not row["published"]]
    return {
        "scope": "DIAGNOSTIC_ONLY", "paired": paired, "wrist_finger_events": events, "native_stage_counts": dict(stage_counts),
        "right_ring": {"controlled_indices": sorted(RIGHT_RING),
                       "changed_axis_opportunities": len(attempts), "published": len(positive),
                       "first_movement_proposal": attempts[0] if attempts else None,
                       "first_denial": denied[0] if denied else None,
                       "last_published": positive[-1] if positive else None,
                       "last_denial": denied[-1] if denied else None},
        "first_failed_arm_sweep": ({"sequence": first_arm_fail["sequence"], "reason": first_arm_fail["reason"]}
                                    if first_arm_fail else None),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "staged", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.baseline, args.staged)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
