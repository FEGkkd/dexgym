"""Verify actual admitted RMB references at the native environment boundary."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from ...p02.model import RobotModel
from ..rmb.reader import RobotReferenceBank
from .physics import ServoProfile, signature
from .reference import RmbReference
from .smoke import make_task


def first_window(partition, length):
    values = partition.arrays
    legal = values["domain_nominal_legal"] & values["domain_mapped_valid"] & values["inputs_safety_permission"]
    generation = values["audit_generation"]
    segment = values["audit_segment"]
    start = None
    for tick, good in enumerate(legal):
        if not good:
            start = None
        elif start is None or generation[tick] != generation[start] or segment[tick] != segment[start]:
            start = tick
        if start is not None and tick - start + 1 >= length:
            return start
    return None


def roll_out(bank, robot, *, train, roots, frames, device):
    profile = ServoProfile()
    reference = RmbReference(bank, device=device, train=train)
    by_root = {}
    for partition in reference.partitions.values():
        by_root.setdefault(partition.root, []).append(partition)
    task, _, _ = make_task(robot, 1, profile, "B0", device)
    checks = []
    for root, partitions in sorted(by_root.items()):
        if len([row for row in checks if row.get("scheduled")]) >= roots:
            break
        witness = next(((p, start) for p in partitions if (start := first_window(p, frames)) is not None), None)
        if witness is None:
            checks.append({"root": root, "scheduled": False, "reason": "no_contiguous_signed_window"})
            continue
        partition, start = witness
        baseline = reference.current([(partition.key, start)])
        command = baseline.nominal
        ids = torch.arange(1, device=device)
        try:
            task.reset(ids, command, torch.zeros_like(command))
        except ValueError as error:
            checks.append({"root": root, "scheduled": True, "partition": partition.key,
                           "first": int(start), "planned": frames, "physical": 0,
                           "reason": str(error), "pass": False})
            continue
        record = {"root": root, "scheduled": True, "partition": partition.key, "first": int(start),
                  "planned": frames, "physical": 0, "fully_valid": 0, "events": []}
        for i in range(start, start + frames):
            target = reference.current([(partition.key, i)])
            obs = task.observe(target)
            if i == start and obs[0, :-task.builder.frame_dim].count_nonzero():
                raise RuntimeError("First RMB observation had unapproved history")
            transition = task.step(torch.zeros_like(command))
            record["physical"] += int(transition.substeps[0] > 0)
            record["fully_valid"] += int(transition.valid[0])
            if transition.events[0]:
                record["events"].append({"row": i, "flags": transition.events[0].item(),
                                         "actual_substeps": transition.substeps[0].item()})
            if transition.terminated[0]:
                break
        record["pass"] = record["fully_valid"] == frames
        checks.append(record)
    signed_count = sum(row["scheduled"] for row in checks)
    return {"scope": "qualified RMB native/warp connection and actual B0 physical smoke",
            "environment_model": signature(robot, profile), "split": reference.split,
            "minimum_sources": roots, "frames_per_source": frames, "source_count": signed_count,
            "physical_transitions": sum(row.get("physical", 0) for row in checks), "rows": checks,
            "pass": signed_count == roots and all(row.get("pass") for row in checks if row["scheduled"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmb", type=Path, required=True)
    parser.add_argument("--rmb-sha", required=True)
    parser.add_argument("--rmb-review", required=True, type=Path)
    parser.add_argument("--rmb-review-sha", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("Train", "Validation"), default="Train")
    parser.add_argument("--roots", type=int, default=16)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args()
    if args.roots < 1 or args.frames < 1:
        raise ValueError("Expected positive source count and per-source sample size")
    bank = RobotReferenceBank(args.rmb, expected_sha=args.rmb_sha, review_receipt=args.rmb_review,
                              expected_review_sha=args.rmb_review_sha)
    result = roll_out(bank, RobotModel(args.model), train=args.split == "Train", roots=args.roots,
                      frames=args.frames, device=args.device)
    result["rmb_sha256"] = args.rmb_sha
    result["rmb_review_sha256"] = args.rmb_review_sha
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"pass={result['pass']} sha256={sha256(args.out.read_bytes()).hexdigest()}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
