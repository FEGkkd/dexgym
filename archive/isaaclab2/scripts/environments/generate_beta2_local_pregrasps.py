"""Generate reproducible local hand-shape variants for native Beta 2 screening.

The output is unqualified. A PhysX PGS Mimic probe must admit each Reset row.
"""

import argparse
import copy
import json
from pathlib import Path
import random


def perturb_local_hand_shape(source: dict, *, count: int, seed: int,
                             wrist_limit_m: float = 0.0005,
                             joint_limit_rad: float = 0.01) -> list[dict]:
    if count < 2 or wrist_limit_m <= 0 or joint_limit_rad <= 0:
        raise ValueError("Positive local perturbation limits and at least two rows are required")
    if source["validated"] or source["metadata"].get("asset_variant") != "beta2":
        raise ValueError("Source must be an unvalidated Beta 2 candidate")
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        row = copy.deepcopy(source)
        row["seed"] = seed * 1000 + index
        offsets = {}
        for side in ("left", "right"):
            state = row[side]
            xyz = [rng.uniform(-wrist_limit_m, wrist_limit_m) for _ in range(3)]
            dq = [rng.uniform(-joint_limit_rad, joint_limit_rad) for _ in range(20)]
            for field in ("root_pose", "wrist_target_pose"):
                for axis in range(3):
                    state[field][axis] += xyz[axis]
            for field in ("joint_position", "joint_target"):
                for joint in range(20):
                    state[field][joint] += dq[joint]
            offsets[side] = {"wrist_xyz_m": xyz, "joint_delta_rad": dq}
        meta = row["metadata"]
        meta["local_shape_seed"] = row["seed"]
        meta["geometry_qualified"] = False
        meta.pop("phase1b_pgs_reach_evidence", None)
        meta.pop("geometry_evidence", None)
        meta["beta2_local_perturbation"] = {
            "parent_candidate_seed": source["seed"],
            "rng_seed": seed, "index": index, "offsets": offsets,
            "same_relative_template_family": True,
        }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent-seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--wrist-limit-m", type=float, default=0.0005)
    parser.add_argument("--joint-limit-rad", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = next((row for row in json.loads(args.source.read_text())
                   if row["seed"] == args.parent_seed), None)
    if source is None:
        raise ValueError("Parent candidate seed was not found")
    rows = perturb_local_hand_shape(
        source, count=args.count, seed=args.seed,
        wrist_limit_m=args.wrist_limit_m, joint_limit_rad=args.joint_limit_rad)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"count": len(rows), "parent_seed": args.parent_seed,
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
