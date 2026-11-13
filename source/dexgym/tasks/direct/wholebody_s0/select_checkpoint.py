"""Select a checkpoint from complete native MuJoCo validation runs."""

import argparse
import json
import math
from pathlib import Path

from .bag import digest
from .prepare import write_json
from .residual import REWARD_WEIGHTS


def tracking_cost(result):
    metrics = result["all_measured"]
    if metrics is None:
        return math.inf
    scales = {
        "wrist_mm": ("wrist_position", 22.),
        "wrist_deg": ("wrist_rotation", math.degrees(.14)),
        "relative_mm": ("relative_position", 33.),
        "relative_deg": ("relative_rotation", math.degrees(.20)),
    }
    cost = sum(REWARD_WEIGHTS[group] * metrics[metric]["p95"] / scale
               for metric, (group, scale) in scales.items())
    return cost + sum(REWARD_WEIGHTS[side + "_hand"] * metrics["local_tip_mm_sides"][side]["p95"] / 12.
                      for side in ("left", "right"))


def select(path):
    rows = json.loads(path.read_text())
    groups, signature = {}, None
    for row in rows:
        if row["split"] != "validation":
            raise ValueError("Checkpoint selection requires the validation split")
        key = tuple(row[name] for name in ("reference_manifest_sha256", "model_sha256", "seed", "clips", "episodes"))
        if signature is None:
            signature = key
        if key != signature:
            raise ValueError("Validation cases use different evaluation contracts")
        if row["checkpoint"] is not None:
            profiles = groups.setdefault(row["checkpoint"], {})
            if row["robust"] in profiles:
                raise ValueError("Duplicate validation case")
            profiles[row["robust"]] = row
    candidates = []
    for checkpoint, profiles in groups.items():
        if profiles.keys() != {False, True}:
            raise ValueError("A checkpoint must have nominal and stress validation")
        nominal, stress = profiles[False], profiles[True]
        cost = .5 * (tracking_cost(nominal) + tracking_cost(stress))
        if math.isfinite(cost):
            candidates.append({
                "checkpoint": checkpoint, "complete_total": nominal["complete"] + stress["complete"],
                "complete_nominal": nominal["complete"], "complete_stress": stress["complete"],
                "measured_intervals_total": nominal["measured_intervals"] + stress["measured_intervals"],
                "tracking_cost": cost,
            })
    if not candidates:
        raise ValueError("No checkpoint produced finite validation tracking results")
    candidates.sort(key=lambda item: (-item["complete_total"], -item["complete_stress"],
                                      -item["measured_intervals_total"], item["tracking_cost"], item["checkpoint"]))
    chosen = Path(candidates[0]["checkpoint"]).resolve(strict=True)
    return {"validation_results": str(path.resolve()), "validation_sha256": digest(path),
            "checkpoint": str(chosen), "checkpoint_sha256": digest(chosen), "candidates": candidates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = select(args.validation)
    write_json(args.output, result)
    print(result["checkpoint"], flush=True)


if __name__ == "__main__":
    main()
