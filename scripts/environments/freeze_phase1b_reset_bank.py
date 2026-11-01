"""Freeze PGS-qualified local preshape seeds and a source-separated eval split."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.reset_split import split_group_indices


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admitted", type=Path, required=True)
    parser.add_argument("--pgs-report", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.admitted.read_text())
    report = json.loads(args.pgs_report.read_text())
    if not rows or report.get("probe_order") != "active_first" or report.get("admitted") != len(rows):
        raise ValueError("Bank requires a complete active-first PGS probe")
    seeds = [row["seed"] for row in rows]
    if len(set(seeds)) != len(rows):
        raise ValueError("Duplicated local hand-shape seed")
    for row in rows:
        meta = row["metadata"]
        evidence = meta["phase1b_pgs_reach_evidence"]
        physx = meta["geometry_evidence"]["native_physx"]
        if (row["validated"] or meta.get("geometry_qualified") is not True
                or meta.get("right_probe_joint_target") is not None
                or evidence["seed"] != row["seed"]
                or not evidence["pgs_compatible"]
                or not evidence["active_contact_reachable"]
                or evidence["zero_action"]["first_bilateral_contact_step"] >= 0
                or evidence["active_contact_probe"]["first_bilateral_contact_step"] <= 0
                or physx.get("solver_type") != "pgs"):
            raise ValueError(f"Unqualified candidate in bank: {row['seed']}")
    args.bank.parent.mkdir(parents=True, exist_ok=True)
    args.bank.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    # Level A was frozen before any PPO. Preserve its original source split
    # when later curriculum levels introduce new source seeds, and reserve
    # those new seeds for evaluation so they cannot leak into A evaluation.
    primary = [row for row in rows if row["metadata"]["phase1b_reset_level"] == "A"]
    primary_wrapped = [SimpleNamespace(metadata=row["metadata"]) for row in primary]
    primary_eval = split_group_indices(primary_wrapped, key="right_natural_source_seed",
                                       evaluation=True)
    primary_sources = {row["metadata"]["right_natural_source_seed"] for row in primary}
    eval_sources = {primary[i]["metadata"]["right_natural_source_seed"] for i in primary_eval}
    eval_sources.update(row["metadata"]["right_natural_source_seed"] for row in rows
                        if row["metadata"]["right_natural_source_seed"] not in primary_sources)
    splits = {}
    for level in ("A", "B"):
        level_rows = [row for row in rows if row["metadata"]["phase1b_reset_level"] == level]
        wrapped = [SimpleNamespace(metadata=row["metadata"]) for row in level_rows]
        train = split_group_indices(wrapped, key="right_natural_source_seed",
                                    evaluation=False, evaluation_groups=eval_sources)
        evaluation = split_group_indices(wrapped, key="right_natural_source_seed",
                                         evaluation=True, evaluation_groups=eval_sources)
        splits[level] = {
            "train_local_shape_seeds": [level_rows[i]["seed"] for i in train],
            "evaluation_local_shape_seeds": [level_rows[i]["seed"] for i in evaluation],
            "train_natural_source_seeds": sorted({level_rows[i]["metadata"][
                "right_natural_source_seed"] for i in train}),
            "evaluation_natural_source_seeds": sorted({level_rows[i]["metadata"][
                "right_natural_source_seed"] for i in evaluation}),
        }
        if set(splits[level]["train_natural_source_seeds"]) & set(
                splits[level]["evaluation_natural_source_seeds"]):
            raise RuntimeError("Natural source leaked across partitions")
    manifest = {
        "schema": "dexgym-phase1b-frozen-reset-split-v2",
        "solver": "pgs_mimic", "policy_hz": 50, "physx_hz": 500,
        "one_relative_template_family_only": True,
        "world_pose_augmentation_is_not_a_new_local_family": True,
        "bank_path": str(args.bank), "bank_sha256": sha(args.bank),
        "candidate_sha256": sha(args.candidate_file),
        "pgs_report_sha256": sha(args.pgs_report),
        "count": len(rows), "splits": splits,
        "evaluation_natural_source_seeds_global": sorted(eval_sources),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"count": len(rows), "bank_sha256": manifest["bank_sha256"],
                      "level_counts": {level: {
                          "train": len(value["train_local_shape_seeds"]),
                          "evaluation": len(value["evaluation_local_shape_seeds"])}
                          for level, value in splits.items()}}))


if __name__ == "__main__":
    main()
