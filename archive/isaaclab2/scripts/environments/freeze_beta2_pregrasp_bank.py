"""Freeze native-qualified Beta 2 pregrasps and a local-shape train/eval split."""

import argparse
import hashlib
import json
from pathlib import Path
import random


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(admitted_paths, report_paths, candidate_paths, bank_path, manifest_path,
           *, split_seed=9104):
    if not (len(admitted_paths) == len(report_paths) == len(candidate_paths)):
        raise ValueError("Each admitted bank requires its matching report and candidate file")
    rows = []
    provenance = []
    for admitted_path, report_path, candidate_path in zip(
            admitted_paths, report_paths, candidate_paths):
        admitted = json.loads(admitted_path.read_text())
        report = json.loads(report_path.read_text())
        candidates = json.loads(candidate_path.read_text())
        candidate_seeds = {row["seed"] for row in candidates}
        passed = {row["seed"] for row in report["results"]
                  if row["pgs_compatible"] and row["active_contact_reachable"]}
        if (report.get("solver") != "pgs_mimic"
                or report.get("probe_order") != "active_first"
                or report.get("admitted") != len(admitted)
                or {row["seed"] for row in admitted} != passed
                or not passed <= candidate_seeds):
            raise ValueError("Native PGS report and admitted rows disagree")
        for row in admitted:
            meta = row["metadata"]
            if (row["validated"] or meta.get("asset_variant") != "beta2"
                    or meta.get("geometry_qualified") is not True
                    or meta.get("geometry_evidence", {}).get("native_physx", {}).get("solver_type") != "pgs"
                    or meta.get("phase1b_pgs_reach_evidence", {}).get("seed") != row["seed"]):
                raise ValueError("Unqualified Beta 2 pregrasp row")
        rows.extend(admitted)
        provenance.append({"admitted_sha256": digest(admitted_path),
                           "report_sha256": digest(report_path),
                           "candidate_sha256": digest(candidate_path),
                           "attempted": report["attempted"],
                           "admitted": report["admitted"]})
    if len(rows) < 4 or len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("Need at least four distinct qualified local seeds")
    if len({row["metadata"].get("right_natural_source_seed") for row in rows}) != 1:
        raise ValueError("This bank must declare its one natural source family")
    rows.sort(key=lambda row: row["seed"])
    seeds = [row["metadata"]["local_shape_seed"] for row in rows]
    if set(seeds) != {row["seed"] for row in rows}:
        raise ValueError("Local-shape seed identity must match the Reset row seed")
    shuffled = seeds[:]
    random.Random(split_seed).shuffle(shuffled)
    eval_count = max(1, len(seeds) // 4)
    eval_seeds = sorted(shuffled[:eval_count])
    train_seeds = sorted(set(seeds) - set(eval_seeds))
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    bank_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": "dexgym-beta2-local-pregrasp-split-v1",
        "solver": "pgs_mimic", "policy_hz": 50, "physx_hz": 500,
        "asset_variant": "beta2", "bank_sha256": digest(bank_path),
        "count": len(rows), "split_seed": split_seed,
        "split_key": "local_shape_seed",
        "train_local_shape_seeds": train_seeds,
        "evaluation_group_values_global": eval_seeds,
        "one_relative_template_family_only": True,
        "cross_family_generalization_claim_allowed": False,
        "world_pose_augmentation_is_not_a_new_local_family": True,
        "source_evidence": provenance,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admitted", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=9104)
    args = parser.parse_args()
    manifest = freeze(args.admitted, args.report, args.candidate,
                      args.bank, args.manifest, split_seed=args.split_seed)
    print(json.dumps({"count": manifest["count"],
                      "train_count": len(manifest["train_local_shape_seeds"]),
                      "evaluation_count": len(manifest["evaluation_group_values_global"]),
                      "bank_sha256": manifest["bank_sha256"]}))


if __name__ == "__main__":
    main()
