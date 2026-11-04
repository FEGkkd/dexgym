"""Rebind Beta 1 pregrasp geometry as unqualified Beta 2 PhysX candidates.

This does not transfer any Beta 1 contact or reachability certificate. Run
validate_phase1b_pregrasp_pgs.py on the output before using it as a Reset bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dexteroushands_lab.assets.wuji_hand2_beta import load_hand2_spec


def rebind_candidates(rows: list[dict], source_sha256: str, left, right) -> list[dict]:
    if len(source_sha256) != 64 or not rows:
        raise ValueError("A nonempty source bank and its SHA-256 are required")
    output = []
    for source in rows:
        if source.get("validated") or source["metadata"].get("asset_variant") != "beta1":
            raise ValueError("Only unvalidated official Beta 1 pregrasps can be rebound")
        row = json.loads(json.dumps(source))
        meta = row["metadata"]
        meta["asset_variant"] = "beta2"
        meta["asset_source_commit"] = left.source_commit
        meta["left_usd_sha256"] = left.usd_sha256
        meta["right_usd_sha256"] = right.usd_sha256
        meta["left_asset_tree_sha256"] = left.asset_tree_sha256
        meta["right_asset_tree_sha256"] = right.asset_tree_sha256
        meta["geometry_qualified"] = False
        meta["hold_validated"] = False
        meta.pop("beta1_tip_marker_mass_override_kg", None)
        meta.pop("phase1b_pgs_reach_evidence", None)
        meta.pop("geometry_evidence", None)
        meta["beta2_candidate_provenance"] = {
            "source_bank_sha256": source_sha256,
            "source_local_shape_seed": row["seed"],
            "beta1_physics_evidence_discarded": True,
            "requires_native_beta2_pgs_requalification": True,
        }
        output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    rows = json.loads(raw)
    left = load_hand2_spec(variant="beta2", side="left")
    right = load_hand2_spec(variant="beta2", side="right")
    candidates = rebind_candidates(rows, hashlib.sha256(raw).hexdigest(), left, right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(candidates, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"count": len(candidates), "output": str(args.output),
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
