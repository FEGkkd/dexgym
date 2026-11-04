"""Generate unqualified local pregrasp proposals; PhysX admission is separate."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

from dexteroushands_lab.assets.bottle import BottleConfig
from dexteroushands_lab.scenes.grasp_init import grasp_sample_from_dict
from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.reset_bank import make_candidate
from scripts.environments.generate_pregrasp_reset_candidates import (
    combined_model, MjcfGeometryScreen,
)


TRIPOD = ("thumb", "index_finger", "middle_finger")


def eligible(metrics, *, level):
    """Conservative MJCF screen; native USD remains a separate required gate."""
    left = [row["gap_m"] for row in metrics["per_digit"]["l"].values()]
    right = [metrics["per_digit"]["r"][digit].get("sidewall_gap_m", math.inf)
             for digit in TRIPOD]
    sites = [metrics["right_fingertip_sites"][digit] for digit in TRIPOD]
    maximum = .005 if level == "A" else .007 if level == "B" else .012
    if (min(left + right) < -.0005 or max(left + right) > maximum
            or min(metrics["minimum_cross_digit_self_gap_m"].values()) <= 0
            or metrics["minimum_hand_hand_gap_m"] <= .003
            or metrics["right_thumb_tripod_site_opposition_rad"] < math.radians(125)
            or any(not -.0005 <= row["radial_gap_to_cap_sidewall_m"] <= .012
                   or abs(row["axial_rel_cap_m"]) > .0115 for row in sites)
            or any(metrics["per_object"][key]["minimum_gap_m"] < 0 for key in
                   ("l_proxy_cap", "r_proxy_bottle", "r_proxy_neck"))):
        return False
    if level == "A" and (min(left) > .001 or min(right) > .0015):
        return False
    if level == "B" and (min(left) < 0 or min(right) < 0):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-probe-input", type=Path)
    parser.add_argument("--per-level", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=12000)
    args = parser.parse_args()
    if args.per_level < 2:
        parser.error("--per-level must be at least two")
    root = Path(__file__).resolve().parents[2]
    stable = grasp_sample_from_dict(json.loads(
        (root / "docs/assets/phase1_mimic_pgs_grasp_init_bank.json").read_text())[0])
    naturals = [grasp_sample_from_dict(row) for row in json.loads(
        (root / "docs/assets/phase0_pregrasp_reset_bank.json").read_text())]
    bottle = BottleConfig()
    screen = MjcfGeometryScreen(combined_model(
        Path(os.environ["WUJI_DESCRIPTION_ROOT"]), bottle), bottle)
    rows = []
    for level_i, level in enumerate(("A", "B", "C")):
        by_source = {}
        for base_index, natural in enumerate(naturals):
            pool = []
            if level == "A":
                grid = [(li, ri, lb, rb)
                        for li in (0., .001, .002)
                        for ri in (0., .001, .002, .003)
                        for lb in (.1, .2, .3)
                        for rb in (0., .1, .2)]
            elif level == "B":
                grid = [(li, ri, lb, rb)
                        for li in (0., .001)
                        for ri in (0., .001, .002)
                        for lb in (0., .1)
                        for rb in (0., .1)]
            else:
                grid = [(0., 0., 0., 0.)]
            for grid_index, (li, ri, lb, rb) in enumerate(grid):
                seed = args.seed_start + level_i * 100000 + base_index * 1000 + grid_index
                sample = make_candidate(
                    stable, natural, level=level, seed=seed,
                    left_inward=li, right_inward=ri,
                    left_blend=lb, right_blend=rb)
                geometry = screen.check(sample)
                if eligible(geometry, level=level):
                    left_gap = max(row["gap_m"] for row in geometry["per_digit"]["l"].values())
                    right_gap = max(geometry["per_digit"]["r"][digit]["sidewall_gap_m"]
                                    for digit in TRIPOD)
                    pool.append((max(left_gap, right_gap), sample, geometry))
            if pool:
                by_source[base_index] = sorted(pool, key=lambda row: row[0])
        selected = []
        for rank in range(8):
            for base_index in sorted(by_source):
                if len(by_source[base_index]) > rank:
                    selected.append(by_source[base_index][rank])
                if len(selected) >= args.per_level:
                    break
            if len(selected) >= args.per_level:
                break
        if len(selected) < args.per_level:
            raise RuntimeError(f"Only {len(selected)} MJCF-screened {level} candidates")
        for _, sample, geometry in selected[:args.per_level]:
            geometry = {**geometry,
                        "phase0_strict_qualified": geometry["qualified"],
                        "qualified": True,
                        "phase1b_criterion": "near_or_weak_contact_safe_mjcf_v1"}
            row = asdict(sample)
            row["metadata"]["geometry_evidence"] = {"official_mjcf": geometry}
            rows.append(row)
    payload = json.dumps(rows, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    if args.native_probe_input:
        wrapper = {"status": "MJCF_SCREEN_ONLY_NOT_NATIVE_USD_QUALIFIED",
                   "accepted": [{"sample": row,
                                 "mjcf_geometry": row["metadata"]["geometry_evidence"]["official_mjcf"]}
                                for row in rows]}
        args.native_probe_input.parent.mkdir(parents=True, exist_ok=True)
        args.native_probe_input.write_text(json.dumps(wrapper, indent=2) + "\n")
    print(json.dumps({"status": "UNQUALIFIED_PROPOSALS", "count": len(rows),
                      "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
