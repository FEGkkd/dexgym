"""MJCF-screen hybrid local pregrasps before PGS Mimic reach testing."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path

from dexteroushands_lab.assets.bottle import BottleConfig
from dexteroushands_lab.scenes.grasp_init import grasp_sample_from_dict, _reframe_sample
from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.reset_bank import make_hybrid_candidate
from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.mjcf_tripod_fit import fit_right_tripod
from scripts.environments.generate_pregrasp_reset_candidates import combined_model, MjcfGeometryScreen

TRIPOD = ("thumb", "index_finger", "middle_finger")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-level", type=int, default=16)
    args = parser.parse_args()
    if args.per_level < 2:
        parser.error("--per-level must be at least two")
    root = Path(__file__).resolve().parents[2]
    source = grasp_sample_from_dict(json.loads((root /
        "docs/assets/phase1_mimic_pgs_grasp_init_bank.json").read_text())[0])
    stable = _reframe_sample(source, (0., 0., .5, 0., 0., 0., 1.))
    stable = replace(stable,
        bottle=replace(stable.bottle, root_pose=(0., 0., .5, 0., 0., 0., 1.)),
        validated=True, validation=source.validation)
    naturals = [grasp_sample_from_dict(row) for row in json.loads((root /
        "docs/assets/phase0_pregrasp_reset_bank.json").read_text())]
    bottle = BottleConfig()
    screen = MjcfGeometryScreen(combined_model(
        Path(os.environ["WUJI_DESCRIPTION_ROOT"]), bottle), bottle)
    rows = []
    for level_i, level in enumerate(("A", "B")):
        pool = {}
        for source_index, natural in enumerate(naturals):
            options = []
            blends = (.03, .06) if level == "A" else (.10, .15)
            for lb in blends:
                for lo in (0., .0002, .0004, .0006):
                    for ri in (0., .001, .002, .003):
                        seed = 45000 + level_i * 100000 + source_index * 1000 + len(options)
                        sample = make_hybrid_candidate(
                            stable, natural, seed=seed, left_open_blend=lb,
                            left_outward_m=lo, right_inward_m=ri, level=level)
                        sample = fit_right_tripod(
                            screen, sample,
                            site_radial_gap_targets_m=(
                                (.0003, .0030, .0035) if level == "A"
                                else (.0015, .0040, .0045)))
                        probe = fit_right_tripod(
                            screen, sample,
                            site_radial_gap_targets_m=(
                                (.0001, .0025, .0025) if level == "A"
                                else (.0003, .0030, .0035)))
                        sample = replace(sample, metadata={
                            **sample.metadata,
                            "right_probe_joint_target": list(probe.right.joint_position)})
                        geometry = screen.check(sample)
                        left = [row["gap_m"] for row in geometry["per_digit"]["l"].values()]
                        right = [geometry["per_digit"]["r"][digit].get(
                            "sidewall_gap_m", math.inf) for digit in TRIPOD]
                        sites = [geometry["right_fingertip_sites"][digit] for digit in TRIPOD]
                        safe = (
                            min(left) > -.001 and max(left) < (.0015 if level == "A" else .0035)
                            and min(right) > -.0005 and max(right) < (.003 if level == "A" else .005)
                            and min(geometry["minimum_cross_digit_self_gap_m"].values()) > 0
                            and geometry["minimum_hand_hand_gap_m"] > .003
                            and geometry["right_thumb_tripod_site_opposition_rad"] >= math.radians(125)
                            and all(-.0005 <= row["radial_gap_to_cap_sidewall_m"] <= .012
                                    and abs(row["axial_rel_cap_m"]) < .0115 for row in sites)
                            and all(geometry["per_object"][key]["minimum_gap_m"] >= 0
                                    for key in ("l_proxy_cap", "r_proxy_bottle", "r_proxy_neck")))
                        if safe:
                            geometry = {**geometry,
                                "phase0_strict_qualified": geometry["qualified"],
                                "qualified": True,
                                "phase1b_criterion": "recoverable_left_contact_opposed_right_tripod_v1"}
                            options.append((max(right), sample, geometry))
            if options:
                pool[source_index] = sorted(options, key=lambda row: row[0])
        selected = []
        for rank in range(8):
            for source_index in sorted(pool):
                if len(pool[source_index]) > rank:
                    selected.append(pool[source_index][rank])
                if len(selected) >= args.per_level:
                    break
            if len(selected) >= args.per_level:
                break
        if len(selected) < args.per_level:
            raise RuntimeError(f"Only {len(selected)} {level} hybrid MJCF candidates")
        for _, sample, geometry in selected[:args.per_level]:
            row = asdict(sample)
            row["metadata"]["geometry_evidence"] = {"official_mjcf": geometry}
            rows.append(row)
    payload = json.dumps(rows, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(json.dumps({"status": "MJCF_ONLY_UNQUALIFIED_NATIVE_AND_PGS", "count": len(rows),
                      "sha256": hashlib.sha256(payload.encode()).hexdigest()}))


if __name__ == "__main__":
    main()
