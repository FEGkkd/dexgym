#!/usr/bin/env python3
"""Recompute sampled DCR-008 geometry from archived commands in signed CPU MuJoCo."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from audit_dcr008_candidate import SCOPE, STATUS, digest, read_json, require
from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel, asset_files
from dexgym.tasks.direct.wholebody_s05.v1.task.native_sweep import NativeSweep


MODEL_SHA = "939b3651e2426e596eed482e70a68c2382720df1fd55b124aa5f0893b640e92d"
ASSETS_SHA = "92c44cabe65b278ab8261c2659c2cfa8c670fad07fd7d3a640e8629bc5c06a0d"
ROOT_QUOTA = 16
CAUSE_QUOTA = 16
WITNESS_QUOTA = 64
POSITION_TOLERANCE_M = 1e-6  # The trace stores the M1 target in float32.
ANGLE_TOLERANCE_RAD = 1e-5
SEED = 49


@dataclass(frozen=True)
class Row:
    root: str
    partition: str
    trace: Path
    index: int

    def identity(self):
        return {"root": self.root, "partition": self.partition, "row": self.index}


def signed_robot(model, assets, identity):
    require(identity["model"] == MODEL_SHA and digest(model) == MODEL_SHA, "unexpected signed model XML")
    require(identity["model_assets"] == ASSETS_SHA and digest(assets) == ASSETS_SHA, "unexpected asset manifest")
    expected = {name.split("/generated/")[-1]: value for name, value in read_json(assets).items()}
    actual = {Path(name).relative_to(model.parent).as_posix(): value for name, value in asset_files(model).items()}
    require(actual == expected, "signed model dependency bytes differ")
    return RobotModel(model)


def rank(row):
    return sha256(f"{SEED}\0{row.root}\0{row.partition}\0{row.index}".encode()).digest()


def add_sample(samples, key, rows, quota):
    selected = samples[key]
    selected.extend(rows)
    selected.sort(key=rank)
    del selected[quota:]


def sample_rows(directory, summary):
    root_samples, causes, witnesses = (defaultdict(list) for _ in range(3))
    support = {"rows_with_score": Counter(), "rows_without_score": Counter(), "failure_causes": Counter(),
               "failure_causes_with_score": Counter(), "witnesses": Counter()}
    trace_hashes, partitions = {}, set()
    for split in summary["split_results"].values():
        for item in split["partitions"]:
            root, partition = item["root"], item["partition"]
            path = (directory / item["trace"]).resolve()
            require(path.is_relative_to(directory) and path.suffix == ".npz", f"invalid trace path for {partition}")
            require(path not in trace_hashes and partition not in partitions, "duplicate trace or partition")
            require(read_json(path.with_suffix(".json")) == item, f"partition sidecar differs for {partition}")
            require(digest(path) == item["trace_sha256"], f"trace hash differs for {partition}")
            trace_hashes[path] = item["trace_sha256"]
            partitions.add(partition)
            with np.load(path, allow_pickle=False) as recorded:
                scored = recorded["score_valid"] & recorded["mapping_valid"] & recorded["provider_observed"]
                state, why, proved = recorded["row_class"], recorded["cause"], recorded["witness_ok"]
            require(len(scored) == item["counts"]["planned"], f"trace size differs for {partition}")
            require(np.isin(state, tuple(map(int, STATUS))).all(), f"unknown row status in {partition}")
            support["rows_with_score"][root] += int(scored.sum())
            support["rows_without_score"][root] += int((~scored).sum())
            support["witnesses"][root] += int(proved.sum())
            require(not (proved & (~scored | (state > 2))).any(), f"unqualified witness in {partition}")
            rows = [Row(root, partition, path, i) for i in range(len(scored))]
            add_sample(root_samples, root, (rows[i] for i in np.flatnonzero(scored)), ROOT_QUOTA)
            add_sample(witnesses, root, (rows[i] for i in np.flatnonzero(proved)), WITNESS_QUOTA)
            for code in (3, 4, 5):
                for reason in np.unique(why[state == code]):
                    category = f"{STATUS[str(code)]}:{reason or 'NO_RECORDED_REASON'}"
                    mask = (state == code) & (why == reason)
                    support["failure_causes"][category] += int(mask.sum())
                    support["failure_causes_with_score"][category] += int((mask & scored).sum())
                    add_sample(causes, category, (rows[i] for i in np.flatnonzero(mask)), CAUSE_QUOTA)
    return root_samples, causes, witnesses, support, trace_hashes


def choose_witnesses(per_root):
    selected = []
    for round_index in range(WITNESS_QUOTA):
        for root in sorted(per_root):
            if len(per_root[root]) > round_index:
                selected.append(per_root[root][round_index])
                if len(selected) == WITNESS_QUOTA:
                    return selected
    return selected


def tip_error(geometry, target):
    return float(np.linalg.norm(geometry.tips - target, axis=-1).max())


def functional_errors(geometry, target, pinch_mask):
    left, right = geometry.wrists
    left_rotation, right_rotation = geometry.rotations
    relative_xyz = left_rotation.T @ (right - left)
    relative_rotation = Rotation.from_matrix(left_rotation.T @ right_rotation)
    rotation_error = (Rotation.from_quat(target["relative_wrist_xyzw"]).inv() * relative_rotation).magnitude()
    pinch = geometry.local[:, 1:, -1] - geometry.local[:, :1, -1]
    cross = geometry.tips[1, None, :, :] - geometry.tips[0, :, None, :]
    return {
        "local_p95_m": float(np.percentile(np.linalg.norm(geometry.local - target["local_landmark_xyz"], axis=-1), 95)),
        "pinch_max_m": float(np.max(np.linalg.norm(pinch - target["pinch_xyz"], axis=-1)[pinch_mask], initial=0.)),
        "cross_max_m": float(np.linalg.norm(cross - target["cross_xyz"], axis=-1).max()),
        "relative_translation_m": float(np.linalg.norm(relative_xyz - target["relative_wrist_xyz"])),
        "relative_rotation_rad": float(rotation_error),
    }


def check_witness(robot, sweep, fields, index, nominal_geometry, nominal_error):
    q = fields["q_nominal"][index]
    witness = fields["witness_q"][index]
    delta = np.abs(witness - q)
    require(np.isfinite(witness).all() and (delta <= np.deg2rad(np.repeat([8., 5.], [14, 40])) + 1e-9).all(),
            "witness violates the frozen per-axis residual envelope")
    require(robot.legality(q)["feasible"] and robot.legality(witness)["feasible"], "illegal witness or nominal endpoint")
    require(sweep.permits(q, witness), "nominal-to-witness continuous path lacks a clearance proof")
    target = {name: fields[f"mapped_{name}"][index] for name in (
        "world_tip_xyz", "local_landmark_xyz", "pinch_xyz", "cross_xyz", "relative_wrist_xyz", "relative_wrist_xyzw",
    )}
    geometry = robot.geometry(witness)
    actual = tip_error(geometry, target["world_tip_xyz"])
    disagreement = abs(actual - float(fields["witness_error_m"][index]))
    require(disagreement <= POSITION_TOLERANCE_M, "archived witness max-tip error differs from CPU FK")
    needed = max(.002, .10 * nominal_error)
    require(nominal_error - actual >= needed - POSITION_TOLERANCE_M, "witness lacks the frozen absolute/relative improvement")
    pinch_mask = fields["pinch_active"][index]
    before = functional_errors(nominal_geometry, target, pinch_mask)
    after = functional_errors(geometry, target, pinch_mask)
    margins = {name: before[name] + (np.deg2rad(1.) if name.endswith("_rad") else .002) - after[name]
               for name in before}
    require(all(margin >= -(ANGLE_TOLERANCE_RAD if name.endswith("_rad") else POSITION_TOLERANCE_M)
                for name, margin in margins.items()), "witness violates a function gate")
    return {"max_arm_delta_deg": float(np.rad2deg(delta[:14].max())),
            "max_hand_delta_deg": float(np.rad2deg(delta[14:].max())),
            "nominal_error_m": nominal_error, "witness_error_m": actual, "required_improvement_m": needed,
            "witness_recompute_difference_m": disagreement, "functional_margins_m_or_rad": margins}


def examine(robot, root_samples, cause_samples, witnesses, trace_hashes):
    sweep = NativeSweep(robot)
    by_trace = defaultdict(set)
    for row in [*witnesses, *(r for samples in root_samples.values() for r in samples),
                *(r for samples in cause_samples.values() for r in samples)]:
        by_trace[row.trace].add(row)
    witness_set = set(witnesses)
    scores, checked_witnesses, missing = {}, {}, []
    for path, samples in sorted(by_trace.items()):
        require(digest(path) == trace_hashes[path], f"trace changed after sampling: {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            fields = {name: archive[name] for name in (
                "score_valid", "mapping_valid", "provider_observed", "q_nominal", "mapped_world_tip_xyz",
                "nominal_error_m", "witness_q", "witness_error_m", "witness_ok", "pinch_active", "cause", "row_class",
                "mapped_local_landmark_xyz", "mapped_pinch_xyz", "mapped_cross_xyz", "mapped_relative_wrist_xyz",
                "mapped_relative_wrist_xyzw",
            )}
        for row in sorted(samples, key=lambda sample: sample.index):
            i = row.index
            if not (fields["score_valid"][i] and fields["mapping_valid"][i] and fields["provider_observed"][i]):
                missing.append(dict(row.identity(), status=STATUS[str(fields["row_class"][i])], cause=str(fields["cause"][i])))
                continue
            try:
                geometry = robot.geometry(fields["q_nominal"][i])
                error = tip_error(geometry, fields["mapped_world_tip_xyz"][i])
                difference = abs(error - float(fields["nominal_error_m"][i]))
                require(difference <= POSITION_TOLERANCE_M, "archived nominal max-tip error differs from CPU FK")
                if fields["row_class"][i] <= 2:
                    require(robot.legality(fields["q_nominal"][i])["feasible"], "active nominal endpoint is illegal")
                scores[row] = difference
                if row in witness_set:
                    require(fields["witness_ok"][i], "archived witness status changed")
                    checked_witnesses[row] = check_witness(robot, sweep, fields, i, geometry, error)
            except ValueError as error:
                raise ValueError(f"{row.root}/{row.partition} row {i}: {error}") from error
    return scores, checked_witnesses, missing


def audit(directory, model, assets, *, allow_bounded=False):
    path = directory / "candidate_comparison.json"
    summary = read_json(path)
    inputs = read_json(directory / "candidate_inputs.json")
    require(summary["schema"] == "s05.dcr008.nominal_evaluation.v1", "unexpected evidence schema")
    require(summary["scope"] == inputs["scope"] == SCOPE and not summary["official_admission"], "unexpected scope")
    require(summary["identity_sha256"] == inputs["identity_sha256"], "summary and run inputs disagree")
    require(summary["full_comparison"] or allow_bounded, "the full paired comparison is absent")
    robot = signed_robot(model, assets, summary["identity_sha256"])
    roots, causes, per_root_witnesses, support, digests = sample_rows(directory, summary)
    witnesses = choose_witnesses(per_root_witnesses)
    scores, checked, missing = examine(robot, roots, causes, witnesses, digests)
    root_checked = {root: sum(row in scores for row in rows) for root, rows in roots.items()}
    return {
        "scope": "INDEPENDENT_DCR008_STATIC_MUJOCO_GEOMETRY",
        "summary_sha256": digest(path), "model_sha256": MODEL_SHA, "model_assets_sha256": ASSETS_SHA,
        "mujoco": mujoco.__version__, "sample_seed": SEED, "trace_files_hashed": len(digests),
        "rows_checked": len(scores), "root_quota": ROOT_QUOTA, "per_root_checked": root_checked,
        "root_stratified_checked": sum(root_checked.values()),
        "root_quota_met": len(root_checked) == 32 and all(count >= ROOT_QUOTA for count in root_checked.values()),
        "max_nominal_recompute_difference_m": max(scores.values(), default=None),
        "failure_cause_sample_counts": {name: len(rows) for name, rows in causes.items()},
        "failure_samples_without_geometry": missing,
        "available_support": support,
        "witness_quota": WITNESS_QUOTA, "witnesses_checked": len(checked),
        "witness_quota_met": len(checked) >= WITNESS_QUOTA,
        "witness_sample": [dict(row.identity(), **checking) for row, checking in checked.items()],
        "physics_steps": 0, "actual_servo_qualification": False, "official_admission": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path(__file__).resolve().parents[2] /
                        "docs/s0-5/evidence/P02/refinement/model_assets_sha256.json")
    parser.add_argument("--allow-bounded", action="store_true")
    args = parser.parse_args()
    result = audit(args.directory.resolve(), args.model.resolve(), args.assets.resolve(), allow_bounded=args.allow_bounded)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
