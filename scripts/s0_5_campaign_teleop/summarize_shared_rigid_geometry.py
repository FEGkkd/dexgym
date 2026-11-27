#!/usr/bin/env python3
"""Aggregate paired historical A and shared-rigid B simulator geometry on the HMB grid."""

import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from hmb_cohort import EXPECTED_SPLITS, IDENTITIES, validate_partitions
from pair_nominal_hmb import DCR_COMPARISON_SHA, digest, normalized_assets, signed_feedback_names
from dexgym.tasks.direct.wholebody_s05.p02.model import FINGERS, RobotModel


VALID_TICKS = {"Train": 85_382, "Validation": 29_150}
GEOMETRY = (
    "wrist_tcp_gap_mm", "right_wrist_in_left_x_mm", "right_wrist_in_left_y_mm", "right_wrist_in_left_z_mm",
    "relative_wrist_so3_deg", "palm_origins_gap_mm", "palm_centres_gap_mm", "right_minus_left_palm_height_mm",
    "right_minus_left_palm_foreaft_mm", "acute_palm_plane_deg", "directed_palm_normals_deg",
    "closest_cross_hand_tip_mm",
)
M1 = ("frozen_m1_max_wrist_position_error_mm", "frozen_m1_max_wrist_rotation_error_deg")
COHERENT = "COHERENT_MEASURED"
AVAILABILITY = {(True, True): "common", (True, False): "A_only",
                (False, True): "B_only", (False, False): "neither"}


def verified_cohort(path):
    raw = path.read_bytes()
    cohort = json.loads(raw)
    if (cohort.get("schema") != "s05.teleop.hmb_source_cohort.v1"
            or cohort.get("native_assets_sha256_checked") is not True
            or cohort.get("identity_sha256") != {name: identity[1] for name, identity in IDENTITIES.items()}):
        raise ValueError("Expected the native-asset-verified full HMB cohort")
    roots = cohort["selected_roots"]
    splits, keys = defaultdict(Counter), set()
    for root in roots:
        partitions = root["hmb_partitions"]
        validate_partitions(root, partitions)
        new = {part["key"] for part in partitions}
        if len(new) != len(partitions) or keys & new:
            raise ValueError("Repeated HMB partition")
        keys.update(new)
        splits[root["split"]].update(roots=1, partitions=len(partitions), planned=root["planned_ticks"])
    actual = {s: (row["roots"], row["partitions"], row["planned"]) for s, row in splits.items()}
    if actual != EXPECTED_SPLITS or len({root["root"] for root in roots}) != len(roots):
        raise ValueError("The comparison needs all 32 signed roots without changing split denominators")
    return roots, sha256(raw).hexdigest()


def pair_files(base, root, cohort_sha, model_sha, joint_pairs):
    identifier = root["root"].split(":")[-1]
    folders = [base / prefix for prefix in (identifier, f"root-{identifier}")
               if (base / prefix / "pair_summary.json").is_file()]
    if not folders:
        return None
    if len(folders) != 1:
        raise ValueError(f"Ambiguous paired summary for {root['root']}")
    directory, = folders
    summary_path = directory / "pair_summary.json"
    raw = summary_path.read_bytes()
    meta = json.loads(raw)
    paths = [directory / name for name in ("paired_ticks.jsonl", "paired_ticks.jsonl.gz")
             if (directory / name).is_file()]
    max_age = meta.get("max_age_ms")
    if (meta.get("schema") != "s05.teleop.hmb_nominal_pair.v1" or meta.get("root") != root["root"]
            or meta.get("split") != root["split"] or meta.get("cohort_sha256") != cohort_sha
            or meta.get("dcr_comparison_sha256") != DCR_COMPARISON_SHA
            or meta.get("simulator_reported_model_sha256") != model_sha
            or meta.get("feedback_joint_pairs_in_signed_order", joint_pairs) != joint_pairs
            or type(max_age) not in (int, float) or not 0 < max_age < 1000
            or len(paths) != 1):
        raise ValueError(f"Pair receipt does not bind the cohort, model and named joint order: {summary_path}")
    return paths[0], meta["output_ticks_sha256"], sha256(raw).hexdigest(), max_age


def paired_rows(files, root):
    path, expected_sha, _, _ = files
    content = sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as source:
        previous = -1
        for part in root["hmb_partitions"]:
            for tick in range(part["planned_ticks"]):
                line = source.readline()
                if not line.endswith(b"\n"):
                    raise ValueError(f"Incomplete planned tick: {path} {part['key']} {tick}")
                content.update(line)
                row = json.loads(line)
                cutoff = row["hmb_cutoff_ns"]
                if (row["root"] != root["root"] or row["split"] != root["split"] or row["partition"] != part["key"]
                        or row["tick"] != tick or row["hmb_tick_ns"] != cutoff or type(cutoff) is not int
                        or cutoff <= previous or type(row["hmb_joint_valid"]) is not bool):
                    raise ValueError(f"Invalid signed chronological row: {path} {part['key']} {tick}")
                previous = cutoff
                yield row
        if source.readline() or content.hexdigest() != expected_sha:
            raise ValueError(f"Unexpected trailing rows or paired tick digest mismatch: {path}")


def state(row):
    if row is None:
        return None, "MISSING_ROOT_PAIR_ARTIFACT"
    reason, value = row["replay_feedback"], row["replay_measured_q_rad"]
    if reason != COHERENT and value is None:
        return None, reason
    q = np.asarray(value, dtype=np.float64)
    if reason != COHERENT or q.shape != (54,) or not np.isfinite(q).all():
        raise ValueError("Measured feedback status disagrees with the named 54-axis state")
    return q, reason


def unit(vectors):
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(lengths <= 1e-12) or not np.isfinite(lengths).all():
        raise ValueError("Degenerate palm landmark geometry")
    return vectors / lengths


def geometry(robot, q):
    actual = robot.geometry(q)
    wrist_world = actual.wrists[1] - actual.wrists[0]
    wrist_relative = actual.rotations[0].T @ wrist_world
    wrist_angle = Rotation.from_matrix(actual.rotations[0].T @ actual.rotations[1]).magnitude()
    knuckles = actual.landmarks[:, 1:, 0]
    centre = (actual.palm + knuckles.sum(axis=1)) / 5
    palm_delta = centre[1] - centre[0]
    forward = unit(knuckles.mean(axis=1) - actual.palm)
    across = knuckles[:, 0] - knuckles[:, -1]
    lateral = unit(across - (across * forward).sum(axis=1, keepdims=True) * forward)
    normal = np.cross(forward, lateral) * np.array([1., -1.])[:, None]
    dot = float(np.clip(normal[0] @ normal[1], -1., 1.))
    cross = np.linalg.norm(actual.tips[0, :, None] - actual.tips[1, None, :], axis=-1)
    left, right = np.unravel_index(cross.argmin(), cross.shape)
    values = np.array([
        np.linalg.norm(wrist_world) * 1000, *(wrist_relative * 1000), np.degrees(wrist_angle),
        np.linalg.norm(actual.palm[1] - actual.palm[0]) * 1000, np.linalg.norm(palm_delta) * 1000,
        palm_delta[2] * 1000, -palm_delta[1] * 1000, np.degrees(np.arccos(abs(dot))),
        np.degrees(np.arccos(dot)), cross[left, right] * 1000,
    ])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite geometry from coherent replay feedback")
    return values, f"{FINGERS[left]}/{FINGERS[right]}"


def m1_wrist(row):
    pos, rot = (row.get(f"online_m1_wrist_{name}") for name in ("pos_max_m", "rot_max_rad"))
    if pos is None or rot is None:
        return None
    if (row.get("wrist_comparability") != "MEASURED_CONFIGURATION_WORLDFRAME_FK_MATCH"
            or not np.isfinite([pos, rot]).all() or min(pos, rot) < 0):
        raise ValueError("Archived frozen-M1 wrist error has no valid FK comparison")
    return pos * 1000, np.degrees(rot)


def collect_root(root, files_a, files_b, robot):
    if files_b and files_a[3] != files_b[3]:
        raise ValueError(f"A/B measured state freshness window differs for {root['root']}")
    counts, reasons = Counter(), defaultdict(Counter)
    counts.update({f"coherent_feedback_{domain}_{membership}": 0
                   for domain in AVAILABILITY.values() for membership in ("planned", "hmb_valid")})
    counts.update(common_geometry_rows=0, common_frozen_m1_wrist_rows=0, closest_tip_pair_changed=0)
    pairs_a, pairs_b = Counter(), Counter()
    measured = np.empty((root["planned_ticks"], 2, len(GEOMETRY)))
    m1 = np.empty((root["planned_ticks"], 2, len(M1)))
    rows_a = paired_rows(files_a, root)
    rows_b = paired_rows(files_b, root) if files_b else (None for _ in range(root["planned_ticks"]))
    for a, b in zip(rows_a, rows_b, strict=True):
        signed = ("partition", "tick", "hmb_cutoff_ns", "hmb_joint_valid", "dcr_row_class")
        if b is not None and any(a[key] != b[key] for key in signed):
            raise ValueError(f"A/B signed cutoff or target lineage differs at {root['root']} {a['partition']}")
        qa, ra = state(a)
        qb, rb = state(b)
        valid = a["hmb_joint_valid"]
        counts["planned"] += 1
        counts["hmb_joint_valid"] += valid
        availability = AVAILABILITY[qa is not None, qb is not None]
        counts[f"coherent_feedback_{availability}_planned"] += 1
        reasons["A_planned"][ra] += 1
        reasons["B_planned"][rb] += 1
        if not valid:
            continue
        counts[f"coherent_feedback_{availability}_hmb_valid"] += 1
        reasons["A_hmb_valid"][ra] += 1
        reasons["B_hmb_valid"][rb] += 1
        if availability != "common":
            continue
        ga, pair_a = geometry(robot, qa)
        gb, pair_b = (ga, pair_a) if np.array_equal(qa, qb) else geometry(robot, qb)
        measured[counts["common_geometry_rows"]] = ga, gb
        counts["common_geometry_rows"] += 1
        pairs_a[pair_a] += 1
        pairs_b[pair_b] += 1
        counts["closest_tip_pair_changed"] += pair_a != pair_b
        ma, mb = m1_wrist(a), m1_wrist(b)
        if ma is not None and mb is not None:
            m1[counts["common_frozen_m1_wrist_rows"]] = ma, mb
            counts["common_frozen_m1_wrist_rows"] += 1
        else:
            reasons["frozen_m1_missing_on_common_feedback"][("A" if ma is None else "") + ("B" if mb is None else "")] += 1
    return {"counts": dict(counts), "feedback_reasons": dict(reasons),
            "closest_tip_pair_A": pairs_a, "closest_tip_pair_B": pairs_b,
            "A_pair_summary_sha256": files_a[2], "B_pair_summary_sha256": files_b[2] if files_b else None,
            "feedback_max_age_ms": files_a[3],
            "geometry": measured[:counts["common_geometry_rows"]], "m1": m1[:counts["common_frozen_m1_wrist_rows"]]}


def stats(values):
    if not len(values):
        return {"n": 0}
    return {"n": len(values), **dict(zip(("p50", "p95", "p99"),
            np.percentile(values, [50, 95, 99]).tolist(), strict=True))}


def paired_statistics(values, names):
    return {name: {"A": stats(a), "B": stats(b), "B_minus_A_on_paired_ticks": stats(b - a)}
            for name, (a, b) in zip(names, values.transpose(2, 1, 0), strict=True)}


def aggregate(rows):
    counters = Counter()
    reasons = defaultdict(Counter)
    finger_pairs = {side: Counter() for side in ("A", "B")}
    for row in rows:
        counters.update(row["counts"])
        for key, values in row["feedback_reasons"].items():
            reasons[key].update(values)
        for side in finger_pairs:
            finger_pairs[side].update(row[f"closest_tip_pair_{side}"])
    return {
        "counts": dict(counters), "feedback_reasons": dict(reasons), "closest_tip_pair_counts": finger_pairs,
        "live_model_relative_geometry": paired_statistics(np.concatenate([r["geometry"] for r in rows]), GEOMETRY),
        "unchanged_frozen_m1_wrist_errors": paired_statistics(np.concatenate([r["m1"] for r in rows]), M1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("cohort", "a", "b", "model", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--historical-a-subset-cohort", type=Path,
                        help="Exact signed one-root receipt used for the historical A cold-start pilot")
    args = parser.parse_args()
    roots, cohort_sha = verified_cohort(args.cohort)
    a_cohorts = {}
    if args.historical_a_subset_cohort:
        raw = args.historical_a_subset_cohort.read_bytes()
        pilot = json.loads(raw)
        selected = pilot["selected_roots"]
        if (pilot["schema"] != "s05.teleop.hmb_source_cohort.v1"
                or pilot["native_assets_sha256_checked"] is not True
                or pilot["identity_sha256"] != {key: value[1] for key, value in IDENTITIES.items()}
                or len(selected) != 1 or selected[0] not in roots):
            raise ValueError("Historical pilot cohort must match one complete root of the full signed cohort")
        a_cohorts[selected[0]["root"]] = sha256(raw).hexdigest()
    robot = RobotModel(args.model)
    model_sha = digest(robot.path)
    joint_pairs = [list(pair) for pair in zip(robot.names, signed_feedback_names(robot), strict=True)]
    all_rows, splits = {}, defaultdict(list)
    for root in roots:
        a = pair_files(args.a, root, a_cohorts.get(root["root"], cohort_sha), model_sha, joint_pairs)
        b = pair_files(args.b, root, cohort_sha, model_sha, joint_pairs)
        if a is None:
            raise ValueError(f"Missing historical A pair needed for the full signed denominator: {root['root']}")
        row = collect_root(root, a, b, robot)
        all_rows[root["root"]] = row
        splits[root["split"]].append(row)
    summary = {key: aggregate(rows) for key, rows in splits.items()}
    if any(summary[key]["counts"]["hmb_joint_valid"] != n for key, n in VALID_TICKS.items()):
        raise ValueError("Observed HMB-valid denominator differs from frozen splits")
    report = {
        "scope": "HISTORICAL_A_VS_SHARED_RIGID_B_ACTUAL_MUJOCO_GEOMETRY_DIAGNOSTIC",
        "cohort_sha256": cohort_sha, "live_model_xml_sha256": model_sha,
        "historical_a_pilot_cohort_sha256_by_root": a_cohorts,
        "live_model_assets_at_scoring_sha256": normalized_assets(robot.path),
        "pairing": "Same signed cutoff, HMB joint-valid, coherent ACTIVE/TASK feedback in both runs",
        "conditions": {"A": "historical native replay", "B": "shared-rigid native replay"},
        "limits": [
            "Historical A is contextual. Common observed frames do not establish a same-binary experiment or active command qualification.",
            "Closest tip distance measures the live robot's two hands. Independent HMB fingertip target error is unavailable without a site bridge.",
            "Frozen-M1 wrist errors are the existing per-row qualified scores, using the same target and cutoff.",
        ],
        "missing_B_roots": sorted(key for key, row in all_rows.items() if row["B_pair_summary_sha256"] is None),
        "overall": aggregate(list(all_rows.values())), "by_split": summary,
        "by_root": {key: {field: value for field, value in row.items() if field not in ("geometry", "m1")}
                    for key, row in all_rows.items()},
        "formal_qualification": "NONE", "training_admission": False,
    }
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
