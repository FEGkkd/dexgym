"""Verify the 32-root shared-START candidate and seal its per-root array identities."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from export_shared_rigid_candidate import ARM_STATES, MAPPING, SLICES
from pair_nominal_hmb import digest
from summarize_shared_rigid_geometry import VALID_TICKS, verified_cohort


def verify_root(path, root, cohort_sha):
    data, manifest = path / "commands_feedback.npz", path / "manifest.json"
    record = json.loads(manifest.read_bytes())
    if (record["root"] != root["root"] or record["split"] != root["split"]
            or record["capture_manifest_sha256"] != root["manifest_sha256"]
            or record["cohort_sha256"] != cohort_sha or record["npz_sha256"] != digest(data)
            or record["schema"] != "s05.teleop.shared_rigid_candidate.v1"
            or record["status"] != "DIAGNOSTIC_CANDIDATE" or record["training_admission"] is not False
            or record["formal_active_coverage_evaluated"] is not False
            or record["mapping"] != MAPPING):
        raise ValueError(f"Candidate identity differs for {root['root']}")
    count = root["planned_ticks"]
    start = 0
    if len(record["partition_rows"]) != len(root["hmb_partitions"]):
        raise ValueError(f"Candidate partition count differs for {root['root']}")
    for partition, signed in zip(record["partition_rows"], root["hmb_partitions"], strict=True):
        if partition != {"partition": signed["key"], "start": start, "count": signed["planned_ticks"]}:
            raise ValueError(f"Candidate partition bounds differ for {root['root']}")
        start += partition["count"]
    with np.load(data, allow_pickle=False) as arrays:
        schema = {key: {"shape": list(arrays[key].shape), "dtype": str(arrays[key].dtype)} for key in arrays.files}
        if schema != record["schema_arrays"] or any(value["shape"][0] != count for value in schema.values()):
            raise ValueError(f"Candidate array schema differs for {root['root']}")
        cutoff, epoch, valid = (arrays[key] for key in ("cutoff_ns", "active_epoch", "hmb_joint_valid"))
        issued, measured = (arrays[key] for key in ("q_issued_rad", "q_measured_rad"))
        group, arm, joined = (arrays[key] for key in (
            "issued_group_valid", "arm_diagnostic_status", "arm_diagnostic_matches_issued",
        ))
        if (np.any(np.diff(cutoff) <= 0) or np.isinf(issued).any() or np.isinf(measured).any()
                or np.any(group & (epoch == 0)[:, None]) or np.any(arm >= len(ARM_STATES))
                or np.any((epoch == 0) & np.isfinite(measured).all(axis=1))
                or not np.array_equal(np.isfinite(measured).all(axis=1), arrays["feedback_status"] == "COHERENT_MEASURED")
                or not np.array_equal(joined, np.isin(arm, (1, 2)))):
            raise ValueError(f"Inconsistent candidate timebase, status or numeric masks for {root['root']}")
        for index, columns in enumerate(SLICES):
            expected = np.broadcast_to(group[:, index:index + 1], issued[:, columns].shape)
            if not np.array_equal(np.isfinite(issued[:, columns]), expected):
                raise ValueError(f"Candidate command mask disagrees with the named group for {root['root']}")
        for label, mask in (("planned", np.ones(count, dtype=bool)), ("hmb_joint_valid", valid)):
            subset = record["metrics"][label]
            for key, n in {
                "rows": mask.sum(), "all_54_issued": (mask & group.all(axis=1)).sum(),
                "all_54_measured": (mask & np.isfinite(measured).all(axis=1)).sum(),
                "both_arms_reported_active_or_limited": (mask & joined.all(axis=1)).sum(),
            }.items():
                if subset[key] != n:
                    raise ValueError(f"Candidate receipt count differs for {root['root']} {label} {key}")
            for i, side in enumerate(("left", "right")):
                if subset["arm_status"][side] != dict(Counter(ARM_STATES[x] for x in arm[mask, i])):
                    raise ValueError(f"Candidate controller status count differs for {root['root']} {side}")
    return {"path": path.name, "manifest_sha256": digest(manifest), **record}


def aggregate(records):
    totals = {}
    for label in ("planned", "hmb_joint_valid"):
        values, arms = Counter(), defaultdict(Counter)
        for record in records:
            metrics = record["metrics"][label]
            values.update({key: value for key, value in metrics.items() if key != "arm_status"})
            for side, statuses in metrics["arm_status"].items():
                arms[side].update(statuses)
        totals[label] = {**values, "arm_status": arms}
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cohort", "candidate", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    roots, cohort_sha = verified_cohort(args.cohort)
    records, split_results = [], defaultdict(list)
    for root in roots:
        record = verify_root(args.candidate / root["root"].split(":")[-1], root, cohort_sha)
        records.append(record)
        split_results[root["split"]].append(record)
    totals = {key: aggregate(rows) for key, rows in split_results.items()}
    if any(totals[key]["hmb_joint_valid"]["rows"] != count for key, count in VALID_TICKS.items()):
        raise ValueError("Verified raw HMB split and joint-valid denominators changed")
    index = {
        "schema": "s05.teleop.shared_rigid_candidate_index.v1", "status": "DIAGNOSTIC_CANDIDATE",
        "cohort_sha256": cohort_sha, "training_admission": False, "formal_active_coverage_evaluated": False,
        "roots": records, "total": aggregate(records), "by_split": totals,
    }
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(index, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
