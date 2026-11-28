"""Seal the complete START rerun and score mapping, command generation and execution separately."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import yaml

from pair_nominal_hmb import DCR_COMPARISON_SHA, digest
from run_start_handoff_cohort import FRAMES_SHA, MODEL_SHA
from summarize_shared_rigid_geometry import VALID_TICKS, verified_cohort


LAYERS = ("m1", "shared_hmb_raw", "native_shared_raw", "native_filtered", "native_rate_limited",
          "native_ik_requested", "issued_fk", "measured_fk")
HMB_CHAIN = ("m1", "shared_hmb_raw", "issued_fk", "measured_fk")
NATIVE_CHAIN = ("m1", "native_shared_raw", "issued_fk", "measured_fk")
PAIRS = (
    ("m1", "shared_hmb_raw"), ("m1", "native_shared_raw"), ("m1", "measured_fk"),
    ("shared_hmb_raw", "native_shared_raw"), ("shared_hmb_raw", "issued_fk"),
    ("native_shared_raw", "native_filtered"), ("native_filtered", "native_rate_limited"),
    ("native_rate_limited", "native_ik_requested"), ("native_ik_requested", "issued_fk"),
    ("native_shared_raw", "issued_fk"), ("issued_fk", "measured_fk"),
)


def checked_root(root, directory, attempt, cohort_sha, home_reference):
    source = directory / "wrist_layers.npz"
    summary_path = directory / "wrist_layer_summary.json"
    summary = json.loads(summary_path.read_bytes())
    native = attempt.get("attempts", [])
    latest = native[-1] if native else {}
    array_sha = digest(source)
    if (summary["root"] != root["root"] or summary["cohort_sha256"] != cohort_sha
            or tuple(summary["wrist_layers"]) != LAYERS or summary["npz_sha256"] != array_sha
            or summary.get("trace_sha256") != latest.get("trace_sha256")
            or summary.get("native_trace_sha256_uncompressed") != latest.get("trace_uncompressed_sha256")
            or summary.get("anchor_sha256") != latest.get("anchor_sha256")
            or summary["source_manifest_sha256"] != root["manifest_sha256"]
            or summary["model_sha256"] != MODEL_SHA or summary["shared_frames_sha256"] != FRAMES_SHA
            or summary["signed_dcr_comparison_sha256"] != DCR_COMPARISON_SHA):
        raise ValueError(f"Unexpected score source, layer order or cohort for {root['root']}")
    with np.load(source, allow_pickle=False) as data:
        rows = {key: data[key] for key in (
            "cutoff_ns", "source_cutoff_ns", "partition_index", "partition_tick", "hmb_joint_valid",
            "hmb_safety_permission", "wrist_pose_world", "wrist_valid", "q_issued_rad", "q_issued_valid",
            "q_measured_rad", "q_measured_valid", "wrist_status", "raw_shared_reconstruction_error_m_rad",
        )}
    n = root["planned_ticks"]
    manifest = Path(root["canonical_episode_uri"]) / "manifest.yaml"
    if digest(manifest) != root["manifest_sha256"]:
        raise ValueError(f"Source capture manifest differs for {root['root']}")
    origin = yaml.safe_load(manifest.read_bytes())["start_monotonic_ns"]
    if (any(len(value) != n for value in rows.values())
            or any(rows[key].shape != (n,) or rows[key].dtype.kind not in "iu" for key in
                   ("cutoff_ns", "source_cutoff_ns", "partition_index", "partition_tick"))
            or any(rows[key].shape != (n,) or rows[key].dtype != np.dtype(bool) for key in
                   ("hmb_joint_valid", "hmb_safety_permission"))
            or np.any(np.diff(rows["cutoff_ns"]) <= 0)
            or np.any(rows["source_cutoff_ns"] - rows["cutoff_ns"] != origin)):
        raise ValueError(f"Signed planned tick chronology or length differs for {root['root']}")
    offset = 0
    for part in root["hmb_partitions"]:
        span = slice(offset, offset + part["planned_ticks"])
        if (not np.all(rows["partition_index"][span] == part["index"])
                or not np.array_equal(rows["partition_tick"][span], np.arange(part["planned_ticks"]))):
            raise ValueError(f"Signed planned partition row count or order differs for {part['key']}")
        offset = span.stop
    pose, actual = rows["wrist_pose_world"], rows["wrist_valid"]
    status_rows = rows["wrist_status"]
    codes = {value: key for key, value in summary["wrist_status_codes"].items()}
    if (pose.shape != (n, len(LAYERS), 2, 7) or actual.shape != pose.shape[:-1]
            or actual.dtype != np.dtype(bool) or not np.isfinite(pose[actual]).all()
            or status_rows.shape != actual.shape or not set(np.unique(status_rows)) <= codes.keys()
            or not np.array_equal(actual, status_rows == summary["wrist_status_codes"]["VALID"])
            or not np.allclose(np.linalg.norm(pose[..., 3:][actual], axis=-1), 1.0, atol=1e-6, rtol=0)):
        raise ValueError(f"Wrist pose or eligibility array differs for {root['root']}")
    q_masks = {}
    groups = (slice(0, 14), slice(14, 34), slice(34, 54))
    for label in ("issued", "measured"):
        q, q_valid = rows[f"q_{label}_rad"], rows[f"q_{label}_valid"]
        if q.shape != (n, 54) or q_valid.shape != (n, 3) or q_valid.dtype != np.dtype(bool):
            raise ValueError(f"Named 54-axis command or feedback shape differs for {root['root']}")
        for i, cols in enumerate(groups):
            if not np.array_equal(np.isfinite(q[:, cols]), np.broadcast_to(q_valid[:, i:i + 1], q[:, cols].shape)):
                raise ValueError(f"Command or feedback validity differs for {root['root']} {label} group {i}")
        q_masks[label] = q_valid
    status = attempt["result"]
    if status == "START_NOT_REACHED" and (latest.get("historical_input_count") != 0
            or actual[:, LAYERS.index("native_shared_raw"):].any() or any(mask.any() for mask in q_masks.values())):
        raise ValueError(f"START miss unexpectedly has a historical scored action or feedback for {root['root']}")
    handoff = latest.get("start_handoff") or {}
    measurement = handoff.get("measurement") or {}
    home = summary.get("home_feedback")
    home_error = None
    if home is not None:
        home_expected = np.array([home_reference[name] for name in summary["native_joint_names"]])
        actual_home = np.array(home["q_rad"])
        if actual_home.shape != (54,) or not np.isfinite(actual_home).all():
            raise ValueError(f"Measured Home is invalid for {root['root']}")
        home_error = float(np.max(np.abs(actual_home - home_expected)))
        if home_error > np.deg2rad(1.) or not 0 <= home["max_measured_speed_rad_s"] <= 0.16:
            raise ValueError(f"Measured Home fails the palms-down reference for {root['root']}")
    if handoff and home_error is None:
        raise ValueError(f"Native START decision lacks measured Home for {root['root']}")
    parity = rows["raw_shared_reconstruction_error_m_rad"]
    if parity.shape != (n, 2, 2):
        raise ValueError(f"Native raw telemetry parity schema differs for {root['root']}")
    paired_parity = parity[np.isfinite(parity).all(axis=-1)]
    tolerance = np.array([1e-4, 1e-3])
    if not np.array_equal(summary["native_raw_parity_tolerance_m_rad"], tolerance):
        raise ValueError(f"Native raw target reconstruction tolerance changed for {root['root']}")
    qualified = actual.copy()
    qualifies_raw = np.isfinite(parity).all(axis=-1) & (parity <= tolerance).all(axis=-1)
    qualified[:, LAYERS.index("native_shared_raw")] &= qualifies_raw
    receipt = {"root": root["root"], "split": root["split"], "partitions": len(root["hmb_partitions"]),
               "run_result": status, "attempts": len(native), "history_messages": latest.get("historical_input_count", 0),
               "anchor_cutoff_ns": latest.get("anchor_source_cutoff_ns"), "start_elapsed_s": handoff.get("elapsed_s"),
               "start_wrist_error_m_deg": measurement.get("wrist_error_m_deg"),
               "start_control_reasons": measurement.get("control_reasons"),
               "max_measured_home_error_rad": home_error,
               "run_error": latest.get("error"), "replay_exit": latest.get("replay_exit"),
               "native_raw_parity": {"wrist_rows": len(paired_parity),
                                     "within_tolerance": int(qualifies_raw.sum()),
                                     "max_m_rad": np.max(paired_parity, axis=0).tolist() if len(paired_parity) else None},
               "wrist_status_counts": {layer: dict(Counter(codes[code] for code in status_rows[:, i].flat))
                                       for i, layer in enumerate(LAYERS)},
               "trace_sha256": latest.get("trace_sha256"), "array_sha256": array_sha,
               "layer_summary_sha256": digest(summary_path), "array_path": f"{directory.name}/{source.name}"}
    return receipt, pose, actual, qualified, rows["hmb_joint_valid"], rows["hmb_safety_permission"], q_masks


def errors(x, y):
    delta = y[..., :3] - x[..., :3]
    angle = 2 * np.degrees(np.arccos(np.clip(np.abs(np.sum(x[..., 3:] * y[..., 3:], axis=-1)), 0., 1.)))
    return np.stack((1000 * np.linalg.norm(delta, axis=-1), angle), axis=-1)


def summarize(values):
    if not len(values):
        return {"n": 0}
    q = np.percentile(values, (50, 95, 99), axis=0)
    return {"n": len(values), "position_mm": dict(zip(("p50", "p95", "p99"), q[:, 0].tolist(), strict=True)),
            "orientation_deg": dict(zip(("p50", "p95", "p99"), q[:, 1].tolist(), strict=True))}


def pair_values(pose, valid, support, pairs):
    values = {}
    for a, b in pairs:
        ix, iy = LAYERS.index(a), LAYERS.index(b)
        include = support & valid[:, ix].all(axis=1) & valid[:, iy].all(axis=1)
        values[f"{a}__{b}"] = errors(pose[include, ix], pose[include, iy]).max(axis=1)
    return values


def common_chain(pose, valid, eligible, chain):
    ix = [LAYERS.index(name) for name in chain]
    support = eligible & valid[:, ix].all(axis=(1, 2))
    aligned = pose[support][:, ix]
    if not len(aligned):
        return np.empty((0, 4, 2, 2)), 0.0
    links = errors(aligned[:, :-1], aligned[:, 1:])
    total = errors(aligned[:, :1], aligned[:, -1:])
    telescoped = np.diff(aligned[..., :3], axis=1).sum(axis=1) - (aligned[:, -1, :, :3] - aligned[:, 0, :, :3])
    return np.concatenate((links, total), axis=1), float(np.max(np.abs(telescoped)))


def aggregate(values, key, splits):
    selected = [row[key] for row in values if row["receipt"]["split"] in splits]
    return np.concatenate(selected, axis=0)


def root_counts(pose, avail, valid, safe, joint_masks):
    counts = {"planned": len(pose), "hmb_joint_valid": int(valid.sum()),
              "hmb_joint_valid_and_safety_permission": int((valid & safe).sum()),
              "layers": {}, "joint_availability": {}}
    for i, name in enumerate(LAYERS):
        paired = avail[:, i].all(axis=1)
        counts["layers"][name] = {"planned_with_both_wrists": int(paired.sum()),
                                  "hmb_valid_with_both_wrists": int((paired & valid).sum())}
    for label, mask in joint_masks.items():
        complete = mask.all(axis=1)
        counts["joint_availability"][label] = {"all54_planned": int(complete.sum()),
                                                "all54_hmb_valid": int((complete & valid).sum())}
    return counts


def chain_summary(values, chain):
    labels = tuple(f"{a}__{b}" for a, b in zip(chain, chain[1:])) + (f"{chain[0]}__{chain[-1]}",)
    summary = {label: summarize(values[:, i].max(axis=1)) for i, label in enumerate(labels)}
    if not len(values):
        return {"n": 0, "links": summary}
    largest_wrist = np.argmax(values[:, -1, :, 0], axis=1)
    row = np.arange(len(values))
    worst = values[row, -1, largest_wrist, 0]
    threshold = float(np.percentile(worst, 95))
    include = worst >= threshold
    tail = values[row, :, largest_wrist, 0][include]
    tied_wrists = np.isclose(values[include, -1, 0, 0], values[include, -1, 1, 0], atol=1e-9, rtol=0)
    tied_links = np.isclose(tail[:, :3], tail[:, :3].max(axis=1, keepdims=True), atol=1e-9, rtol=0).sum(axis=1) > 1
    unique = ~(tied_wrists | tied_links)
    winner = np.bincount(np.argmax(tail[unique, :3], axis=1), minlength=3)
    return {"n": len(values), "links": summary, "total_position_tail_cutoff_mm": threshold, "tail_rows": len(tail),
            "largest_position_link_on_the_worst_wrist": {
                **dict(zip(labels[:3], winner.tolist(), strict=True)), "tied_unassigned": int((~unique).sum())}}


def report(rows, splits):
    selected = [row for row in rows if row["receipt"]["split"] in splits]
    outcomes = Counter(row["receipt"]["run_result"] for row in selected)
    counts = {key: sum(row["receipt"]["counts"][key] for row in selected)
              for key in ("planned", "hmb_joint_valid", "hmb_joint_valid_and_safety_permission")}
    counts["layers"] = {name: {key: sum(row["receipt"]["counts"]["layers"][name][key] for row in selected)
                                for key in ("planned_with_both_wrists", "hmb_valid_with_both_wrists")} for name in LAYERS}
    counts["joint_availability"] = {label: {key: sum(row["receipt"]["counts"]["joint_availability"][label][key]
                                                    for row in selected) for key in ("all54_planned", "all54_hmb_valid")}
                                     for label in ("issued", "measured")}
    pairwise = {}
    for label in selected[0]["pair_values"]:
        root_values = [row["pair_values"][label] for row in selected if len(row["pair_values"][label])]
        pooled = summarize(np.concatenate(root_values)) if root_values else {"n": 0}
        medians = np.array([np.percentile(values, 95, axis=0) for values in root_values])
        pairwise[label] = {"frames": pooled, "root_support": len(root_values),
                           "median_of_root_p95_mm_deg": np.median(medians, axis=0).tolist() if len(medians) else None}
    return {"root_outcomes": outcomes, "counts": counts, "pair_specific_hmb_valid": pairwise,
            "hmb_shared_on_same_four_layer_support": chain_summary(aggregate(rows, "hmb_chain", splits), HMB_CHAIN),
            "native_shared_on_same_four_layer_support": chain_summary(aggregate(rows, "native_chain", splits), NATIVE_CHAIN)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cohort", "run", "layers", "home-proof", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    roots, sha = verified_cohort(args.cohort)
    proof = json.loads(args.home_proof.read_bytes())
    if (proof["model_sha256"] != MODEL_SHA or len(proof["named_q_rad"]) != 54
            or not all(0 <= value <= 5 for value in proof["palm_normal_angle_to_world_down_deg"])):
        raise ValueError("Home reference lacks verified downward palm orientation for the loaded model")
    run = json.loads(args.run.read_bytes())
    attempts = {row["root"]: row for row in run["roots"]}
    if (run["cohort_sha256"] != sha or len(attempts) != len(roots) or len(run["roots"]) != len(roots)
            or any(not attempts[root["root"]]["selected"] for root in roots)):
        raise ValueError("Strict native attempts must account for all signed roots")
    rows = []
    for root in roots:
        directory = args.layers / root["root"].split(":")[-1]
        receipt, pose, actual, qualified, valid, safe, masks = checked_root(
            root, directory, attempts[root["root"]], sha, proof["named_q_rad"],
        )
        receipt["counts"] = root_counts(pose, actual, valid, safe, masks)
        pairs = pair_values(pose, qualified, valid, PAIRS)
        receipt["pair_specific_hmb_valid"] = {key: summarize(value) for key, value in pairs.items()}
        hmb, hmb_residual = common_chain(pose, actual, valid, HMB_CHAIN)
        native, native_residual = common_chain(pose, qualified, valid, NATIVE_CHAIN)
        receipt["max_position_decomposition_closure_m"] = max(hmb_residual, native_residual)
        rows.append({"receipt": receipt, "pair_values": pairs, "hmb_chain": hmb, "native_chain": native})
    by_split = {split: report(rows, {split}) for split in VALID_TICKS}
    if any(by_split[split]["counts"]["hmb_joint_valid"] != n for split, n in VALID_TICKS.items()):
        raise ValueError("Immutable HMB-valid denominator differs")
    result = {"schema": "s05.teleop.start_handoff_layers_index.v1", "scope": "DIAGNOSTIC_CANDIDATE",
              "cohort_sha256": sha, "cohort_run_sha256": digest(args.run), "training_admission": False,
              "home_proof_sha256": digest(args.home_proof),
              "overall": report(rows, set(VALID_TICKS)), "by_split": by_split,
              "by_root": [row["receipt"] for row in rows], "layers": list(LAYERS),
              "interpretation": ["Position link vectors telescope, their lengths and percentiles do not.",
                                  "SO(3) geodesics are scored separately per link; they do not add.",
                                  "A link uses the larger error of the two wrists on each complete signed cutoff.",
                                  "Pair-specific and same-four-layer scoring have different availability masks.",
                                  "Native raw target scoring additionally requires successful independent map reconstruction.",
                                  "Tail attribution sorts by the M1-to-measured position error on matched support. It is observational."]}
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
