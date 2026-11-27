"""Bind the signed 32-root HMB lineage to its recorded nominal Teleop inputs."""

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path


IDENTITIES = {
    "plan": ("P01_E_E1/production_plan.json", "0f94b842e2558ec04da257aff94d469ae9b965968fff1a4593382fc2eea09404"),
    "hmb_manifest": ("P01_E_E1/hmb_manifest.json", "e63df2698feb812bc300593d9bf8d1ad3dcae1d2dc59aa4d3a3dfa96851d8074"),
    "hmb_index": ("P01_E_E1/build_index.json", "fecb0c31ee78c6d3ecc0923fe2101aba2f1b06bd7e878786839fe22d40f7cc1d"),
    "handoff": ("P01_F/p02_handoff_manifest.json", "cc1f21ead157e576113a184dba48a5d2596f4835ffdf53e554823cb67eb3fb97"),
    "whitelist": ("P01_D/products/source_whitelist.json", "1228c92b5ffff81df9058878ab5f555dd933781934f2ca9cc6ff567bfa2dd93b"),
    "reconstruction": ("P01_F/raw_reconstruction_report.json", "477cc0b64e967a7dfb9d3d2ceb2eaf27da42f1306d372cf8e42a06e1120f63d3"),
    "config": ("P01_E_E0/candidate_config.json", "6aeaa203e39bea392bc20d5fceee26ea5f85752d219824f98f96b2c852846317"),
}
TELEOP_COMMIT = "6d0b0b5b2601618f1b0052d93ef2403eea9b064a"
MANUS_SCHEMA_SHA = "5635ec6a4cabdea549c9db39d48405a7baa9530447d965f8d85d9a1ea6036b97"
TRACKER_SCHEMA_SHA = "05bace1fb626861602e6d3650893707b91359faa75f7f0c3df50f2fcecf0930e"
OBSERVATION_SCHEMA_SHA = "f2d71b37a6b6415a40dd38745c357d7746f18b8c0072d5504af9f35d5ec58d9a"
INPUT_ROLES = {
    "manus_left": ("/manus_glove_left", "manus_ros2_msgs/msg/ManusGlove", MANUS_SCHEMA_SHA),
    "manus_right": ("/manus_glove_right", "manus_ros2_msgs/msg/ManusGlove", MANUS_SCHEMA_SHA),
    "tracker_left": ("/vive/tracker/left_wrist", "tianji_interfaces/msg/TrackerPose", TRACKER_SCHEMA_SHA),
    "tracker_right": ("/vive/tracker/right_wrist", "tianji_interfaces/msg/TrackerPose", TRACKER_SCHEMA_SHA),
}
EXCLUDED_ROLES = {"tracker_left_raw", "tracker_right_raw", "tracker_safety"}
EXPECTED_SPLITS = {"Train": (24, 228, 86_338), "Validation": (8, 80, 30_275)}
TICK_NS = 20_000_000


def checked(path, expected):
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError(f"Pinned evidence SHA-256 differs: {path}")
    return data


def unique_records(rows, field):
    index = {row[field]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Duplicate {field} in source evidence")
    return index


def validate_partitions(source, rows):
    start, stop = source["interval_ns"]
    if (len(rows) != source["partitions"] or not rows
            or rows[0]["start_ns"] != start or rows[-1]["stop_ns"] != stop
            or sum(row["planned_ticks"] for row in rows) != source["planned_ticks"]):
        raise ValueError(f"Partition count or interval differs: {source['root']}")
    previous = start
    for row in rows:
        if (row["parent"] != source["root"] or row["split"] != source["split"]
                or row["start_ns"] != previous or row["stop_ns"] <= previous
                or row["planned_ticks"] != (row["stop_ns"] - previous + TICK_NS - 1) // TICK_NS):
            raise ValueError(f"Inconsistent signed partition {row['key']}")
        previous = row["stop_ns"]


def asset_path(row, asset, registry, verify_assets):
    record = registry[asset["id"]]
    path = Path(record["canonical_uri"])
    if (record["file_sha256"] != asset["sha256"] or record["size_bytes"] != asset["bytes"]
            or not path.is_absolute() or path.suffix.lower() != ".mcap"):
        raise ValueError(f"Registry asset identity or URI differs: {asset['id']}")
    row["canonical_uri"] = str(path)
    row["asset_check"] = "REGISTRY_IDENTITY_ONLY"
    if verify_assets:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if before.st_size != asset["bytes"] or file_digest(stream, "sha256").hexdigest() != asset["sha256"]:
                raise ValueError(f"Actual MCAP bytes differ: {asset['id']}")
            after = os.fstat(stream.fileno())
        with path.open("rb") as stream:
            current = os.fstat(stream.fileno())

        def fingerprint(status):
            return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns

        if fingerprint(before) != fingerprint(after) or fingerprint(before) != fingerprint(current):
            raise ValueError(f"Actual MCAP changed during hashing: {asset['id']}")
        row["asset_check"] = "SHA256_AND_SIZE_MATCH"


def selected_stream(src, witness, role, registry, verify_assets):
    (asset,) = src["assets"][role]
    (proof,) = witness["sources"][role]
    topic, ros_type, schema = INPUT_ROLES[role]
    if (proof["topic"] != topic or proof["stream"] != role or proof["schema_sha256"] != schema
            or proof["observation_schema_sha256"] != OBSERVATION_SCHEMA_SHA or proof["part_id"] != asset["id"]
            or proof["asset_sha256"] != asset["sha256"] or proof["observation_message"] != proof["physical_message"] + 1):
        raise ValueError(f"Independent native input proof differs for {src['root_episode_id']}: {role}")
    row = {
        "role": role, "topic": topic, "ros_type": ros_type, "payload_encoding": "cdr",
        "schema_sha256": schema, "asset_id": asset["id"], "sha256": asset["sha256"], "bytes": asset["bytes"],
        "canonical_uri": None, "asset_check": "REGISTRY_NOT_PROVIDED",
        "observation_topic": f"/dex/collection/observations/{role}",
        "observation_schema_sha256": OBSERVATION_SCHEMA_SHA,
        "observation_is_replay_input": False,
        "independent_raw_witness": {
            k: proof[k] for k in ("physical_message", "observation_message", "cdr_sha256", "observation_cdr_sha256",
                                  "collector_mono_ns", "source_instance", "writer_sequence")
        },
    }
    if registry is not None:
        asset_path(row, asset, registry, verify_assets)
    return row


def cohort(docs, *, root_selection=(), registry_path=None, verify_assets=False):
    if verify_assets and registry_path is None:
        raise ValueError("Byte verification requires a pinned registry")
    evidence = docs / "evidence"
    data = {label: json.loads(checked(evidence / name, digest)) for label, (name, digest) in IDENTITIES.items()}
    if (data["hmb_manifest"]["index_sha256"] != IDENTITIES["hmb_index"][1]
            or data["handoff"]["e1_plan_sha256"] != IDENTITIES["plan"][1]
            or data["handoff"]["e1_manifest_sha256"] != IDENTITIES["hmb_manifest"][1]
            or [row["partition"] for row in data["hmb_index"]["partitions"]] != data["plan"]["partitions"]):
        raise ValueError("Signed HMB artifact relationships disagree")
    sources = unique_records(data["plan"]["sources"], "root")
    whitelist = unique_records(data["whitelist"]["sources"], "root_episode_id")
    witnesses = unique_records(data["reconstruction"]["source_witnesses"], "root")
    if sources.keys() != whitelist.keys() or sources.keys() != witnesses.keys():
        raise ValueError("Signed plan, asset whitelist and raw reconstruction roots differ")
    root_selection = set(root_selection) or set(sources)
    if not root_selection <= sources.keys():
        raise ValueError(f"Requested roots are not signed HMB sources: {sorted(root_selection - sources.keys())}")
    expected_registry_sha = data["config"]["registry_sha256"]
    registry = None
    if registry_path is not None:
        registry = unique_records([json.loads(line) for line in checked(registry_path, expected_registry_sha).splitlines()],
                                  "id")

    blocks = defaultdict(list)
    for entry in data["hmb_index"]["partitions"]:
        row, artifact = entry["partition"], entry["evidence"]
        if entry["status"] != "PUBLISHED":
            raise ValueError(f"Requested signed HMB partition is not published: {row['key']}")
        blocks[row["recording_id"]].append({
            **row, "archive_relative_path": artifact["path"], "partition_manifest_sha256": artifact["manifest_sha256"],
        })
    result, all_asset_ids, input_ids = [], set(), set()
    totals = {split: Counter() for split in EXPECTED_SPLITS}
    for root, source in sources.items():
        approval, witness = whitelist[root], witnesses[root]
        if (any(source[key] != approval[key] for key in ("recording_id", "manifest_sha256", "split", "task"))
                or source["interval_ns"] != approval["recorded_interval_ns"]
                or set(approval["assets"]) != INPUT_ROLES.keys() | EXCLUDED_ROLES):
            raise ValueError(f"Signed source and whitelist disagree: {root}")
        rows = blocks[source["recording_id"]]
        validate_partitions(source, rows)
        totals[source["split"]].update(roots=1, partitions=len(rows), planned_ticks=source["planned_ticks"])
        for role, (asset,) in approval["assets"].items():
            if asset["id"] in all_asset_ids:
                raise ValueError(f"Aliased asset within the 32-root cohort: {asset['id']}")
            all_asset_ids.add(asset["id"])
            if role in INPUT_ROLES:
                input_ids.add(asset["id"])
        if root not in root_selection:
            continue
        episode_uri = None
        if registry is not None:
            episode = registry[source["recording_id"]]
            episode_uri = episode["canonical_uri"]
            if (episode["root_episode_id"] != root or episode["recorded_interval_ns"] != source["interval_ns"]
                    or episode["producer_kind"] != "LIVE_TELEOP" or not Path(episode_uri).is_absolute()):
                raise ValueError(f"Registry recording identity differs: {root}")
            if verify_assets:
                checked(Path(episode_uri) / "manifest.yaml", source["manifest_sha256"])
        result.append({
            **source, "canonical_episode_uri": episode_uri, "recorded_wrist_calibration_sha256": approval["calibration_sha256"],
            "registered_wrist_frame": "openvr_world", "collector_instance_id": approval["collector_instance_id"],
            "inputs": [selected_stream(approval, witness, role, registry, verify_assets) for role in INPUT_ROLES],
            "excluded_asset_ids": {role: approval["assets"][role][0]["id"] for role in sorted(EXCLUDED_ROLES)},
            "hmb_partitions": rows,
        })
    actual_splits = {split: tuple(totals[split][metric] for metric in ("roots", "partitions", "planned_ticks"))
                     for split in EXPECTED_SPLITS}
    if len(all_asset_ids) != 224 or len(input_ids) != 128 or actual_splits != EXPECTED_SPLITS:
        raise ValueError("Signed aggregate HMB root, partition, tick or asset totals disagree")
    return {
        "schema": "s05.teleop.hmb_source_cohort.v1", "scope": "READ_ONLY_NOMINAL_DIAGNOSTIC",
        "teleop_source_reference": TELEOP_COMMIT,
        "identity_sha256": {label: digest for label, (_, digest) in IDENTITIES.items()},
        "signed_hmb_ubuntu_products": data["handoff"]["ubuntu_products"],
        "registry_expected_sha256": expected_registry_sha, "registry_checked": registry is not None,
        "native_assets_sha256_checked": verify_assets,
        "signed_cohort_denominator": {key: dict(value) for key, value in totals.items()},
        "signed_raw_asset_count": len(all_asset_ids), "signed_human_input_asset_count": len(input_ids),
        "selected_root_count": len(result), "selected_roots": result,
        "replay_topics": [role[0] for role in INPUT_ROLES.values()],
        "start_and_mode_reconstruction": "UNKNOWN_NATIVE_INPUT_ASSETS_HAVE_NO_LIFECYCLE_TOPICS",
        "availability_clock": "RECORDER_MONOTONIC_FROM_PAIRED_SAMPLE_OBSERVATION",
        "clock_scope": "PROXY_NO_ORIGINAL_CONTROLLER_ARRIVAL",
        "hardware_or_policy_authority": False, "formal_d1_gate_authority": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default=Path("docs/s0-5"), type=Path)
    parser.add_argument("--registry", type=Path, help="Frozen A1 registry.jsonl on the capture host")
    parser.add_argument("--verify-assets", action="store_true", help="Hash actual selected source MCAPs")
    parser.add_argument("--root", action="append", default=[], help="Optional signed root; repeat to select multiple")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = cohort(args.docs, root_selection=args.root, registry_path=args.registry, verify_assets=args.verify_assets)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(2, f"Cohort proof blocked: {exc}\n")


if __name__ == "__main__":
    main()
