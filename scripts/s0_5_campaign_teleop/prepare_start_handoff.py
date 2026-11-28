#!/usr/bin/env python3
"""Bind the first usable signed HMB START snapshot to four native replay records."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import yaml

from dexgym.tasks.direct.wholebody_s05.data.synchronize import CORE_STREAMS
from dexgym.tasks.direct.wholebody_s05.hmb_reader import HMBReader, SIMULATED
from hmb_cohort import IDENTITIES, INPUT_ROLES, validate_partitions


def read_identity(path, root_id):
    raw = path.read_bytes()
    cohort = json.loads(raw)
    if (cohort.get("schema") != "s05.teleop.hmb_source_cohort.v1"
            or cohort.get("native_assets_sha256_checked") is not True
            or cohort.get("identity_sha256") != {name: item[1] for name, item in IDENTITIES.items()}):
        raise ValueError("Expected the approved byte-verified native HMB cohort receipt")
    matches = [root for root in cohort["selected_roots"] if root["root"] == root_id]
    if len(matches) != 1:
        raise ValueError("Expected one signed source for the requested root")
    root = matches[0]
    validate_partitions(root, root["hmb_partitions"])
    inputs = {entry["role"]: entry for entry in root["inputs"]}
    if (tuple(sorted(inputs)) != tuple(sorted(CORE_STREAMS)) or len(root["inputs"]) != len(CORE_STREAMS)
            or len({entry["asset_id"] for entry in inputs.values()}) != len(CORE_STREAMS)
            or any((entry["topic"], entry["ros_type"], entry["schema_sha256"]) != INPUT_ROLES[role]
                   or entry["asset_check"] != "SHA256_AND_SIZE_MATCH" for role, entry in inputs.items())):
        raise ValueError("Native cohort does not bind exactly the four distinct signed human inputs")
    manifest_path = Path(root["canonical_episode_uri"]) / "manifest.yaml"
    raw_manifest = manifest_path.read_bytes()
    if sha256(raw_manifest).hexdigest() != root["manifest_sha256"]:
        raise ValueError("Signed capture manifest SHA-256 differs")
    manifest = yaml.safe_load(raw_manifest)
    origin = manifest["start_monotonic_ns"]
    capture_end = manifest["stop_monotonic_ns"]
    if type(origin) is not int or type(capture_end) is not int or not 0 < origin < capture_end:
        raise ValueError("Capture manifest has no valid recorder monotonic interval")
    return cohort, root, sha256(raw).hexdigest(), inputs, origin, capture_end


def first_anchor(reader, root):
    previous = -1
    for part in root["hmb_partitions"]:
        data, meta, sources, frames = reader.read(part["key"])
        identity = meta["identity"]
        cutoffs = data["cutoff_ns"]
        if (identity["root_episode_id"] != root["root"]
                or identity["source_manifest_sha256"] != root["manifest_sha256"]
                or identity["development_split"] != root["split"]
                or len(cutoffs) != part["planned_ticks"] or not np.array_equal(cutoffs, data["tick_ns"])
                or int(cutoffs[0]) <= previous or np.any(np.diff(cutoffs) <= 0)):
            raise ValueError(f"HMB partition does not bind the signed chronological input timeline: {part['key']}")
        previous = int(cutoffs[-1])
        valid = data["joint_valid"] & data["safety_permission"] & data["pose_valid"][:, :len(CORE_STREAMS)].all(axis=1)
        candidates = np.flatnonzero(valid)
        if not len(candidates):
            continue
        tick = int(candidates[0])
        cutoff = int(cutoffs[tick])
        selected = {}
        for column, role in enumerate(CORE_STREAMS):
            source, alternate = data["source_index"][tick, column]
            if not 0 <= source < len(sources) or alternate != -1:
                raise ValueError(f"Expected one causal HMB source for {role} at {part['key']}:{tick}")
            row = sources[source]
            if (row["stream"] != role or not row["source_eligible"] or not row["fresh"] or not row["raw_valid"]
                    or row["arrival_bounds_ns"][1] > cutoff):
                raise ValueError(f"The earliest eligible HMB tick has an invalid source for {role}")
            selected[role] = row
        geometry = {}
        for column, role in enumerate(CORE_STREAMS):
            mask = data["node_mask"][tick, column]
            pose = data["pose"][tick, column, mask]
            if not len(pose) or not np.isfinite(pose).all():
                raise ValueError(f"Expected populated finite source geometry for {role}")
            geometry[role] = {"frame_contract": frames["manus_pose" if role.startswith("manus_") else "registered_wrist"],
                              "recorded_frame": selected[role]["recorded_frame"],
                              "pose_xyz_xyzw": pose.tolist(), "age_ns": int(data["age_ns"][tick, column])}
        return part, tick, cutoff, selected, geometry
    raise ValueError("No jointly valid, safety-permitted four-input anchor in the signed root")


def bind_inputs(rows, assets, events, cutoff_ns):
    wanted = {}
    for role, row in rows.items():
        record, asset = row["record"], assets[role]
        index = record["physical_message"]
        if (record["part_id"] != asset["asset_id"] or record["asset_sha256"] != asset["sha256"]
                or record["topic"] != asset["topic"] or record["schema_sha256"] != asset["schema_sha256"]
                or type(index) is not int or index < 0 or record["observation_message"] != index + 1):
            raise ValueError(f"Raw HMB endpoint differs from verified cohort asset: {role}")
        wanted[asset["topic"], index] = row
    matched = {}
    for event in events:
        key = event.topic, event.record_index
        if key not in wanted:
            continue
        row = wanted[key]
        clock = row["raw_clock"]
        digest = sha256(event.data).hexdigest()
        if (event.topic in matched or digest != row["record"]["cdr_sha256"]
                or event.schedule_ns != clock["collector_monotonic"]["value"]
                or event.log_ns != clock["collector_realtime"]["value"]
                or not 0 < event.schedule_ns <= cutoff_ns):
            raise ValueError(f"Raw replay CDR, order or recorder clocks differ: {key}")
        matched[event.topic] = {"record_index": event.record_index, "cdr_sha256": digest}
    if set(matched) != {topic for topic, _ in wanted}:
        raise ValueError("The earliest eligible HMB tick cannot bind all four messages in the signed replay window")
    return matched


def prepare(args):
    from nominal_live_replay import signed_events

    cohort, root, cohort_sha, assets, origin, capture_end = read_identity(args.cohort, args.root)
    evidence = args.docs / "evidence"
    reader = HMBReader(evidence / IDENTITIES["hmb_manifest"][0], evidence / IDENTITIES["plan"][0],
                       evidence / IDENTITIES["config"][0], args.hmb_products or cohort["signed_hmb_ubuntu_products"],
                       root["split"], SIMULATED)
    part, tick, cutoff, rows, geometry = first_anchor(reader, root)
    absolute = origin + cutoff
    if not origin <= absolute < capture_end:
        raise ValueError("HMB anchor lies outside the signed monotonic capture interval")
    events, receipt = signed_events(args.cohort, args.root, None)
    if receipt["root"] != root or receipt["cohort_receipt_sha256"] != cohort_sha:
        raise ValueError("Native events are from a different signed root or cohort")
    inputs = bind_inputs(rows, assets, events, absolute)
    return {"schema": "s05.teleop.preposition_start_handoff.v1", "root": args.root, "split": root["split"],
            "cohort_sha256": cohort_sha, "source_manifest_sha256": root["manifest_sha256"],
            "partition": part["key"], "partition_tick": tick, "hmb_cutoff_ns": cutoff,
            "source_cutoff_ns": absolute, "inputs": inputs, "hmb_geometry": geometry,
            "scope": "SIGNED_HUMAN_INPUT_ANCHOR_SIMULATION_DIAGNOSTIC"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--docs", type=Path, default=Path("docs/s0-5"))
    parser.add_argument("--hmb-products", type=Path)
    args = parser.parse_args()
    try:
        report = prepare(args)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"START anchor rejected: {error}\n")


if __name__ == "__main__":
    main()
