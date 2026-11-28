#!/usr/bin/env python3
"""Bind recorded MANUS writer events to signed HMB inputs for diagnostic parity."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path

from mcap.reader import NonSeekingReader
import numpy as np

from dexgym.tasks.direct.wholebody_s05.data.raw_events import clocks, manus_geometry, message_fields
from dexgym.tasks.direct.wholebody_s05.data.schema_registry import EmbeddedDecoders
from dexgym.tasks.direct.wholebody_s05.data.synchronize import CORE_STREAMS
from dexgym.tasks.direct.wholebody_s05.hmb_reader import HMBReader, SIMULATED, eligible_history, file_sha
from dexgym.tasks.direct.wholebody_s05.v1.rmb.mapping import order_for_sdk
from hmb_cohort import IDENTITIES, checked
from prepare_start_handoff import read_identity


ROOT05 = "capture:b63bc72d2e8b2ee395e966f6"
DOCS = Path(__file__).resolve().parents[2] / "docs/s0-5"
ROOT05_RECEIPT = Path("campaign/evidence/dcr010-offline-rmb/run02-receipt.json")
ROOT05_RECEIPT_SHA = "d6d83d40b44d29447e1d4a1e357bc0424b123dbabfcc113eef8218d330c605df"
OBSERVATION_TYPE = "dex_collection_interfaces/msg/SampleObservation"
SIDES = ("left", "right")
ROLES = tuple(f"manus_{side}" for side in SIDES)
NODE_IDENTITY = ("node_id", "parent_node_id", "chain_type", "joint_type")


def fingerprint(stream):
    status = os.fstat(stream.fileno())
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns


def native_events(asset, origin, interval):
    """Read physical file order and preserve every validated adjacent writer event."""
    path = Path(asset["canonical_uri"])
    role = asset["role"]
    side = role.removeprefix("manus_")
    witness = asset["independent_raw_witness"]
    definitions = {
        asset["topic"]: (asset["ros_type"], asset["schema_sha256"]),
        asset["observation_topic"]: (OBSERVATION_TYPE, asset["observation_schema_sha256"]),
    }
    bank = EmbeddedDecoders()
    events = []
    pending = None
    with path.open("rb") as stream:
        before = fingerprint(stream)
        if before[2] != asset["bytes"] or file_digest(stream, "sha256").hexdigest() != asset["sha256"]:
            raise ValueError(f"Native MCAP size or SHA-256 differs for {role}: {path}")
        stream.seek(0)
        reader = NonSeekingReader(stream, validate_crcs=True, record_size_limit=32 * 2**20)
        for index, (schema, channel, message) in enumerate(reader.iter_messages(log_time_order=False)):
            if (schema is None or channel.message_encoding != "cdr" or schema.encoding != "ros2msg"
                    or definitions.get(channel.topic) != (schema.name, sha256(schema.data).hexdigest())):
                raise ValueError(f"Unexpected channel or embedded schema: {path}:{index}")
            if channel.topic == asset["topic"]:
                if pending is not None:
                    raise ValueError(f"Native MANUS message is missing its adjacent observation: {path}:{index}")
                pending = index, message, schema
                continue
            observed, invalid = message_fields(bank.decode(channel.message_encoding, schema, message.data))
            if (pending is None or pending[0] + 1 != index or invalid
                    or observed["stream_id"] != role or observed["source_instance_id"] != witness["source_instance"]
                    or observed["episode_sample_index"] != len(events)
                    or not pending[1].log_time == message.log_time == observed["host_realtime_ns"]):
                raise ValueError(f"Native MANUS observation identity or adjacency differs: {path}:{index}")
            physical, raw, raw_schema = pending
            fields, nonfinite = message_fields(bank.decode(channel.message_encoding, raw_schema, raw.data))
            geometry = manus_geometry(fields, side)
            mono = observed["host_observed_monotonic_ns"]
            timing = clocks(fields, raw.log_time, raw.publish_time, manus=True, native=True)
            cdr_sha = sha256(raw.data).hexdigest()
            observation_sha = sha256(message.data).hexdigest()
            record = {
                "source_index": f"{asset['asset_id']}:{physical}",
                "side": side, "stream": role, "topic": asset["topic"], "glove_id": fields["glove_id"],
                "declared_side": fields["side"], "frame_id": (fields.get("header") or {}).get("frame_id"),
                "in_recorded_interval": interval[0] <= raw.log_time < interval[1],
                "receive_ns": mono - origin, "collector_mono_ns": mono,
                "collector_realtime_ns": observed["host_realtime_ns"],
                "collector_receive_ros_ns": observed["receive_ros_timestamp_ns"],
                "manus_header_ns": timing["ros_header"]["value"],
                "sdk_source_stamp_raw": timing["sdk_source"]["value"],
                "sdk_receive_us": timing["sdk_receive"]["value"],
                "mcap_log_ns": raw.log_time, "mcap_publish_ns": raw.publish_time,
                "observation_mcap_log_ns": message.log_time, "observation_mcap_publish_ns": message.publish_time,
                "mcap_sequence": raw.sequence, "observation_mcap_sequence": message.sequence,
                "episode_sample_index": observed["episode_sample_index"],
                "source_instance": observed["source_instance_id"], "writer_sequence": observed["source_sequence_id"],
                "physical_message": physical, "observation_message": index,
                "asset_id": asset["asset_id"], "asset_sha256": asset["sha256"],
                "schema_sha256": asset["schema_sha256"],
                "observation_schema_sha256": asset["observation_schema_sha256"],
                "cdr_sha256": cdr_sha, "observation_cdr_sha256": observation_sha,
                "topology_id": geometry["topology_sha256"], "geometry_issues": geometry["issues"],
                "nonfinite": nonfinite,
                "raw_nodes": [{**{key: node[key] for key in NODE_IDENTITY},
                               "xyz": node["node_position"], "xyzw": node["node_orientation"]}
                              for node in geometry["nodes"]],
            }
            events.append(record)
            pending = None
        if pending is not None or not events or before != fingerprint(stream):
            raise ValueError(f"Native MANUS MCAP was empty, unpaired or changed: {path}")
    with path.open("rb") as stream:
        if fingerprint(stream) != before:
            raise ValueError(f"Native MANUS MCAP path changed during reading: {path}")
    match = next((row for row in events if row["physical_message"] == witness["physical_message"]), None)
    if match is None or any(match[key] != value for key, value in witness.items()):
        raise ValueError(f"Independent native source witness differs for {role}")
    return events


def cadence(values, scale_ns=1):
    stamps = [value for value in values if value is not None]
    delta = np.diff(np.asarray(stamps, dtype=np.int64))
    positive = delta[delta > 0] * scale_ns
    duration = (stamps[-1] - stamps[0]) * scale_ns if len(stamps) > 1 else 0
    return {"provided": len(stamps), "missing": len(values) - len(stamps),
            "first": stamps[0] if stamps else None, "last": stamps[-1] if stamps else None,
            "observed_rate_hz": (len(stamps) - 1) * 1e9 / duration if duration > 0 else None,
            "backsteps": int(np.count_nonzero(delta < 0)), "equal_neighbors": int(np.count_nonzero(delta == 0)),
            "interval_ns_p50_p95_p99_max": np.quantile(positive, [.5, .95, .99, 1]).tolist()
            if len(positive) else None}


def repeated(values):
    values = [value for value in values if value is not None]
    return {"provided": len(values), "distinct": len(set(values)), "repeat_occurrences": len(values) - len(set(values)),
            "equal_neighbors": sum(first == second for first, second in zip(values, values[1:]))}


def event_statistics(rows):
    fields = ("collector_mono_ns", "collector_realtime_ns", "collector_receive_ros_ns", "manus_header_ns",
              "mcap_log_ns", "mcap_publish_ns", "observation_mcap_log_ns", "observation_mcap_publish_ns")
    ordering = {key: cadence([row[key] for row in rows]) for key in fields}
    ordering["sdk_receive_us"] = cadence([row["sdk_receive_us"] for row in rows], 1000)
    seq = np.diff([row["writer_sequence"] for row in rows])
    sdk = repeated([row["sdk_source_stamp_raw"] for row in rows])
    span = rows[-1]["collector_mono_ns"] - rows[0]["collector_mono_ns"]
    sdk["distinct_rate_over_collector_span_hz"] = (sdk["distinct"] - 1) * 1e9 / span if span > 0 else None
    return {"events": len(rows), "in_recorded_interval": sum(row["in_recorded_interval"] for row in rows),
            "origin": rows[0]["source_instance"], "glove_ids": sorted({row["glove_id"] for row in rows}),
            "topologies": dict(Counter(row["topology_id"] for row in rows)),
            "clocks": ordering, "mcap_log_publish_differences": sum(row["mcap_log_ns"] != row["mcap_publish_ns"] for row in rows),
            "sdk_source_stamp_raw": sdk,
            "sdk_source_and_receive_stamp": repeated([(row["sdk_source_stamp_raw"], row["sdk_receive_us"]) for row in rows]),
            "cdr_sha256": repeated([row["cdr_sha256"] for row in rows]),
            "writer_identity_repeats": repeated([row["writer_sequence"] for row in rows])["repeat_occurrences"],
            "writer_sequence_first": rows[0]["writer_sequence"], "writer_sequence_last": rows[-1]["writer_sequence"],
            "writer_sequence_backsteps": int(np.count_nonzero(seq < 0)),
            "writer_sequence_gap_events": int(np.count_nonzero(seq > 1)),
            "writer_sequence_missing": int(np.maximum(0, seq - 1).sum()),
            "mcap_sequence": repeated([row["mcap_sequence"] for row in rows]),
            "invalid_geometry_events": sum(bool(row["geometry_issues"]) for row in rows),
            "nonfinite_events": sum(bool(row["nonfinite"]) for row in rows)}


def bind_ref(source, role, events, cutoff, delay):
    record = source["record"]
    token = record["raw_index_token"]
    event = events.get(token)
    fields = ("asset_sha256", "schema_sha256", "cdr_sha256", "physical_message", "observation_message",
              "observation_cdr_sha256", "observation_schema_sha256", "source_instance", "writer_sequence",
              "topic", "collector_mono_ns", "stream")
    if (event is None or source["stream"] != role or event["stream"] != role
            or event["asset_id"] != record["part_id"] or any(event[key] != record[key] for key in fields)):
        raise ValueError(f"HMB raw endpoint differs: {role} {cutoff} {token}")
    mono = event["receive_ns"]
    raw_clock = source["raw_clock"]
    clock_fields = {"collector_monotonic": "collector_mono_ns", "collector_realtime": "collector_realtime_ns",
                    "collector_receive": "collector_receive_ros_ns", "mcap_log": "mcap_log_ns",
                    "mcap_publish": "mcap_publish_ns", "ros_header": "manus_header_ns",
                    "sdk_source": "sdk_source_stamp_raw", "sdk_receive": "sdk_receive_us"}
    if (any(raw_clock[key]["value"] != event[field] for key, field in clock_fields.items())
            or list(source["capture_bounds_ns"]) != [mono, mono]
            or list(source["arrival_bounds_ns"]) != [mono + delay, mono + delay] or mono + delay > cutoff):
        raise ValueError(f"HMB source clock or release differs: {role} {cutoff} {token}")
    return event


def hmb_selections(reader, root, events, window=None):
    selections = []
    previous = -1
    delay = reader.frozen_parameters["arrival_delay_ns"]
    for part in root["hmb_partitions"]:
        key = part["key"]
        data, meta, sources, _ = reader.read(key)
        cutoffs = data["cutoff_ns"]
        if (len(cutoffs) != part["planned_ticks"] or not np.array_equal(cutoffs, data["tick_ns"])
                or int(cutoffs[0]) <= previous or np.any(np.diff(cutoffs) <= 0)):
            raise ValueError(f"Signed HMB timeline differs at {key}")
        previous = int(cutoffs[-1])
        topologies = {(entry["stream"], entry["version"]): entry for entry in meta["topologies"]}
        orders = {key: order_for_sdk(entry).tolist() for key, entry in topologies.items() if key[0] in ROLES}
        for tick, cutoff in enumerate(map(int, cutoffs)):
            pose_valid = data["pose_valid"][tick, :len(CORE_STREAMS)]
            source_indices = data["source_index"][tick, :len(CORE_STREAMS), 0]
            joint_valid = bool(data["joint_valid"][tick])
            safety = bool(data["safety_permission"][tick])
            eligible = bool(joint_valid and safety and pose_valid.all() and (source_indices >= 0).all()
                            and (data["capture_bounds_ns"][tick, :len(CORE_STREAMS), 1] <= cutoff).all()
                            and (data["arrival_bounds_ns"][tick, :len(CORE_STREAMS), 1] <= cutoff).all())
            row = {"partition": key, "partition_tick": tick, "tick_ns": cutoff, "joint_valid": joint_valid,
                   "safety_permission": safety, "nominal_pair_eligible": eligible,
                   "history_eligible": eligible_history(data, tick), "pose_valid": pose_valid[:2].tolist(),
                   "in_start_window": None if window is None else window[0] <= cutoff < window[1],
                   "source_index": [None, None], "partition_source_index": [None, None],
                   "writer_identity": [None, None], "arrival_ns": [None, None], "receive_ns": [None, None],
                   "common": [None, None], "order": [None, None]}
            for side, role in enumerate(ROLES):
                index, upper = map(int, data["source_index"][tick, side])
                if upper != -1 or index < -1 or index >= len(sources) or (index < 0 and pose_valid[side]):
                    raise ValueError(f"Expected one causal endpoint: {key}:{tick} {role}")
                if index == -1:
                    continue
                source = sources[index]
                event = bind_ref(source, role, events, cutoff, delay)
                row["source_index"][side] = event["source_index"]
                row["partition_source_index"][side] = index
                row["writer_identity"][side] = [event["source_instance"], event["writer_sequence"]]
                row["arrival_ns"][side] = event["receive_ns"] + delay
                row["receive_ns"][side] = event["receive_ns"]
                if not pose_valid[side]:
                    continue
                topology = (role, source["topology_id"])
                nodes = event["raw_nodes"]
                mask = data["node_mask"][tick, side]
                common = data["sdk_common_pose"][tick, side, mask]
                order = orders[topology]
                if (not all(source[field] for field in ("source_eligible", "fresh", "raw_valid"))
                        or event["topology_id"] != source["topology_id"] or mask.sum() != len(nodes)
                        or not np.array_equal(np.flatnonzero(mask), np.arange(len(nodes)))
                        or not np.array_equal(common, [node["xyz"] + node["xyzw"] for node in nodes])
                        or data["arrival_bounds_ns"][tick, side].tolist() != source["arrival_bounds_ns"]
                        or data["capture_bounds_ns"][tick, side].tolist() != source["capture_bounds_ns"]):
                    raise ValueError(f"Signed SDK-common pose, raw topology or causal bounds differ: {key}:{tick} {role}")
                row["common"][side] = common[:, :3].tolist()
                row["order"][side] = order
            selections.append(row)
    return selections


def selection_statistics(rows):
    return {"planned_ticks": len(rows), "joint_valid_ticks": sum(row["joint_valid"] for row in rows),
            "nominal_pair_eligible_ticks": sum(row["nominal_pair_eligible"] for row in rows),
            "five_frame_ticks": sum(row["history_eligible"] for row in rows),
            "sides": {side: {"matched_endpoints": sum(row["source_index"][i] is not None for row in rows),
                             "valid_poses_exactly_equal": sum(row["pose_valid"][i] for row in rows),
                             "distinct_selected_events": len({row["source_index"][i] for row in rows
                                                              if row["source_index"][i] is not None})}
                      for i, side in enumerate(SIDES)}}


def write_rows(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def run(args):
    cohort, root, cohort_sha, assets, origin, capture_end = read_identity(args.cohort, args.root)
    products = args.hmb_products or Path(cohort["signed_hmb_ubuntu_products"])
    evidence = args.docs / "evidence"
    reader = HMBReader(*(evidence / IDENTITIES[key][0] for key in ("hmb_manifest", "plan", "config")),
                       products, root["split"], SIMULATED)
    window = None
    if args.root == ROOT05:
        previous_receipt = json.loads(checked(args.docs / ROOT05_RECEIPT, ROOT05_RECEIPT_SHA))
        prior = previous_receipt["window"]
        window = [prior[name] for name in ("hmb_cutoff_start_ns", "hmb_cutoff_stop_exclusive_ns")]
        if [origin + cutoff for cutoff in window] != [prior[name] for name in (
                "source_cutoff_start_ns", "source_cutoff_stop_exclusive_ns")]:
            raise ValueError("Signed ROOT05 comparison window and native recorder origin differ")
    by_side = {side: native_events(assets[role], origin, root["interval_ns"])
               for side, role in zip(SIDES, ROLES, strict=True)}
    merged = sorted((row for rows in by_side.values() for row in rows),
                    key=lambda row: (row["receive_ns"], row["side"], row["physical_message"]))
    events = {row["source_index"]: row for row in merged}
    if len(events) != len(merged):
        raise ValueError("Native physical event identity repeated")
    selected = hmb_selections(reader, root, events, window)
    selection = {"full_root": selection_statistics(selected)}
    if window is not None:
        selection["start_window"] = selection_statistics([row for row in selected if row["in_start_window"]])
        for scope, reference in (("full_root", previous_receipt["full_root_hmb"]), ("start_window", prior)):
            for key, expected in (("planned_ticks", "plan_ticks"), ("joint_valid_ticks", "hmb_joint_valid_ticks"),
                                  ("five_frame_ticks", "hmb_five_frame_candidate_ticks")):
                if selection[scope][key] != reference[expected]:
                    raise ValueError(f"ROOT05 {scope} denominator differs for {key}")
    receipt = {
        "schema": "s05.teleop.parity_manus_events.v1", "scope": "DIAGNOSTIC_ONLY", "root": args.root,
        "script_sha256": file_sha(Path(__file__)), "cohort_sha256": cohort_sha,
        "signed_identities": cohort["identity_sha256"], "source_manifest_sha256": root["manifest_sha256"],
        "manus_assets": {role: {key: assets[role][key] for key in ("canonical_uri", "sha256", "bytes", "asset_id")}
                         for role in ROLES},
        "recorded_interval_realtime_ns": root["interval_ns"], "collector_origin_mono_ns": origin,
        "capture_end_mono_ns": capture_end, "hmb_arrival_delay_ns": reader.frozen_parameters["arrival_delay_ns"],
        "historical_teleop_callback_clock": "UNOBSERVED", "raw_receive_scope": "RECORDER_MONOTONIC_OFFSET",
        "hmb_arrival_scope": "SIMULATED_RECORDER_MONOTONIC_PLUS_FROZEN_DELAY",
        "cross_side_tie_break": "COLLECTOR_MONOTONIC_SIDE_PHYSICAL_INDEX", "start_window_cutoffs_ns": window,
        "same_collector_mono_cross_side_neighbors": sum(
            a["receive_ns"] == b["receive_ns"] and a["side"] != b["side"] for a, b in zip(merged, merged[1:])),
        "events": {side: event_statistics(rows) for side, rows in by_side.items()}, "hmb": selection,
    }
    if window is not None:
        receipt["start_window_reference_sha256"] = ROOT05_RECEIPT_SHA
    args.out.mkdir(parents=True, exist_ok=False)
    records_path = args.out / "manus-events.jsonl"
    selected_path = args.out / "hmb-selections.jsonl"
    write_rows(records_path, merged)
    write_rows(selected_path, selected)
    receipt["artifact_sha256"] = {path.name: file_sha(path) for path in (records_path, selected_path)}
    with (args.out / "event-audit.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--root", default=ROOT05)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--docs", default=DOCS, type=Path)
    parser.add_argument("--hmb-products", type=Path)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"MANUS parity evidence rejected: {error}\n")


if __name__ == "__main__":
    main()
