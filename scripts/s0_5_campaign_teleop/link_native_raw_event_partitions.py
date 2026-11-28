#!/usr/bin/env python3
"""Add storage-partition provenance to every original ROOT05 writer event."""

import argparse
from bisect import bisect_right
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

from native_event_replay import ARCHIVE_SHA256, ROOT, SIDES, bind_ticks, digest, event_order, read_rows


EVENTS_NAME = "native-writer-events-by-partition.jsonl.gz"
RECEIPT_NAME = "native-writer-event-partition-summary.json"
EVENT_MARKER = b',"event":'


def partition_spans(ticks, stop):
    grid, partitions = bind_ticks(ticks, stop)
    for row in grid.values():
        partition = partitions[row["partition"]]
        partition.setdefault("first_signed_cutoff_ns", row["tick_ns"])
        partition["last_signed_cutoff_ns"] = row["tick_ns"]
    spans = list(partitions.values())
    if not spans:
        raise ValueError("No signed HMB partition cutoffs")
    for index, span in enumerate(spans):
        end = spans[index + 1]["first_signed_cutoff_ns"] if index + 1 < len(spans) else stop
        span["storage_stop_exclusive_ns"] = end
    return spans


def storage_assignment(receive, spans, starts):
    index = bisect_right(starts, receive) - 1
    if index < 0:
        return None, "OUTSIDE_BEFORE_FIRST_SIGNED_CUTOFF"
    if receive >= spans[index]["storage_stop_exclusive_ns"]:
        return None, "OUTSIDE_AT_OR_AFTER_CAPTURE_END"
    return spans[index]["partition"], "ASSIGNED"


def verify_mcaps(audit, overrides):
    assets = {}
    for side in SIDES:
        recorded = audit["manus_assets"][f"manus_{side}"]
        path = (overrides[side] or Path(recorded["canonical_uri"])).resolve()
        if path.stat().st_size != recorded["bytes"] or digest(path) != recorded["sha256"]:
            raise ValueError(f"Original {side} MCAP bytes differ from the verified event audit")
        assets[side] = {**recorded, "verified_path": str(path)}
    return assets


def annotated_rows(events, output, audit, spans, assets):
    starts = [span["first_signed_cutoff_ns"] for span in spans]
    counts = {span["partition"]: Counter() for span in spans}
    outside = Counter()
    side_counts = Counter()
    flags = {side: Counter() for side in SIDES}
    source_hash = sha256()
    previous_order = None
    with events.open("rb") as stream, output.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for line in stream:
                if not line.endswith(b"\n") or line.endswith(b"\r\n"):
                    raise ValueError("Verified writer events must use newline-terminated LF JSONL")
                event = json.loads(line)
                order = event_order(event)
                side = event["side"]
                asset = assets[side]
                if (previous_order is not None and order <= previous_order
                        or event["receive_ns"] != event["collector_mono_ns"] - audit["collector_origin_mono_ns"]
                        or event["asset_id"] != asset["asset_id"] or event["asset_sha256"] != asset["sha256"]):
                    raise ValueError(f"Native file order, clock or MCAP identity differs at {event['source_index']}")
                previous_order = order
                partition, status = storage_assignment(event["receive_ns"], spans, starts)
                validity = {"in_recorded_interval": event["in_recorded_interval"],
                            "geometry_issue_free": not event["geometry_issues"],
                            "finite_payload_fields": not event["nonfinite"]}
                metadata = {"root": ROOT, "storage_partition": partition, "partition_assignment": status,
                            "validity": validity}
                prefix = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode("utf-8")
                compressed.write(prefix[:-1] + EVENT_MARKER + line[:-1] + b"}\n")
                source_hash.update(line)
                side_counts[side] += 1
                flags[side].update(key for key, value in validity.items() if value)
                if partition is None:
                    outside[status, side] += 1
                else:
                    counts[partition][side] += 1
    return source_hash.hexdigest(), side_counts, counts, outside, flags


def recover_source_digest(sidecar):
    """Hash the original JSONL bytes extracted losslessly in order from the sidecar."""
    recovered = sha256()
    rows = 0
    with gzip.open(sidecar, "rb") as stream:
        for line in stream:
            metadata, marker, event = line.partition(EVENT_MARKER)
            if not metadata.startswith(b'{"root":') or marker != EVENT_MARKER or not event.endswith(b"}}\n"):
                raise ValueError(f"Malformed annotated writer event at row {rows}")
            recovered.update(event[:-2] + b"\n")
            rows += 1
    return recovered.hexdigest(), rows


def run(args):
    for field in ("ticks", "event_audit"):
        if digest(getattr(args, field)) != ARCHIVE_SHA256[field]:
            raise ValueError(f"{field} differs from the published ROOT05 archive")
    audit = json.loads(args.event_audit.read_bytes())
    if audit["root"] != ROOT or audit["historical_teleop_callback_clock"] != "UNOBSERVED":
        raise ValueError("Native writer event audit scope differs")
    stop = audit["capture_end_mono_ns"] - audit["collector_origin_mono_ns"]
    spans = partition_spans(read_rows(args.ticks), stop)
    if sum(span["ticks"] for span in spans) != audit["hmb"]["full_root"]["planned_ticks"]:
        raise ValueError("Signed HMB plan denominator differs")
    args.out.mkdir(parents=True, exist_ok=False)
    assets = verify_mcaps(audit, {side: getattr(args, f"mcap_{side}") for side in SIDES})
    target = args.out / EVENTS_NAME
    source_hash, sides, counts, outside, flags = annotated_rows(args.events, target, audit, spans, assets)
    recovered_hash, recovered_rows = recover_source_digest(target)
    if (source_hash != ARCHIVE_SHA256["events"] or recovered_hash != source_hash
            or recovered_rows != sum(sides.values())
            or any(sides[side] != audit["events"][side]["events"] for side in SIDES)):
        raise ValueError("Annotated events do not round-trip to the complete verified writer stream")
    receipt = {
        "schema": "s05.teleop.native_raw_event_partitions.v1", "scope": "DIAGNOSTIC_ONLY", "root": ROOT,
        "formal_coverage_credit": False, "historical_teleop_callback_clock": "UNOBSERVED",
        "assignment_clock": "RECORDER_MONOTONIC_OFFSET", "collector_origin_mono_ns": audit["collector_origin_mono_ns"],
        "storage_interval": "FIRST_SIGNED_CUTOFF_INCLUSIVE_TO_NEXT_PARTITION_FIRST_SIGNED_CUTOFF_EXCLUSIVE",
        "final_storage_stop_exclusive_ns": stop, "no_resampling": True, "native_callback_calls": 0,
        "partition_tags_scope": "STORAGE_ONLY", "partition_session_resets": 0,
        "mcaps": assets, "writer_events": sides, "validity_counts": flags,
        "partitions": [{**span, "writer_events": dict(counts[span["partition"]])} for span in spans],
        "outside": {status: {side: outside[status, side] for side in SIDES} for status in
                    ("OUTSIDE_BEFORE_FIRST_SIGNED_CUTOFF", "OUTSIDE_AT_OR_AFTER_CAPTURE_END")},
        "source_sha256": {name: ARCHIVE_SHA256[name] for name in ("events", "ticks", "event_audit")},
        "recovered_writer_events_sha256": recovered_hash, "roundtrip_exact": True,
        "output_sha256": {EVENTS_NAME: digest(target)}, "script_sha256": digest(Path(__file__)),
    }
    with (args.out / RECEIPT_NAME).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("events", "ticks", "event-audit", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    for side in SIDES:
        parser.add_argument(f"--mcap-{side}", type=Path, help="Verify the same MCAP bytes at a relocated path")
    try:
        run(parser.parse_args())
    except (OSError, KeyError, ValueError) as error:
        parser.exit(2, f"Native writer partition link rejected: {error}\n")


if __name__ == "__main__":
    main()
