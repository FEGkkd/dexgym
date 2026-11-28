#!/usr/bin/env python3
"""Replay the verified ROOT05 MANUS sidecar through the original native hand callback."""

import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
from hashlib import file_digest
import io
from itertools import zip_longest
import json
from pathlib import Path


ROOT = "capture:b63bc72d2e8b2ee395e966f6"
SIDES = ("left", "right")
ZOH_NS = 2_000_000
SAMPLE_NS = 20_000_000
GRID_TOLERANCE_NS = 1_000
UPDATE_NAME = "wuji-c-updates.jsonl"
ZOH_NAME = "native-zoh-500hz.jsonl.gz"
SAMPLE_NAME = "native-samples-50hz.jsonl.gz"
RECEIPT_NAME = "native-event-replay-summary.json"
UPDATE_ID = ("source_index", "side", "receive_ns")
ARCHIVE_SHA256 = {
    "events": "c32a2987d55ab6137ac44e28cf3e1661012c5e88c1c6cd7a1fba178494585ffa",
    "ticks": "e99addfcec5841ff893f399ef8942822b1da01c0024cb4463827fd1593490fc0",
    "event_audit": "5c2911706e52c6b3b1ad66a7bc74ed2772ffcb784a8f5c0a1ff377bdc2cd55b8",
    "archived_a": "e279269347fa071c93550c003000f87c40f27249a0259c737678e50710763ec2",
    "archived_a_summary": "7622a45fef7564295a4131e6dd74b0931ef01fef54f59e8748e7bdcec7cfaec0",
}


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def write_row(stream, row):
    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


@contextmanager
def compressed_rows(path):
    with path.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as stream:
            yield stream


def check_archive(args):
    hashes = {field: digest(getattr(args, field)) for field in ARCHIVE_SHA256}
    for field, value in hashes.items():
        if value != ARCHIVE_SHA256[field]:
            raise ValueError(f"{field} differs from the published ROOT05 parity archive")
    event_audit = json.loads(args.event_audit.read_bytes())
    archived = json.loads(args.archived_a_summary.read_bytes())
    counts = {side: event_audit["events"][side]["events"] for side in SIDES}
    signed = event_audit["hmb"]["full_root"]["planned_ticks"]
    if (event_audit["root"] != ROOT or event_audit["scope"] != "DIAGNOSTIC_ONLY"
            or event_audit["historical_teleop_callback_clock"] != "UNOBSERVED"
            or archived["raw_event_clock"] != "RECONSTRUCTED_COLLECTOR_ORDER"
            or archived["raw_event_gate"] != "DIAGNOSTIC_OPEN"
            or archived["inputs_in_scope"] != {"raw_events": sum(counts.values()), "signed_ticks": signed}):
        raise ValueError("ROOT05 input and A scope or denominator differs")
    return hashes, event_audit, archived


def event_order(row):
    return row["receive_ns"], row["side"], row["physical_message"]


class EventReplay:
    """Consume each writer event once; hold the latest publication independently per hand."""

    def __init__(self, native, events, stream, origin):
        self.native = native
        self.events = iter(events)
        self.pending = next(self.events, None)
        if self.pending is None:
            raise ValueError("Empty native event stream")
        self.stream = stream
        self.origin = origin
        self.callback_offset = min(0, self.pending["receive_ns"])
        self.previous_order = None
        self.previous_cutoff = None
        self.source_indices = set()
        self.writer_identities = set()
        self.latest = [None, None]
        self.calls = Counter()
        self.stages = Counter()
        self.published = Counter()
        self.last_physical = [-1, -1]

    def consume(self):
        event = self.pending
        side = event["side"]
        index = SIDES.index(side)
        receipt = event["receive_ns"]
        order = event_order(event)
        physical = event["physical_message"]
        token = event["source_index"]
        writer = (event["source_instance"], event["writer_sequence"])
        writer_key = (side, *writer)
        if token in self.source_indices or writer_key in self.writer_identities:
            raise ValueError(f"Repeated native event identity at {token}")
        if ((self.previous_order is not None and order <= self.previous_order)
                or physical <= self.last_physical[index]):
            raise ValueError(f"Native event is out of collector or writer file order at {token}")
        if type(receipt) is not int or event["collector_mono_ns"] - self.origin != receipt:
            raise ValueError(f"Native event does not belong to the collector clock at {token}")
        self.source_indices.add(token)
        self.writer_identities.add(writer_key)
        self.previous_order = order
        self.last_physical[index] = physical
        probe = self.native.step(event, receipt - self.callback_offset, reconstructed=True, gate_open=True)
        row = {key: event[key] for key in UPDATE_ID}
        row.update(probe)
        row["in_recorded_interval"] = event["in_recorded_interval"]
        write_row(self.stream, row)
        ordinal = sum(self.calls.values())
        self.calls[side] += 1
        self.stages[f"{side}/{probe['stage']}"] += 1
        if probe["published"]:
            self.latest[index] = {
                "source_index": token, "writer_identity": writer, "receive_ns": receipt,
                "event_ordinal": ordinal, "side_event_ordinal": self.calls[side] - 1,
                "filtered_rad": probe["filtered_rad"],
            }
            self.published[side] += 1
        self.pending = next(self.events, None)

    def at(self, cutoff):
        if self.previous_cutoff is not None and cutoff <= self.previous_cutoff:
            raise ValueError("Native sampling cutoffs must strictly increase")
        while self.pending is not None and self.pending["receive_ns"] <= cutoff:
            self.consume()
        self.previous_cutoff = cutoff
        return [{**last, "age_ns": cutoff - last["receive_ns"]} if last is not None else None
                for last in self.latest]

    def finish(self):
        while self.pending is not None:
            self.consume()


def bind_ticks(ticks, stop):
    """Associate each signed row with the latest causal 500 Hz grid row."""
    partitions = {}
    grid_samples = {}
    previous_cutoff = -1
    for index, tick in enumerate(ticks):
        cutoff = tick["tick_ns"]
        key = tick["partition"]
        grid_index = cutoff // ZOH_NS
        if (not previous_cutoff < cutoff < stop or abs(cutoff - index * SAMPLE_NS) > GRID_TOLERANCE_NS
                or grid_index in grid_samples):
            raise ValueError(f"Signed HMB sample grid differs at {key}:{tick['partition_tick']}")
        part = partitions.setdefault(key, {"partition": key, "first_root_tick": index, "ticks": 0})
        if (tick["partition_tick"] != part["ticks"] or part["first_root_tick"] + part["ticks"] != index):
            raise ValueError(f"Signed HMB partition sequence differs at {key}:{tick['partition_tick']}")
        part["ticks"] += 1
        grid_samples[grid_index] = tick
        previous_cutoff = cutoff
    return grid_samples, partitions


def sample_grids(replay, ticks, stop, zoh_stream, sample_stream):
    grid_samples, partitions = bind_ticks(ticks, stop)
    supported = Counter()
    max_age = Counter()
    max_grid_lag = 0
    rows_500 = 0
    for grid_index, cutoff in enumerate(range(0, stop, ZOH_NS)):
        hands = replay.at(cutoff)
        zoh = {"root": ROOT, "zoh_index": grid_index, "zoh_cutoff_ns": cutoff, "hands": hands}
        write_row(zoh_stream, zoh)
        rows_500 += 1
        tick = grid_samples.get(grid_index)
        if tick is None:
            continue
        sample_lag = tick["tick_ns"] - cutoff
        max_grid_lag = max(max_grid_lag, sample_lag)
        row = {**zoh, "partition": tick["partition"], "partition_tick": tick["partition_tick"],
               "tick_ns": tick["tick_ns"], "sample_lag_ns": sample_lag,
               "hmb": {key: tick[key] for key in ("in_start_window", "joint_valid", "safety_permission",
                       "nominal_pair_eligible", "history_eligible", "pose_valid", "source_index", "writer_identity",
                       "arrival_ns")}}
        write_row(sample_stream, row)
        for side, hand in zip(SIDES, hands, strict=True):
            if hand is not None:
                supported[side] += 1
                max_age[side] = max(max_age[side], hand["age_ns"] + sample_lag)
        if tick["partition_tick"] == 0:
            partitions[tick["partition"]]["first_sample"] = {
                "tick_ns": tick["tick_ns"], "zoh_cutoff_ns": cutoff,
                "held_receive_ns": [hand["receive_ns"] if hand is not None else None for hand in hands],
                "held_source_index": [hand["source_index"] if hand is not None else None for hand in hands],
                "native_calls": dict(replay.calls),
            }
    return {"rows_500hz": rows_500, "rows_50hz": len(grid_samples), "supported_50hz": dict(supported),
            "max_grid_lag_ns": max_grid_lag, "max_receive_to_sample_age_ns": dict(max_age),
            "partitions": list(partitions.values())}


def compare_updates(recovered, archived):
    """Compare the independently generated callback outputs at identical writer events."""
    layers = {"L2": ("node_ids", "sdk_keypoints"), "L3": ("raw_rad",), "L4": ("filtered_rad",)}
    per_side = {side: {layer: {"present": 0, "exact": 0, "missing": 0} for layer in layers} for side in SIDES}
    for index, (left, right) in enumerate(zip_longest(read_rows(recovered), read_rows(archived))):
        if left is None or right is None:
            raise ValueError("Recovered and archived callback row counts differ")
        fields = (*UPDATE_ID, "stage", "published", "callback_error", "in_recorded_interval")
        if any(left[field] != right[field] for field in fields):
            raise ValueError(f"Recovered and archived callback identity or status differs at row {index}")
        for layer, names in layers.items():
            a = [left[name] for name in names]
            b = [right[name] for name in names]
            if a != b:
                raise ValueError(f"Recovered {layer} differs from archived A at row {index} ({left['source_index']})")
            ledger = per_side[left["side"]][layer]
            if any(value is None for value in a):
                ledger["missing"] += 1
            else:
                ledger["present"] += 1
                ledger["exact"] += 1
    return per_side


def loaded_wuji(sdk):
    # The dynamic loader can consult LD_LIBRARY_PATH before the bridge's RUNPATH.
    paths = set()
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        columns = line.split(maxsplit=5)
        if len(columns) == 6 and Path(columns[-1]).name.startswith("libwuji_sdk_c.so"):
            paths.add(Path(columns[-1]).resolve())
    if paths != {sdk}:
        raise ValueError(f"Loaded Wuji SDK differs from the verified library: {paths}")
    return {"path": str(sdk), "sha256": digest(sdk)}


def run(args):
    from parity_wuji_probe import (REVISION, VENDOR, NativePair, checked_source, native_build, profile)

    source_hashes, event_audit, archived = check_archive(args)
    out = args.out.resolve()
    teleop = args.teleop.resolve()
    sdk = (args.wuji or teleop / VENDOR / "lib/libwuji_sdk_c.so").resolve()
    configs, config_hashes = profile(teleop, args.effective_params)
    reference = archived["provenance"]
    if (reference["teleop_revision"] != REVISION or configs != archived["profile_by_side"]
            or sorted(config_hashes.values()) != sorted(archived["profile_sha256"].values())
            or digest(sdk) != reference["wuji_release"]["library_sha256"]):
        raise ValueError("Wuji library, Teleop revision or callback profile differs from archived A")
    out.mkdir(parents=True, exist_ok=False)
    if args.bridge is not None:
        bridge = args.bridge.resolve()
        if digest(bridge) != reference["bridge_sha256"]:
            raise ValueError("Compiled callback differs from the archived native bridge")
        for name in reference["source_sha256"]:
            checked_source(teleop, Path(name))
        provenance = {**reference, "bridge_path": str(bridge), "bridge_reused": True}
    else:
        bridge, provenance = native_build(teleop, out / "build", args.cxx, sdk)
        provenance.update(bridge_path=str(bridge), bridge_reused=False)
    for key in ("source_sha256", "wrapper_cpp_sha256", "python_sha256", "instrumented_callback_sha256"):
        if provenance[key] != reference[key]:
            raise ValueError(f"Native callback implementation differs from archived A at {key}")
    wrapper = Path(__file__).with_name("parity_wuji_probe.py")
    if digest(wrapper) != reference["python_sha256"]:
        raise ValueError("Native ctypes wrapper differs from archived A")

    origin = event_audit["collector_origin_mono_ns"]
    stop = event_audit["capture_end_mono_ns"] - origin
    updates = out / UPDATE_NAME
    native = NativePair(bridge, configs)
    try:
        provenance["loaded_wuji"] = loaded_wuji(sdk)
        joint_names = dict(zip(SIDES, native.joint_names(), strict=True))
        if joint_names != archived["joint_names"]:
            raise ValueError("Native Wuji joint order differs from archived A")
        with updates.open("x", encoding="utf-8", newline="\n") as stream:
            with compressed_rows(out / ZOH_NAME) as zoh, compressed_rows(out / SAMPLE_NAME) as samples:
                replay = EventReplay(native, read_rows(args.events), stream, origin)
                grids = sample_grids(replay, read_rows(args.ticks), stop, zoh, samples)
                replay.finish()
    finally:
        native.close()

    expected_calls = {side: event_audit["events"][side]["events"] for side in SIDES}
    if (dict(replay.calls) != expected_calls or replay.stages != archived["raw_event_callback_status"]
            or grids["rows_50hz"] != event_audit["hmb"]["full_root"]["planned_ticks"]):
        raise ValueError("Replayed callbacks or samples differ from the archived ROOT05 denominator")
    comparison = compare_updates(updates, args.archived_a)
    output_hashes = {name: digest(out / name) for name in (UPDATE_NAME, ZOH_NAME, SAMPLE_NAME)}
    summary = {
        "schema": "s05.teleop.native_event_replay.v1", "root": ROOT, "scope": "DIAGNOSTIC_ONLY",
        "formal_coverage_credit": False, "historical_teleop_callback_clock": "UNOBSERVED",
        "historical_callback_gate": "UNOBSERVED", "replay_callback_clock": "RECONSTRUCTED_COLLECTOR_ORDER",
        "replay_callback_gate": "DIAGNOSTIC_OPEN", "collector_origin_mono_ns": origin,
        "capture_stop_exclusive_ns": stop, "callback_offset_ns": replay.callback_offset,
        "callback_order": "COLLECTOR_MONOTONIC_SIDE_PHYSICAL_INDEX", "published": dict(replay.published),
        "callback_calls": dict(replay.calls), "callback_stages": dict(replay.stages), "joint_names": joint_names,
        "session_creations_per_side": 1, "explicit_session_resets": 0, "partition_resets": 0,
        "zoh_scope": "NATIVE_L4_PRE_DRIVER", "zoh_step_ns": ZOH_NS, "signed_sample_nominal_step_ns": SAMPLE_NS,
        "zoh_rule": "LATEST_PUBLISHED_PER_SIDE_WITH_RECEIVE_NS_LE_ZOH_CUTOFF_NS",
        "sample_rule": "LATEST_500HZ_ROW_WITH_ZOH_CUTOFF_NS_LE_SIGNED_HMB_TICK_NS",
        "grids": grids, "provenance": provenance, "profile_by_side": dict(zip(SIDES, configs, strict=True)),
        "signed_identities": event_audit["signed_identities"], "source_sha256": source_hashes,
        "script_sha256": digest(Path(__file__)), "output_sha256": output_hashes,
        "exact_against_archived_a": comparison,
        "callback_output_byte_identical_to_a": output_hashes[UPDATE_NAME] == source_hashes["archived_a"],
    }
    with (out / RECEIPT_NAME).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teleop", "events", "ticks", "event-audit", "archived-a", "archived-a-summary", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, help="Reuse the verified compiled callback from archived A")
    parser.add_argument("--wuji", type=Path)
    parser.add_argument("--effective-params", type=Path)
    parser.add_argument("--cxx", default="g++")
    try:
        run(parser.parse_args())
    except (OSError, KeyError, RuntimeError, ValueError) as error:
        parser.exit(2, f"Native event replay rejected: {error}\n")


if __name__ == "__main__":
    main()
