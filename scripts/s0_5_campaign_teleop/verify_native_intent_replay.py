#!/usr/bin/env python3
"""Independently verify the ROOT05 replay receipts and paired fixed-window traces."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import gzip
from hashlib import file_digest, sha256
import json
from math import isfinite
from pathlib import Path


ROOT = "capture:b63bc72d2e8b2ee395e966f6"
SIDES = ("left", "right")
MODES = ("A", "B", "C", "D")
SERVO_NS = 2_000_000
HMB_NS = 20_000_000
SPEED_CAP = 5.00002
FROZEN = {
    "audit": "5c2911706e52c6b3b1ad66a7bc74ed2772ffcb784a8f5c0a1ff377bdc2cd55b8",
    "events": "c32a2987d55ab6137ac44e28cf3e1661012c5e88c1c6cd7a1fba178494585ffa",
    "ticks": "e99addfcec5841ff893f399ef8942822b1da01c0024cb4463827fd1593490fc0",
    "archived_a": "e279269347fa071c93550c003000f87c40f27249a0259c737678e50710763ec2",
    "windows": "0931007455bf9aaf178f6260726e037e74cc69a404b9345439c186263c9b822c",
    "baseline": "d6d83d40b44d29447e1d4a1e357bc0424b123dbabfcc113eef8218d330c605df",
    "sdk_summary": "7622a45fef7564295a4131e6dd74b0931ef01fef54f59e8748e7bdcec7cfaec0",
    "sdk_ticks": "f51b74ceeb4052c8a985ad5ea6b43b1017dad27d42f1b8694d2ad3b1d479ff24",
    "model_receipt": "1d67a6dad1831574af10a1fdf6349638097af6457e9d8527c9bd77e150e876fe",
    "model": "939b3651e2426e596eed482e70a68c2382720df1fd55b124aa5f0893b640e92d",
    "wuji": "1bdc160c6c27a9b1bfa727dfb7c33c8b01fbb5acbefc8cfdfa17b2fd84c75282",
}
NATIVE_PIN = "6d0b0b5b2601618f1b0052d93ef2403eea9b064a"
HMB_FIELDS = ("in_start_window", "joint_valid", "safety_permission", "nominal_pair_eligible",
              "history_eligible", "pose_valid", "source_index", "writer_identity", "arrival_ns")
ROW_KEY = ("partition", "partition_tick", "tick_ns")
FEEDBACK = ("q_issued", "q_applied", "q_measured", "qd_measured", "feedback_available_ns")
NATIVE_TARGET = ("sources", "q_driver_command")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def lines(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        yield from stream


def rows(path):
    yield from map(json.loads, lines(path))


def select(row, keys):
    return {key: row[key] for key in keys}


def verify_timeline(path, audit):
    ticks, partition_starts, partition_sizes = [], {}, Counter()
    counts = {"full_root": Counter(), "start_window": Counter()}
    start, stop = audit["start_window_cutoffs_ns"]
    previous = -1
    for row in rows(path):
        part, i, cutoff = (row[key] for key in ROW_KEY)
        require(i == partition_sizes[part] and previous < cutoff, f"HMB order {part}/{i}")
        if i == 0:
            partition_starts[part] = cutoff
        require(part == next(reversed(partition_starts)), f"Interleaved partition {part}")
        require(row["in_start_window"] == (start <= cutoff < stop), f"START mask at {part}/{i}")
        observed = {"planned_ticks": 1, "joint_valid_ticks": row["joint_valid"],
                    "five_frame_ticks": row["history_eligible"], "nominal_pair_eligible_ticks": row["nominal_pair_eligible"]}
        counts["full_root"].update(observed)
        if row["in_start_window"]:
            counts["start_window"].update(observed)
        ticks.append(select(row, (*ROW_KEY, *HMB_FIELDS)))
        partition_sizes[part] += 1
        previous = cutoff
    require(len(partition_starts) == 13, "Original partition count")
    for scope, expected in (("full_root", (4844, 4831, 4741)), ("start_window", (4804, 4794, 4708))):
        keys = ("planned_ticks", "joint_valid_ticks", "five_frame_ticks")
        require(tuple(counts[scope][key] for key in keys) == expected, f"Original denominator {scope}")
        require(all(audit["hmb"][scope][key] == value for key, value in counts[scope].items()), f"Audit counts {scope}")
    return ticks, partition_starts, partition_sizes, counts


def verify_events(args, audit, parts, sizes, receipt, stop):
    require(receipt["root"] == ROOT and receipt["partition_tags_scope"] == "STORAGE_ONLY", "Raw storage scope")
    for side in SIDES:
        asset = receipt["mcaps"][side]
        original = audit["manus_assets"][f"manus_{side}"]
        mcap = Path(asset["verified_path"])
        require(select(asset, original) == original and mcap.stat().st_size == original["bytes"]
                and digest(mcap) == original["sha256"], f"Original {side} MCAP")
    hashes = {name: sha256() for name in ("events", "archived_a", "recovered_c")}
    names, starts = tuple(parts), tuple(parts.values())
    previous = None
    side_count, storage = Counter(), Counter()
    sources, publications = {}, []
    for i, (packed, a, c) in enumerate(zip(lines(args.raw_partitions), lines(args.archived_a),
                                          lines(args.recovered_c), strict=True)):
        _, marker, raw = packed.partition(b',"event":')
        require(marker and raw.endswith(b"}}\n"), f"Lossless writer payload {i}")
        hashes["events"].update(raw[:-2] + b"\n")
        hashes["archived_a"].update(a)
        hashes["recovered_c"].update(c)
        require(a == c, f"Archived A and independent C differ at event {i}")
        tagged, command = json.loads(packed), json.loads(c)
        event = tagged["event"]
        token, side, receive = (event[key] for key in ("source_index", "side", "receive_ns"))
        writer = [event["source_instance"], event["writer_sequence"]]
        key = (receive, side, event["physical_message"])
        require(side in SIDES and token not in sources and (previous is None or previous < key)
                and receive == event["collector_mono_ns"] - audit["collector_origin_mono_ns"], f"Writer order at {i}")
        previous = key
        index = bisect_right(starts, receive) - 1
        part = names[index] if index >= 0 and receive < stop else None
        assignment = ("ASSIGNED" if part else "OUTSIDE_BEFORE_FIRST_SIGNED_CUTOFF" if index < 0
                      else "OUTSIDE_AT_OR_AFTER_CAPTURE_END")
        validity = {"in_recorded_interval": event["in_recorded_interval"],
                    "geometry_issue_free": not event["geometry_issues"], "finite_payload_fields": not event["nonfinite"]}
        require(tagged["root"] == ROOT and tagged["storage_partition"] == part
                and tagged["partition_assignment"] == assignment and tagged["validity"] == validity, f"Raw assignment {i}")
        identity = ("source_index", "side", "receive_ns", "in_recorded_interval")
        require(select(command, identity) == select(event, identity) and command["published"]
                and command["stage"] == "PUBLISHED" and not command["callback_error"], f"L0-to-L4 identity {i}")
        require(len(command["node_ids"]) == len(set(command["node_ids"])) == 21
                and len(command["sdk_keypoints"]) == 21
                and all(len(xyz) == 3 and all(map(isfinite, xyz)) for xyz in command["sdk_keypoints"])
                and all(len(command[field]) == 20 for field in ("raw_rad", "filtered_rad"))
                and all(isfinite(x) for field in ("raw_rad", "filtered_rad") for x in command[field]),
                f"L2/L3/L4 finite vectors {i}")
        sources[token] = (side, writer, receive)
        publications.append((side, {"source_index": token, "writer_identity": writer, "receive_ns": receive,
                                    "event_ordinal": i, "side_event_ordinal": side_count[side],
                                    "filtered_rad": command["filtered_rad"]}))
        side_count[side] += 1
        storage[part or assignment, side] += 1
    result_hashes = {name: value.hexdigest() for name, value in hashes.items()}
    require(result_hashes == {"events": FROZEN["events"], "archived_a": FROZEN["archived_a"],
                              "recovered_c": FROZEN["archived_a"]}, "Raw round trip and full L2-L4 byte identity")
    require(side_count == {"left": 11621, "right": 11621} == receipt["writer_events"], "Full callback denominator")
    require(receipt["recovered_writer_events_sha256"] == FROZEN["events"] and receipt["roundtrip_exact"],
            "Raw storage round-trip receipt")
    require([p["partition"] for p in receipt["partitions"]] == list(names), "Ordered raw storage partitions")
    for i, span in enumerate(receipt["partitions"]):
        part = span["partition"]
        require(span["ticks"] == sizes[part] and span["first_signed_cutoff_ns"] == starts[i]
                and span["storage_stop_exclusive_ns"] == (starts[i + 1] if i + 1 < len(starts) else stop)
                and all(span["writer_events"].get(side, 0) == storage[part, side] for side in SIDES), f"Storage span {part}")
    for status, counts in receipt["outside"].items():
        require(all(counts[side] == storage[status, side] for side in SIDES), f"Raw outside counts {status}")
    return publications, sources, result_hashes


def verify_grids(args, ticks, publications, sources, stop, delay):
    expected_50hz = {row["tick_ns"] // SERVO_NS: row for row in ticks}
    require(len(expected_50hz) == len(ticks), "Distinct signed samples on 500 Hz grid")
    held = {side: None for side in SIDES}
    used, count = 0, 0
    ages, support = Counter(), Counter()
    sample_rows = rows(args.samples)
    maximum_lag = 0
    for i, (cutoff, row) in enumerate(zip(range(0, stop, SERVO_NS), rows(args.zoh), strict=True)):
        while used < len(publications) and publications[used][1]["receive_ns"] <= cutoff:
            side, command = publications[used]
            held[side] = command
            used += 1
        hands = [None if held[side] is None else {**held[side], "age_ns": cutoff - held[side]["receive_ns"]}
                 for side in SIDES]
        expected = {"root": ROOT, "zoh_index": i, "zoh_cutoff_ns": cutoff, "hands": hands}
        require(row == expected, f"500 Hz latest-publication selection {i}")
        tick = expected_50hz.get(i)
        if tick is None:
            continue
        lag = tick["tick_ns"] - cutoff
        maximum_lag = max(maximum_lag, lag)
        expected.update(select(tick, ROW_KEY), sample_lag_ns=lag, hmb=select(tick, HMB_FIELDS))
        require(next(sample_rows, None) == expected, f"Signed 50 Hz availability at {tick['tick_ns']}")
        for j, side in enumerate(SIDES):
            token = tick["source_index"][j]
            if token is not None:
                event_side, writer, receive = sources[token]
                require((event_side, writer, receive + delay) == (side, tick["writer_identity"][j], tick["arrival_ns"][j])
                        and receive + delay <= tick["tick_ns"], f"Frozen HMB writer causal link at {tick['tick_ns']}/{side}")
            if hands[j] is not None:
                support[side] += 1
                ages[side] = max(ages[side], hands[j]["age_ns"] + lag)
        count += 1
    require(next(sample_rows, None) is None and count == 4844, "50 Hz total")
    return {"rows_500hz": i + 1, "rows_50hz": count, "supported_50hz": support,
            "max_grid_lag_ns": maximum_lag, "max_receive_to_sample_age_ns": ages, "publications_by_capture_end": used}


def verify_physics(path, ticks, windows, receipt, publications):
    require(receipt["windows"] == windows and receipt["root"] == ROOT and receipt["scope"] == "DIAGNOSTIC_ONLY"
            and not receipt["formal_coverage_credit"] and (receipt["physics_hz"], receipt["rmb_hz"]) == (500, 50),
            "Frozen physics scope")
    controls = ("valid_hold", "smoothstep_and_clip", "independent_hands", "duplicate_rejected", "stale_rejected",
                "future_rejected", "timeout_latches_measured_once_per_side")
    require(receipt["driver_source"]["native_revision"] == NATIVE_PIN
            and all(receipt["driver_controls"].get(name) is True for name in controls),
            "Native driver seam controls")
    compare_ac = (*NATIVE_TARGET, *FEEDBACK, "admitted_events", "driver_watchdog", "q_candidate",
                  "pre_issue_signed_snapshot", "common_proof", "status", "cause", "event", "torque_nm",
                  "minimum_gap_m", "depth_m", "contact_pairs")
    publication_times = [p[1]["receive_ns"] for p in publications]
    counts = {mode: Counter() for mode in MODES}
    ac, shadow, lead_offsets, window_rows = Counter(), Counter(), Counter(), Counter()
    active, prior = None, {}
    start = next(row["tick_ns"] for row in ticks if row["in_start_window"])
    for i, (tick, row) in enumerate(zip((row for row in ticks if row["in_start_window"]), rows(path), strict=True)):
        t, lead = tick["tick_ns"], row["control_cutoff_ns"]
        require(select(row, ROW_KEY) == select(tick, ROW_KEY) and row["input_source_index"] == tick["source_index"],
                f"Frozen physics/HMB alignment {i}")
        window = next((j for j, w in enumerate(windows) if w["start_row"] <= i < w["stop_row_exclusive"]), None)
        require(row["window"] == window and lead == start + i * HMB_NS + 1000 and lead >= t,
                f"Frozen window and physical cutoff {i}")
        lead_offsets[lead - t] += 1
        window_rows[window] += 1
        if window != active:
            prior = {}
            window_start = lead
            active = window
        experiments = row["experiments"]
        for k, steps in enumerate(zip(*(experiments[mode] for mode in MODES), strict=True)):
            cutoff = lead + k * SERVO_NS
            require(all(step["cutoff_ns"] == cutoff for step in steps), f"2 ms substep alignment {i}/{k}")
            a, _, c, d = steps
            if window is not None:
                ac["compared_substeps"] += 1
                ac.update(f"different_{key}" for key in compare_ac if a.get(key) != c.get(key))
            if "common_proof" in c:
                require(c["current_extra_v2"] is not None and len(c["pre_issue_signed_snapshot"]) == 3,
                        f"Missing same-snapshot C/D shadow {i}/{k}")
                shadow[f"{c['common_proof']}/{c['current_extra_v2']}"] += 1
            if "common_proof" in c and "common_proof" in d:
                same = all(c[key] == d[key] for key in ("pre_issue_signed_snapshot", "q_candidate"))
                shadow["same_live_C_D_snapshot" if same else "diverged_live_C_D_snapshot"] += 1
                if same and c["current_extra_v2"] != d["current_extra_v2"]:
                    shadow["different_live_V2_at_equal_snapshot"] += 1
            for mode, step in zip(MODES, steps, strict=True):
                if window is None:
                    require(step["status"] == "NOT_EXECUTED" and "q_measured" not in step, f"Out-of-window physics {i}/{k}")
                    continue
                if mode != "B" and "admitted_events" in step:
                    begin = (bisect_left(publication_times, cutoff) if cutoff == window_start
                             else bisect_right(publication_times, cutoff - SERVO_NS))
                    end = bisect_right(publication_times, cutoff)
                    expected = [(side, command["source_index"], command["receive_ns"])
                                for side, command in publications[begin:end]]
                    require([(event["side"], event["source_index"], event["receive_ns"])
                             for event in step["admitted_events"]] == expected, f"Driver dispatch {mode}/{i}/{k}")
                    counts[mode]["native_accepted"] += sum(e["driver_accepted"] for e in step["admitted_events"])
                    counts[mode]["native_rejected"] += sum(not e["driver_accepted"] for e in step["admitted_events"])
                if "q_measured" not in step:
                    continue
                require(step["feedback_available_ns"] == cutoff + SERVO_NS, f"Physical feedback timestamp {mode}/{i}/{k}")
                speed = max(map(abs, step["qd_measured"][14:]))
                counter = counts[mode]
                counter["substeps"] += 1
                counter["maximum_hand_speed_rad_s"] = max(counter["maximum_hand_speed_rad_s"], speed)
                counter["velocity_excesses"] += speed > SPEED_CAP
                counter["events"] += bool(step["event"])
                counter["penetration_steps"] += step["depth_m"] > 0
                counter["named_contact_steps"] += bool(step["contact_pairs"])
        for mode, steps in experiments.items():
            require(len(steps) == 10, f"Missing 500 Hz records {mode}/{i}")
            for cutoff, field, keys, strict in (
                (t, "rmb_feedback_at_exact_signed_tick", (*NATIVE_TARGET, *FEEDBACK), True),
                (lead, "rmb_feedback_at_leading_cutoff", FEEDBACK, False),
            ):
                observed = [s for s in prior.get(mode, []) if s.get("feedback_available_ns", cutoff + 1) <= cutoff]
                expected = None
                eligible = observed and (cutoff - observed[-1]["feedback_available_ns"] < SERVO_NS if strict else len(observed) == 10)
                if eligible and not any(s["event"] for s in observed):
                    expected = select(observed[-1], keys)
                require(row[field][mode] == expected, f"50 Hz actual availability {mode}/{i}/{field}")
                counts[mode][field] += expected is not None
        prior = experiments
    require(window_rows[None] == 4189 and sum(window_rows[i] for i in range(7)) == 615, "Physical window denominator")
    for mode, counter in counts.items():
        summary = receipt["physics"][mode]
        require(counter["substeps"] == summary.get("physics_steps", 0) and counter["events"] == summary.get("physics_events", 0)
                and counter["penetration_steps"] == summary.get("penetrating_steps", 0)
                and counter["maximum_hand_speed_rad_s"] == summary.get("max_hand_speed_rad_s"), f"Physics receipt {mode}")
    return {"window_rows": {str(key): value for key, value in window_rows.items()}, "modes": counts,
            "A_C_paired_fields": ac, "C_D_shadow": shadow, "leading_offset_min_max_ns": [min(lead_offsets), max(lead_offsets)]}


def main(args):
    hashes = {name: digest(getattr(args, name)) for name in ("audit", "ticks", "windows", "raw_partitions",
              "partition_receipt", "replay_receipt", "samples", "zoh", "physics", "physics_receipt")}
    require(all(hashes[name] == FROZEN[name] for name in ("audit", "ticks", "windows")), "Frozen prior evidence identity")
    audit, raw, replay, physics = (json.loads(getattr(args, name).read_bytes()) for name in
                                   ("audit", "partition_receipt", "replay_receipt", "physics_receipt"))
    require(audit["root"] == replay["root"] == ROOT and audit["historical_teleop_callback_clock"] ==
            replay["historical_teleop_callback_clock"] == "UNOBSERVED", "ROOT05 historical clock scope")
    require(replay["provenance"]["teleop_revision"] == NATIVE_PIN and replay["provenance"]["loaded_wuji"]["sha256"] == FROZEN["wuji"]
            and replay["session_creations_per_side"] == 1 and replay["explicit_session_resets"] == replay["partition_resets"] == 0,
            "Pinned native SDK and continuous per-side session")
    for receipt, source, name in ((raw, "raw_partitions", args.raw_partitions.name), (replay, "samples", args.samples.name),
                                  (replay, "zoh", args.zoh.name)):
        require(receipt["output_sha256"][name] == hashes[source], f"Receipt file identity {name}")
    stop = audit["capture_end_mono_ns"] - audit["collector_origin_mono_ns"]
    ticks, starts, sizes, counts = verify_timeline(args.ticks, audit)
    publications, sources, event_hashes = verify_events(args, audit, starts, sizes, raw, stop)
    hashes.update(event_hashes)
    require(replay["output_sha256"][args.recovered_c.name] == hashes["recovered_c"] and
            replay["callback_output_byte_identical_to_a"], "C replay output receipt")
    require(replay["callback_calls"] == replay["published"] == {"left": 11621, "right": 11621}
            and all(layer == {"present": 11621, "exact": 11621, "missing": 0}
                    for side in SIDES for layer in replay["exact_against_archived_a"][side].values()), "L2-L4 receipt")
    grids = verify_grids(args, ticks, publications, sources, stop, audit["hmb_arrival_delay_ns"])
    require(all(replay["grids"][key] == value for key, value in grids.items() if key in replay["grids"]), "500/50 Hz grid receipt")
    require(physics["sha256"]["ledger"] == hashes["physics"], "Physics ledger digest")
    require(all(physics["sha256"][name] == FROZEN[name] for name in
                ("baseline", "windows", "sdk_summary", "sdk_ticks", "model_receipt", "model"))
            and physics["sha256"]["native_a"] == physics["sha256"]["native_c"] == hashes["recovered_c"], "Physics identity chain")
    windows = json.loads(args.windows.read_bytes())["physical_windows"]
    trace = verify_physics(args.physics, ticks, windows, physics, publications)
    print(json.dumps({"scope": "DIAGNOSTIC_ONLY", "root": ROOT, "denominators": counts,
                      "partitions": sizes, "writer_events": len(publications), "exact_L2_L3_L4_events": len(publications),
                      "grids": grids, "physics": trace, "sha256": {**hashes, "verifier": digest(Path(__file__))}},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("audit", "ticks", "windows", "raw-partitions", "partition-receipt", "archived-a", "recovered-c",
                 "replay-receipt", "samples", "zoh", "physics", "physics-receipt"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    main(parser.parse_args())
