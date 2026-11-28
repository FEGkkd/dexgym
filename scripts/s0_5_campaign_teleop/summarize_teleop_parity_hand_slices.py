#!/usr/bin/env python3
"""Summarize root-five parity physics on windows identified from original MANUS geometry."""

import argparse
from collections import Counter
import gzip
from hashlib import file_digest
from itertools import zip_longest
import json
from math import cos, dist, isfinite, pi, sqrt
from pathlib import Path


ROOT = "capture:b63bc72d2e8b2ee395e966f6"
MODES = ("A", "B", "C", "D")
SIDES = ("left", "right")
FINGERS = ("index", "middle", "ring", "little")
PERIOD_NS = 20_000_000
MIN_TICKS = 3
RAPID_MAX_NS = 500_000_000
RAPID_TRANSITIONS = 4
MOTION_RAD = 0.001
RULES = {
    "natural_open": "All four non-thumb fingers have curl at most 0.25.",
    "fist": "Mean non-thumb curl is at least 0.60 and every non-thumb curl is at least 0.45.",
    "thumb_index_pinch": "Thumb-index gap at most 20 mm and at least 5 mm smaller than every other thumb gap.",
    "thumb_middle_pinch": "Thumb-middle gap at most 20 mm and at least 5 mm smaller than every other thumb gap.",
    "spread": "All four curls at most 0.35, index-little tip span at least 1.35 times proximal span, "
              "and at least 25 degrees between their proximal-to-tip chords.",
    "one_finger_flexion": "One non-thumb curl is at least 0.60 and every other non-thumb curl is at most 0.30.",
    "asymmetric_hands": "Mean absolute left-right difference across the four corresponding curls is at least 0.30.",
    "rapid_repeated_open_close": "Five alternating stable open/fist runs on the same hand, with at most 0.50 s "
                                 "between the onset of successive runs. Missing valid observations break a sequence.",
}
STATIC = tuple(RULES)[:-1]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def delta(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


def length(v):
    return sqrt(dot(v, v))


def shape(hand):
    """Finger extension measured against the palm-to-proximal-centre direction."""
    if len(hand) != 5 or any(len(finger) != 4 for finger in hand):
        raise ValueError("Expected five semantically ordered four-point fingers")
    if any(len(xyz) != 3 or not all(map(isfinite, xyz)) for finger in hand for xyz in finger):
        raise ValueError("Expected finite cartesian hand landmarks")
    forward = tuple(sum(finger[0][axis] for finger in hand[1:]) / 4 for axis in range(3))
    chords = [delta(finger[-1], finger[0]) for finger in hand[1:]]
    arcs = [sum(dist(a, b) for a, b in zip(finger, finger[1:])) for finger in hand[1:]]
    proximal_span = dist(hand[1][0], hand[-1][0])
    chord_norm = length(chords[0]) * length(chords[-1])
    norm = length(forward)
    if min(norm, proximal_span, *arcs) <= 1e-7 or chord_norm <= 1e-14:
        return None
    return {
        "curl": [1 - dot(chord, forward) / (arc * norm) for chord, arc in zip(chords, arcs, strict=True)],
        "thumb_gaps_mm": [1000 * dist(hand[0][-1], finger[-1]) for finger in hand[1:]],
        "spread_ratio": dist(hand[1][-1], hand[-1][-1]) / proximal_span,
        "outer_chord_cosine": dot(chords[0], chords[-1]) / chord_norm,
    }


def from_input(row):
    hands = []
    for common, order, valid in zip(row["common"], row["order"], row["pose_valid"], strict=True):
        if not valid:
            hands.append(None)
            continue
        if len(order) != 21 or len(set(order)) != 21 or min(order) < 0 or max(order) >= len(common):
            raise ValueError("Invalid signed MANUS semantic landmark order")
        origin = common[order[0]]
        relative = [delta(common[i], origin) for i in order[1:]]
        hands.append(shape([relative[i:i + 4] for i in range(0, 20, 4)]))
    return hands


def margin(name, hands, side):
    """Zero is the threshold and a positive value meets every clause of the proxy."""
    if side == "both":
        if any(h is None for h in hands):
            return None
        return sum(abs(a - b) for a, b in zip(hands[0]["curl"], hands[1]["curl"], strict=True)) / 4 - 0.30
    hand = hands[SIDES.index(side)]
    if hand is None:
        return None
    curl = hand["curl"]
    if name == "natural_open":
        return 0.25 - max(curl)
    if name == "fist":
        return min(sum(curl) / 4 - 0.60, min(curl) - 0.45)
    if name == "spread":
        return min(0.35 - max(curl), hand["spread_ratio"] - 1.35, cos(25 * pi / 180) - hand["outer_chord_cosine"])
    if name == "one_finger_flexion":
        high, next_high = sorted(curl, reverse=True)[:2]
        return min(high - 0.60, 0.30 - next_high)
    finger = ("thumb_index_pinch", "thumb_middle_pinch").index(name)
    gap = hand["thumb_gaps_mm"][finger]
    other = min(x for i, x in enumerate(hand["thumb_gaps_mm"]) if i != finger)
    return min(20 - gap, other - gap - 5)


def continuous(rows, start, stop):
    return all(abs(rows[i]["tick_ns"] - rows[i - 1]["tick_ns"] - PERIOD_NS) <= 2_000
               for i in range(start + 1, stop))


def stable_runs(rows, mask, side):
    start = None
    sides = range(2) if side == "both" else [SIDES.index(side)]
    for i, yes in enumerate([*mask, False]):
        if start is not None and (not yes or not continuous(rows, i - 1, i + 1)):
            if i - start >= MIN_TICKS and all(
                len({r["source_index"][s] for r in rows[start:i] if r["source_index"][s] is not None}) >= MIN_TICKS
                for s in sides
            ):
                yield start, i
            start = None
        if yes and start is None:
            start = i


def rapid_runs(rows, hands, side):
    states = sorted((start, stop, state) for state in ("natural_open", "fist")
                    for start, stop in stable_runs(rows, [passes(margin(state, h, side)) for h in hands], side))
    chain = []
    for start, stop, state in states:
        if chain:
            previous_start, previous_stop, previous_state = chain[-1]
            if (state == previous_state or rows[start]["tick_ns"] - rows[previous_start]["tick_ns"] > RAPID_MAX_NS
                    or not continuous(rows, previous_stop - 1, start + 1)
                    or any(h[SIDES.index(side)] is None for h in hands[previous_stop:start])):
                chain = []
        chain.append((start, stop, state))
        if len(chain) == RAPID_TRANSITIONS + 1:
            yield chain[0][0], stop
            chain.pop(0)


def passes(value):
    return value is not None and value >= 0


def digest(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def jsonl(path):
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt" if path.suffix == ".gz" else "r", encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def check_hashes(args):
    ledger = args.physics / "fixed-arm-parity.jsonl.gz"
    receipt = args.physics / "fixed-arm-parity-summary.json"
    physics = json.loads(receipt.read_bytes())
    sdk_receipt = args.sdk_summary or args.sdk_ticks.with_name("wuji-summary.json")
    sdk = json.loads(sdk_receipt.read_bytes())
    paths = {"input_ticks": args.input_ticks, "sdk_ticks": args.sdk_ticks, "sdk_summary": sdk_receipt,
             "physics_ledger": ledger, "physics_summary": receipt, "script": Path(__file__)}
    hashes = {key: digest(path) for key, path in paths.items()}
    if (physics["root"] != ROOT or physics["scope"] != "DIAGNOSTIC_ONLY" or physics["formal_coverage_credit"]
            or sdk["source_sha256"]["selection_jsonl"] != hashes["input_ticks"]
            or sdk["output_sha256"][args.sdk_ticks.name] != hashes["sdk_ticks"]
            or any(physics["sha256"][key] != hashes[key] for key in ("sdk_ticks", "sdk_summary"))
            or physics["sha256"]["ledger"] != hashes["physics_ledger"]):
        raise ValueError("Signed input, SDK output or fixed-arm physics receipt association differs")
    return physics, ledger, hashes


def tick_identity(row):
    return row["partition"], row["partition_tick"], row["tick_ns"], row["source_index"]


def collect(args, physics, ledger):
    rows, descriptors, observations = [], [], []
    full_count = 0
    physical = jsonl(ledger)
    first = physics["window"]["hmb_cutoff_start_ns"]
    end = physics["window"]["hmb_cutoff_stop_exclusive_ns"]
    for raw, sdk in zip_longest(jsonl(args.input_ticks), jsonl(args.sdk_ticks)):
        full_count += 1
        if raw is None or sdk is None or tick_identity(raw) != tick_identity(sdk) or any(
            raw[key] != sdk[key] for key in ("in_start_window", "pose_valid", "nominal_pair_eligible")
        ):
            raise ValueError(f"HMB and SDK rows differ at row {full_count}")
        in_window = first <= raw["tick_ns"] < end
        if raw["in_start_window"] is not in_window:
            raise ValueError("Signed HMB START window differs from physics scope")
        if not in_window:
            continue
        fixed = next(physical, None)
        if fixed is None or tick_identity(raw) != tick_identity(fixed):
            raise ValueError(f"Physical tick absent or out of sequence at HMB row {full_count}")
        rows.append(raw)
        descriptors.append(from_input(raw))
        observations.append(observe(fixed, sdk))
    if (full_count != physics["full"]["plan_ticks"] or len(rows) != physics["window"]["plan_ticks"]
            or next(physical, None) is not None or not continuous(rows, 0, len(rows))):
        raise ValueError("Original signed timeline or its fixed-arm replay denominator differs")
    return rows, descriptors, observations


def observe(fixed, sdk):
    result = {}
    for mode in MODES:
        traces = fixed["experiments"][mode]
        samples = []
        if len(traces) != 5:
            raise ValueError("Expected five 4-ms rollout decisions per planned tick")
        for sub, trace in enumerate(traces):
            if "snapshot" in trace and (trace["subcycle"] != sub
                    or trace["physical_selection_ns"] != sdk["control_cutoff_ns"] + 4_000_000 * sub):
                raise ValueError("Fixed-arm rollout decision differs from signed control grid")
            applied = trace.get("physical", [])
            if len(applied) > 2:
                raise ValueError("Expected at most two 2-ms physics samples per decision")
            samples.extend(applied)
        complete = len(samples) == 10
        geometry = traces[-1].get("measured_fk", {}).get("local_landmarks_m")
        if complete != (geometry is not None):
            raise ValueError("Fixed-arm end-of-tick FK and substep count disagree")
        moved = []
        issued = []
        for offset in (14, 34):
            group = slice(offset, offset + 20)
            start = traces[0]["snapshot"][2][group] if samples else None
            moved.append(start is not None and max(abs(a - b) for a, b in zip(
                start, samples[-1]["measured"][group], strict=True)) >= MOTION_RAD)
            issued.append(sum(t["status"] == "ISSUED" and t["issued"][group] != t["snapshot"][0][group]
                              for t in traces))
        result[mode] = {
            "substeps": len(samples), "moved": moved, "changed_commands": issued,
            "holds": sum(t["status"] == "HOLD" for t in traces),
            "contact_substeps": sum(bool(s["contact_pairs"]) for s in samples),
            "penetrating_substeps": sum(s["depth_m"] > 0 for s in samples),
            "event_substeps": sum(bool(s["event"]) for s in samples),
            "minimum_gap_mm": 1000 * min(s["minimum_gap_m"] for s in samples) if samples else None,
            "geometry": list(map(shape, geometry)) if complete else [None, None],
        }
    shadow = [t for t in fixed["experiments"]["C"] if t.get("guard_v2") is not None]
    common_permit = [t for t in shadow if t["common_proof"] == "PERMIT"]
    veto = [t for t in common_permit if t["guard_v2"] != "PERMIT"]
    result["guard"] = {"evaluated": len(shadow), "common_permit": len(common_permit),
                       "veto": Counter(t["guard_v2"] for t in veto),
                       "changed_command_veto": Counter(t["guard_v2"] for t in veto if t["status"] == "ISSUED"
                                                       and t["nominal"][14:] != t["snapshot"][0][14:])}
    return result


def chosen_ticks(windows):
    return {i for start, stop in windows for i in range(start, stop)}


def location(row):
    return {key: row[key] for key in ("partition", "partition_tick", "tick_ns", "source_index", "receive_ns")}


def mode_counts(mode, observations, selected, side, matching):
    indices = range(2) if side == "both" else [SIDES.index(side)]
    traces = [observations[i][mode] for i in sorted(selected)]
    gaps = [trace["minimum_gap_mm"] for trace in traces if trace["minimum_gap_mm"] is not None]
    return {
        "complete_ticks": sum(t["substeps"] == 10 for t in traces),
        "partial_ticks": sum(0 < t["substeps"] < 10 for t in traces),
        "unexecuted_ticks": sum(t["substeps"] == 0 for t in traces),
        "observed_substeps": sum(t["substeps"] for t in traces),
        "moved_ticks": sum(any(t["moved"][i] for i in indices) for t in traces),
        "changed_command_hand_decisions": sum(t["changed_commands"][i] for t in traces for i in indices),
        "hold_decisions": sum(t["holds"] for t in traces),
        "contact_ticks": sum(t["contact_substeps"] > 0 for t in traces),
        "contact_substeps": sum(t["contact_substeps"] for t in traces),
        "penetrating_substeps": sum(t["penetrating_substeps"] for t in traces),
        "event_substeps": sum(t["event_substeps"] for t in traces),
        "minimum_observed_gap_mm": min(gaps, default=None),
        "measured_matching_proxy_ticks": len(selected & matching),
    }


def guard_counts(observations, selected):
    counts = [observations[i]["guard"] for i in sorted(selected)]
    veto, changed = Counter(), Counter()
    for count in counts:
        veto.update(count["veto"])
        changed.update(count["changed_command_veto"])
    return {"planned_decisions": 5 * len(selected), "evaluated_decisions": sum(c["evaluated"] for c in counts),
            "common_permit_decisions": sum(c["common_permit"] for c in counts),
            "veto_decisions": veto.total(), "veto_ticks": sum(bool(c["veto"]) for c in counts),
            "veto_reasons": veto, "changed_command_veto_decisions": changed.total(),
            "changed_command_veto_reasons": changed}


def slice_summary(name, side, rows, descriptors, observations):
    rapid = name == "rapid_repeated_open_close"
    source_margins = [margin(name, h, side) for h in descriptors] if not rapid else []
    candidate = [i for i, value in enumerate(source_margins) if passes(value)]
    windows = (list(rapid_runs(rows, descriptors, side)) if rapid else
               list(stable_runs(rows, [passes(value) for value in source_margins], side)))
    selected = chosen_ticks(windows)
    matching = {}
    for mode in MODES:
        measured = [o[mode]["geometry"] for o in observations]
        matching[mode] = (chosen_ticks(rapid_runs(rows, measured, side)) if rapid else
                          {i for i in selected if passes(margin(name, measured[i], side))})
    best = max((i for i, value in enumerate(source_margins) if value is not None),
               key=source_margins.__getitem__, default=None)
    exemplar = None
    if best is not None:
        metrics = descriptors[best] if side == "both" else descriptors[best][SIDES.index(side)]
        exemplar = {"row": location(rows[best]), "input_metrics": metrics,
                    "threshold_margin": source_margins[best], "in_stable_window": best in selected}
        if name == "one_finger_flexion":
            exemplar["most_flexed_finger"] = max(zip(metrics["curl"], FINGERS))[1]
    return selected, matching, {
        "status": "OBSERVED_INPUT_PROXY_WINDOWS" if windows else "INSUFFICIENT_INPUT_EVIDENCE",
        "instantaneous_candidate_ticks": None if rapid else len(candidate),
        "planned_ticks": len(selected), "planned_substeps_per_mode": 10 * len(selected),
        "modes": {mode: mode_counts(mode, observations, selected, side, matching[mode]) for mode in MODES},
        "same_snapshot_d_guard": guard_counts(observations, selected), "best_input_exemplar": exemplar,
        "windows": [{"first": location(rows[start]), "last": location(rows[stop - 1]),
                     "elapsed_ms_first_to_last": (rows[stop - 1]["tick_ns"] - rows[start]["tick_ns"]) / 1e6,
                     "planned_ticks": stop - start,
                     "complete_ticks_by_mode": {mode: sum(observations[i][mode]["substeps"] == 10
                                                         for i in range(start, stop)) for mode in MODES},
                     "same_snapshot_d_veto_decisions": guard_counts(observations, range(start, stop))["veto_decisions"]}
                    for start, stop in windows],
    }


def summarize(args):
    physics, ledger, hashes = check_hashes(args)
    rows, descriptors, observations = collect(args, physics, ledger)
    slices = {}
    for name in RULES:
        sides = ("both",) if name == "asymmetric_hands" else SIDES
        summaries = {side: slice_summary(name, side, rows, descriptors, observations) for side in sides}
        selected = set().union(*(value[0] for value in summaries.values()))
        matching = {mode: set().union(*(value[0] & value[1][mode] for value in summaries.values())) for mode in MODES}
        slices[name] = {"planned_ticks": len(selected), "planned_substeps_per_mode": 10 * len(selected),
                        "complete_ticks_by_mode": {mode: sum(observations[i][mode]["substeps"] == 10 for i in selected)
                                                   for mode in MODES},
                        "measured_matching_proxy_ticks_by_mode": {mode: len(matching[mode]) for mode in MODES},
                        "same_snapshot_d_guard": guard_counts(observations, selected),
                        "by_side": {side: value[2] for side, value in summaries.items()}}
    return {
        "schema": "s05.teleop.parity_hand_slices.v1", "scope": "DIAGNOSTIC_ONLY",
        "formal_coverage_credit": False, "root": ROOT, "source_sha256": hashes,
        "signed_start_window": physics["window"], "planned_ticks": len(rows),
        "recorded_pose_geometry_ticks_by_side": {side: sum(h[i] is not None for h in descriptors)
                                                 for i, side in enumerate(SIDES)},
        "definitions": {
            "source": "Unrotated recorded MANUS positions from signed HMB selections. No historical gesture labels.",
            "stability": "At least three consecutive 50 Hz ticks and three distinct selected raw identities per side. "
                         "Overlapping windows are counted once per tick within each aggregate.",
            "curl": "One minus the projection of proximal-to-tip chord on the unit wrist-to-mean-proximal direction, "
                    "divided by the four-landmark path length. Finger order is index, middle, ring, little.",
            "rules": RULES,
            "physics": "Complete means ten archived 2 ms samples for that planned 20 ms tick, including holds. "
                       "Measured movement is a net change of at least 0.001 rad in any named joint of the selected hand(s). "
                       "Contact includes any archived MuJoCo contact pair on either hand or arm.",
            "measured_matching_proxy": "Uses the same formulas on measured end-of-tick robot palm-local landmarks, "
                                        "with the robot palm body as origin. Human and robot proximal landmarks differ. "
                                        "Rapid counts overlap of input windows with robot-qualified windows for the same hand.",
            "guard_veto": "C's extra V2 shadow result differs from PERMIT when the shared common gate permitted. "
                          "Changed-command vetoes also require that C issued a different hand target.",
            "clocks": "Tick and endpoints are signed HMB cutoff ns on the recorder-relative timeline. The receive_ns "
                      "pairs identify collector-relative original source times.",
        },
        "slices": slices,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ticks", type=Path, required=True)
    parser.add_argument("--sdk-ticks", type=Path, required=True)
    parser.add_argument("--sdk-summary", type=Path)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
