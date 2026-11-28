#!/usr/bin/env python3
"""Freeze the six-root DCR-009 Train-only Pilot and causal START anchors."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.data.synchronize import CORE_STREAMS
from dexgym.tasks.direct.wholebody_s05.hmb_reader import HMBReader, SIMULATED
from dexgym.tasks.direct.wholebody_s05.p02.intent import semantic_order
from hmb_cohort import EXPECTED_SPLITS, IDENTITIES, TICK_NS
from prepare_start_handoff import bind_inputs, read_identity


RUN_ID = "HOME-ANCHORED-PILOT-DCR009-01"
PLAN_SCHEMA = "s05.teleop.home_anchored_pilot_plan.v1"
ANCHOR_SCHEMA = "s05.teleop.home_anchored_start_anchor.v1"
SEARCH_NS = 2_000_000_000
REPLAY_NS = 120_000_000_000
AGE_NS = 60_000_000
SKEW_NS = 35_000_000
ARRIVAL_DELAY_NS = 8_000_000
RELATIVE_PAIRS = ((0, 1), (2, 3), (0, 2), (1, 3))
PALM_CLASSES = {"down", "up", "facing", "mixed"}
STABLE_IDENTITY_FIELDS = ("clock_generation", "calibration_id", "topology_id")
MOTION = {"block_ticks": 10, "minimum_distinct_per_side_block": 3, "minimum_source_interval_ns": 100_000_000,
          "maximum_source_interval_ns": 320_000_000, "fast_m_s": .12, "still_m_s": .035,
          "reversal_m_s": .10, "reversal_cos_max": -.50, "same_side_event_spacing_ns": 400_000_000}
POISE = {"vertical_cos_min": 2**-.5, "facing_pair_cos_max": -.5, "facing_inward_cos_min": .5}


@dataclass(frozen=True)
class StableRule:
    ticks: int
    unique: int
    wrist_speed: float
    wrist_angular: float
    wrist_move: float
    wrist_rotation_deg: float
    finger_speed: float
    finger_move: float


PRIMARY = StableRule(11, 4, .05, .35, .008, 4., .08, .010)
SENSITIVITY = StableRule(21, 8, .035, .25, .006, 3., .06, .008)


@dataclass
class Row:
    part: str
    tick: int
    time: int
    segment: int
    versions: tuple
    status: str
    samples: tuple = ()
    source_ns: object = None
    wrists: object = None
    tips: object = None
    sources: tuple = ()


def file_sha(path):
    with path.open("rb") as handle:
        from hashlib import file_digest
        return file_digest(handle, "sha256").hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def proper(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.shape[-2:] != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[..., 3, :], [0, 0, 0, 1], atol=1e-9)):
        raise ValueError("Expected finite homogeneous proper transforms")
    rot = matrix[..., :3, :3]
    if not np.allclose(rot @ np.swapaxes(rot, -1, -2), np.eye(3), atol=1e-8) or not np.allclose(np.linalg.det(rot), 1, atol=1e-8):
        raise ValueError("Expected a proper right-handed rotation")
    return matrix


def pose_matrices(raw):
    raw = np.asarray(raw, dtype=float)
    if not np.isfinite(raw).all() or np.any(abs(np.linalg.norm(raw[..., 3:], axis=-1) - 1) > 1e-5):
        raise ValueError("Invalid signed pose quaternion")
    result = np.broadcast_to(np.eye(4), (*raw.shape[:-1], 4, 4)).copy()
    result[..., :3, 3] = raw[..., :3]
    result[..., :3, :3] = Rotation.from_quat(raw[..., 3:].reshape(-1, 4)).as_matrix().reshape(*raw.shape[:-1], 3, 3)
    return result


def start_transform(world_rotation, anatomical_start_h, home_anatomical_world):
    """Keep the measured rotation fixed and translate both homologous roots together."""
    target = np.eye(4)
    target[:3, :3] = world_rotation
    anatomy = proper(anatomical_start_h)
    home = proper(home_anatomical_world)
    if anatomy.shape != (2, 4, 4) or home.shape != (2, 4, 4):
        raise ValueError("Expected paired corresponding anatomical frames")
    target[:3, 3] = home[:, :3, 3].mean(axis=0) - world_rotation @ anatomy[:, :3, 3].mean(axis=0)
    return proper(target)


def role_identity(source):
    record = source["record"]
    instance, sequence = record["source_instance"], record["writer_sequence"]
    if not isinstance(instance, str) or not instance or type(sequence) is not int or sequence < 0:
        raise ValueError("Source sample has no stable writer instance and sequence")
    return instance, sequence


def clock_status(data, tick):
    time = int(data["tick_ns"][tick])
    clock = np.asarray(data["capture_bounds_ns"][tick, :4])
    receipt = data["arrival_bounds_ns"][tick, :4]
    age = data["age_ns"][tick, :4]
    if (np.any(clock[:, 0] < 0) or np.any(clock[:, 0] > clock[:, 1]) or np.any(clock[:, 1] > time)
            or np.any(receipt > time) or not np.array_equal(receipt, clock + ARRIVAL_DELAY_NS)
            or not np.array_equal(age, time - clock[:, 0]) or np.any(age > AGE_NS)):
        return "SOURCE_TIME_OR_AGE"
    if any(max(clock[a, 1], clock[b, 1]) - min(clock[a, 0], clock[b, 0]) > SKEW_NS
           for a, b in RELATIVE_PAIRS):
        return "SOURCE_SKEW"
    return "VALID"


def source_rows(data, sources, tick):
    rows = []
    for role, (source, alternate) in zip(CORE_STREAMS, data["source_index"][tick, :4], strict=True):
        if not 0 <= source < len(sources) or alternate != -1:
            raise ValueError("Expected a single HMB as-of endpoint for every input role")
        row = sources[source]
        if row["stream"] != role or any(row[key] is not True for key in ("raw_valid", "fresh", "source_eligible")):
            raise ValueError("Eligible HMB tick carries a mismatched or invalid source")
        rows.append(row)
    return tuple(rows)


def hand_tips(data, sources, topologies, tick):
    wrists = []
    for side, row in enumerate(sources[:2]):
        root, fingers = topologies[row["stream"], row["topology_id"]]
        pose = data["pose"][tick, side]
        if not data["node_mask"][tick, side, root] or not data["node_mask"][tick, side, fingers[:, -1]].all():
            raise ValueError("Signed MANUS record lacks its named palm or fingertips")
        points = pose[fingers[:, -1], :3] - pose[root, :3]
        if not np.isfinite(points).all():
            raise ValueError("Signed MANUS tips are nonfinite")
        wrists.append(points)
    return np.asarray(wrists)


def anchor_rows(data, meta, sources, part, mounts, stop):
    used = data["tick_ns"] <= stop
    topologies = {(entry["stream"], entry["version"]): semantic_order(entry) for entry in meta["topologies"]}
    for tick in np.flatnonzero(used):
        time = int(data["cutoff_ns"][tick])
        versions = tuple(tuple(int(value) for value in data[key][tick, :4]) for key in STABLE_IDENTITY_FIELDS)
        segment = int(data["segment_id"][tick])
        status = "HMB_JOINT_INVALID_OR_UNSAFE"
        source = ()
        samples = source_ns = wrists = tips = None
        if data["joint_valid"][tick] and data["safety_permission"][tick] and data["pose_valid"][tick, :4].all():
            status = clock_status(data, tick) if segment >= 0 else "NO_SIGNED_CONTINUOUS_SEGMENT"
            if status == "VALID":
                source = source_rows(data, sources, tick)
                samples = tuple(role_identity(item) for item in source)
                source_ns = data["capture_bounds_ns"][tick, :4, 0]
                wrists = pose_matrices(data["pose"][tick, 2:4, 0]) @ mounts
                tips = hand_tips(data, source, topologies, tick)
        yield Row(part, int(tick), time, segment, versions, status, samples, source_ns, wrists, tips, source)


def stable_window(rows, rule):
    if len(rows) != rule.ticks:
        return None, "INCOMPLETE_WINDOW"
    head = rows[0]
    if any(row.status != "VALID" for row in rows):
        return None, next(row.status for row in rows if row.status != "VALID")
    if any(row.part != head.part or row.segment != head.segment or row.versions != head.versions
           or row.time != head.time + i * TICK_NS for i, row in enumerate(rows)):
        return None, "DISCONTINUOUS_WINDOW"
    measurements = {}
    for index, role in enumerate(CORE_STREAMS):
        keys = [row.samples[index] for row in rows]
        side = index % 2
        feature = "tips" if index < 2 else "wrists"
        for a, b in zip(rows, rows[1:]):
            if a.samples[index] == b.samples[index] and (
                a.source_ns[index] != b.source_ns[index]
                or not np.array_equal(getattr(a, feature)[side], getattr(b, feature)[side])
            ):
                return None, "REPEATED_SOURCE_GEOMETRY_CHANGED"
        chosen = [0] + [i for i in range(1, len(rows)) if keys[i] != keys[i - 1]]
        unique = [keys[i] for i in chosen]
        if len({key[0] for key in unique}) != 1:
            return None, "WRITER_GENERATION_CHANGED"
        times = np.asarray([rows[i].source_ns[index] for i in chosen], dtype=np.int64)
        if len(unique) != len(set(unique)) or any(b[1] <= a[1] for a, b in zip(unique, unique[1:])) or np.any(np.diff(times) <= 0):
            return None, "SOURCE_SEQUENCE_OR_CAPTURE_RECEDES"
        if len(unique) < rule.unique:
            return None, "SOURCE_SUPPORT_TOO_LOW"
        dt = np.diff(times) / 1e9
        if index < 2:
            features = np.asarray([rows[i].tips[side] for i in chosen])
            speed = np.linalg.norm(np.diff(features, axis=0), axis=-1) / dt[:, None]
            move = np.linalg.norm(features - features[0], axis=-1)
            values = {"speed_m_s": float(speed.max()), "from_first_m": float(move.max())}
            if values["speed_m_s"] > rule.finger_speed or values["from_first_m"] > rule.finger_move:
                return None, "MANUS_MOTION_EXCEEDS_STABLE_WINDOW"
        else:
            poses = np.asarray([rows[i].wrists[side] for i in chosen])
            points = poses[:, :3, 3]
            rot = Rotation.from_matrix(poses[:, :3, :3])
            speed = np.linalg.norm(np.diff(points, axis=0), axis=-1) / dt
            move = np.linalg.norm(points - points[0], axis=-1)
            angular = (rot[:-1].inv() * rot[1:]).magnitude() / dt
            turn = np.degrees((rot[0].inv() * rot).magnitude())
            values = {"speed_m_s": float(speed.max()), "from_first_m": float(move.max()),
                      "speed_rad_s": float(angular.max()), "from_first_deg": float(turn.max())}
            if (values["speed_m_s"] > rule.wrist_speed or values["from_first_m"] > rule.wrist_move
                    or values["speed_rad_s"] > rule.wrist_angular or values["from_first_deg"] > rule.wrist_rotation_deg):
                return None, "ANATOMICAL_WRIST_MOTION_EXCEEDS_STABLE_WINDOW"
        measurements[role] = {"unique": len(unique), **values}
    return {"first_cutoff_ns": head.time, "last_cutoff_ns": rows[-1].time,
            "partition": head.part, "first_tick": head.tick, "last_tick": rows[-1].tick,
            "streams": measurements}, "STABLE"


def choose_anchor(rows, rule, explicit_status, explicit_cutoff=None):
    attempts = Counter()
    if explicit_status == "PRESENT_INELIGIBLE":
        return None, {"status": "START_UNQUALIFIED", "reason": explicit_status, "window_attempts_by_reason": {}}
    if explicit_status == "PRESENT_QUALIFIED_EVENT":
        candidates = [i for i, row in enumerate(rows) if row.time == explicit_cutoff]
        success = "EXPLICIT_START"
    elif explicit_status == "UNOBSERVED_IN_SIGNED_NATIVE_INPUTS":
        candidates = range(rule.ticks - 1, len(rows))
        success = "RECONSTRUCTED_START"
    else:
        raise ValueError("Explicit lifecycle evidence state needs adjudication")
    for index in candidates:
        window = rows[max(0, index - rule.ticks + 1):index + 1]
        stats, reason = stable_window(window, rule)
        attempts[reason] += 1
        if stats is not None:
            return rows[index], {"status": success, **stats, "window_attempts_by_reason": dict(attempts)}
    reason = "EXPLICIT_EVENT_WINDOW_UNQUALIFIED" if success == "EXPLICIT_START" else "NO_STABLE_WINDOW_WITHIN_2S"
    return None, {"status": "START_UNQUALIFIED", "reason": reason, "window_attempts_by_reason": dict(attempts)}


def palm_class(wrists, world_rotation, local_normals):
    local = np.asarray(local_normals, dtype=float)
    if local.shape != (2, 3) or not np.isfinite(local).all() or not np.allclose(np.linalg.norm(local, axis=1), 1., atol=1e-8):
        raise ValueError("Expected two verified unit palm normals in the corresponding A frames")
    normals = (world_rotation @ wrists[:, :3, :3] @ local[..., None])[..., 0]
    side = world_rotation @ (wrists[1, :3, 3] - wrists[0, :3, 3])
    span = np.linalg.norm(side)
    if span <= 1e-8:
        raise ValueError("Cannot classify coincident corresponding anatomical roots")
    side /= span
    up = normals[:, 2]
    dots = np.array([normals[0] @ side, -normals[1] @ side])
    if np.all(up <= -POISE["vertical_cos_min"]):
        label = "down"
    elif np.all(up >= POISE["vertical_cos_min"]):
        label = "up"
    elif normals[0] @ normals[1] <= POISE["facing_pair_cos_max"] and np.all(dots >= POISE["facing_inward_cos_min"]):
        label = "facing"
    else:
        label = "mixed"
    return {"class": label, "world_normals": normals.tolist(), "inward_cos": dots.tolist(),
            "pair_cos": float(normals[0] @ normals[1]), "startup_anatomical_gap_m": float(span)}


def hmb_five_rows(data):
    """Existing five-frame support in the signed partition, separate from START eligibility."""
    mask = data["joint_valid"] & data["safety_permission"] & data["pose_valid"][:, :4].all(axis=1)
    count = len(mask)
    if count < 5:
        return np.zeros(count, dtype=bool)
    ends = np.ones(count - 4, dtype=bool)
    for n in range(5):
        ends &= mask[n:n+count-4]
    segment = data["segment_id"]
    ends &= segment[:-4] >= 0
    for n in range(1, 5):
        ends &= (segment[n:n+count-4] == segment[:-4])
        ends &= (data["tick_ns"][n:n+count-4] - data["tick_ns"][:-4] == n * TICK_NS)
        for key in STABLE_IDENTITY_FIELDS:
            values = data[key]
            ends &= (values[n:n+count-4, :4] == values[:-4, :4]).all(axis=1)
    return np.r_[np.zeros(4, dtype=bool), ends]


def motion_events(times, segments, poses, sources, valid, active):
    """Compare two causal-source displacement blocks within one signed valid segment."""
    block = MOTION["block_ticks"]
    retained = []
    for side in range(2):
        previous = -MOTION["same_side_event_spacing_ns"]
        for i in range(block, len(times) - block):
            x, y = i - block, i + block
            t = int(times[i])
            if t - previous < MOTION["same_side_event_spacing_ns"] or not active[x:y + 1].all():
                continue
            if segments[x] < 0 or not np.all(segments[x:y + 1] == segments[x]) or not valid[x:y + 1].all():
                continue
            physical = sources[x:y + 1, side]
            if (np.any(np.diff(physical) < 0)
                    or min(len(set(physical[:block + 1])), len(set(physical[block:]))) < MOTION["minimum_distinct_per_side_block"]):
                continue
            dt = np.diff(sources[[x, i, y], side])
            if np.any(dt < MOTION["minimum_source_interval_ns"]) or np.any(dt > MOTION["maximum_source_interval_ns"]):
                continue
            v = np.diff(poses[[x, i, y], side], axis=0) / (dt[:, None] / 1e9)
            before, after = np.linalg.norm(v, axis=-1)
            if before >= MOTION["fast_m_s"] and after <= MOTION["still_m_s"]:
                event = "stop"
            elif before <= MOTION["still_m_s"] and after >= MOTION["fast_m_s"]:
                event = "start"
            elif min(before, after) >= MOTION["reversal_m_s"] and (v[0] @ v[1]) / (before * after) <= MOTION["reversal_cos_max"]:
                event = "reversal"
            else:
                continue
            retained.append({"cutoff_ns": t, "side": ("left", "right")[side], "kind": event,
                             "before_speed_m_s": float(before), "after_speed_m_s": float(after)})
            previous = t
    return sorted(retained, key=lambda row: (row["cutoff_ns"], row["side"]))


def scan_root(reader, root, mounts, world_rotation, normals, origin, capture_end, explicit_status):
    parts, near = [], []
    start = previous = None
    for part in root["hmb_partitions"]:
        raw, meta, sources, _ = reader.read(part["key"])
        identity = meta["identity"]
        times = raw["tick_ns"]
        if (identity["root_episode_id"] != root["root"] or identity["source_manifest_sha256"] != root["manifest_sha256"]
                or identity["development_split"] != "Train" or len(times) != part["planned_ticks"]
                or not np.array_equal(times, raw["cutoff_ns"]) or not np.all(np.diff(times) == TICK_NS)
                or (previous is not None and times[0] <= previous)):
            raise ValueError("Root or signed chronological HMB timetable differs")
        start = int(times[0]) if start is None else start
        previous = int(times[-1])
        if times[0] <= start + SEARCH_NS:
            near.extend(anchor_rows(raw, meta, sources, part["key"], mounts, start + SEARCH_NS))
        valid = raw["joint_valid"] & raw["safety_permission"] & raw["pose_valid"][:, :4].all(axis=1)
        positions = np.full((len(times), 2, 3), np.nan)
        if np.any(valid):
            poses = pose_matrices(raw["pose"][valid, 2:4, 0]) @ mounts
            positions[valid] = poses[..., :3, 3]
        parts.append({"key": part["key"], "times": times.copy(), "joint": raw["joint_valid"].copy(),
                      "five": hmb_five_rows(raw), "segment": raw["segment_id"].copy(), "valid": valid,
                      "positions": positions, "sources": raw["capture_bounds_ns"][:, 2:4, 0].copy()})
    anchor, primary = choose_anchor(near, PRIMARY, explicit_status)
    _, sensitivity = choose_anchor(near, SENSITIVITY, explicit_status)
    gaps = np.concatenate([np.linalg.norm(p["positions"][p["valid"], 1] - p["positions"][p["valid"], 0], axis=-1) for p in parts])
    stats = {"root": root["root"], "split": root["split"], "task": root["task"], "source_manifest_sha256": root["manifest_sha256"],
             "explicit_event_status": explicit_status, "stable_primary": primary, "stable_sensitivity": sensitivity,
             "original_planned_start_ns": start, "original_planned_last_ns": previous,
             "full_root_denominators": {"partitions": len(parts), "plan_ticks": root["planned_ticks"],
                 "hmb_joint_valid_ticks": int(sum(p["joint"].sum() for p in parts)),
                 "hmb_five_frame_candidate_ticks": int(sum(p["five"].sum() for p in parts))},
             "pose_support_ticks": len(gaps), "median_anatomical_gap_m": float(np.median(gaps)) if len(gaps) else None,
             "ineligibility_reason": None if anchor else primary["reason"]}
    if anchor is None:
        return stats, None
    end = min(previous + TICK_NS, anchor.time + REPLAY_NS, capture_end - origin)
    segments = []
    events = []
    for part in parts:
        use = (anchor.time <= part["times"]) & (part["times"] < end)
        if not use.any():
            continue
        indices = np.flatnonzero(use)
        segments.append({"partition": part["key"], "first_tick": int(indices[0]), "last_tick": int(indices[-1]),
                         "plan_ticks": len(indices), "hmb_joint_valid_ticks": int(part["joint"][use].sum()),
                         "hmb_five_frame_candidate_ticks": int(part["five"][use].sum())})
        events.extend(motion_events(part["times"], part["segment"], part["positions"], part["sources"], part["valid"], use))
    if not segments or origin + anchor.time >= capture_end:
        stats["ineligibility_reason"] = "NO_SIGNED_REPLAY_WINDOW_AFTER_ANCHOR"
        return stats, None
    win = {"source_cutoff_start_ns": origin + anchor.time, "source_cutoff_stop_exclusive_ns": origin + end,
           "hmb_cutoff_start_ns": anchor.time, "hmb_cutoff_stop_exclusive_ns": end, "partitions": segments,
           **{k: sum(part[k] for part in segments) for k in ("plan_ticks", "hmb_joint_valid_ticks", "hmb_five_frame_candidate_ticks")}}
    stats.update(palm=palm_class(anchor.wrists, world_rotation, normals), motion_events=events,
                 contains_start_stop_or_reversal=bool(events), historical_window=win)
    return stats, anchor


def select_six(all_roots):
    if any(row["split"] != "Train" for row in all_roots):
        raise ValueError("Pilot selection can only access Train root features")
    gaps = [root["median_anatomical_gap_m"] for root in all_roots if root["median_anatomical_gap_m"] is not None]
    if not gaps:
        return (), (), None
    lower, upper = np.quantile(gaps, [.25, .75], method="linear")
    if not lower < upper:
        return (), (), [float(lower), float(upper)]
    candidates = []
    for root in all_roots:
        if root["median_anatomical_gap_m"] is None:
            continue
        gap = root["median_anatomical_gap_m"]
        root["gap_stratum"] = "lower_quartile" if gap <= lower else ("upper_quartile" if gap >= upper else "middle")
        if root["ineligibility_reason"] is None:
            candidates.append(root)
    ordered = sorted(candidates, key=lambda root: root["root"])
    for group in combinations(ordered, 6):
        if ({root["palm"]["class"] for root in group} != PALM_CLASSES
                or not {"lower_quartile", "upper_quartile"} <= {root["gap_stratum"] for root in group}
                or sum(root["contains_start_stop_or_reversal"] for root in group) < 2):
            continue
        repeats = tuple(next(root["root"] for root in group if root["gap_stratum"] == key)
                        for key in ("lower_quartile", "upper_quartile"))
        return group, repeats, [float(lower), float(upper)]
    return (), (), [float(lower), float(upper)]


def calibrated_geometry(calibration_path, frames_path, model_path=None):
    cal = json.loads(calibration_path.read_bytes())
    frames_sha = file_sha(frames_path)
    if cal.get("scope") != "MUJOCO_ONLY" or cal.get("status") != "QUALIFIED" or cal["transforms"]["sha256"] != frames_sha:
        raise ValueError("Stage II requires a qualified independent MuJoCo-only calibration bound to the exact frames")
    transforms = np.repeat(np.eye(4)[None], 7, axis=0)
    transforms[:, :3] = np.loadtxt(frames_path).reshape(7, 3, 4)
    proper(transforms)
    axis = cal.get("palm_normal_axis_A")
    normal_provenance = "qualified calibration receipt"
    if axis is None:
        if model_path is None or cal.get("anatomical_frame_reference") != "LIVE_MODEL_PALM_BODY":
            raise ValueError("Calibration does not qualify the signed palm normal axes in A")
        from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel
        model_sha = file_sha(model_path)
        if model_sha != cal["live_model_sha256"]:
            raise ValueError("Palm normal model differs from the qualified live MJCF")
        robot = RobotModel(model_path)
        home = robot.geometry(robot.home)
        knuckles = home.landmarks[:, 1:, 0]
        forward = knuckles.mean(axis=1) - home.palm
        forward /= np.linalg.norm(forward, axis=1, keepdims=True)
        across = knuckles[:, 0] - knuckles[:, -1]
        lateral = across - (across * forward).sum(axis=1, keepdims=True) * forward
        lateral /= np.linalg.norm(lateral, axis=1, keepdims=True)
        rotation = np.swapaxes(home.palm_rotation, -1, -2)
        normal = np.cross(forward, lateral) * np.array([1., -1.])[:, None]
        axis = (rotation @ normal[..., None])[..., 0]
        normal_provenance = f"fixed anatomical convention on exact live-model static Home {model_sha}"
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (2, 3) or not np.isfinite(axis).all() or not np.allclose(np.linalg.norm(axis, axis=1), 1., atol=1e-8):
        raise ValueError("Invalid signed anatomical palm normal axes")
    return transforms, axis, cal, normal_provenance, frames_sha


def train_reader(cohort_path, docs, hmb_products):
    raw = cohort_path.read_bytes()
    cohort = json.loads(raw)
    if (cohort.get("schema") != "s05.teleop.hmb_source_cohort.v1" or cohort.get("native_assets_sha256_checked") is not True
            or cohort.get("identity_sha256") != {key: item[1] for key, item in IDENTITIES.items()}):
        raise ValueError("Expected the byte-verified signed HMB native input receipt")
    if cohort.get("start_and_mode_reconstruction") != "UNKNOWN_NATIVE_INPUT_ASSETS_HAVE_NO_LIFECYCLE_TOPICS":
        raise ValueError("Actual lifecycle evidence must first be adjudicated before choosing an explicit or reconstructed START")
    selected = sorted((row for row in cohort["selected_roots"] if row["split"] == "Train"), key=lambda row: row["root"])
    denom = len(selected), sum(row["partitions"] for row in selected), sum(row["planned_ticks"] for row in selected)
    if denom != EXPECTED_SPLITS["Train"] or len({row["root"] for row in selected}) != len(selected):
        raise ValueError("Selector must use exactly the signed Train-only cohort")
    evidence = docs / "evidence"
    reader = HMBReader(evidence / IDENTITIES["hmb_manifest"][0], evidence / IDENTITIES["plan"][0],
                       evidence / IDENTITIES["config"][0], hmb_products or cohort["signed_hmb_ubuntu_products"],
                       "Train", SIMULATED)
    return selected, reader, sha256(raw).hexdigest()


def verify_native(inputs, data, anchor, assets, cohort_path, cohort_sha, root):
    from nominal_live_replay import signed_events

    native, receipt = signed_events(cohort_path, root["root"], None)
    if receipt["root"] != root or receipt["cohort_receipt_sha256"] != cohort_sha:
        raise ValueError("Raw replay receipt changed the signed input provenance")
    rows = dict(zip(CORE_STREAMS, anchor.sources, strict=True))
    verified = bind_inputs(rows, assets, native, data["source_cutoff_ns"])
    if verified != inputs:
        raise ValueError("Raw native Event binding does not match the frozen HMB anchor")
    window = data["historical_window"]
    replay_roles = {event.topic for event in native if window["source_cutoff_start_ns"] <= event.schedule_ns < window["source_cutoff_stop_exclusive_ns"]}
    if replay_roles != set(inputs):
        raise ValueError("Predeclared historical window does not have all four replay input roles")


def frozen_anchor(stats, anchor, assets, cohort_sha, frames_sha, cal_sha):
    inputs = {}
    source_evidence = {}
    for role, source in zip(CORE_STREAMS, anchor.sources, strict=True):
        asset = assets[role]
        record = source["record"]
        topic = asset["topic"]
        inputs[topic] = {"record_index": record["physical_message"], "cdr_sha256": record["cdr_sha256"]}
        source_evidence[role] = {"topic": topic, "part_id": record["part_id"], "source_instance": record["source_instance"],
                                 "writer_sequence": record["writer_sequence"], "capture_bounds_ns": source["capture_bounds_ns"],
                                 "arrival_bounds_ns": source["arrival_bounds_ns"], "declared_clock_scope": source["clock_generation"]}
    return {"schema": ANCHOR_SCHEMA, "scope": "SIMULATION_ONLY_STAGE_II_DCR009_PILOT",
            "calibration_scope": "MUJOCO_ONLY", "calibration_receipt_sha256": cal_sha, "shared_frames_sha256": frames_sha,
            "cohort_sha256": cohort_sha, "source_manifest_sha256": stats["source_manifest_sha256"],
            "root": stats["root"], "split": "Train", "partition": anchor.part, "partition_tick": anchor.tick,
            "hmb_cutoff_ns": anchor.time, "source_cutoff_ns": stats["historical_window"]["source_cutoff_start_ns"],
            "inputs": inputs, "selected_source_evidence": source_evidence,
            "anatomical_start_h": anchor.wrists.tolist(), "hand_tips_local_start_m": anchor.tips.tolist(),
            **{key: stats[key] for key in ("full_root_denominators", "palm", "stable_primary", "stable_sensitivity",
                                           "explicit_event_status", "historical_window")}}


def build(args):
    transforms, normals, calibration, normal_provenance, frames_sha = calibrated_geometry(args.calibration, args.frames, args.model)
    cohort_rows, reader, cohort_sha = train_reader(args.cohort, args.docs, args.hmb_products)
    cal_sha = file_sha(args.calibration)
    all_train, anchors = [], {}
    for root in cohort_rows:
        _, signed, _, assets, origin, end = read_identity(args.cohort, root["root"])
        stats, anchor = scan_root(reader, signed, transforms[3:5], transforms[0, :3, :3], normals, origin, end,
                                  "UNOBSERVED_IN_SIGNED_NATIVE_INPUTS")
        all_train.append(stats)
        if anchor is not None:
            anchors[root["root"]] = anchor, assets, signed
    signed_denominators = {key: sum(root["full_root_denominators"][key] for root in all_train)
                           for key in ("partitions", "plan_ticks", "hmb_joint_valid_ticks", "hmb_five_frame_candidate_ticks")}
    if tuple(signed_denominators.values()) != (228, 86_338, 85_382, 82_559):
        raise ValueError(f"Unexpected original Train denominators: {signed_denominators}")
    chosen, repeat_roots, quartiles = select_six(all_train)
    selection = []
    files = {}
    failures = []
    for stats in chosen:
        root = stats["root"]
        anchor, assets, signed = anchors[root]
        data = frozen_anchor(stats, anchor, assets, cohort_sha, frames_sha, cal_sha)
        try:
            verify_native(data["inputs"], data, anchor, assets, args.cohort, cohort_sha, signed)
        except (OSError, KeyError, ValueError) as error:
            stats["ineligibility_reason"] = "NATIVE_EVENT_BINDING_FAILED"
            failures.append({"root": root, "reason": str(error)})
            continue
        filename = f"anchors/{root.split(':')[-1]}.json"
        raw = json_bytes(data)
        files[filename] = raw
        selection.append({key: stats[key] for key in ("root", "split", "task", "gap_stratum", "palm",
                                                     "stable_primary", "stable_sensitivity", "historical_window",
                                                     "full_root_denominators", "contains_start_stop_or_reversal")})
        selection[-1].update(anchor_path=filename, anchor_sha256=sha256(raw).hexdigest(), repeat_cold_start=root in repeat_roots,
                             replay_planned_ticks=stats["historical_window"]["plan_ticks"], eligibility="PRIMARY_STABLE_WINDOW_AND_NATIVE_RECORDS_VERIFIED")
    if failures:
        selection, files, repeat_roots = [], {}, ()
    chosen_ids = {row["root"] for row in selection}
    for stats in all_train:
        stats["selection_disposition"] = ("SELECTED" if stats["root"] in chosen_ids else
                                           "INELIGIBLE_PRIMARY" if stats["ineligibility_reason"] else "ELIGIBLE_NOT_SELECTED")
    plan = {"schema": PLAN_SCHEMA, "run_id": RUN_ID, "scope": "SIMULATION_ONLY_STAGE_II_DCR009_PILOT",
            "selection_status": ("PILOT_PLAN_READY_FOR_REVIEW" if selection else
                                 "BLOCKED_NATIVE_EVENT_BINDING" if failures else "INCONCLUSIVE_INSUFFICIENT_TRAIN_SUPPORT"),
            "cohort_sha256": cohort_sha, "shared_frames_sha256": frames_sha, "calibration_receipt_sha256": cal_sha,
            "calibration_scope": calibration["scope"], "palm_normal_basis": normal_provenance,
            "palm_normal_axis_A": normals.tolist(), "full_signed_train_denominators": signed_denominators,
            "selector_sha256": file_sha(Path(__file__)),
            "selection_policy": {"sort": "lexicographic signed root ID", "cohort": "existing signed Train only",
                                 "required_palm_classes": sorted(PALM_CLASSES), "minimum_roots_with_start_stop_or_reversal": 2,
                                 "gap_quartiles_population": "median paired anatomical gap across jointly valid full planned Train root",
                                 "gap_quartiles_m": quartiles, "repeat_policy": "earliest sorted selected low- and high-quartile roots",
                                 "palm_criteria": POISE, "motion_criteria": MOTION},
            "stable_primary_rule": asdict(PRIMARY), "stable_sensitivity_rule": asdict(SENSITIVITY),
            "search_ns": SEARCH_NS, "replay_limit_ns": REPLAY_NS, "attempt_limit": 8, "repeat_roots": list(repeat_roots),
            "selected_roots": selection, "roots": all_train, "native_event_binding_failures": failures,
            "proposed_roots_before_native_readback": [stats["root"] for stats in chosen],
            "selection_consumed_execution_outcomes": False,
            "validation_opened": False, "pilot_approved": False, "full_execution_authorized": False}
    return plan, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cohort", "frames", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--calibration", "--calibration-receipt", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Exact pinned live model, only when CAL approves static Home normal derivation")
    parser.add_argument("--docs", type=Path, default=Path(__file__).resolve().parents[2] / "docs/s0-5")
    parser.add_argument("--hmb-products", type=Path)
    args = parser.parse_args()
    plan, files = build(args)
    if args.out.exists() and (args.out / "pilot_plan.json").exists():
        raise FileExistsError("Pilot selection plan already exists")
    args.out.mkdir(parents=True, exist_ok=True)
    for filename, raw in files.items():
        artifact = args.out / filename
        artifact.parent.mkdir(exist_ok=True)
        with artifact.open("xb") as handle:
            handle.write(raw)
    with (args.out / "pilot_plan.json").open("xb") as handle:
        handle.write(json_bytes(plan))
    print(plan["selection_status"], len(plan["selected_roots"]), file_sha(args.out / "pilot_plan.json"))
    return 0 if files else 3


if __name__ == "__main__":
    raise SystemExit(main())
