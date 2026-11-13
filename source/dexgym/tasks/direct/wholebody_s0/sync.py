"""Timestamp alignment with explicit interpolation and source availability."""

import numpy as np
from scipy.spatial.transform import Rotation

from . import SIDES
from .bag import EPOCH_NS, MAIN_STREAMS, MS_NS, stats

RATE_HZ = 50
PERIOD_NS = EPOCH_NS // RATE_HZ
MANUS_GAP_NS = 40 * MS_NS
TRACKER_GAP_NS = 25 * MS_NS


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(edges[::2], edges[1::2]))


def slerp(a, b, fraction):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    angle = np.arccos(np.clip(np.abs(dot), 0, 1))
    # sinc gives the linear limit when the angle tends to zero.
    w0 = (1 - fraction) * np.sinc((1 - fraction) * angle / np.pi) / np.sinc(angle / np.pi)
    w1 = fraction * np.sinc(fraction * angle / np.pi) / np.sinc(angle / np.pi)
    value = w0 * a + w1 * b
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def sample_indices(stamps, valid, query, max_gap, omit=None):
    """Find bracket indices without crossing an invalid sample or a long gap."""
    valid = np.asarray(valid, dtype=bool)
    keep = stamps > 0
    if omit is not None:
        keep &= ~omit | ~valid
    order = np.flatnonzero(keep)
    order = order[np.argsort(stamps[order], kind="stable")]
    if len(order) == 0:
        return np.full((len(query), 2), -1, dtype=np.int64)
    unique = np.r_[True, np.diff(stamps[order]) > 0]
    order = order[unique]
    time = stamps[order]
    insertion = np.searchsorted(time, query)
    right = np.minimum(insertion, len(time) - 1)
    exact = time[right] == query
    left = np.where(exact, right, np.maximum(insertion - 1, 0))
    a, b = order[left], order[right]
    inside = (time[left] <= query) & (query <= time[right])
    good = inside & valid[a] & valid[b] & ((time[right] - time[left]) <= max_gap)
    return np.where(good[:, None], np.column_stack((a, b)), -1)


def interpolate(values, stamps, query, indexes, *, poses=False):
    out = np.full((len(query), *values.shape[1:]), np.nan, dtype=np.float32)
    valid = indexes[:, 0] >= 0
    a, b = indexes[valid].T
    if not len(a):
        return out
    dt = np.maximum(stamps[b] - stamps[a], 1)
    ratio = (query[valid] - stamps[a]) / dt
    ratio = ratio.reshape((-1,) + (1,) * (values.ndim - 1))
    if poses:
        out[valid, ..., :3] = (1 - ratio) * values[a, ..., :3] + ratio * values[b, ..., :3]
        out[valid, ..., 3:] = slerp(values[a, ..., 3:], values[b, ..., 3:], ratio)
    else:
        out[valid] = (1 - ratio) * values[a] + ratio * values[b]
    return out


def causal_index(track, query, max_age):
    arrival = track.arrays["receive_ros_ns"]
    order = np.argsort(arrival, kind="stable")
    latest = np.searchsorted(arrival[order], query, side="right") - 1
    index = order[np.maximum(latest, 0)]
    age = query - track.header[index]
    good = (latest >= 0) & track.valid[index] & (age >= 0) & (age <= max_age)
    return np.where(good, index, -1)


def validate_calibration(tracks, configuration):
    """Compare raw and calibrated wrists at the exact same ROS header stamp."""
    if configuration is None:
        return {"status": "snapshot_unavailable"}
    result = {"status": "checked", "sides": {}}
    for side in SIDES:
        raw, calibrated = tracks.get(f"tracker_{side}_raw"), tracks.get(f"tracker_{side}")
        if raw is None or calibrated is None:
            result["sides"][side] = {"status": "stream_unavailable"}
            continue
        t, a, b = np.intersect1d(raw.header, calibrated.header, return_indices=True)
        good = raw.valid[a] & calibrated.valid[b]
        a, b = a[good], b[good]
        if not len(a):
            result["sides"][side] = {"status": "no_valid_matches", "matched_headers": int(len(t))}
            continue
        p = raw.arrays["pose"][a]
        q = calibrated.arrays["pose"][b]
        raw_rotation = Rotation.from_quat(p[:, 3:])
        offset = configuration[f"{side}_wrist_translation"]
        delta = Rotation.from_quat(configuration[f"{side}_wrist_quaternion_xyzw"])
        position_error = np.linalg.norm(p[:, :3] + raw_rotation.apply(offset) - q[:, :3], axis=1)
        angular_error = (Rotation.from_quat(q[:, 3:]).inv() * raw_rotation * delta).magnitude()
        result["sides"][side] = {
            "status": "checked", "matched_headers": int(len(t)), "compared_valid_pairs": int(len(a)),
            "position_error_mm": stats(1000 * position_error),
            "angular_error_deg": stats(np.degrees(angular_error)),
            "calibration_mismatch_count": int(np.count_nonzero((position_error > 0.0001) | (angular_error > np.deg2rad(0.02)))),
        }
    return result


def align_episode(tracks, start_ns, stop_ns):
    query = np.arange(start_ns, stop_ns, PERIOD_NS, dtype=np.int64)
    aligned = {"time_ns": query, "time_s": (query - start_ns) / EPOCH_NS}
    alignment = {"rate_hz": RATE_HZ, "frames": len(query), "max_manus_gap_ms": MANUS_GAP_NS / MS_NS,
                 "max_tracker_gap_ms": TRACKER_GAP_NS / MS_NS, "streams": {}}
    for key in (*MAIN_STREAMS, "tracker_left_raw", "tracker_right_raw"):
        track = tracks.get(key)
        max_gap = MANUS_GAP_NS if key.startswith("manus") else TRACKER_GAP_NS
        if track is None or "pose" not in track.arrays:
            aligned[key + "_valid"] = np.zeros(len(query), dtype=bool)
            alignment["streams"][key] = {"valid_frames": 0}
            continue
        omit = track.arrays["duplicate_source"] if key.startswith("manus") else None
        indexes = sample_indices(track.header, track.valid, query, max_gap, omit)
        valid = indexes[:, 0] >= 0
        aligned[key + "_pose"] = interpolate(track.arrays["pose"], track.header, query, indexes, poses=True)
        aligned[key + "_valid"] = valid
        aligned[key + "_brackets"] = indexes
        aligned[key + "_causal_index"] = causal_index(track, query, max_gap)
        ages = np.abs(query[valid, None] - track.header[indexes[valid]]) / MS_NS
        alignment["streams"][key] = {
            "valid_frames": int(valid.sum()), "causal_available_frames": int((aligned[key + "_causal_index"] >= 0).sum()),
            "nearest_header_offset_ms": stats(ages.min(axis=1)),
        }
    safety = tracks.get("tracker_safety")
    if safety is not None:
        safety_index = causal_index(safety, query, TRACKER_GAP_NS)
        aligned["tracker_safety_valid"] = safety_index >= 0
    else:
        aligned["tracker_safety_valid"] = np.ones(len(query), dtype=bool)
    both = np.logical_and.reduce([aligned[key + "_valid"] for key in MAIN_STREAMS])
    both &= aligned["tracker_safety_valid"]
    aligned["human_valid"] = both
    alignment["safety_stream_present"] = safety is not None
    alignment["valid_frames"] = int(both.sum())
    alignment["invalid_intervals_s"] = [[a / RATE_HZ, b / RATE_HZ] for a, b in runs(~both)]
    for left, right, name in (("manus_left", "manus_right", "manus"), ("tracker_left", "tracker_right", "tracker")):
        if left in tracks and right in tracks and both.any():
            stamps = []
            for key in (left, right):
                clocks = tracks[key].header[aligned[key + "_brackets"][both]]
                nearest = np.abs(query[both, None] - clocks).argmin(axis=1)
                stamps.append(clocks[np.arange(len(nearest)), nearest])
            alignment[name + "_left_right_nearest_header_skew_ms"] = stats(np.abs(stamps[0] - stamps[1]) / MS_NS)
    return aligned, alignment
