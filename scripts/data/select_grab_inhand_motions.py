"""Rank GRAB windows by right-hand in-hand rotation, translation, and contact exchange."""

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from dexgym.data.grab_mano import FINGER_JOINT_IDS, load_mano_model
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

BODY_RIGHT_HAND_IDS = np.arange(7342, 8129)
FPS = 30
WINDOW_FRAMES = (60, 90)
WINDOW_STRIDE = 30


def _ply_vertices(path: Path, count: int) -> np.ndarray:
    with path.open("rb") as stream:
        while stream.readline().strip() != b"end_header":
            if stream.tell() > 4096:
                raise ValueError(f"Invalid PLY header: {path}")
        vertices = np.frombuffer(stream.read(count * 12), dtype="<f4").reshape(count, 3)
    return vertices.astype(np.float64)


def _finger_body_vertices(data_root: Path, template: str, weights: np.ndarray) -> tuple[list[np.ndarray], dict]:
    hand_path = data_root / template
    body_path = hand_path.with_name(hand_path.name.replace("_rhand.ply", ".ply"))
    hand = _ply_vertices(hand_path, 778)
    body = _ply_vertices(body_path, 10475)
    body_points = body[BODY_RIGHT_HAND_IDS]
    tree = cKDTree(body_points)
    shift = body_points.mean(axis=0) - hand.mean(axis=0)
    for _ in range(15):
        distance, nearest = tree.query(hand + shift)
        matched = distance < 0.012
        if matched.sum() < 500:
            raise ValueError(f"Cannot align right hand to body contact mesh: {template}")
        next_shift = (body_points[nearest[matched]] - hand[matched]).mean(axis=0)
        if np.linalg.norm(next_shift - shift) < 1e-7:
            shift = next_shift
            break
        shift = next_shift
    distance, nearest = tree.query(hand + shift)
    mapped = BODY_RIGHT_HAND_IDS[nearest]
    finger_ids = []
    for chain in FINGER_JOINT_IDS:
        skinning = weights[:, list(chain)].sum(axis=1)
        ids = np.unique(mapped[(skinning > 0.5) & (distance < 0.001)])
        if len(ids) < 80:
            raise ValueError(f"Too few mapped finger contact vertices in {template}: {len(ids)}")
        finger_ids.append(ids)
    quality = {
        "template": template,
        "matched_mano_vertices_under_1mm": int((distance < 0.001).sum()),
        "median_template_mapping_mm": float(np.median(distance) * 1000),
        "finger_body_vertex_counts": [len(ids) for ids in finger_ids],
    }
    return finger_ids, quality


def _contact_exchange(active: np.ndarray) -> tuple[int, int, int, int]:
    bins = np.asarray([active[i:i + 8].mean(axis=0) >= 0.5 for i in range(0, len(active), 8)])
    changes = np.diff(bins.astype(np.int8), axis=0)
    if len(changes) > 2:
        changes = changes[1:-1]  # exclude grasp onset and final release
    gains = int((changes > 0).sum())
    losses = int((changes < 0).sum())
    changed_fingers = int(np.any(changes != 0, axis=0).sum())
    changes_total = int(np.any(changes != 0, axis=1).sum())
    return gains, losses, changed_fingers, changes_total


def _longest_no_contact_gap(active_count: np.ndarray) -> int:
    longest = current = 0
    for absent in active_count == 0:
        current = current + 1 if absent else 0
        longest = max(longest, current)
    return longest


def _score_window(start: int, stop: int, relative_rotation: Rotation, relative_position: np.ndarray,
                  finger_active: np.ndarray, record: dict, object_name: str) -> dict:
    beginning = relative_rotation[start:start + 5].mean()
    ending = relative_rotation[stop - 5:stop].mean()
    rotation_deg = float(np.degrees((beginning.inv() * ending).magnitude()))
    translation_mm = float(np.linalg.norm(
        np.median(relative_position[stop - 5:stop], axis=0)
        - np.median(relative_position[start:start + 5], axis=0)
    ) * 1000)
    active = finger_active[start:stop]
    active_count = active.sum(axis=1)
    hand_contact = float((active_count >= 1).mean())
    two_finger_contact = float((active_count >= 2).mean())
    endpoint_frames = min(10, len(active) // 4)
    start_contact = float((active_count[:endpoint_frames] >= 1).mean())
    end_contact = float((active_count[-endpoint_frames:] >= 1).mean())
    no_contact_gap = _longest_no_contact_gap(active_count)
    gains, losses, changed_fingers, changes_total = _contact_exchange(active)
    held = (hand_contact >= 0.80 and two_finger_contact >= 0.40 and start_contact >= 0.70
            and end_contact >= 0.70 and no_contact_gap <= 9)
    rotation = held and rotation_deg >= 45 and translation_mm <= 250
    translation = held and 30 <= translation_mm <= 180 and rotation_deg <= 45
    regrasp = (hand_contact >= 0.75 and two_finger_contact >= 0.35 and start_contact >= 0.70
               and end_contact >= 0.70 and no_contact_gap <= 9 and gains >= 1 and losses >= 1
               and changed_fingers >= 2 and changes_total >= 2)
    return {
        "source": record["origin_source"],
        "processed_file": record["file"],
        "split": record["split"],
        "object": object_name,
        "start_frame": start,
        "stop_frame_exclusive": stop,
        "start_s": round(start / FPS, 3),
        "stop_s": round(stop / FPS, 3),
        "processed_start_frame": start - int(record["source_start_frame"]),
        "processed_stop_frame_exclusive": stop - int(record["source_start_frame"]),
        "relative_rotation_deg": round(rotation_deg, 2),
        "relative_translation_mm": round(translation_mm, 2),
        "hand_contact_fraction": round(hand_contact, 3),
        "two_finger_contact_fraction": round(two_finger_contact, 3),
        "start_contact_fraction": round(start_contact, 3),
        "end_contact_fraction": round(end_contact, 3),
        "max_no_contact_gap_frames": no_contact_gap,
        "finger_contact_fraction": [round(float(x), 3) for x in active.mean(axis=0)],
        "contact_gains": gains,
        "contact_losses": losses,
        "fingers_changed": changed_fingers,
        "contact_state_changes": changes_total,
        "rotation_candidate": rotation,
        "translation_candidate": translation,
        "regrasp_candidate": regrasp,
    }


def scan_source(source: Path, records: list[dict], finger_ids: list[np.ndarray]) -> list[dict]:
    with np.load(source, allow_pickle=True) as raw:
        hand = raw["rhand"].item()["params"]
        obj = raw["object"].item()["params"]
        fps = float(raw["framerate"])
        stride = round(fps / FPS)
        if not np.isclose(fps / FPS, stride) or stride < 1:
            raise ValueError(f"Unsupported source frame rate {fps}: {source}")
        indices = np.arange(0, len(hand["transl"]), stride)
        hand_rotation = Rotation.from_rotvec(np.asarray(hand["global_orient"])[indices])
        object_rotation = Rotation.from_rotvec(np.asarray(obj["global_orient"])[indices])
        relative_rotation = hand_rotation.inv() * object_rotation
        relative_position = hand_rotation.inv().apply(
            np.asarray(obj["transl"])[indices] - np.asarray(hand["transl"])[indices]
        )
        body_contact = raw["contact"].item()["body"]
        contact_30hz = body_contact[indices]
        finger_counts = np.stack(
            [np.count_nonzero(contact_30hz[:, ids], axis=1) for ids in finger_ids], axis=1
        )
        object_name = str(raw["obj_name"].item())
    finger_active = uniform_filter1d((finger_counts >= 2).astype(np.float32), size=7, axis=0) >= 4 / 7
    rows = []
    for record in records:
        segment_start = int(record["source_start_frame"])
        segment_stop = int(record["source_stop_frame_exclusive"])
        if segment_stop > len(indices):
            raise ValueError(f"Processed segment exceeds source motion: {source}")
        for length in WINDOW_FRAMES:
            if segment_stop - segment_start < length:
                continue
            starts = set(range(segment_start, segment_stop - length + 1, WINDOW_STRIDE))
            starts.add(segment_stop - length)
            for start in sorted(starts):
                rows.append(_score_window(
                    start, start + length, relative_rotation, relative_position,
                    finger_active, record, object_name,
                ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/ubuntu/dexgym/data"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("/home/ubuntu/dexgym/data/processed/hand21_right_geometry_feasible/manifest.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("/home/ubuntu/dexgym/artifacts/hand21/grab_inhand_windows.json"))
    parser.add_argument("--limit", type=int, help="Analyze the first N source files for a quick check")
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    manifest = json.loads(args.manifest.expanduser().read_text())
    by_source = defaultdict(list)
    for record in manifest["motions"]:
        by_source[record["origin_source"]].append(record)
    sources = sorted(by_source)
    if args.limit is not None:
        sources = sources[:args.limit]
    model = load_mano_model(data_root / "models/mano_v1_2/MANO_RIGHT.pkl")
    mapping = {}
    rows, errors = [], []
    for index, name in enumerate(sources, 1):
        records = by_source[name]
        subject = name.split("/")[0]
        try:
            if subject not in mapping:
                mapping[subject] = _finger_body_vertices(data_root, records[0]["template"], model.weights)
            rows.extend(scan_source(data_root / "grab" / name, records, mapping[subject][0]))
        except (KeyError, ValueError, OSError) as error:
            errors.append({"source": name, "error": f"{type(error).__name__}: {error}"})
        if index % 100 == 0 or index == len(sources):
            print(f"SCAN_PROGRESS {index}/{len(sources)} windows={len(rows)} errors={len(errors)}", flush=True)
            gc.collect()
    result = {
        "definition": "30 Hz right-hand relative object motion with mapped GRAB body-to-object finger contacts",
        "source_count": len(sources),
        "window_count": len(rows),
        "criteria": {
            "window_lengths_frames": list(WINDOW_FRAMES),
            "hand_contact_fraction_min": 0.80,
            "two_finger_contact_fraction_min": 0.40,
            "endpoint_contact_fraction_min": 0.70,
            "max_no_contact_gap_frames": 9,
            "rotation_min_deg": 45,
            "rotation_max_translation_mm": 250,
            "translation_min_mm": 30,
            "translation_max_mm": 180,
            "translation_max_rotation_deg": 45,
            "contact_smoothing_frames": 7,
            "contact_vertices_per_finger_min": 2,
        },
        "mapping_quality": {name: value[1] for name, value in mapping.items()},
        "errors": errors,
        "windows": rows,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"SCAN_DONE {output} windows={len(rows)} errors={len(errors)}", flush=True)


if __name__ == "__main__":
    main()
