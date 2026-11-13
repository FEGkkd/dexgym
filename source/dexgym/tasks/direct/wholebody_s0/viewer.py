"""Create an offline browser bundle for the original samples and 54-DoF reference."""

from base64 import b64encode
import json
from pathlib import Path
from shutil import copyfile

import numpy as np

from .bag import digest
from .library import MotionLibrary
from .native_replay import checked_clip
from .pipeline import read_json, read_npz
from .robot import Robot

STATIC = Path(__file__).parent / "web"


def packed(values, scale=None):
    values = np.asarray(values)
    finite = np.isfinite(values)
    if scale and (not finite.any() or np.max(np.abs(values[finite])) / scale < 32766):
        encoded = np.full(values.shape, -32768, dtype="<i2")
        encoded[finite] = np.rint(values[finite] / scale).astype("<i2")
        kind = "i16"
    elif values.dtype == bool:
        encoded = values.astype("u1")
        kind = "u8"
    else:
        encoded = values.astype("<f4")
        kind, scale = "f32", None
    return {"kind": kind, "scale": scale, "shape": values.shape,
            "data": b64encode(encoded.tobytes()).decode("ascii")}


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def export_retimed(library, entries, root, robot):
    retimed = [entry for entry in entries if entry.get("timing") == "retimed"]
    if retimed:
        (root / "clips").mkdir(exist_ok=True)
    for entry in retimed:
        clip = checked_clip(library, entry)
        frames = entry["frames"]
        fields = {key: packed(clip[key], 0.0001) for key in ("q_ref", "target_tcp_world")}
        fields["robot_wrist_pose"] = packed(clip["robot_wrist_pose"], 0.0001)
        fields["robot_skeleton"] = packed(robot.geometry(clip["q_ref"])["robot_skeleton"], 0.0001)
        fields.update({key: packed(clip[key], 0.00001) for key in (
            "static_penetration_m", "arm_position_error_m", "arm_orientation_error_rad",
        )})
        for i, side in enumerate(("left", "right")):
            fields[f"tracker_{side}"] = packed(clip["tracker_pose"][:, i], 0.0001)
            fields[f"manus_{side}"] = packed(clip[f"manus_{side}_pose"][..., :3], 0.0001)
        valid = packed(np.ones(frames, dtype=bool))
        for key in ("manus_left_valid", "manus_right_valid", "tracker_left_valid", "tracker_right_valid",
                    "tracker_safety_valid", "human_valid", "reference_usable", "in_library"):
            fields[key] = valid
        source_seconds = (clip["source_time_ns"] - entry["episode_start_time_ns"]) / 1e9
        fields["source_time_s"] = packed(source_seconds)
        payload = {"id": entry["id"], "sourceEpisode": entry["source_episode"], "fields": fields}
        (root / "clips" / f"{entry['id']}.js").write_text(
            "window.wholebodyReceiveClip(" + compact_json(payload) + ");\n", encoding="utf-8"
        )


def export_viewer(output):
    dataset = read_json(output / "dataset.json")
    root = output / "viewer"
    episodes = root / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "style.css", "app.js"):
        copyfile(STATIC / name, root / name)
    index = {
        "date": dataset["recording_date"], "timezone": dataset["timezone"], "rate": 50,
        "jointNames": dataset["joint_names"], "robotLinks": dataset["robot_links"], "splits": dataset["splits"],
        "episodes": [],
    }
    final_path = output / "native_replay/motion_library/manifest.json"
    final = MotionLibrary.open(final_path) if final_path.is_file() else None
    final_by_episode = {}
    if final is not None:
        index["splits"] = final.manifest["splits"]
        index["replayApproved"] = True
        for entry in final.manifest["clips"]:
            final_by_episode.setdefault(entry["source_episode"], []).append(entry)
    retimed = []
    for episode in dataset["episodes"]:
        episode_id = episode["id"]
        folder = output / "episodes" / episode_id
        quality = read_json(folder / "quality.json")
        config = read_json(folder / "configuration.json")
        aligned = read_npz(folder / "aligned.npz")
        trace = read_npz(folder / "reference.npz") if episode["reference"]["status"] == "built" else {}
        entries = final_by_episode.get(episode_id, [])
        slow = [entry for entry in entries if entry.get("timing") == "retimed"]
        retimed.extend({**entry, "episode_start_time_ns": int(aligned["time_ns"][0])} for entry in slow)
        approved = np.zeros(len(aligned["time_ns"]), dtype=bool)
        for entry in entries:
            if entry.get("timing", "original") == "original":
                approved[entry["source_start_frame"]:entry["source_stop_frame_exclusive"]] = True
        payload = {
            "id": episode_id, "fields": {}, "topology": {}, "streams": quality["streams"],
            "calibration": quality["calibration"], "configurationOrigin": config["origin"],
            "reference": episode["reference"], "alignment": quality["alignment"],
            "captureControl": quality["capture_control_state"],
        }
        for key in ("manus_left", "manus_right", "tracker_left", "tracker_right",
                    "tracker_left_raw", "tracker_right_raw"):
            pose = aligned.get(key + "_pose")
            if pose is not None:
                if key.startswith("manus_"):
                    payload["fields"][key] = packed(pose[..., :3], 0.0001)
                    payload["topology"][key] = quality["streams"][key]["topology"]
                else:
                    payload["fields"][key] = packed(pose, 0.0001)
            payload["fields"][key + "_valid"] = packed(aligned[key + "_valid"])
        for key in ("human_valid", "tracker_safety_valid"):
            payload["fields"][key] = packed(aligned[key])
        for key, scale in (
            ("q_ref", 0.0001), ("robot_skeleton", 0.0001), ("robot_wrist_pose", 0.0001),
            ("target_tcp_world", 0.0001), ("native_tcp_world", 0.0001), ("static_penetration_m", 0.00001),
            ("arm_position_error_m", 0.0001), ("arm_orientation_error_rad", 0.0001),
            ("reference_usable", None), ("in_library", None), ("rejection_bits", None), ("arm_accepted", None),
        ):
            if key in trace:
                payload["fields"][key] = packed(trace[key], scale)
        if final is not None:
            payload["fields"]["in_library"] = packed(approved)
        (episodes / f"{episode_id}.js").write_text(
            "window.wholebodyReceive(" + compact_json(payload) + ");\n", encoding="utf-8"
        )
        copyfile(folder / "quality.json", episodes / f"{episode_id}-quality.json")
        copyfile(folder / "retarget_quality.json", episodes / f"{episode_id}-retarget.json")
        wrist_gaps = []
        for side in ("left", "right"):
            stream = quality["streams"][f"tracker_{side}"]
            timing = stream.get("header_clock", {}).get("interval_ms") or {}
            wrist_gaps.append(max(timing.get("max", 0), stream.get("capture_last_to_stop_ms", 0)))
        index["episodes"].append({
            "id": episode_id, "task": episode["task"], "trial": episode["trial"],
            "duration": episode["duration_s"], "start": episode["start_local"],
            "frames": episode["alignment"]["frames"], "synced": episode["alignment"]["valid_frames"],
            "captureQuality": episode["capture_quality"], "integrity": episode["integrity_status"],
            "reference": episode["reference"]["status"],
            "libraryFrames": (sum(entry["frames"] for entry in entries) if final is not None
                              else episode["reference"].get("library_frames", 0)),
            "clips": len(entries) if final is not None else len(episode["reference"]["clips"]),
            "split": episode["split"],
            "retimed": [{"id": entry["id"], "frames": entry["frames"], "factor": entry["time_dilation_factor"],
                         "sourceFrame": entry["source_start_frame"]} for entry in slow],
            "trackerFrame": episode["tracker_frame"], "missingRobot": episode["missing_robot_streams"],
            "maxWristGapMs": max(wrist_gaps),
        })
    if retimed:
        info = final.manifest["provenance"]["robot_model"]
        if digest(info["path"]) != info["sha256"]:
            raise ValueError("Viewer robot model does not match the final reference library")
        robot = Robot(info["path"])
        export_retimed(final, retimed, root, robot)
    (root / "dataset.js").write_text("window.WHOLEBODY_DATASET=" + compact_json(index) + ";\n", encoding="utf-8")
    print(f"{len(index['episodes'])} episode payloads. Open {root / 'index.html'}")
