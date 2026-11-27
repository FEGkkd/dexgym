#!/usr/bin/env python3
"""Render named feedback from the live MuJoCo replay and export its issued commands."""

import argparse
import bisect
import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

import mujoco
import numpy as np


ARM = ("/left_arm/joint_states", "/right_arm/joint_states")
HAND = ("/left_hand/state", "/right_hand/state")
COMMANDS = {"/tianji/joint_command", "/left_hand/command", "/right_hand/command"}
FINGER = dict(zip(("TH", "FF", "MF", "RF", "LF"),
                  ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")))
JOINTS = {
    finger: (("cmc_flex", "cmc_abd", "mcp", "ip") if finger == "TH"
             else ("mcp_flex", "mcp_abd", "pip", "dip"))
    for finger in FINGER
}


def qpos_index(model, name):
    if name.startswith(("left_j", "right_j")):
        side, number = name.split("_j")
        candidates = [f"Joint{number}_{side[0].upper()}"]
    else:
        side, stem = name.split("_")
        finger, index = stem.split("J")
        alias = f"{side}_{FINGER[finger]}_{JOINTS[finger][int(index)]}"
        candidates = [name, alias]
        if finger == "LF":
            candidates.extend([alias.replace("_pinky_", "_pinky_finger_"),
                               alias.replace("_pinky_", "_little_finger_")])
    for candidate in candidates:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, candidate)
        if joint >= 0:
            return model.jnt_qposadr[joint]
    raise ValueError(f"No model joint for {name}, checked {candidates}")


def load_trace(trace, command_csv):
    frames = collections.defaultdict(dict)
    counts = collections.Counter()
    starts = {}
    active_model = None
    ages = []
    last_input_offset = 0.0
    opener = gzip.open if trace.suffix == ".gz" else open
    with opener(trace, "rt") as source, gzip.open(command_csv, "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("observed_original_log_ns", "message_stamp_ns",
                         "topic", "sequence", "session_id", "source", "joint_name", "issued_rad"))
        for line in source:
            row = json.loads(line)
            if row["kind"] == "model":
                active_model = row
            if row["stage"] != "historical_human_inputs":
                continue
            if row["kind"] == "timebase":
                starts = row
                continue
            topic = row["topic"]
            if row["kind"] == "input":
                counts["input " + topic] += 1
                input_ns = row.get("original_schedule_ns", row["original_log_ns"])
                offset = input_ns - starts.get("origin_schedule_ns", starts["origin_log_ns"])
                last_input_offset = offset / 1e9
                ages.append((row["mono_ns"] - starts["origin_mono_ns"] - offset) / 1e6)
                continue
            value = row["value"]
            counts[topic] += 1
            stamp = value["header"]["stamp"]
            stamp_ns = stamp["sec"] * 1_000_000_000 + stamp["nanosec"]
            if topic in ARM + HAND:
                if topic in HAND and (not value["connected"] or not all(value["position_valid"])
                                      or not all(value["online"]) or any(value["stale"])):
                    continue
                frames[stamp_ns][topic] = (row["log_est_ns"], dict(zip(
                    value.get("name", value.get("joint_names")), value["position"], strict=True
                )))
            elif topic in COMMANDS:
                positions = value.get("positions", value.get("position"))
                for name, position in zip(value["joint_names"], positions, strict=True):
                    writer.writerow((row["log_est_ns"], stamp_ns, topic, value["sequence"],
                                     value.get("session_id", ""), value.get("source", ""),
                                     name, position))
    samples = []
    subscriber_lags = []
    for stamp in sorted(frames):
        groups = frames[stamp]
        if set(groups) != set(ARM + HAND):
            continue
        arrival = max(group[0] for group in groups.values())
        qpos = {name: value for topic in ARM + HAND for name, value in sorted(groups[topic][1].items())}
        if len(qpos) != 54 or not all(math.isfinite(value) for value in qpos.values()):
            raise ValueError(f"Invalid joint manifest or feedback at {stamp}")
        feedback_ns = stamp - starts["epoch_delta_ns"]
        offset = (feedback_ns - starts["origin_log_ns"]) / 1e9
        if 0 <= offset <= last_input_offset:
            samples.append((offset, qpos))
            subscriber_lags.append((arrival - feedback_ns) / 1e6)
    if not samples:
        raise ValueError("No coherent 54-joint frames found")
    samples.sort(key=lambda sample: sample[0])
    return samples, dict(sorted(counts.items())), ages, subscriber_lags, active_model


def render(model, samples, output, poster):
    times = [sample[0] for sample in samples]
    addresses = {name: qpos_index(model, name) for name in samples[0][1]}
    if len(set(addresses.values())) != 54:
        raise ValueError("Joint name resolution is not one-to-one")
    width, height, fps = 720, 540, 30
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    origins = np.array(list(samples[0][1].values()))
    contrast = [np.linalg.norm(np.array(list(sample[1].values())) - origins) for sample in samples]
    poster_index = int(round((times[int(np.argmax(contrast))] - times[0]) * fps))
    label = (
        "drawbox=x=0:y=0:w=iw:h=52:color=0x132333@0.93:t=fill,"
        "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
        "text='S0-5 | Native Teleop nominal replay':x=18:y=8:fontsize=16:fontcolor=white,"
        "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
        "text='MuJoCo feedback  /  54 joints  /  1x  /  diagnostic START':x=18:y=30:fontsize=12:fontcolor=0xb9d2e6"
    )
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "-", "-vf", label, "-c:v", "libx264", "-threads", "3",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        with subprocess.Popen(args, stdin=subprocess.PIPE) as encoder:
            for frame in range(int((times[-1] - times[0]) * fps) + 1):
                index = bisect.bisect_right(times, times[0] + frame / fps) - 1
                for name, value in samples[index][1].items():
                    data.qpos[addresses[name]] = value
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera="home_front_top")
                encoder.stdin.write(renderer.render().tobytes())
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("ffmpeg video encoding failed")
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", str(poster_index / fps), "-i", str(output), "-frames:v", "1", str(poster)],
                   check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    video = args.output.with_suffix(".mp4")
    samples, counts, ages, subscriber_lags, active_model = load_trace(args.trace, args.output.with_suffix(".csv.gz"))
    with args.model.open("rb") as stream:
        model_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if active_model is not None and model_sha256 != active_model["sha256"]:
        raise ValueError("Rendering model differs from the model loaded by the simulator")
    names = tuple(samples[0][1])
    positions = np.array([[qpos[name] for name in names] for _, qpos in samples])
    result = {
        "coherent_feedback_samples": len(samples),
        "first_feedback_original_offset_seconds": samples[0][0],
        "last_feedback_original_offset_seconds": samples[-1][0],
        "input_publish_lag_p50_p95_p99_max_ms": np.percentile(ages, [50, 95, 99, 100]).tolist(),
        "subscriber_lag_p50_p95_p99_max_ms": np.percentile(subscriber_lags, [50, 95, 99, 100]).tolist(),
        "max_joint_range_deg": dict(zip(names, np.degrees(np.ptp(positions, axis=0)).tolist())),
        "render_model_sha256": model_sha256,
        "loaded_model_sha256_verified": active_model is not None,
        "records": counts,
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    render(model, samples, video, args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
