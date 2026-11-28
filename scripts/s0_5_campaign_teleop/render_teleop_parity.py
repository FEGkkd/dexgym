#!/usr/bin/env python3
"""Four-column, timestamp-aligned rendering of the archived parity rollout."""

import argparse
import gzip
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dexgym.tasks.direct.wholebody_s05.v1.rmb.schema import file_sha, write_json
from swept_live_guard import verified_robot


MODES = ("A", "B", "C", "D")
CANVAS = (1600, 752)
COLORS = ("#166394", "#a34720")


def trace(draw, xyz, x, y, *, color, scale=360, width=2):
    if xyz is None:
        return
    xyz = np.asarray(xyz, float)
    if xyz.shape != (5, 4, 3) or not np.isfinite(xyz).all():
        raise ValueError("Invalid figure geometry")
    # A fixed oblique camera in each hand's local frame, in metres.
    for finger in xyz:
        points = [(x + scale * (a - .28 * c), y - scale * (b + .22 * c)) for a, b, c in finger]
        draw.line(points, fill=color, width=width)
        xx, yy = points[-1]
        draw.ellipse((xx - 2, yy - 2, xx + 2, yy + 2), fill=color)


def q_geometry(robot, joints):
    return None if joints is None else robot.geometry(np.asarray(joints)).local


def render(args):
    summary_path = args.physics / "fixed-arm-parity-summary.json"
    summary = json.loads(summary_path.read_bytes())
    ledger = args.physics / "fixed-arm-parity.jsonl.gz"
    for name, path in (("ledger", ledger), ("model", args.model), ("model_receipt", args.model_receipt),
                       ("sdk_ticks", args.sdk_ticks)):
        if file_sha(path) != summary["sha256"][name]:
            raise ValueError(f"Parity render identity mismatch: {name}")
    signed = json.loads(args.model_receipt.read_bytes())["model_identity"]["signed"]
    robot = verified_robot(args.model, signed["xml_sha256"], signed)
    sdk = {row["tick_ns"]: row for row in map(json.loads, args.sdk_ticks.read_text(encoding="utf-8").splitlines())}
    manifests = {row["tick_ns"]: row for row in map(json.loads, args.input_ticks.read_text(encoding="utf-8").splitlines())}
    font = ImageFont.truetype("DejaVuSans.ttf", 16)
    head = ImageFont.truetype("DejaVuSans-Bold.ttf", 21)
    args.out.mkdir()
    video = args.out / "teleop-parity-root05.mp4"
    command = ("ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{CANVAS[0]}x{CANVAS[1]}", "-r", "25", "-i", "-", "-an", "-c:v", "libx264",
               "-crf", "22", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video))
    frames = 0
    measured = {mode: 0 for mode in MODES}
    with gzip.open(ledger, "rt", encoding="utf-8") as source, subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        for i, row in enumerate(map(json.loads, source)):
            if i % 2:
                continue
            tick = row["tick_ns"]
            comparison = sdk[tick]
            raw = manifests[tick]
            if comparison["source_index"] != raw["source_index"] or row["source_index"] != raw["source_index"]:
                raise ValueError("Raw, SDK and measured frames have different HMB provenance")
            original = []
            for common, order in zip(raw["common"], raw["order"], strict=True):
                if common is None:
                    original.append(None)
                else:
                    xyz = np.asarray(common)[order]
                    original.append((xyz[1:] - xyz[0]).reshape(5, 4, 3))
            image = Image.new("RGB", CANVAS, "#f3f7fb")
            draw = ImageDraw.Draw(image)
            t = (tick - summary["window"]["hmb_cutoff_start_ns"]) / 1e9
            draw.text((18, 12), f"Root 5 | Fixed arm targets | Recorded timeline {t:06.2f} s | Diagnostic only",
                      font=head, fill="#183247")
            for col, name in enumerate(("MANUS recorded HMB sample", "Wuji raw + independent target",
                                        "Processed goal (nominal FK)", "MuJoCo actual feedback")):
                draw.text((18 + 400 * col, 55), name, font=head, fill="#183247")
            for m, mode in enumerate(MODES):
                y = 93 + 161 * m
                for col in range(4):
                    x = 12 + 400 * col
                    draw.rounded_rectangle((x, y, x + 376, y + 148), 8, fill="#ffffff", outline="#d6e3eb")
                draw.text((19, y + 7), mode, font=head, fill="#183247")
                traces = row["experiments"][mode]
                trace_goal = next((v for v in reversed(traces) if "nominal" in v), None)
                physical = [s for v in traces for s in v.get("physical", [])]
                native_key = "c" if mode == "D" else mode.lower()
                raw_sdk = comparison[f"{native_key}_raw_rad"]
                raw_q = robot.home.copy()
                raw_mask = [q is not None for q in raw_sdk]
                for side, jnts in enumerate(raw_sdk):
                    if jnts is not None:
                        raw_q[14 + side * 20:34 + side * 20] = jnts
                sdk_fk = q_geometry(robot, raw_q) if any(raw_mask) else None
                nominal_fk = q_geometry(robot, trace_goal["nominal"]) if trace_goal else None
                actual_fk = q_geometry(robot, physical[-1]["measured"]) if physical else None
                measured[mode] += bool(physical)
                # The human samples in column 1 share the HMB clock. A's SDK
                # column is the native trace causally sampled at that time.
                for side, color in enumerate(COLORS):
                    x = 123 + 156 * side
                    trace(draw, original[side], x, y + 111, color=color)
                    trace(draw, None if not raw_mask[side] else sdk_fk[side], x + 400, y + 111, color=color)
                    if row["target_local_m"] is not None:
                        trace(draw, row["target_local_m"][side], x + 400, y + 111, color="#9eacb4", width=1)
                    trace(draw, None if nominal_fk is None else nominal_fk[side], x + 800, y + 111, color=color)
                    trace(draw, None if actual_fk is None else actual_fk[side], x + 1200, y + 111, color=color)
                status = f"Actual {len(physical)}/10 steps" if physical else "NOT_EXECUTED"
                draw.text((1220, y + 122), status, font=font, fill="#973819" if len(physical) < 10 else "#23614b")
                if trace_goal:
                    draw.text((820, y + 122), trace_goal["status"], font=font, fill="#405362")
            draw.text((18, 730), "Left and right hands are blue and orange. Grey is the independent engineering shape proxy.",
                      font=font, fill="#405362")
            if not frames:
                image.save(args.out / "poster.png")
            process.stdin.write(image.tobytes())
            frames += 1
        process.stdin.close()
        if process.wait() or frames != 2402:
            raise ValueError("Parity video encoder or frame count failed")
    write_json(args.out / "video.json", {"scope": "DIAGNOSTIC_ONLY", "fps": 25, "frames": frames,
               "actual_feedback_frames": measured, "interpolated": False, "video_sha256": file_sha(video),
               "physics_summary_sha256": file_sha(summary_path), "input_ticks_sha256": file_sha(args.input_ticks)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("physics", "sdk-ticks", "input-ticks", "model", "model-receipt", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    render(parser.parse_args())
