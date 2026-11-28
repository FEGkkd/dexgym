#!/usr/bin/env python3
"""Render original hand geometry, independent intent, native commands, and measured FK."""

import argparse
import gzip
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

from dexgym.tasks.direct.wholebody_s05.v1.rmb.schema import file_sha, write_json
from render_teleop_parity import CANVAS, COLORS, MODES, q_geometry, trace
from swept_live_guard import verified_robot


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def render(args):
    execution = json.loads((args.physics / "fixed-arm-native-summary.json").read_bytes())
    functional = json.loads((args.functional / "summary.json").read_bytes())
    raw = args.physics / "fixed-arm-native.jsonl.gz"
    goal = args.functional / "rows.jsonl.gz"
    if (file_sha(raw) != execution["sha256"]["ledger"] or file_sha(goal) != functional["sha256"]["ledger"]
            or file_sha(args.model) != execution["sha256"]["model"]
            or file_sha(args.model_receipt) != execution["sha256"]["model_receipt"]):
        raise ValueError("Video inputs differ from the two independent receipts")
    signed = json.loads(args.model_receipt.read_bytes())["model_identity"]["signed"]
    robot = verified_robot(args.model, signed["xml_sha256"], signed)
    handmade = {r["tick_ns"]: r for r in read(goal)}
    font = ImageFont.truetype("DejaVuSans.ttf", 16)
    bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    args.out.mkdir()
    video = args.out / "root05-native-event-and-intent.mp4"
    command = ("ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{CANVAS[0]}x{CANVAS[1]}", "-r", "25", "-i", "-", "-an", "-c:v", "libx264",
               "-crf", "22", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video))
    frames = 0
    visible = {mode: 0 for mode in MODES}
    with subprocess.Popen(command, stdin=subprocess.PIPE) as pipe:
        for i, row in enumerate(read(raw)):
            if i % 2:
                continue
            tick = row["tick_ns"]
            b = handmade[tick]
            if (b["partition"], b["partition_tick"], b["source_index"]) != (
                    row["partition"], row["partition_tick"], row["input_source_index"]):
                raise ValueError("Video frame joins nonidentical signed history")
            elapsed = (tick - execution["window"]["hmb_cutoff_start_ns"]) / 1e9
            hands = b.get("HUMAN_FUNCTIONAL_INTENT") or {}
            witness = b.get("KINEMATIC_WITNESS") or {}
            permitted = witness.get("status") == "SIGNED_HOME_PATH_PERMITTED"
            adapted = witness if permitted else b.get("ROBOT_FUNCTIONAL_CANDIDATE")
            image = Image.new("RGB", CANVAS, "#f3f7fb")
            draw = ImageDraw.Draw(image)
            draw.text((18, 13), f"Root 5   Native event replay   {elapsed:06.2f} s   |   DIAGNOSTIC ONLY",
                      fill="#183247", font=bold)
            for col, label in enumerate(("Original MANUS", "Independent hand intent", "Wuji and native driver target",
                                        "MuJoCo measured")):
                draw.text((18 + 400 * col, 56), label, fill="#183247", font=bold)
            for m, mode in enumerate(MODES):
                y = 92 + 160 * m
                for col in range(4):
                    x = 12 + 400 * col
                    draw.rounded_rectangle((x, y, x + 376, y + 148), 8, fill="#fff", outline="#d6e3eb")
                draw.text((20, y + 8), f"{mode}   {['dense reference','HMB reference','native replay','native + V2'][m]}",
                          fill="#183247", font=font)
                steps = row["experiments"][mode]
                decision = next((s for s in reversed(steps) if "q_driver_command" in s), None)
                actual = row["fk_at_last_observed_substep"].get(mode)
                visible[mode] += bool(actual)
                driver_fk = q_geometry(robot, decision["q_driver_command"]) if decision else None
                q_raw = robot.home.copy()
                sdk_mask = [False, False]
                if decision:
                    for side, command_state in enumerate(decision["sources"]):
                        if command_state is not None:
                            sdk_mask[side] = True
                            q_raw[14 + 20 * side:34 + 20 * side] = command_state["raw_rad"]
                sdk_fk = q_geometry(robot, q_raw) if any(sdk_mask) else None
                for side, label in enumerate(("left", "right")):
                    x, color = 118 + 156 * side, COLORS[side]
                    raw_hand = hands.get(label, {})
                    trace(draw, raw_hand.get("raw_manus_landmarks_m"), x, y + 108, color=color)
                    trace(draw, raw_hand.get("rotated_raw_landmarks_m"), x + 400, y + 108, color="#909da6", width=1)
                    if adapted and b["candidate_mask"][side]:
                        trace(draw, adapted["score_local_landmarks_m"][side], x + 400, y + 108,
                              color=color if permitted else "#c69767", width=1)
                    if sdk_mask[side]:
                        trace(draw, sdk_fk[side], x + 800, y + 108, color="#909da6", width=1)
                    if driver_fk is not None:
                        trace(draw, driver_fk[side], x + 800, y + 108, color=color)
                    if actual:
                        trace(draw, actual["landmark_local_m"][side], x + 1200, y + 108, color=color)
                count = sum("q_measured" in s for s in steps)
                draw.text((1220, y + 122), f"Observed {count}/10 steps" if count else "NOT_EXECUTED",
                          fill="#23614b" if count == 10 else "#973819", font=font)
                if adapted:
                    draw.text((420, y + 122), "Certified static witness" if permitted else "Candidate: uncertified",
                              fill="#405362", font=font)
                if decision:
                    draw.text((820, y + 122), decision["status"], fill="#405362", font=font)
            draw.text((18, 735), "Grey shows the unscaled human intent in column 2 and raw SDK FK in column 3.",
                      fill="#405362", font=font)
            if not frames:
                image.save(args.out / "poster.png")
            pipe.stdin.write(image.tobytes())
            frames += 1
        pipe.stdin.close()
        if pipe.wait() or frames != 2402:
            raise ValueError("Native-intent video recording did not reach the signed time-axis frame count")
    write_json(args.out / "video.json", {"scope": "DIAGNOSTIC_ONLY", "fps": 25, "frames": frames,
               "actual_feedback_frames": visible, "source_frames_interpolated": False,
               "execution_ledger_sha256": file_sha(raw), "functional_ledger_sha256": file_sha(goal),
               "video_sha256": file_sha(video)})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("physics", "functional", "model", "model-receipt", "out"):
        p.add_argument("--" + option, type=Path, required=True)
    render(p.parse_args())
