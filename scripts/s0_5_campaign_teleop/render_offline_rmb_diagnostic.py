#!/usr/bin/env python3
"""Render the original axis, engineering target and actual offline MuJoCo state."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel
from dexgym.tasks.direct.wholebody_s05.v1.rmb.schema import file_sha, write_json
from render_dcr009_guarded_diagnostic import DrawVideo, FRAME_SIZE, MUTED, VIRTUAL_R
from replay_home_anchored_pilot import palm_poses


READ = ("tick_ns", "raw_tracker", "raw_tracker_mask", "raw_manus_local", "raw_manus_local_mask",
        "mapped", "mapped_mask", "target_local_landmark_xyz", "target_local_landmark_xyz_mask",
        "target_wrist_xyz", "target_wrist_xyzw", "q_measured", "q_measured_mask", "applied_substeps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "anchor", "model", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--fps", type=float, choices=(12.5, 25., 50.), default=12.5)
    args = parser.parse_args()
    receipt_path = args.run / "receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    if (receipt["scope"] != "DIAGNOSTIC_ONLY" or receipt["formal_coverage_credit"]
            or file_sha(args.anchor) != receipt["anchor_sha256"]
            or file_sha(args.model) not in receipt["robot_model_servo"]["assets"].values()):
        raise ValueError("Unexpected source, anchor or model")
    start_path = args.run / "start.json"
    events_path = args.run / "diagnostic-events.jsonl"
    if (file_sha(start_path) != receipt["start_sha256"]
            or file_sha(events_path) != receipt["diagnostic_events_sha256"]):
        raise ValueError("Runtime evidence bytes changed")
    anchor = json.loads(args.anchor.read_bytes())
    start = json.loads(start_path.read_bytes())
    with events_path.open(encoding="utf-8") as f:
        events = {(row["partition"], row["tick"]): row for row in map(json.loads, f)}
    robot = RobotModel(args.model)
    home = robot.geometry(robot.home)
    home_tcp = np.repeat(np.eye(4)[None], 2, axis=0)
    home_tcp[:, :3, :3], home_tcp[:, :3, 3] = home.rotations, home.wrists
    tcp_from_palm = np.linalg.inv(home_tcp) @ palm_poses(home)
    view = DrawVideo(receipt["root"], anchor, np.asarray(start["locked_transform"]), raw_human=True)
    raw_view = np.eye(4)
    raw_view[:3, :3] = VIRTUAL_R
    args.out.mkdir(exist_ok=False)
    video = args.out / "root-five-offline.mp4"
    fps = args.fps
    stride = int(50 / fps)
    command = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{FRAME_SIZE[0]}x{FRAME_SIZE[1]}", "-r", str(fps), "-i", "-", "-an",
               "-c:v", "libx264", "-crf", "22", "-threads", "2", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(video)]
    frames = measured = ticks = 0
    first_raw_center = None
    with subprocess.Popen(command, stdin=subprocess.PIPE) as ffmpeg:
        for part in receipt["partitions"]:
            arrays = {}
            for field in READ:
                path = args.run / part["directory"] / (field + ".npy")
                spec = part["arrays"][field]
                if file_sha(path) != spec["sha256"]:
                    raise ValueError(f"Diagnostic model/data mismatch for {part['partition']}/{field}")
                arrays[field] = np.load(path, allow_pickle=False)
            for i, tick in enumerate(arrays["tick_ns"]):
                event = events.get((part["partition"], i))
                if event is None:
                    continue
                status = event["status"]
                ticks += 1
                if (ticks - 1) % stride != 0 and ticks != receipt["window"]["plan_ticks"]:
                    continue
                image = Image.new("RGB", FRAME_SIZE, "#f3f7fb")
                draw = ImageDraw.Draw(image)
                elapsed = (int(tick) - receipt["window"]["hmb_cutoff_start_ns"]) / 1e9
                title = "DCR-010   Four-group offline RMB diagnostic" if receipt.get("group_admission") else (
                    "DCR-010   Offline RMB / Simulation diagnostic")
                view.text(draw, (18, 17), title, "title")
                view.text(draw, (18, 53), f"Original timeline   {elapsed:06.3f} s   |   {status}", "body",
                          "#963419" if status != "ACTIVE" else "#205d4a")
                for x, title, caption in (
                    (18, "Original human input", "Raw Tracker wrist markers and MANUS local skeleton"),
                    (494, "Independent mapped target", "Locked wrist proxy and signed-model morphology"),
                    (970, "MuJoCo measured", "Only recorded simulated feedback is displayed"),
                ):
                    view.panel(draw, x, title, caption)
                raw = raw_view @ arrays["raw_tracker"][i] if arrays["raw_tracker_mask"][i] else None
                if raw is not None and first_raw_center is None:
                    first_raw_center = raw[:, :3, 3].mean(axis=0)
                view.wrist_panel(draw, 18, raw, first_raw_center)
                view.local_hands(draw, 18, [arrays["raw_manus_local"][i, side]
                                 if arrays["raw_manus_local_mask"][i, side] else None for side in range(2)])
                target = arrays["mapped"][i] if arrays["mapped_mask"][i] else None
                fingers = None
                if arrays["target_local_landmark_xyz_mask"][i].all():
                    tcp = home_tcp.copy()
                    tcp[:, :3, :3] = Rotation.from_quat(arrays["target_wrist_xyzw"][i]).as_matrix()
                    tcp[:, :3, 3] = arrays["target_wrist_xyz"][i]
                    palm = tcp @ tcp_from_palm
                    fingers = (np.einsum("sij,sfkj->sfki", palm[:, :3, :3], arrays["target_local_landmark_xyz"][i])
                               + palm[:, None, None, :3, 3])
                view.wrist_panel(draw, 494, target, view.world_center, robot_fingers=fingers)
                view.text(draw, (511, 492), "Engineering wrist/palm bridge. Anatomy UNKNOWN.", "small", MUTED)
                if arrays["q_measured_mask"][i]:
                    g = robot.geometry(arrays["q_measured"][i])
                    poses = home_tcp.copy()
                    poses[:, :3, :3], poses[:, :3, 3] = g.rotations, g.wrists
                    arms = [np.vstack(([robot.data.joint(name).xanchor.copy() for name in robot.names[s*7:(s+1)*7]],
                                       g.wrists[s])) for s in range(2)]
                    view.wrist_panel(draw, 970, poses, view.world_center, robot_fingers=g.landmarks, arm=arms)
                    view.text(draw, (987, 492), f"Executed substeps: {arrays['applied_substeps'][i]}/10", "small", MUTED)
                    measured += 1
                else:
                    view.wrist_panel(draw, 970, None, None)
                    view.text(draw, (987, 492), "No historical physics executed for this row.", "small", "#963419")
                if receipt.get("group_admission"):
                    statuses = []
                    for g, name in enumerate(("Left arm", "Right arm", "Left hand", "Right hand")):
                        labels = [e["groups"][g].get("label", "NO_COMMIT") for e in event["safety"]
                                  if len(e.get("groups", ())) == 4]
                        inactive = [label for label in labels if not label.endswith("_ACTIVE")]
                        state = inactive[-1] if inactive else ("ACTIVE" if len(labels) == 5 else "NO_FULL_INTERVAL")
                        statuses.append(f"{name}  {state}")
                    view.text(draw, (987, 519), "   |   ".join(statuses[:2]), "small", MUTED)
                    view.text(draw, (987, 545), "   |   ".join(statuses[2:]), "small", MUTED)
                view.text(draw, (18, 616), "Recorded input time is retained. Missing and rejected commands stay in the denominator.",
                          "footer", MUTED)
                if not frames:
                    image.save(args.out / "poster.png")
                ffmpeg.stdin.write(image.tobytes())
                frames += 1
        ffmpeg.stdin.close()
        if ffmpeg.wait() or ticks != receipt["window"]["plan_ticks"]:
            raise RuntimeError("Incomplete render or history")
    write_json(args.out / "video.json", {"scope": "DIAGNOSTIC_ONLY", "rendered_frames": frames,
              "rendered_frames_with_measured_feedback": measured, "duration_seconds": frames / fps,
              "sample_fps": fps, "source_frames_interpolated": False,
              "original_history_ticks": ticks, "video_sha256": file_sha(video),
              "poster_sha256": file_sha(args.out / "poster.png"), "run_receipt_sha256": file_sha(receipt_path)})


if __name__ == "__main__":
    main()
