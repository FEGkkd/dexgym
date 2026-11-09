"""Render one Newton Hand 2.1 PPO play beside its active GRAB joint target."""

import argparse
import os
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from dexgym.assets.wuji_hand21 import OFFICIAL_MJCF
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.states.expanduser().resolve(), allow_pickle=False) as states:
        actual = np.asarray(states["q_actual"], dtype=np.float32)
        goal = np.asarray(states["q_goal"], dtype=np.float32)
        errors = np.asarray(states["tip_error_mm"], dtype=np.float32)
        goal_frame = np.asarray(states["goal_frame"], dtype=np.int32)
        span = np.asarray(states["goal_span"], dtype=np.int32)
        reached = np.asarray(states["reached"], dtype=bool)
        source = str(states["source"].item())
        fps = float(states["fps"].item())
    if actual.ndim != 2 or actual.shape[1] != 20 or goal.shape != actual.shape:
        parser.error("Expected matching actual and goal arrays of shape [T,20]")
    if errors.shape != (len(actual), 5) or any(len(a) != len(actual) for a in (goal_frame, span, reached)):
        parser.error("Step annotations do not match the joint trajectories")
    if len(actual) == 0 or not np.isfinite(actual).all() or not np.isfinite(goal).all():
        parser.error("Expected nonempty, finite joint trajectories")

    model = mujoco.MjModel.from_xml_path(str(OFFICIAL_MJCF))
    model.vis.headlight.ambient[:] = [0.4, 0.4, 0.4]
    model.vis.headlight.diffuse[:] = [0.85, 0.85, 0.85]
    actual_data, goal_data = mujoco.MjData(model), mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, -0.08]
    camera.distance = 0.27
    camera.azimuth = 135
    camera.elevation = -15
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", "1280x540", "-framerate", str(fps), "-i", "-", "-c:v", "libx264",
        "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 18) if os.path.isfile(font_path) else ImageFont.load_default()
    small = ImageFont.truetype(font_path, 15) if os.path.isfile(font_path) else font
    with mujoco.Renderer(model, width=640, height=480) as renderer:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            assert encoder.stdin is not None
            for step in range(len(actual)):
                panels = []
                for data, q in ((actual_data, actual[step]), (goal_data, goal[step])):
                    data.qpos[:] = q
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    panels.append(renderer.render().copy())
                image = Image.new("RGB", (1280, 540), (10, 10, 14))
                image.paste(Image.fromarray(panels[0]), (0, 40))
                image.paste(Image.fromarray(panels[1]), (640, 40))
                draw = ImageDraw.Draw(image)
                draw.text((16, 9), "Policy play | Newton physical state", font=font, fill="white")
                draw.text((656, 9), "Active GRAB subgoal | Hand 2.1", font=font, fill="white")
                max_error = float(errors[step].max())
                status = "REACHED" if reached[step] else ("within 15 mm" if max_error < 15 else "outside 15 mm")
                color = "#8ceaac" if max_error < 15 else "#ff9a91"
                draw.text((16, 493), f"{source} | t={step/fps:.2f}s | goal frame={goal_frame[step]} | span={span[step]}",
                          font=small, fill="white")
                draw.text((16, 515), f"Max fingertip error: {max_error:.1f} mm | {status}", font=small, fill=color)
                encoder.stdin.write(np.asarray(image).tobytes())
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("ffmpeg video encoding failed")
    print(f"VIDEO_DONE {output} frames={len(actual)} duration_s={len(actual)/fps:.2f}", flush=True)


if __name__ == "__main__":
    main()
