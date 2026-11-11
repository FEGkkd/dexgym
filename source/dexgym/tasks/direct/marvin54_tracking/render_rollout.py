"""Render measured native MuJoCo states beside the reference motion."""

import argparse
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw


def render(model, reference, trace, output):
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.27, 0, 0.83]
    camera.distance = 2.45
    camera.azimuth = 150
    camera.elevation = -13
    width, height = 480, 480
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".tmp.mp4")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width * 2}x{height}",
        "-framerate", "25", "-i", "-", "-c:v", "libx264", "-crf", "21",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial),
    ]
    with mujoco.Renderer(model, height=height, width=width) as renderer, subprocess.Popen(
        command, stdin=subprocess.PIPE
    ) as encoder:
        for i in range(0, len(trace["qpos"]), 2):
            panels = []
            for pose in (reference[i], trace["qpos"][i]):
                data.qpos[:] = pose
                mujoco.mj_fwdPosition(model, data)
                renderer.update_scene(data, camera=camera)
                panels.append(renderer.render().copy())
            frame = Image.fromarray(np.concatenate(panels, axis=1))
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 2 * width, 29), fill=(20, 24, 32))
            draw.text((10, 8), "Retargeted reference", fill="white")
            draw.text((width + 10, 8), "Closed-loop policy in MuJoCo", fill="white")
            draw.text((2 * width - 73, 8), f"{trace['time_s'][i]:.2f} s", fill="white")
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("ffmpeg could not encode the replay")
    partial.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    with np.load(args.reference, allow_pickle=False) as reference, np.load(args.trace, allow_pickle=False) as trace:
        render(model, reference["qpos"], trace, args.output)


if __name__ == "__main__":
    main()
