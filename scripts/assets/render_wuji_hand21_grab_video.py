"""Render a direct Hand 2.1 joint replay beside its GRAB retarget reference."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from dexgym.assets.wuji_hand21 import OFFICIAL_MJCF
from dexgym.data.wuji_retarget import RightHandFK
from PIL import Image, ImageDraw, ImageFont


def replay_mujoco(q_ref: np.ndarray) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(OFFICIAL_MJCF))
    model.opt.timestep = 1 / 480
    data = mujoco.MjData(model)
    data.qpos[:] = q_ref[0]
    data.ctrl[:] = q_ref[0]
    mujoco.mj_forward(model, data)
    actual = np.empty_like(q_ref)
    for frame, command in enumerate(q_ref):
        data.ctrl[:] = command
        for _ in range(16):
            mujoco.mj_step(model, data)
        actual[frame] = data.qpos
    return actual


def render(q_ref: np.ndarray, q_actual: np.ndarray, label: str, source: str, output: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(OFFICIAL_MJCF))
    model.vis.headlight.ambient[:] = [0.4, 0.4, 0.4]
    model.vis.headlight.diffuse[:] = [0.85, 0.85, 0.85]
    actual_data = mujoco.MjData(model)
    reference_data = mujoco.MjData(model)
    fk = RightHandFK(OFFICIAL_MJCF)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, -0.08]
    camera.distance = 0.27
    camera.azimuth = 135
    camera.elevation = -15
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "1280x520",
        "-framerate",
        "30",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 18) if os.path.isfile(font_path) else ImageFont.load_default()
    small = ImageFont.truetype(font_path, 15) if os.path.isfile(font_path) else font
    with mujoco.Renderer(model, width=640, height=480) as renderer:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            assert encoder.stdin is not None
            for frame in range(len(q_ref)):
                panels = []
                for data, q in ((actual_data, q_actual[frame]), (reference_data, q_ref[frame])):
                    data.qpos[:] = q
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    panels.append(renderer.render().copy())
                image = Image.new("RGB", (1280, 520), (10, 10, 14))
                image.paste(Image.fromarray(panels[0]), (0, 40))
                image.paste(Image.fromarray(panels[1]), (640, 40))
                draw = ImageDraw.Draw(image)
                draw.text((16, 9), f"Actual: {label}", font=font, fill="white")
                draw.text((656, 9), "Reference: GRAB / official Wuji SDK", font=font, fill="white")
                error_mm = np.linalg.norm(fk.tips(q_actual[frame]) - fk.tips(q_ref[frame]), axis=-1).max() * 1000
                draw.text(
                    (16, 490),
                    f"{source}  |  t={frame / 30:.2f}s  |  max tip error={error_mm:.1f} mm",
                    font=small,
                    fill="white",
                )
                encoder.stdin.write(np.asarray(image).tobytes())
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("ffmpeg video encoding failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/hand21_right_geometry_feasible/manifest.json")
    )
    parser.add_argument("--source", default="s9/spheresmall_inspect_1.npz")
    parser.add_argument("--states", type=Path, help="Newton q_actual[T,20] NPZ; otherwise run MuJoCo position control")
    parser.add_argument("--output", type=Path, default=Path("artifacts/hand21/hand21_grab_direct_replay.mp4"))
    args = parser.parse_args()
    manifest = args.manifest.expanduser().resolve()
    bank = json.loads(manifest.read_text())
    candidates = [row for row in bank["motions"] if row.get("origin_source", row["source"]) == args.source]
    if len(candidates) != 1:
        parser.error(f"Expected one unbroken segment for {args.source}; found {len(candidates)}")
    row = candidates[0]
    with np.load(manifest.parent / row["file"], allow_pickle=False) as motion:
        q_ref = np.asarray(motion["q_ref"], dtype=np.float32)
    if args.states:
        with np.load(args.states.expanduser().resolve(), allow_pickle=False) as states:
            q_actual = np.asarray(states["q_actual"], dtype=np.float32)
        label = "Newton USD position control"
    else:
        q_actual = replay_mujoco(q_ref)
        label = "MuJoCo position control"
    if q_actual.shape != q_ref.shape or not np.isfinite(q_actual).all():
        raise ValueError("Actual joint trajectory must match the finite 30-Hz reference")
    render(q_ref, q_actual, label, args.source, args.output.expanduser().resolve())
    print(f"video={args.output.expanduser().resolve()} frames={len(q_ref)} duration_s={len(q_ref) / 30:.2f}")


if __name__ == "__main__":
    main()
