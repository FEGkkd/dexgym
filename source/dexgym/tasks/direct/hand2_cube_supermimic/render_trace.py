"""Render recorded native poses without re-simulating or changing the policy."""

import argparse
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from dexgym.assets.wuji_hand2_cube import ROOT


def render_model():
    import mujoco

    hand = ET.parse(ROOT / "generated/hand2_cube_rig.xml").getroot()
    for mesh in hand.findall("asset/mesh"):
        mesh.set("file", str(ROOT / "generated/meshes" / Path(mesh.get("file")).name))
    hand.find("compiler").attrib.pop("meshdir", None)
    cube = ET.parse(ROOT / "cube/xmls/cube.xml").getroot()
    for element in cube.iter():
        for key, value in list(element.attrib.items()):
            if key.startswith("file"):
                element.set(key, str(ROOT / "cube" / value))
    for section in ("asset", "default", "worldbody"):
        source = cube.find(section)
        target = hand.find(section)
        if target is None:
            target = ET.SubElement(hand, section)
        if source is not None:
            target.extend(list(source))
    world = hand.find("worldbody")
    ET.SubElement(world, "geom", name="ground", type="plane", size="2 2 .01", rgba=".18 .20 .23 1")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1", diffuse="1 1 1")
    return mujoco.MjModel.from_xml_string(ET.tostring(hand, encoding="unicode"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    with np.load(args.trace, allow_pickle=False) as archive:
        frames = archive["frames"]
        dt = float(archive["frame_dt"])
        names = archive["joint_names"].tolist()
        metadata = json.loads(str(archive["metadata"]))
    if not 0 <= args.env_index < frames.shape[1] or args.seconds <= 0:
        parser.error("Invalid recorded environment index or duration")
    model = render_model()
    data = mujoco.MjData(model)
    palm = model.body("right_palm_link").id
    assert model.body_parentid[palm] == 0
    qids = [model.joint(name).qposadr[0] for name in names]
    free = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
    cube_q = int(model.jnt_qposadr[free])
    cube_body = int(model.jnt_bodyid[free])
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-0.07, 0.005, 0.55]
    camera.distance = 0.42
    camera.azimuth = 135
    camera.elevation = -25
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
        "640x480",
        "-framerate",
        str(1 / dt),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        str(args.output),
    ]
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    count = min(len(frames), int(args.seconds / dt))
    scene_revision = metadata.get("evaluated_scene_revision", "unknown")
    max_pose_error = 0.0
    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for i, row in enumerate(frames[:count, args.env_index]):
                data.qpos[qids] = row[:20]
                model.body_pos[palm] = row[20:23]
                model.body_quat[palm] = row[[26, 23, 24, 25]]
                data.qpos[cube_q : cube_q + 3] = row[27:30]
                data.qpos[cube_q + 3 : cube_q + 7] = row[[33, 30, 31, 32]]
                mujoco.mj_forward(model, data)
                max_pose_error = max(
                    max_pose_error,
                    float(np.abs(data.xpos[palm] - row[20:23]).max()),
                    float(np.abs(data.xpos[cube_body] - row[27:30]).max()),
                )
                desired = data.xmat[palm].reshape(3, 3) @ row[34:43].reshape(3, 3)
                relative = data.xmat[cube_body].reshape(3, 3).T @ desired
                angle = np.arccos(np.clip((np.trace(relative) - 1) * 0.5, -1, 1))
                renderer.update_scene(data, camera=camera)
                picture = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(picture)
                draw.rectangle((0, 0, 640, 68), fill=(20, 24, 30))
                draw.text(
                    (12, 5), "Recorded native simulation | " + metadata["checkpoint_kind"], font=font, fill="white"
                )
                draw.text(
                    (12, 27),
                    f"t={i * dt:.2f}s  env={args.env_index}  episode={int(row[44])}  goals={int(row[43])}",
                    font=font,
                    fill="white",
                )
                draw.text(
                    (12, 48),
                    f"Target error={np.degrees(angle):.1f} deg | scene={scene_revision}",
                    font=font,
                    fill="white",
                )
                if i == 0:
                    picture.save(args.output.with_suffix(".png"))
                encoder.stdin.write(np.asarray(picture).tobytes())
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("Video encoder failed")
    finally:
        renderer.close()
    assert max_pose_error < 1e-6, max_pose_error
    report = dict(
        trace=str(args.trace),
        output=str(args.output),
        frames=count,
        fps=1 / dt,
        recorded_env_index=args.env_index,
        root_pose_max_error_m=max_pose_error,
        mode="recorded native poses; no physics resimulation",
        checkpoint=metadata,
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "checkpoint"}), flush=True)


if __name__ == "__main__":
    main()
