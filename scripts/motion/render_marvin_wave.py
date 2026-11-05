"""Offscreen human, FK, and Newton physical-state videos with distinct labels."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from retarget_marvin_wave import BVH, MJCF, OUT, read_human
from scipy.spatial.transform import Rotation

os.environ.setdefault("MUJOCO_GL", "egl")
W, H = 640, 480
FONT = ImageFont.load_default()
AXIS_COLORS = ((240, 80, 80), (70, 225, 120), (80, 150, 255))


def _txt(draw, xy, text, color=(245, 245, 245)):
    draw.text(xy, text, font=FONT, fill=color)


def _project(point, origin=(0, 1.25), scale=320):
    x, y, z = point
    return (
        int(W / 2 + (x - origin[0]) * scale + z * 0.28 * scale),
        int(260 - (y - origin[1]) * scale + z * 0.12 * scale),
    )


def _human_panel(m1, time_s, hand_q, title):
    image = Image.new("RGB", (W, H), (16, 21, 30))
    d = ImageDraw.Draw(image)
    t = m1["time"]

    def interp(k, side):
        p = m1[k][:, side, :3]
        return np.array([np.interp(time_s, t, p[:, i]) for i in range(3)])

    shoulder = [interp("shoulder_pose", s) for s in range(2)]
    elbow = [interp("elbow_pose", s) for s in range(2)]
    hand = [interp("hand_pose", s) for s in range(2)]
    torso = (shoulder[0] + shoulder[1]) / 2
    base = torso - np.array([0, 0.34, 0])
    d.line([_project(base), _project(torso)], fill=(170, 175, 185), width=6)
    d.line([_project(shoulder[0]), _project(shoulder[1])], fill=(170, 175, 185), width=6)
    d.ellipse(
        (
            *np.subtract(_project(torso + np.array([0, 0.17, 0])), 15),
            *np.add(_project(torso + np.array([0, 0.17, 0])), 15),
        ),
        outline=(170, 175, 185),
        width=3,
    )
    for side in range(2):
        color = (160, 180, 205) if side == 0 else (245, 155, 75)
        d.line([_project(shoulder[side]), _project(elbow[side]), _project(hand[side])], fill=color, width=7)
        for p in (shoulder[side], elbow[side], hand[side]):
            u, v = _project(p)
            d.ellipse((u - 6, v - 6, u + 6, v + 6), fill=color)
    axes = Rotation.from_quat(hand_q).as_matrix()
    for j, color in enumerate(AXIS_COLORS):
        d.line([_project(hand[1]), _project(hand[1] + axes[:, j] * 0.16)], fill=color, width=4)
    _txt(d, (14, 14), title)
    _txt(d, (14, 34), f"CMU 141_16  source time {time_s:5.2f} s")
    _txt(d, (14, H - 22), "Y up | right arm orange | hand SO(3) axes RGB")
    return image


class RobotRender:
    def __init__(self, mjcf):
        self.model = mujoco.MjModel.from_xml_path(str(mjcf))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=H, width=W)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0, -0.3, 1.1]
        self.camera.distance = 1.8
        self.camera.azimuth = 90
        self.camera.elevation = -10
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp_R")
        self.elbow = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Link4_R")
        self.data.qpos[:] = self.model.key_qpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")]
        mujoco.mj_forward(self.model, self.data)
        shoulder = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Link1_R")
        self.shoulder_origin = self.data.xpos[shoulder].copy()

    def _axes(self, p, q, alpha=1.0, length=0.13):
        r = Rotation.from_quat(q).as_matrix()
        for j, color in enumerate(AXIS_COLORS):
            if self.renderer.scene.ngeom >= self.renderer.scene.maxgeom:
                break
            geom = self.renderer.scene.geoms[self.renderer.scene.ngeom]
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.004, p, p + length * r[:, j])
            geom.rgba[:] = [*[c / 255 for c in color], alpha]
            self.renderer.scene.ngeom += 1

    def _dot(self, p, color, size=0.012):
        geom = self.renderer.scene.geoms[self.renderer.scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size] * 3), p, np.eye(3).ravel(), color)
        self.renderer.scene.ngeom += 1

    def _screen(self, p):
        cam = self.renderer.scene.camera[0]
        origin = (self.renderer.scene.camera[0].pos + self.renderer.scene.camera[1].pos) * 0.5
        forward = np.asarray(cam.forward)
        up = np.asarray(cam.up)
        right = np.cross(forward, up)
        vector = np.asarray(p) - origin
        depth = max(0.01, float(np.dot(vector, forward)))
        tangent = float(cam.frustum_top / cam.frustum_near)
        horizontal = tangent * W / H
        return (
            int(W / 2 + np.dot(vector, right) / depth / horizontal * W / 2),
            int(H / 2 - np.dot(vector, up) / depth / tangent * H / 2),
        )

    def _overlay_axes(self, draw, p, q, prefix, soft=False):
        rotation = Rotation.from_quat(q).as_matrix()
        start = self._screen(p)
        for j, color in enumerate(AXIS_COLORS):
            end = self._screen(p + rotation[:, j] * 0.18)
            if soft:
                color = tuple(int(c * 0.65 + 80) for c in color)
            draw.line([start, end], fill=color, width=2 if soft else 4)
            draw.ellipse((end[0] - 3, end[1] - 3, end[0] + 3, end[1] + 3), fill=color)
        draw.ellipse((start[0] - 5, start[1] - 5, start[0] + 5, start[1] + 5), outline=(250, 250, 250), width=2)
        _txt(draw, (start[0] + 8, start[1] - 14), prefix)

    def frame(self, q, target_p, target_q, title, actual_p=None, actual_q=None, target_elbow=None):
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        self._axes(target_p, target_q, alpha=0.65, length=0.15)
        p = self.data.site_xpos[self.site]
        current = Rotation.from_matrix(self.data.site_xmat[self.site].reshape(3, 3)).as_quat()
        self._axes(p, current, alpha=1.0, length=0.11)
        if target_elbow is not None:
            self._dot(target_elbow, [0.1, 1, 1, 0.7])
            self._dot(self.data.xpos[self.elbow], [1, 0.8, 0.1, 1])
        image = Image.fromarray(self.renderer.render())
        d = ImageDraw.Draw(image)
        self._overlay_axes(d, target_p, target_q, "T", soft=True)
        self._overlay_axes(d, p, current, "A", soft=False)
        if target_elbow is not None:
            for point, color in ((target_elbow, (80, 240, 240)), (self.data.xpos[self.elbow], (255, 220, 80))):
                u, v = self._screen(point)
                d.ellipse((u - 5, v - 5, u + 5, v + 5), outline=color, width=2)
        _txt(d, (14, 14), title)
        _txt(d, (14, 34), "Target axes long/translucent | realized TCP axes short")
        err = np.linalg.norm(p - target_p) * 1000
        rot = np.rad2deg((Rotation.from_quat(current) * Rotation.from_quat(target_q).inv()).magnitude())
        _txt(d, (14, H - 22), f"TCP FK error to target: {err:5.1f} mm  {rot:5.1f} deg")
        return image

    def close(self):
        self.renderer.close()


def _writer(path, fps, width, height):
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "24",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def _write(proc, image):
    proc.stdin.write(np.asarray(image).tobytes())


def _close(proc):
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def _curve_png(human, path):
    raw = human["local_right_raw_quat"]
    filt = human["local_right_filtered_quat"]
    dt = human["meta"]["original_frame_time_s"]

    def speed(q):
        return np.r_[0, np.rad2deg((Rotation.from_quat(q[1:]) * Rotation.from_quat(q[:-1]).inv()).magnitude()) / dt]

    def deviation(q):
        return np.rad2deg((Rotation.from_quat(q) * Rotation.from_quat(q[0]).inv()).magnitude())

    curves = [
        (deviation(raw), deviation(filt), "Local right hand SO(3) from first frame (deg)", 180),
        (speed(raw), speed(filt), "Local right hand angular speed (deg/s)", 2500),
    ]
    im = Image.new("RGB", (1000, 600), (18, 23, 32))
    d = ImageDraw.Draw(im)
    for k, (a, b, title, limit) in enumerate(curves):
        top = 50 + k * 275
        bottom = top + 210
        left = 75
        right = 965
        d.rectangle((left, top, right, bottom), outline=(130, 140, 160), width=1)
        _txt(d, (left, top - 25), title)
        for values, color in ((a, (245, 110, 90)), (b, (60, 210, 180))):
            points = [
                (
                    int(left + i / (len(values) - 1) * (right - left)),
                    int(bottom - min(v, limit) / limit * (bottom - top)),
                )
                for i, v in enumerate(values)
            ]
            d.line(points, fill=color, width=3)
    _txt(d, (75, 580), "raw red | SO(3) filtered teal | 120 Hz BVH; frame 0 artificial T-pose removed")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def render(reference, output, newton_states=None, mujoco_states=None):
    ref = np.load(reference, allow_pickle=False)
    m1 = np.load(OUT.parent / "human_motion_v1/diagnostic_motions/141_16.npz", allow_pickle=False)
    human = read_human(BVH)
    _curve_png(human, output / "cmu_141_16_right_wrist_filter.png")
    source = np.arange(0, 2.48 + 0.0001, 0.04)
    writer = _writer(output / "cmu_141_16_human.mp4", 25, 2 * W, H)
    for t in source:
        nearest = np.argmin(abs(ref["original_bvh_time"] - t))
        raw = ref["human_hand_raw_quat_xyzw"][nearest, 1]
        filtered = ref["human_hand_filtered_quat_xyzw"][nearest, 1]
        pair = Image.new("RGB", (2 * W, H))
        pair.paste(_human_panel(m1, t, raw, "Original BVH FK"), (0, 0))
        pair.paste(_human_panel(m1, t, filtered, "SO(3) filtered hand"), (W, 0))
        _write(writer, pair)
    _close(writer)
    robot = RobotRender(MJCF)
    writer = _writer(output / "cmu_141_16_human_vs_marvin.mp4", 50 / 3, 2 * W, H)
    for i in range(0, len(ref["time"]), 3):
        source_t = float(ref["original_bvh_time"][i])
        human_frame = _human_panel(m1, source_t, ref["human_hand_filtered_quat_xyzw"][i, 1], "CMU filtered FK")
        robot_frame = robot.frame(
            ref["q_full"][i],
            ref["mapped_target_pos"][i, 1],
            ref["mapped_target_quat_xyzw"][i, 1],
            "Marvin kinematic FK reference",
            target_elbow=ref["mapped_right_elbow_target_pos"][i],
        )
        pair = Image.new("RGB", (2 * W, H))
        pair.paste(human_frame, (0, 0))
        pair.paste(robot_frame, (W, 0))
        _write(writer, pair)
    _close(writer)
    generated = [str(output / "cmu_141_16_human.mp4"), str(output / "cmu_141_16_human_vs_marvin.mp4")]
    if newton_states and Path(newton_states).exists():
        states = np.load(newton_states, allow_pickle=False)
        if len(states["time"]) != len(ref["time"]):
            raise ValueError("Newton states are not full reference")
        writer = _writer(output / "cmu_141_16_newton_pd.mp4", 50 / 3, 2 * W, H)
        for i in range(0, len(ref["time"]), 3):
            target_p = ref["tcp_pos"][i, 1]
            target_q = ref["tcp_quat_xyzw"][i, 1]
            left = robot.frame(ref["q_full"][i], target_p, target_q, "Reference FK (kinematic)")
            right = robot.frame(states["q"][i, 0], target_p, target_q, "Newton MJWarp physical PD state")
            pair = Image.new("RGB", (2 * W, H))
            pair.paste(left, (0, 0))
            pair.paste(right, (W, 0))
            _write(writer, pair)
        _close(writer)
        generated.append(str(output / "cmu_141_16_newton_pd.mp4"))
    if mujoco_states and Path(mujoco_states).exists():
        states = np.load(mujoco_states, allow_pickle=False)
        if len(states["time"]) != len(ref["time"]):
            raise ValueError("MuJoCo states are not full reference")
        writer = _writer(output / "cmu_141_16_mujoco_pd.mp4", 50 / 3, 2 * W, H)
        for i in range(0, len(ref["time"]), 3):
            target_p = ref["tcp_pos"][i, 1]
            target_q = ref["tcp_quat_xyzw"][i, 1]
            left = robot.frame(ref["q_full"][i], target_p, target_q, "Reference FK (kinematic)")
            right = robot.frame(states["q"][i], target_p, target_q, "MuJoCo physical mj_step PD state")
            pair = Image.new("RGB", (2 * W, H))
            pair.paste(left, (0, 0))
            pair.paste(right, (W, 0))
            _write(writer, pair)
        _close(writer)
        generated.append(str(output / "cmu_141_16_mujoco_pd.mp4"))
    robot.close()
    output.joinpath("render_manifest.json").write_text(
        json.dumps(
            {
                "reference": str(reference),
                "reference_q_full_sha256": hashlib.sha256(ref["q_full"].tobytes()).hexdigest(),
                "newton_states": str(newton_states) if newton_states else None,
                "mujoco_states": str(mujoco_states) if mujoco_states else None,
                "video_paths": generated,
                "newton_video_source": "recorded Newton MJWarp physical joint states rendered with MuJoCo EGL",
                "kinematic_video_source": "Marvin reference q_full rendered with MuJoCo EGL",
            },
            indent=2,
        )
    )
    return generated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path)
    ap.add_argument("--output", type=Path, default=OUT / "previews")
    ap.add_argument("--newton-states", type=Path)
    ap.add_argument("--mujoco-states", type=Path)
    args = ap.parse_args()
    print(json.dumps(render(args.reference, args.output, args.newton_states, args.mujoco_states), indent=2))


if __name__ == "__main__":
    main()
