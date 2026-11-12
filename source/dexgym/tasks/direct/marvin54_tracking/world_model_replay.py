"""Compare recorded Newton states with fixed-horizon World Model forecasts."""

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from dexgym.algorithms.supermimic.dex_reference.networks import WorldModel

from .model import sha256
from .task import DT, JOINTS, TASK_ID, TrackingGeometry, TrackingTask


HORIZONS = (1, 4, 8, 16, 32)
VIDEO_HORIZONS = (1, 16, 32)


def summary(errors):
    return {
        key: {"mean": float(value.mean()), "p95": float(np.percentile(value, 95))}
        for key, value in errors.items()
    }


@torch.inference_mode()
def forecast(wm, geometry, state, target):
    """Roll each window from its measured initial state using its recorded targets."""
    results, arrays = {}, {}
    actual = geometry.pose(state[:, :JOINTS])

    def errors(x, horizon):
        wrist, _, _, tip, _ = geometry.pose(x[:, :JOINTS])
        return {
            "wrist_mm": ((wrist - actual[0][horizon:]).norm(dim=-1) * 1000).cpu().numpy(),
            "tip_mm": ((tip - actual[3][horizon:]).norm(dim=-1) * 1000).cpu().numpy(),
            "joint_deg": (x[:, :JOINTS] - state[horizon:, :JOINTS]).abs().rad2deg().cpu().numpy(),
            "joint_velocity_rad_s": (x[:, JOINTS:] - state[horizon:, JOINTS:]).abs().cpu().numpy(),
        }

    predicted = state
    for horizon in range(1, max(HORIZONS) + 1):
        previous = predicted[:-1]
        predicted, _ = wm(previous, target[horizon - 1:], previous)
        if horizon not in HORIZONS:
            continue
        if not torch.isfinite(predicted).all():
            raise FloatingPointError(f"Nonfinite forecast at horizon {horizon}")
        initial = state[:-horizon]
        velocity = initial[:, JOINTS:]
        extrapolated = torch.cat((initial[:, :JOINTS] + DT * horizon * velocity, velocity), dim=-1)
        measured_errors = errors(predicted, horizon)
        results[str(horizon)] = {
            "horizon_ms": int(horizon * DT * 1000),
            "windows": len(predicted),
            "world_model": summary(measured_errors),
            "hold_initial_state": summary(errors(initial, horizon)),
            "constant_velocity": summary(errors(extrapolated, horizon)),
        }
        arrays[f"qpos_h{horizon}"] = predicted[:, :JOINTS].cpu().numpy()
        arrays[f"qvel_h{horizon}"] = predicted[:, JOINTS:].cpu().numpy()
        for key, value in measured_errors.items():
            arrays[f"{key}_h{horizon}"] = value
    return results, arrays


def render(model, time, qpos, forecasts, output):
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.28, 0, 0.96]
    camera.distance = 2.07
    camera.azimuth = 148
    camera.elevation = -11
    data = mujoco.MjData(model)
    width, height, header = 560, 416, 54
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 17) if font_path.is_file() else ImageFont.load_default()
    partial = output.with_suffix(".tmp.mp4")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width * 2}x{height * 2}",
        "-framerate", "25", "-i", "-", "-c:v", "libx264", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial),
    ]
    with mujoco.Renderer(model, height=height, width=width) as renderer, subprocess.Popen(
        command, stdin=subprocess.PIPE
    ) as encoder:
        for frame_index in range(0, len(time), 2):
            frame = Image.new("RGB", (width * 2, height * 2), (23, 28, 38))
            draw = ImageDraw.Draw(frame)
            for panel, horizon in enumerate((0, *VIDEO_HORIZONS)):
                x, y = panel % 2 * width, panel // 2 * height
                if not horizon:
                    pose, title = qpos[frame_index], f"Newton measured state   |   {time[frame_index]:.2f} s"
                    detail = "Recorded policy rollout"
                else:
                    title = f"World Model   |   {horizon}-step forecast ({horizon * 20} ms)"
                    i = frame_index - horizon
                    pose = forecasts[f"qpos_h{horizon}"][i] if i >= 0 else None
                    if pose is None:
                        detail = "Waiting for first complete prediction window"
                    else:
                        wrist = forecasts[f"wrist_mm_h{horizon}"][i].mean()
                        tip = forecasts[f"tip_mm_h{horizon}"][i].mean()
                        detail = f"Wrist error {wrist:.1f} mm   |   Fingertip error {tip:.1f} mm"
                if pose is not None:
                    data.qpos[:] = pose
                    mujoco.mj_fwdPosition(model, data)
                    renderer.update_scene(data, camera=camera)
                    frame.paste(Image.fromarray(renderer.render()), (x, y))
                draw.rectangle((x, y, x + width - 1, y + header), fill=(23, 28, 38))
                draw.text((x + 12, y + 6), title, font=font, fill="white")
                draw.text((x + 12, y + 29), detail, font=font, fill=(169, 214, 239))
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("ffmpeg could not encode the World Model comparison")
    partial.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    provenance = json.loads((args.checkpoint.parent / "provenance.json").read_text())
    evaluation = json.loads((args.trace.parent / "evaluation.json").read_text())
    if (sha256(args.model) != provenance["model_sha256"]
            or Path(evaluation["checkpoint"]).resolve() != args.checkpoint.resolve()
            or args.trace.name != "policy.npz"):
        raise ValueError("Use the model and policy trace from this checkpoint's Newton evaluation")
    model = mujoco.MjModel.from_xml_path(str(args.model))
    names = [model.joint(i).name for i in np.argsort(model.jnt_qposadr)]
    if names != provenance["joint_names"] or (model.nq, model.nv) != (JOINTS, JOINTS):
        raise ValueError("The model does not match the checkpoint's joint order")
    with np.load(args.trace, allow_pickle=False) as trace:
        qpos, qvel, target, time = (trace[key] for key in ("qpos", "qvel", "target", "time_s"))
    n = len(time)
    if (n <= max(HORIZONS) or qpos.shape != (n, JOINTS) or qvel.shape != (n, JOINTS)
            or target.shape != (n - 1, JOINTS) or not np.allclose(time, np.arange(n) * DT)
            or not all(np.isfinite(v).all() for v in (qpos, qvel, target))):
        raise ValueError("Expected a finite 50 Hz Newton trace with one target per transition")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload["format_version"] != 1 or payload["task_id"] != TASK_ID:
        raise ValueError("Unsupported checkpoint task or format")
    cfg = payload["config"]
    wm = WorldModel(
        TrackingTask.dynamics_dim, TrackingTask.contact_dim, TrackingTask.codec,
        TrackingTask.action_dim, TrackingTask.error_floor, tuple(cfg["wm_hidden"]), False,
        control_error_features=cfg.get("control_error_features", False),
    )
    wm.load_state_dict(payload["modules"]["wm"])
    wm.to(device).eval()
    iteration = payload["iteration"]
    del payload
    geometry = TrackingGeometry(model, SimpleNamespace(names=names), device)
    state = torch.as_tensor(np.concatenate((qpos, qvel), axis=1), dtype=torch.float32, device=device)
    controls = torch.as_tensor(target, dtype=torch.float32, device=device)
    scores, arrays = forecast(wm, geometry, state, controls)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "prediction_metrics.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()), "iteration": iteration, "trace": str(args.trace.resolve()),
        "method": "Every forecast starts from the measured state at t-h, applies the recorded Newton targets, "
                  "and is compared with the measured state at t.",
        "control_dt_s": DT, "frames": n, "horizons": scores,
    }, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.output / "prediction_trace.npz", time_s=time, measured_qpos=qpos,
                        measured_qvel=qvel, target=target, **arrays)
    render(model, time, qpos, arrays, args.output / "world_model_comparison.mp4")


if __name__ == "__main__":
    main()
