"""Render recorded Newton policy states against the pinned Marvin FK reference."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "motion"))
from preflight import REFERENCE, SOURCE, verify  # noqa: E402
from render_marvin_wave import H, RobotRender, W, _close, _write, _writer  # noqa: E402


def render(states_path: Path, output: Path, stride: int = 3):
    verify()
    if stride < 1:
        raise ValueError("stride must be positive")
    with np.load(REFERENCE, allow_pickle=False) as ref, np.load(states_path, allow_pickle=False) as states:
        q = states["q"]
        if q.ndim != 3 or q.shape[1:] != (1, 54) or len(q) > len(ref["time"]):
            raise ValueError("Recorded Newton states must be [T,1,54] within the reference")
        output.mkdir(parents=True, exist_ok=True)
        video_path = output / "newton_policy_vs_reference.mp4"
        writer = _writer(video_path, 50 / stride, 2 * W, H)
        robot = RobotRender(SOURCE)
        try:
            for i in range(0, len(q), stride):
                target_p = ref["tcp_pos"][i, 1]
                target_q = ref["tcp_quat_xyzw"][i, 1]
                left_target = (ref["tcp_pos"][i, 0], ref["tcp_quat_xyzw"][i, 0])
                reference = robot.frame(
                    ref["q_full"][i], target_p, target_q, "Pinned Marvin reference FK", left_target=left_target
                )
                physical = robot.frame(
                    q[i, 0],
                    target_p,
                    target_q,
                    "Recorded Newton policy physical state",
                    left_target=left_target,
                )
                panel = Image.new("RGB", (2 * W, H))
                panel.paste(reference, (0, 0))
                panel.paste(physical, (W, 0))
                _write(writer, panel)
        finally:
            _close(writer)
            robot.close()
        manifest = {
            "video": str(video_path),
            "recorded_newton_states": str(states_path),
            "reference": str(REFERENCE),
            "recorded_frames": len(q),
            "reference_frames": len(ref["time"]),
            "full_motion_recorded": len(q) == len(ref["time"]),
            "rendering": (
                "Recorded Newton MJWarp physical states visualized offline with MuJoCo EGL; no MuJoCo physics stepping"
            ),
        }
        (output / "render_manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(render(args.states, args.output, args.stride), indent=2))
