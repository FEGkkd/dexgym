"""Render audited human upper-body motion and measured local wrist orientation."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from scipy.spatial.transform import Rotation

ROOT = Path("/home/ubuntu/dexgym_data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("motion_id")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    file = args.root / "human_motion_v1/diagnostic_motions" / f"{args.motion_id}.npz"
    data = np.load(file)
    time = data["time"]
    shoulder, elbow, hand, forearm = (data[key] for key in ("shoulder_pose", "elbow_pose", "hand_pose", "forearm_pose"))
    out = args.root / "human_motion_v1/previews"
    out.mkdir(parents=True, exist_ok=True)
    relative = np.zeros((len(time), 2, 3))
    for side in range(2):
        fr = Rotation.from_quat(forearm[:, side, 3:])
        hr = Rotation.from_quat(hand[:, side, 3:])
        relative[:, side] = np.rad2deg((fr.inv() * hr).as_rotvec())
    fig, (ax, trace) = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": None})
    # Replace the first planar axes with a 3-D view while keeping trace planar.
    ax.remove()
    ax = fig.add_subplot(121, projection="3d")
    center = np.mean(np.concatenate((shoulder[..., :3], hand[..., :3]), axis=1).reshape(-1, 3), axis=0)
    span = max(0.55, float(np.max(np.abs(np.concatenate((shoulder[..., :3], hand[..., :3]), axis=1) - center))))
    colors = ("tab:blue", "tab:orange")
    axis_colors = ("r", "g", "b")
    for side, name in enumerate(("left", "right")):
        for component, label in enumerate("XYZ"):
            trace.plot(
                time,
                relative[:, side, component],
                color=axis_colors[component],
                alpha=1 if side == 0 else 0.5,
                linestyle="-" if side == 0 else "--",
                label=f"{name} local {label}",
            )
    trace.set_xlabel("time (s)")
    trace.set_ylabel("hand relative to forearm, rotvec (deg)")
    trace.legend(loc="upper right", fontsize=7)
    trace.grid(True)
    cursor = trace.axvline(time[0], color="k", alpha=0.4)

    def draw(i: int) -> None:
        ax.cla()
        for side in range(2):
            pts = np.stack((shoulder[i, side, :3], elbow[i, side, :3], hand[i, side, :3]))
            ax.plot(pts[:, 0], pts[:, 2], pts[:, 1], color=colors[side], marker="o", lw=2)
            rotation = Rotation.from_quat(hand[i, side, 3:]).as_matrix()
            for component in range(3):
                end = hand[i, side, :3] + 0.11 * rotation[:, component]
                ax.plot(
                    [pts[-1, 0], end[0]], [pts[-1, 2], end[2]], [pts[-1, 1], end[1]], color=axis_colors[component], lw=2
                )
        ax.plot(shoulder[i, :, 0], shoulder[i, :, 2], shoulder[i, :, 1], color="k", lw=2)
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[2] - span, center[2] + span)
        ax.set_zlim(center[1] - span, center[1] + span)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z forward (m)")
        ax.set_zlabel("Y up (m)")
        ax.set_title(f"{args.motion_id}: BVH reconstructed hand axes, t={time[i]:.2f}s")
        cursor.set_xdata([time[i], time[i]])

    draw(0)
    fig.tight_layout()
    fig.savefig(out / f"{args.motion_id}_orientation.png", dpi=150)
    with FFMpegWriter(fps=25, codec="libx264", bitrate=1600).saving(
        fig, str(out / f"{args.motion_id}_orientation.mp4"), dpi=110
    ) as writer:
        for i in range(0, len(time), args.stride):
            draw(i)
            writer.grab_frame()
    plt.close(fig)
    print(out / f"{args.motion_id}_orientation.mp4")


if __name__ == "__main__":
    main()
