"""Plot real Newton TCP errors, world-model prediction, and training losses."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot(run_dir: Path, output: Path, eval_dir: Path | None = None):
    output.mkdir(parents=True, exist_ok=True)
    eval_dir = eval_dir or run_dir
    with np.load(eval_dir / "newton_policy_states.npz", allow_pickle=False) as states:
        time = states["time"]
        position = states["tcp_position_error_mm"][:, 0]
        rotation = states["tcp_orientation_error_deg"][:, 0]
    phase_metrics = {}
    for name, start, stop in (("opening", 0.0, 1.0), ("wave_peak", 8.0, 12.0), ("active", 9.0, 16.0)):
        selected = (time >= start) & (time < stop)
        if not selected.any():
            continue
        phase_metrics[name] = {"start_s": start, "stop_s": stop, "frames": int(selected.sum())}
        for side, label in enumerate(("left", "right")):
            for metric_name, values in (
                ("position_mm", position[selected, side]),
                ("orientation_deg", rotation[selected, side]),
            ):
                phase_metrics[name][f"{label}_{metric_name}"] = {
                    "mean": float(values.mean()),
                    "p95": float(np.percentile(values, 95)),
                    "max": float(values.max()),
                }
    (output / "phase_metrics.json").write_text(json.dumps(phase_metrics, indent=2))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for side, label in enumerate(("left", "right")):
        axes[0].plot(time, position[:, side], label=label)
        axes[1].plot(time, rotation[:, side], label=label)
    axes[0].axhline(50, color="gray", linestyle="--", linewidth=1)
    axes[1].axhline(15, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("TCP position error (mm)")
    axes[1].set_ylabel("TCP SO(3) error (deg)")
    axes[1].set_xlabel("Newton physical time (s)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle("Recorded Newton policy tracking; early end means safety termination")
    fig.tight_layout()
    fig.savefig(output / "newton_tcp_errors.png", dpi=160)
    plt.close(fig)

    metrics = json.loads((run_dir / "metrics.json").read_text())
    prediction = metrics.get("world_prediction")
    if prediction is None:
        checks = [entry["prediction"] for entry in metrics["world_model_trust_history"] if "prediction" in entry]
        if not checks:
            raise ValueError("No held-out world-model prediction was recorded")
        prediction = checks[-1]
    horizons = ("1", "8", "32")
    actual = [prediction[h]["tcp_position_mean_mm"] for h in horizons]
    constant = [prediction[h]["constant_velocity_tcp_mean_mm"] for h in horizons]
    fig, axis = plt.subplots(figsize=(7, 4))
    x = np.arange(3)
    axis.bar(x - 0.18, actual, 0.36, label="learned model")
    axis.bar(x + 0.18, constant, 0.36, label="constant velocity")
    axis.set_xticks(x, [f"{h} step" for h in horizons])
    axis.set_ylabel("Held-out TCP prediction mean error (mm)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "world_model_prediction.png", dpi=160)
    plt.close(fig)

    curves_npz = run_dir / "learning_curves.npz"
    if curves_npz.exists():
        with np.load(curves_npz, allow_pickle=False) as saved:
            curves = {key: saved[key] for key in ("world_loss", "actor_loss", "critic_loss")}
    else:
        import torch

        curves = torch.load(run_dir / "learning_curves.pt", map_location="cpu", weights_only=True)
    if curves:
        fig, axes = plt.subplots(3, 1, figsize=(9, 7))
        for axis, key in zip(axes, ("world_loss", "actor_loss", "critic_loss")):
            values = curves[key]
            if len(values):
                axis.plot(values)
            axis.set_ylabel(key)
            axis.grid(alpha=0.2)
        axes[-1].set_xlabel("Gradient update")
        fig.tight_layout()
        fig.savefig(output / "training_curves.png", dpi=160)
        plt.close(fig)
    return [str(path) for path in sorted(output.glob("*.png"))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(plot(args.run_dir, args.output, args.eval_dir), indent=2))
