#!/usr/bin/env python3
"""Compare archived held-outs with static MuJoCo model and virtual-axis evidence."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def relative_angle(first, second):
    difference = np.asarray(first).T @ np.asarray(second)
    return np.degrees(np.arccos(np.clip((np.trace(difference) - 1) / 2, -1, 1)))


def labeled_bars(ax, labels, values, color, *, limit=None, units, digits=2):
    bars = ax.barh(labels, values, height=0.55, color=color)
    ax.invert_yaxis()
    ax.set_xlabel(units)
    ax.set_xlim(0, max(max(values), limit or 0) * 1.22)
    ax.bar_label(bars, labels=[f"{value:.{digits}f}" for value in values], padding=4, fontsize=9)
    if limit is not None:
        ax.axvline(limit, linestyle="--", color="#202a35", linewidth=1.1, label=f"design max {limit:g}")
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(axis="x", alpha=0.17)
    ax.set_axisbelow(True)


def draw(calibration, model, virtual, output):
    archive = calibration["evidence"]["tracker_archive"]
    heading = archive["heading"]
    references = heading["references"]
    sides = [model["sides"][key]["metrics"] for key in ("l", "r")]
    wrist = [metrics["wrist_scoring_tcp"]["max"][0] * 1000 for metrics in sides]
    fingertip = [
        np.max(np.asarray(metrics["world_landmark_euclidean"]["max"])[:, -1]) * 1000
        for metrics in sides
    ]
    simulated = virtual["virtual_world_from_tracker_rotation"]
    archived = heading["old_rotation"]
    revised = heading["historical_candidate_rotation"]
    angle_values = [relative_angle(archived, revised), relative_angle(archived, simulated),
                    relative_angle(revised, simulated)]

    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "text.color": "#162532", "axes.labelcolor": "#162532"}):
        fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.0), layout="constrained",
                                 gridspec_kw={"width_ratios": (1.05, 1, 1.22)})
        fig.suptitle("DCR-009 calibration preflight", fontsize=18, fontweight="bold")
        labeled_bars(axes[0], [entry["bag"][-6:] for entry in references],
                     [abs(entry["directed_holdout_angle_deg"]) for entry in references],
                     "#9c411c", limit=1, units="Absolute held-out heading error (°)")
        axes[0].set_title("Archived heading sessions", loc="left", fontsize=12, fontweight="bold")
        labeled_bars(axes[1], ["Old → new fit", "Old → virtual", "New fit → virtual"], angle_values,
                     "#276e87", units="SO(3) difference (°)", digits=1)
        axes[1].set_title("Coordinate conventions", loc="left", fontsize=12, fontweight="bold")
        labeled_bars(axes[2], ["Left wrist TCP", "Right wrist TCP", "Left fingertips", "Right fingertips"],
                     wrist + fingertip, "#365b48", units="Max Euclidean FK difference (mm)")
        axes[2].set_title("Installed vs signed MuJoCo model", loc="left", fontsize=12, fontweight="bold")
        fig.supxlabel("Archived recordings and static FK checks  •  Six-root replay not run", fontsize=9, color="#43586a")
        fig.savefig(output, dpi=180, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("calibration", "model", "virtual-axes", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    draw(read(args.calibration), read(args.model), read(args.virtual_axes), args.out)


if __name__ == "__main__":
    main()
