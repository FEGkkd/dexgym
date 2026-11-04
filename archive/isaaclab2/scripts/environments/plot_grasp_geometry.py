"""Plot pinned bottle cross-section, contact sites, and PhysX replay geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def _draw(ax, radius, bottle_xy, sites, title):
    ax.add_patch(Circle(bottle_xy, radius, facecolor="#dcecf2",
                        edgecolor="#176b86", linewidth=2, alpha=.8))
    colors = {"thumb": "#c44e52", "index": "#4c72b0", "middle": "#55a868",
              "ring": "#8172b3", "pinky": "#dd8452"}
    for name, pos in sites.items():
        digit = next((key for key in colors if key in name), name)
        ax.scatter(pos[0], pos[1], s=45, color=colors.get(digit, "black"),
                   label=digit)
        ax.annotate(digit, (pos[0], pos[1]), xytext=(3, 3),
                    textcoords="offset points", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-.085, .085)
    ax.set_ylim(-.085, .085)
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.set_title(title)
    ax.grid(alpha=.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reach", type=Path, required=True)
    parser.add_argument("--synthesis", type=Path, required=True)
    parser.add_argument("--proposal_index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reach = json.loads(args.reach.read_text())
    synthesis = json.loads(args.synthesis.read_text())
    proposal = next(p for p in reach["best"] if p["index"] == args.proposal_index)
    result = next(r for r in synthesis["results"] if
                  r["reach_proposal_index"] == args.proposal_index)
    settle = min((entry for entry in result["trace"] if entry["phase"] == "settle"),
                 key=lambda entry: abs(entry["t_s"] - 1.0))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    _draw(axes[0], reach["radius_m"], (0, 0), proposal["tip_sites_w_m"],
          "Official MJCF geometric proposal (unvalidated)")
    _draw(axes[1], reach["radius_m"], settle["bottle_pose_w"][:2],
          settle["official_tip_sites_w_m"],
          f"Native USD/PhysX after gradual closing, t={settle['t_s']:.2f}s")
    fig.suptitle("WujiHand2 Beta 1 left-hand fingertip site vs bottle cross-section")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
