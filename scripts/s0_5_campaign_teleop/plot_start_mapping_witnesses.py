#!/usr/bin/env python3
"""Plot four synchronized hand poses and uncentered wrist midpoints per witness."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


PANELS = {
    "original_human": ("Original human", "Historical calibration, shared placement", "#52587e", "D"),
    "m1": ("M1", "Independent mapped target", "#007c75", "s"),
    "shared": ("Shared rigid", "Mapped target", "#936016", "^"),
    "mujoco": ("MuJoCo", "Measured feedback", "#803e49", "o"),
}
SIDES = (("L", "#176694", "o"), ("R", "#a44528", "^"))
INK = "#233547"
MUTED = "#536477"
GRID = "#d8e1eb"
NORMAL_LENGTH_MM = 55
AXIS_LENGTH_MM = 40
CAMERA = (25, -54)


def numeric(panel, key, shape):
    value = panel.get(key)
    if value is None:
        return np.full(shape, np.nan)
    values = np.asarray(value, dtype=float)
    if values.shape != shape or np.isinf(values).any():
        raise ValueError(f"{key} must have shape {shape} and use null for missing coordinates")
    values = values.copy()
    values[~np.isfinite(values).all(axis=-1)] = np.nan
    return values


def normalized(vector):
    length = np.linalg.norm(vector)
    return vector / length if np.isfinite(length) and length > 1e-12 else np.full(3, np.nan)


def geometry(panel):
    panel = panel or {}
    wrists = numeric(panel, "wrist_xyz_m", (2, 3))
    fingers = numeric(panel, "finger_landmarks_xyz_m", (2, 5, 4, 3))
    centres = numeric(panel, "palm_centres_xyz_m", (2, 3))
    normals = numeric(panel, "palm_normals_world", (2, 3))
    origins = numeric(panel, "palm_origins_xyz_m", (2, 3))
    for side in range(2):
        landmarks = fingers[side, 1:, 0]
        if np.isfinite(origins[side]).all() and np.isfinite(landmarks).all():
            if not np.isfinite(centres[side]).all():
                centres[side] = (origins[side] + landmarks.sum(axis=0)) / 5
            if not np.isfinite(normals[side]).all():
                forward = normalized(landmarks.mean(axis=0) - origins[side])
                across = landmarks[0] - landmarks[-1]
                lateral = normalized(across - (across @ forward) * forward)
                normals[side] = np.cross(forward, lateral) * (1 if side == 0 else -1)
        normals[side] = normalized(normals[side])
    midpoint = wrists.mean(axis=0) if np.isfinite(wrists).all() else None
    status = panel.get("status") or "Missing paired wrist geometry"
    return wrists, fingers, centres, normals, origins, midpoint, status


def local_coordinates(panel):
    wrist, fingers, centres, normals, origins, midpoint, _ = panel
    if midpoint is None:
        return None
    return tuple((points - midpoint) * 1000 for points in (wrist, fingers, centres, origins)) + (normals,)


def equal_extent(panels):
    limits = []
    for panel in panels:
        current = local_coordinates(panel)
        if current is not None:
            wrists, fingers, centres, origins, normals = current
            points = np.vstack((wrists, fingers.reshape(-1, 3), origins, centres, centres + normals * NORMAL_LENGTH_MM))
            finite = points[np.isfinite(points).all(axis=-1)]
            limits.append(float(np.max(np.abs(finite))))
    return max(75., np.ceil((max(limits, default=0.) + 22) / 25) * 25)


def plot_span(ax, wrists):
    ax.plot(*wrists.T, color="#7e8995", linestyle=(0, (3, 3)), linewidth=1.2)
    ax.scatter(0, 0, 0, marker="+", s=54, color=INK, zorder=8, depthshade=False, linewidth=1.5)


def plot_midpoint_axes(ax):
    for direction, label in zip(np.eye(3), ("X", "Y", "Z"), strict=True):
        ax.quiver(0, 0, 0, *direction, length=AXIS_LENGTH_MM, color="#708193", linewidth=.85, arrow_length_ratio=.24)
        ax.text(*(direction * (AXIS_LENGTH_MM + 8)), label, color=MUTED, fontsize=7)


def plot_hand(ax, wrist, fingers, centre, normal, origin, style):
    side, color, symbol = style
    ax.scatter(*wrist, marker=symbol, s=47, facecolors="white", edgecolors=color,
               linewidth=1.4, depthshade=False, zorder=10)
    ax.text(*(wrist + [0, 0, -12]), side, color=color, fontsize=10, fontweight="bold")
    if np.isfinite(origin).all() and np.isfinite(fingers[1:, 0]).all():
        hull = np.vstack((origin, fingers[1:, 0]))
        ax.add_collection3d(Poly3DCollection([hull], facecolors=color, edgecolors=color, linewidth=.55, alpha=.13))
    for line in fingers:
        points = np.vstack((origin, line)) if np.isfinite(origin).all() else line
        ax.plot(*points.T, color=color, linewidth=1.7, solid_capstyle="round")
        if np.isfinite(line[-1]).all():
            ax.scatter(*line[-1], s=13, marker=symbol, color=color, depthshade=False)
    if np.isfinite(centre).all() and np.isfinite(normal).all():
        ax.quiver(*centre, *normal, length=NORMAL_LENGTH_MM, linewidth=2.2,
                  color=color, arrow_length_ratio=.27, normalize=True)


def number(value):
    return f"{value:.0f}" if np.isfinite(value) else "\N{EM DASH}"


def describe(panel):
    wrists, fingers, _, normals, _, midpoint, _ = panel
    if midpoint is None:
        return "Wrist gap \N{EM DASH}   Palm angle \N{EM DASH}\nWorld midpoint \N{EM DASH}"
    gap = np.linalg.norm(wrists[1] - wrists[0]) * 1000
    dot = float(np.clip(normals[0] @ normals[1], -1., 1.))
    obliquity = np.degrees(np.arccos(abs(dot)))
    directed = np.degrees(np.arccos(dot))
    tips = fingers[:, :, -1]
    cross = np.linalg.norm(tips[0, :, None] - tips[1, None, :], axis=-1)
    finite = cross[np.isfinite(cross)]
    nearest = f"{number(finite.min() * 1000)} mm" if len(finite) else "\N{EM DASH}"
    location = ", ".join(f"{component:+.0f}" for component in midpoint * 1000)
    angle = f"{number(obliquity)}\N{DEGREE SIGN} plane / {number(directed)}\N{DEGREE SIGN} normals" if np.isfinite(dot) else "\N{EM DASH}"
    return (f"Wrist gap  {number(gap)} mm   Palm  {angle}"
            f"\nWorld midpoint  ({location}) mm"
            f"\nClosest plotted fingertips  {nearest}")


def local_view(ax, panel, extent, title, subtitle):
    ax.set_proj_type("ortho")
    ax.view_init(elev=CAMERA[0], azim=CAMERA[1])
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent))
    ax.set_box_aspect((1., 1., 1.), zoom=1.26)
    ax.set_axis_off()
    ax.text2D(.02, .995, title, transform=ax.transAxes, fontsize=15, fontweight="bold", color=INK, va="top")
    ax.text2D(.02, .934, subtitle, transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top")
    current = local_coordinates(panel)
    if current is None:
        text = str(panel[-1]).replace("_", " ")
        ax.text2D(.5, .50, text, transform=ax.transAxes, ha="center", va="center", color=MUTED,
                  fontsize=9, wrap=True)
    else:
        wrists, fingers, centres, origins, normals = current
        plot_midpoint_axes(ax)
        plot_span(ax, wrists)
        for side, style in enumerate(SIDES):
            plot_hand(ax, wrists[side], fingers[side], centres[side], normals[side], origins[side], style)
    ax.text2D(.02, .01, describe(panel), transform=ax.transAxes, fontsize=8.1,
              color=INK, va="bottom", linespacing=1.45)


def world_views(axes, panels):
    points = [panel[-2] * 1000 for panel in panels if panel[-2] is not None]
    if not points:
        for ax in axes:
            ax.set_axis_off()
        return
    points = np.asarray(points)
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    halfspan = max(60., np.ptp(points, axis=0).max() * .65 + 10)
    for ax, vertical, label in zip(axes, (1, 2), ("Top view", "Front view"), strict=True):
        for (_, (_, _, color, symbol)), panel in zip(PANELS.items(), panels, strict=True):
            midpoint = panel[-2]
            if midpoint is not None:
                p = midpoint * 1000
                ax.scatter(p[0], p[vertical], color=color, marker=symbol, s=47, edgecolor="white", linewidth=.65, zorder=5)
        ax.set_xlim(centre[0] - halfspan, centre[0] + halfspan)
        ax.set_ylim(centre[vertical] - halfspan, centre[vertical] + halfspan)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(label, color=INK, fontsize=10, loc="left", pad=5)
        ax.set_xlabel("World X (mm)", fontsize=8, color=MUTED, labelpad=1)
        ax.set_ylabel(f"World {'Y' if vertical == 1 else 'Z'} (mm)", fontsize=8, color=MUTED, labelpad=1)
        ax.tick_params(labelsize=7, colors=MUTED, length=2)
        ax.grid(color=GRID, linewidth=.55)
        for spine in ax.spines.values():
            spine.set_color(GRID)


def draw(sample, out):
    panels = [geometry(sample["panels"].get(key)) for key in PANELS]
    extent = equal_extent(panels)
    figure = plt.figure(figsize=(15, 6.7), dpi=155, facecolor="#f4f7fb")
    grids = figure.add_gridspec(2, 4, left=.035, right=.975, top=.84, bottom=.095,
                               height_ratios=(4.0, 1.95), wspace=.075, hspace=.19)
    title = "Bimanual mapping at the same recorded cutoff"
    figure.text(.035, .951, title, fontsize=22, fontweight="bold", color=INK, va="top")
    cutoff = int(sample["cutoff_ns"])
    selector = sample.get("selection")
    details = f"{sample['root']}  |  {sample['split']}  |  HMB cutoff {cutoff / 1e9:.6f} s ({cutoff:,} ns)"
    if isinstance(selector, str) and selector:
        details += f"  |  {selector}"
    figure.text(.037, .887, details, fontsize=10, color=MUTED, va="top")
    for column, (panel, (title, subtitle, _, _)) in enumerate(zip(panels, PANELS.values(), strict=True)):
        axis = figure.add_subplot(grids[0, column], projection="3d", facecolor="white")
        local_view(axis, panel, extent, title, subtitle)
    world_x = [figure.add_subplot(grids[1, column], facecolor="white") for column in (1, 2)]
    world_views(world_x, panels)
    left = figure.add_subplot(grids[1, 0])
    left.set_axis_off()
    left.text(.02, .89, "Uncentered world midpoints", fontsize=10, fontweight="bold", color=INK, transform=left.transAxes)
    left.text(.02, .69, "The two views at right retain the\nabsolute world midpoint offsets.",
              fontsize=8.7, color=MUTED, linespacing=1.5, transform=left.transAxes, va="top")
    right = figure.add_subplot(grids[1, 3])
    right.set_axis_off()
    proxies = [Line2D([0], [0], marker=marker, linestyle="None", color=color, markersize=6, label=label)
               for label, _, color, marker in PANELS.values()]
    right.legend(handles=proxies, loc="upper left", frameon=False, borderaxespad=0.1, labelcolor=INK,
                 fontsize=8, title="World midpoint", title_fontsize=9)
    figure.text(.037, .025,
                "Hand panels are translated to their own wrist midpoint. They share world XYZ orientation, orthographic camera and millimetre scale. "
                "Gray XYZ axes are 40 mm; colored arrows show directed palm normals.",
                color=MUTED, fontsize=8.5)
    figure.savefig(out, facecolor=figure.get_facecolor())
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witnesses", "--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.witnesses.read_bytes())
    samples = data["samples"]
    if not samples:
        raise ValueError("No mapping witnesses supplied")
    identities = set()
    artefacts = []
    for sample in samples:
        key = sample["root"], sample["cutoff_ns"]
        if key in identities or not set(sample["panels"]) <= set(PANELS):
            raise ValueError(f"Repeated witness or unexpected panel name: {key}")
        identities.add(key)
        tag = "".join(c for c in key[0].split(":")[-1][:12] if c.isalnum())
        artefacts.append((sample, args.out / f"{tag}_{int(key[1])}_mapping_witness.png"))
    args.out.mkdir(parents=True, exist_ok=False)
    for sample, image in artefacts:
        draw(sample, image)
        print(image)


if __name__ == "__main__":
    main()
