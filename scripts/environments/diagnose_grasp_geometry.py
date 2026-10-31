"""Inspect the official Beta 1 geometry and first PhysX contact of a grasp pose.

The probe resets only at episode boundaries. It never advances with a
teleported hand or bottle and does not treat distal body origins as tip sites.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path)
parser.add_argument("--radius", type=float, default=0.035)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
import warp as wp  # noqa: E402
from dexteroushands_lab.assets.bottle import BottleConfig  # noqa: E402
from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402


def _tensor(x):
    return x.detach().cpu().tolist()


def _contact_points(fixture):
    records = []
    for env, source, filters, view in fixture.monitor._views:
        if env != 0 or not source.endswith("/Bottle"):
            continue
        forces, points, normals, distances, counts, starts = view.get_contact_data(dt=fixture.dt)
        forces, points, normals, distances = map(wp.to_torch, (forces, points, normals, distances))
        counts, starts = map(wp.to_torch, (counts, starts))
        for i, path in enumerate(filters):
            count = int(counts.reshape(-1)[i].item())
            if not count:
                continue
            start = int(starts.reshape(-1)[i].item())
            for j in range(start, start + count):
                records.append({"pair": f"{source}|{path}", "point_w_m": _tensor(points[j]),
                                "normal_w": _tensor(normals[j]),
                                "separation_m": float(distances.reshape(-1)[j]),
                                "normal_force_n": float(forces[j].norm())})
    return sorted(records, key=lambda x: x["separation_m"])


def main():
    fixture = BimanualPhase0Fixture(num_envs=1, device=args.device,
                                    config=BottleConfig(bottle_radius=args.radius),
                                    include_hand_hand=False)
    original = fixture.initializer.generate(mode="stable_pregrasp", seed=0)
    right = replace(original.right,
                    root_pose=(original.right.root_pose[0] + 0.4, *original.right.root_pose[1:]),
                    wrist_target_pose=(original.right.wrist_target_pose[0] + 0.4,
                                       *original.right.wrist_target_pose[1:]))
    output = {"radius_m": args.radius, "seed": 0, "cases": []}
    for x in (-0.11, -0.13, -0.15, -0.17):
        for grip in ("legacy", "open"):
            pose = (x, *original.left.root_pose[1:])
            joints = original.left.joint_position if grip == "legacy" else (0.0,) * 20
            left = replace(original.left, root_pose=pose, wrist_target_pose=pose,
                           joint_position=joints, joint_target=joints)
            fixture.reset_manager.reset([0], [replace(original, left=left, right=right)])
            before = {"bottle_pose": _tensor(fixture.bottle.data.root_pose_w.torch[0]),
                      "left_pose": _tensor(fixture.left.data.root_pose_w.torch[0]),
                      "left_velocity": _tensor(fixture.left.data.root_vel_w.torch[0]),
                      "bottle_velocity": _tensor(fixture.bottle.data.root_vel_w.torch[0])}
            fixture.step()
            summary = fixture.monitor.summary(0)
            contacts = _contact_points(fixture)
            result = {"wrist_x_m": x, "grip": grip, "before": before,
                      "left_joint_names": fixture.left.joint_names,
                      "joint_positions_rad": _tensor(fixture.left.data.joint_pos.torch[0]),
                      "tip_body_origins_w": {name: _tensor(fixture.left.data.body_link_pose_w.torch[
                          0, fixture.left.body_names.index(name), :3])
                          for name in fixture.specs[0].fingertip_bodies},
                      "official_tip_sites_w": {name: _tensor(fixture.left.data.body_link_pose_w.torch[
                          0, fixture.left.body_names.index(name), :3])
                          for name in (f"l_{digit}_tip" for digit in
                                       ("thumb", "index_finger", "middle_finger",
                                        "ring_finger", "pinky"))},
                      "contact_count": len(contacts), "worst_contacts": contacts[:15],
                      "left_bottle": summary.get("left_bottle"),
                      "after_bottle_velocity": _tensor(fixture.bottle.data.root_vel_w.torch[0]),
                      "after_left_velocity": _tensor(fixture.left.data.root_vel_w.torch[0])}
            output["cases"].append(result)
            print("GEOMETRY_PROBE " + json.dumps({"x": x, "grip": grip,
                  "min_separation_m": contacts[0]["separation_m"] if contacts else None,
                  "peak_force_n": summary.get("left_bottle", {}).get("peak_force_n", 0.0),
                  "contacts": len(contacts),
                  "bottle_speed_m_s": float(torch.linalg.vector_norm(
                      fixture.bottle.data.root_vel_w.torch[0, :3]))}), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")


try:
    main()
finally:
    app.close()
