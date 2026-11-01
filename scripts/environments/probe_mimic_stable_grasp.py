"""One-family diagnostic: replay a legacy stable grasp with a hard-pitch cap.

The old cap pose violates the new screw relation by about 1.2 mm. This
probe shifts the right wrist by the same bottle-local axial correction and
lets PhysX determine the cap link pose. It never writes the cap link or adds
object support. Passing this probe alone would not validate a new bank.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bank-index", type=int, default=0)
parser.add_argument("--duration", type=float, default=1.0)
parser.add_argument("--left-only", action="store_true",
                    help="Move right hand away to test left-hand support through fingers")
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.bank_index not in range(4) or not 0 < args.duration <= 2.0:
    parser.error("bank-index must be 0..3 and duration must be in (0, 2] s")
app = AppLauncher(args).app

import torch
import warp as wp
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab_physx.physics import PhysxManager
from pxr import Gf, UsdPhysics

from dexteroushands_lab.assets.bottle import BottleConfig
from dexteroushands_lab.assets.bottle.generate_usd import build_bottle_usd
from dexteroushands_lab.assets.bottle.mimic_contract import mimic_reset_geometry
from dexteroushands_lab.assets.bottle.mimic_thread import author_mimic_thread
from dexteroushands_lab.assets.wuji_hand2_beta import load_hand2_spec
from dexteroushands_lab.scenes.batched_contacts import BatchedContactMonitor
from dexteroushands_lab.scenes.bimanual_bottle import make_bimanual_bottle_scene_cfg


def _axis(q):
    x, y, z, w = q
    return (2 * (x * z + w * y), 2 * (y * z - w * x),
            1 - 2 * (x * x + y * y))


def _angle(q0, q1):
    dot = abs(float(torch.dot(q0, q1)))
    return 2 * math.acos(min(1.0, dot))


def main():
    root = Path(__file__).resolve().parents[2]
    row = json.loads((root / "docs/assets/phase0e_bimanual_grasp_init_bank.json")
                     .read_text())[args.bank_index]
    cfg = BottleConfig(**row["metadata"]["bottle_parameters"])
    geometry = mimic_reset_geometry(row["bottle"]["root_pose"],
                                    row["cap"]["root_pose"],
                                    row["thread_initial_angle_rad"], cfg)
    if geometry["lateral_m"] > 0.0001 or abs(geometry["twist_error_rad"]) > 0.01:
        raise RuntimeError("Legacy cap has more than an axial hard-pitch mismatch")
    correction = tuple(-geometry["pitch_error_m"] * component
                       for component in _axis(row["bottle"]["root_pose"][3:]))
    left_pose = tuple(row["left"]["root_pose"])
    right_pose = tuple((row["right"]["root_pose"][i] + correction[i])
                       for i in range(3)) + tuple(row["right"]["root_pose"][3:])
    if args.left_only:
        right_pose = (right_pose[0] + 0.35, *right_pose[1:])
    asset_dir = Path(tempfile.mkdtemp(prefix="dexgym-mimic-grasp-probe-"))
    build_bottle_usd(cfg, asset_dir)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.002, device=args.device))
    scene = InteractiveScene(make_bimanual_bottle_scene_cfg(
        num_envs=1, asset_dir=asset_dir, bottle=cfg,
        wrist_control_mode="fixed_wrist", bottle_center_z=0.5,
        left_initial_pose=left_pose, right_initial_pose=right_pose))
    for side, pose in (("LeftHand", left_pose), ("RightHand", right_pose)):
        joint = UsdPhysics.Joint(sim.stage.GetPrimAtPath(
            f"/World/envs/env_0/{side}/root_joint"))
        joint.CreateLocalPos0Attr(Gf.Vec3f(*pose[:3]))
        joint.CreateLocalRot0Attr(Gf.Quatf(pose[6], *pose[3:6]))
    author_mimic_thread(
        sim.stage, "/World/envs/env_0", cfg,
        cap_local_pose_xyzw=(0., 0., 0.5 + cfg.cap_initial_z, 0., 0., 0., 1.))
    sim.reset()
    left, right = scene["left_hand"], scene["right_hand"]
    bottle, cap = scene["bottle_body"], scene["bottle_cap"]
    physics_view = PhysxManager.get_physics_sim_view()
    articulation = physics_view.create_articulation_view("/World/envs/env_*/Bottle")
    names = list(articulation.shared_metatype.dof_names)
    spin_id, slide_id = names.index("SpinJoint"), names.index("SlideJoint")
    ids = wp.from_torch(torch.tensor([0], dtype=torch.int32, device=args.device))
    dofs = wp.to_torch(articulation.get_dof_positions()).reshape(1, -1).clone()
    theta = row["thread_initial_angle_rad"]
    dofs[0, spin_id] = theta
    dofs[0, slide_id] = cfg.thread_sign * cfg.thread_pitch * theta / (2 * math.pi)
    articulation.set_dof_positions(wp.from_torch(dofs), ids)
    articulation.set_dof_velocities(wp.from_torch(torch.zeros_like(dofs)), ids)
    targets = wp.to_torch(articulation.get_dof_position_targets()).reshape(1, -1).clone()
    targets[0, spin_id] = theta
    articulation.set_dof_position_targets(wp.from_torch(targets), ids)
    bottle_pose = torch.tensor(row["bottle"]["root_pose"], device=args.device,
                               dtype=torch.float32)[None]
    bottle.write_root_pose_to_sim_index(root_pose=bottle_pose)
    bottle.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=args.device))
    for hand, side, pose in ((left, "left", left_pose), (right, "right", right_pose)):
        spec = load_hand2_spec(variant="beta1", side=side)
        order = [spec.joint_names.index(name) for name in hand.joint_names]
        hand_pose = torch.tensor(pose, device=args.device, dtype=torch.float32)[None]
        hand.write_root_pose_to_sim_index(root_pose=hand_pose)
        hand.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=args.device))
        q = torch.tensor(row[side]["joint_position"], device=args.device,
                         dtype=torch.float32)[order][None]
        target = torch.tensor(row[side]["joint_target"], device=args.device,
                              dtype=torch.float32)[order][None]
        hand.write_joint_position_to_sim_index(position=q)
        hand.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        hand.set_joint_position_target_index(target=target)
    physics_view.update_articulations_kinematic()
    contacts = BatchedContactMonitor(
        physics_view, 1, load_hand2_spec(variant="beta1", side="left"),
        load_hand2_spec(variant="beta1", side="right"), device=args.device)
    point_views = {}
    for obj, side, hand_name in (("Bottle", "left", "LeftHand"),
                                 ("Cap", "right", "RightHand")):
        point_views[(obj, side)] = physics_view.create_rigid_contact_view(
            f"/World/envs/env_0/{obj}",
            filter_patterns=[f"/World/envs/env_0/{hand_name}/{body}"
                             for body in contacts.body_names[side]],
            max_contact_data_count=256)
    steps = math.ceil(args.duration / 0.002)
    left_count = right_count = 0
    digit_steps = {side: {digit: 0 for digit in (
        "thumb", "index_finger", "middle_finger", "ring_finger", "pinky")}
        for side in ("left", "right")}
    peak_left = peak_right = 0.0
    max_pitch_error = 0.0
    trajectory = []
    initial_contact_points = {}
    start = bottle_pose[0].clone()
    for step in range(steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(0.002)
        contacts.update(0.002)
        if step == 0:
            for obj, side in (("Bottle", "left"), ("Cap", "right")):
                data = point_views[(obj, side)].get_contact_data(dt=0.002)
                forces, points, normals, distances, counts, starts = map(wp.to_torch, data)
                counts, starts = counts.reshape(-1), starts.reshape(-1)
                if int(counts.to(torch.int64).sum()) > len(forces):
                    raise RuntimeError("Initial contact manifold exceeded 256-point buffer")
                rows = []
                for body_index, body_name in enumerate(contacts.body_names[side]):
                    begin = int(starts[body_index])
                    for point_index in range(begin, begin + int(counts[body_index])):
                        normal_force = float(torch.linalg.vector_norm(forces[point_index]))
                        if normal_force < contacts.force_floor_n:
                            continue
                        rows.append({"body": body_name,
                                     "normal_force_n": normal_force,
                                     "separation_m": float(distances.reshape(-1)[point_index]),
                                     "position_w_m": points[point_index].detach().cpu().tolist(),
                                     "normal_w": normals[point_index].detach().cpu().tolist()})
                initial_contact_points[f"{side}_{obj.lower()}"] = sorted(
                    rows, key=lambda item: item["separation_m"])[:30]
        left_force = float(contacts.force_magnitudes("Bottle", "left").max())
        right_force = float(contacts.force_magnitudes("Cap", "right").max())
        for obj, side in (("Bottle", "left"), ("Cap", "right")):
            magnitudes = contacts.force_magnitudes(obj, side)[0]
            for digit in digit_steps[side]:
                indices = [index for index, name in enumerate(contacts.body_names[side])
                           if digit in name]
                digit_steps[side][digit] += bool(
                    (magnitudes[indices] >= contacts.force_floor_n).any())
        left_count += left_force > contacts.force_floor_n
        right_count += right_force > contacts.force_floor_n
        peak_left = max(peak_left, left_force)
        peak_right = max(peak_right, right_force)
        dofs_now = wp.to_torch(articulation.get_dof_positions()).reshape(1, -1)
        error = float((dofs_now[0, slide_id] - cfg.thread_sign * cfg.thread_pitch
                       * dofs_now[0, spin_id] / (2 * math.pi)).abs())
        max_pitch_error = max(max_pitch_error, error)
        if step % 50 == 0 or step == steps - 1:
            trajectory.append({"t_s": (step + 1) * 0.002,
                               "bottle_position_w_m": bottle.data.root_pose_w.torch[0, :3]
                               .detach().cpu().tolist(),
                               "left_force_n": left_force,
                               "right_force_n": right_force})
    end = bottle.data.root_pose_w.torch[0]
    result = {"bank_index": args.bank_index,
              "left_only": args.left_only,
              "legacy_pitch_error_m": geometry["pitch_error_m"],
              "right_wrist_correction_w_m": correction,
              "duration_s": steps * 0.002,
              "left_bottle_contact_ratio": left_count / steps,
              "right_cap_contact_ratio": right_count / steps,
              "digit_contact_ratios": {
                  side: {digit: count / steps for digit, count in counts.items()}
                  for side, counts in digit_steps.items()},
              "peak_left_force_n": peak_left, "peak_right_force_n": peak_right,
              "bottle_displacement_m": float(torch.linalg.vector_norm(end[:3] - start[:3])),
              "bottle_rotation_rad": _angle(end[3:], start[3:]),
              "max_mimic_pitch_error_m": max_pitch_error,
              "bottle_ground_contact": bool(contacts.ground_contact("Bottle")[0]),
              "initial_contact_points": initial_contact_points,
              "trajectory": trajectory}
    all_initial = [point for rows in initial_contact_points.values() for point in rows]
    result["initial_min_separation_m"] = min(
        (point["separation_m"] for point in all_initial), default=None)
    result["initial_max_point_force_n"] = max(
        (point["normal_force_n"] for point in all_initial), default=0.0)
    result["hold_physics_pass"] = (
        result["duration_s"] >= 1.0
        and result["left_bottle_contact_ratio"] > 0.8
        and (args.left_only or result["right_cap_contact_ratio"] > 0.8)
        and result["bottle_displacement_m"] < 0.02
        and result["bottle_rotation_rad"] < math.radians(10)
        and result["max_mimic_pitch_error_m"] < 0.00005
        and result["initial_min_separation_m"] is not None
        and result["initial_min_separation_m"] >= -0.002
        and result["initial_max_point_force_n"] < 100.0
        and not result["bottle_ground_contact"])
    left_digits = result["digit_contact_ratios"]["left"]
    right_digits = result["digit_contact_ratios"]["right"]
    result["phase1_contact_topology_pass"] = (
        left_digits["thumb"] > 0.8
        and sum(left_digits[digit] > 0.8 for digit in (
            "index_finger", "middle_finger", "ring_finger", "pinky")) >= 2
        and all(right_digits[digit] > 0.8 for digit in (
            "thumb", "index_finger", "middle_finger")))
    result["diagnostic_hold_pass"] = (result["hold_physics_pass"]
                                      and result["phase1_contact_topology_pass"])
    print("MIMIC_GRASP_PROBE", json.dumps({k: v for k, v in result.items()
                                           if k not in ("trajectory", "initial_contact_points")}),
          flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    app.close()


if __name__ == "__main__":
    main()
