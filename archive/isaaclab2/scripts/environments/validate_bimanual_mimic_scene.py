"""GPU contract: existing two-hand InteractiveScene with a hard-pitch cap.

This test catches an important integration failure: the old Bottle/Cap
RigidObject handles must still expose correct dynamic link states after the
pair becomes one floating articulation. It does not claim grasp success.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import tempfile
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--tilt-deg", type=float, default=0.0)
parser.add_argument("--initial-angle-rad", type=float, default=0.0)
parser.add_argument("--reset-repetitions", type=int, default=0)
parser.add_argument("--random-reset-ids", action="store_true")
parser.add_argument("--contact-audit", action="store_true")
parser.add_argument("--implicit-running-drive", action="store_true",
                    help="Update the bounded PhysX breakaway/running spin drive every substep")
parser.add_argument("--right-pregrasp-contact", action="store_true",
                    help="Diagnostic scripted right wrist approach from the geometric bank")
parser.add_argument("--approach-speed", type=float, default=0.05,
                    help="Right contact probe speed in m/s, clipped to 0.1")
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import warp as wp
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab_physx.physics import PhysxManager

from dexteroushands_lab.assets.bottle import BottleConfig
from dexteroushands_lab.assets.bottle.generate_usd import build_bottle_usd
from dexteroushands_lab.assets.bottle.mimic_thread import author_mimic_thread
from dexteroushands_lab.assets.bottle.mimic_thread_view import MimicThreadView
from dexteroushands_lab.assets.bottle.mimic_drive import BatchedMimicDrive
from dexteroushands_lab.assets.bottle.batched_thread import _rotate
from dexteroushands_lab.assets.wuji_hand2_beta import load_hand2_spec
from dexteroushands_lab.scenes.batched_contacts import BatchedContactMonitor
from dexteroushands_lab.scenes.bimanual_bottle import make_bimanual_bottle_scene_cfg


def main():
    cfg = BottleConfig()
    if args.right_pregrasp_contact and args.num_envs != 1:
        raise ValueError("right-pregrasp-contact is a one-environment diagnostic")
    if args.reset_repetitions < 0 or args.reset_repetitions > 1000:
        raise ValueError("reset-repetitions must be in [0, 1000]")
    if args.right_pregrasp_contact and args.reset_repetitions:
        raise ValueError("Contact probe and repeated Reset are separate diagnostics")
    if args.random_reset_ids and (args.num_envs < 2 or not args.reset_repetitions):
        raise ValueError("random-reset-ids requires at least 2 envs and reset repetitions")
    if not math.isfinite(args.approach_speed) or not 0 < args.approach_speed <= 0.1:
        raise ValueError("approach-speed must be in (0, 0.1] m/s")
    pregrasp = None
    if args.right_pregrasp_contact:
        bank = Path(__file__).resolve().parents[2] / "docs/assets/phase0_pregrasp_reset_bank.json"
        pregrasp = json.loads(bank.read_text())[0]
        normal = pregrasp["metadata"]["geometry_evidence"]["native_physx"]["contacts"][0][
            "normal_w"]
        approach_dir = (-normal[0], -normal[1], 0.0)
    tilt = math.radians(args.tilt_deg)
    if not math.isfinite(tilt) or abs(tilt) > math.pi / 2:
        raise ValueError("tilt-deg must be finite and within +/-90 degrees")
    if not math.isfinite(args.initial_angle_rad) or abs(args.initial_angle_rad) > 2 * math.pi:
        raise ValueError("initial-angle-rad must be finite and within +/-one turn")
    if args.right_pregrasp_contact and (tilt != 0.0 or args.initial_angle_rad != 0.0):
        raise ValueError("The saved right pregrasp contact probe requires upright zero-angle cap")
    orientation = (0.0, math.sin(tilt / 2), 0.0, math.cos(tilt / 2))
    asset_dir = Path(tempfile.mkdtemp(prefix="dexgym-bimanual-mimic-scene-"))
    build_bottle_usd(cfg, asset_dir)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.002, device=args.device))
    scene = InteractiveScene(make_bimanual_bottle_scene_cfg(
        num_envs=args.num_envs, asset_dir=asset_dir, bottle=cfg,
        bottle_center_z=0.5, bottle_tilt_radians=tilt,
        wrist_control_mode="pose_command_wrist",
        right_initial_pose=tuple(pregrasp["right"]["root_pose"]) if pregrasp else None))
    if pregrasp is not None:
        from pxr import Gf, UsdPhysics
        joint = UsdPhysics.Joint(sim.stage.GetPrimAtPath(
            "/World/envs/env_0/RightHand/root_joint"))
        pose = pregrasp["right"]["root_pose"]
        joint.CreateLocalPos0Attr(Gf.Vec3f(*pose[:3]))
        joint.CreateLocalRot0Attr(Gf.Quatf(pose[6], *pose[3:6]))
    for env in range(args.num_envs):
        author_mimic_thread(
            sim.stage, f"/World/envs/env_{env}", cfg,
            cap_local_pose_xyzw=(cfg.cap_initial_z * math.sin(tilt), 0.0,
                                 0.5 + cfg.cap_initial_z * math.cos(tilt),
                                 *orientation))
    sim.reset()
    bottle, cap = scene["bottle_body"], scene["bottle_cap"]
    left, right = scene["left_hand"], scene["right_hand"]
    physics_view = PhysxManager.get_physics_sim_view()
    contacts = None
    if args.contact_audit or args.right_pregrasp_contact:
        contacts = BatchedContactMonitor(
            physics_view, args.num_envs,
            load_hand2_spec(variant="beta1", side="left"),
            load_hand2_spec(variant="beta1", side="right"),
            device=args.device)
    art = physics_view.create_articulation_view("/World/envs/env_*/Bottle")
    if art.count != args.num_envs or art.shared_metatype.fixed_base:
        raise AssertionError("Expected one free bottle articulation per environment")
    names = list(art.shared_metatype.dof_names)
    links = list(art.shared_metatype.link_names)
    if set(names) != {"SlideJoint", "SpinJoint"} or set(links) != {"Bottle", "Carriage", "Cap"}:
        raise AssertionError(f"Wrong screw topology: {names}, {links}")
    slide_id, spin_id, cap_id = names.index("SlideJoint"), names.index("SpinJoint"), links.index("Cap")
    thread_view = MimicThreadView(physics_view, art, cfg, device=args.device,
                                  num_envs=args.num_envs)
    drive = (BatchedMimicDrive(art, cfg, device=args.device, num_envs=args.num_envs)
             if args.implicit_running_drive else None)
    if drive is not None:
        initial_drive_state = thread_view.read()
        drive.reset(torch.arange(args.num_envs, device=args.device),
                    initial_drive_state.theta_rad)
    if len(left.joint_names) != 20 or len(right.joint_names) != 20:
        raise AssertionError("Official hand active finger joints changed")
    if pregrasp is not None:
        spec = load_hand2_spec(variant="beta1", side="right")
        order = [spec.joint_names.index(name) for name in right.joint_names]
        finger = torch.tensor(pregrasp["right"]["joint_position"], device=args.device,
                              dtype=torch.float32)[order][None]
        right.write_joint_position_to_sim_index(position=finger)
        right.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(finger))
        right.set_joint_position_target_index(target=finger)
    # Selective Reset must write the floating articulation root and both
    # screw DOFs together. A direct Cap RigidObject transform write is invalid.
    first_id = torch.tensor([0], dtype=torch.long, device=args.device)
    if pregrasp is not None:
        initial = thread_view.read()
        thread_view.reset(first_id, root_pose_w=initial.root_pose_w[0:1],
                          root_velocity_w=torch.zeros(1, 6, device=args.device),
                          theta_rad=torch.zeros(1, device=args.device))
        if drive is not None:
            drive.reset(first_id, torch.zeros(1, device=args.device))
    if pregrasp is None:
        before = thread_view.read()
        moved_bottle = before.root_pose_w[0:1].clone()
        moved_bottle[:, 0] += 0.02
        thread_view.reset(first_id, root_pose_w=moved_bottle,
                          root_velocity_w=torch.zeros(1, 6, device=args.device),
                          theta_rad=torch.tensor([args.initial_angle_rad], device=args.device))
        if drive is not None:
            drive.reset(first_id, torch.tensor([args.initial_angle_rad], device=args.device))
        after = thread_view.read()
        if not torch.allclose(after.root_pose_w[0], moved_bottle[0], atol=1e-5):
            raise AssertionError("Selective root Reset did not restore Bottle pose")
        local_axis = _rotate(moved_bottle[:, 3:], torch.tensor(
            [[0.0, 0.0, 1.0]], device=args.device))[0]
        expected_cap_pos = before.cap_pose_w[0, :3] + torch.tensor(
            [0.02, 0.0, 0.0], device=args.device)
        expected_cap_pos += (cfg.thread_sign * cfg.thread_pitch
                             * args.initial_angle_rad / (2 * math.pi)) * local_axis
        if not torch.allclose(after.cap_pose_w[0, :3], expected_cap_pos, atol=1e-4):
            raise AssertionError("Cap link did not follow reset root and local screw travel")
        if abs(float(after.theta_rad[0]) - args.initial_angle_rad) > 1e-5:
            raise AssertionError("Selective screw Reset did not restore angle")
        if args.num_envs > 1 and (
            not torch.equal(after.root_pose_w[1:], before.root_pose_w[1:])
            or not torch.equal(after.theta_rad[1:], before.theta_rad[1:])
            or not torch.equal(after.cap_pose_w[1:], before.cap_pose_w[1:])
        ):
            raise AssertionError("Selective screw Reset altered peer environments")
    max_peer_reset_change = 0.0
    reset_unique_envs = 0
    mean_reset_ms = None
    mean_read_ms = None
    if args.reset_repetitions:
        initial_state = thread_view.read()
        generator = random.Random(2026)
        selected_envs = set()
        for repetition in range(args.reset_repetitions):
            chosen = (generator.randrange(args.num_envs) if args.random_reset_ids else 0)
            selected_envs.add(chosen)
            reset_id = torch.tensor([chosen], dtype=torch.long, device=args.device)
            before_repetition = thread_view.read()
            target_root = before_repetition.root_pose_w[chosen:chosen + 1].clone()
            if args.random_reset_ids:
                target_root[:, 0] += 0.001 if repetition % 2 else -0.001
            angle = args.initial_angle_rad + (0.1 if repetition % 2 else -0.1)
            thread_view.reset(reset_id, root_pose_w=target_root,
                              root_velocity_w=torch.zeros(1, 6, device=args.device),
                              theta_rad=torch.tensor([angle], device=args.device))
            if drive is not None:
                drive.reset(reset_id, torch.tensor([angle], device=args.device))
            current = thread_view.read()
            if abs(float(current.opening_rad[chosen])) > 1e-5:
                raise AssertionError("Opening progress was not zeroed by selective Reset")
            if args.num_envs > 1:
                peer_mask = torch.ones(args.num_envs, device=args.device, dtype=torch.bool)
                peer_mask[chosen] = False
                for field in ("root_pose_w", "root_velocity_w", "cap_pose_w",
                              "theta_rad", "axial_m", "opening_rad"):
                    difference = (getattr(current, field)[peer_mask]
                                  - getattr(before_repetition, field)[peer_mask]).abs().max()
                    max_peer_reset_change = max(max_peer_reset_change, float(difference))
                    if float(difference) > 1e-6:
                        raise AssertionError(f"Selective Reset changed peer {field}")
        reset_unique_envs = len(selected_envs)
        all_envs = torch.arange(args.num_envs, device=args.device, dtype=torch.long)
        thread_view.reset(all_envs, root_pose_w=initial_state.root_pose_w,
                          root_velocity_w=initial_state.root_velocity_w,
                          theta_rad=initial_state.theta_rad,
                          spin_velocity_rad_s=initial_state.spin_velocity_rad_s)
        if drive is not None:
            drive.reset(all_envs, initial_state.theta_rad)
        restored = thread_view.read()
        if (not torch.allclose(restored.root_pose_w, initial_state.root_pose_w, atol=1e-5)
                or not torch.allclose(restored.theta_rad, initial_state.theta_rad, atol=1e-5)
                or not torch.allclose(restored.cap_pose_w, initial_state.cap_pose_w, atol=1e-5)):
            raise AssertionError("Full restoration after randomized selective Resets failed")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_reset = time.perf_counter()
        for repetition in range(args.reset_repetitions):
            angle = args.initial_angle_rad + (0.1 if repetition % 2 else -0.1)
            thread_view.reset(first_id, root_pose_w=moved_bottle,
                              root_velocity_w=torch.zeros(1, 6, device=args.device),
                              theta_rad=torch.tensor([angle], device=args.device))
            if drive is not None:
                drive.reset(first_id, torch.tensor([angle], device=args.device))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mean_reset_ms = 1000 * (time.perf_counter() - start_reset) / args.reset_repetitions
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_read = time.perf_counter()
        for _ in range(args.reset_repetitions):
            _ = thread_view.read()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mean_read_ms = 1000 * (time.perf_counter() - start_read) / args.reset_repetitions
        thread_view.reset(all_envs, root_pose_w=initial_state.root_pose_w,
                          root_velocity_w=initial_state.root_velocity_w,
                          theta_rad=initial_state.theta_rad,
                          spin_velocity_rad_s=initial_state.spin_velocity_rad_s)
        if drive is not None:
            drive.reset(all_envs, initial_state.theta_rad)
    force = torch.zeros((args.num_envs, 3, 3), device=args.device)
    if pregrasp is None:
        force[0, cap_id, 0] = 0.5 * math.sin(tilt)
        force[0, cap_id, 2] = 0.5 * math.cos(tilt)
    ids = wp.from_torch(torch.arange(args.num_envs, dtype=torch.int32, device=args.device))
    peak_right_cap_force = 0.0
    contact_made = False
    approach_offset = 0.0
    first_contact_points = []
    for step in range(args.steps):
        if drive is not None:
            q = wp.to_torch(art.get_dof_positions()).reshape(args.num_envs, 2)
            qd = wp.to_torch(art.get_dof_velocities()).reshape(args.num_envs, 2)
            drive.step(q[:, spin_id], qd[:, spin_id], q[:, slide_id])
        if pregrasp is not None:
            # This is a measured positive-contact probe, never a policy or
            # grasp validation. The bounded wrist follows the falling cap
            # while approaching along the measured cap-to-finger normal.
            # Stop on
            # first measured contact to limit penetration.
            t = (step + 1) * 0.002
            if not contact_made:
                approach_offset = min(args.approach_speed * t, 0.003)
            root = torch.tensor(pregrasp["right"]["root_pose"], device=args.device,
                                dtype=torch.float32)[None]
            root[0, 0] += approach_dir[0] * approach_offset
            root[0, 1] += approach_dir[1] * approach_offset
            root[0, 2] -= 0.5 * 9.81 * t * t
            right.write_root_pose_to_sim_index(root_pose=root)
            right.write_root_velocity_to_sim_index(root_velocity=torch.tensor(
                [[0.0 if contact_made else approach_dir[0] * args.approach_speed,
                  0.0 if contact_made else approach_dir[1] * args.approach_speed,
                  -9.81 * t, 0.0, 0.0, 0.0]], device=args.device))
        scene.write_data_to_sim()
        art.apply_forces_and_torques_at_position(wp.from_torch(force), None, None, ids, True)
        sim.step()
        scene.update(0.002)
        if contacts is not None:
            contacts.update(0.002)
            peak_right_cap_force = max(peak_right_cap_force,
                                       float(contacts.force_magnitudes("Cap", "right").max()))
            if pregrasp is not None and not contact_made and peak_right_cap_force > 0.01:
                data = contacts.views[("Cap", "right")].get_contact_data(dt=0.002)
                normal_forces, points, normals, distances, counts, starts = map(
                    wp.to_torch, data)
                counts, starts = counts.reshape(-1), starts.reshape(-1)
                cap_pose = cap.data.root_pose_w.torch[0]
                inverse_q = torch.cat((-cap_pose[3:6], cap_pose[6:7]))
                for body_index, body_name in enumerate(contacts.body_names["right"]):
                    begin = int(starts[body_index])
                    end = begin + int(counts[body_index])
                    for point_id in range(begin, end):
                        force_n = float(torch.linalg.vector_norm(normal_forces[point_id]))
                        if force_n < contacts.force_floor_n:
                            continue
                        local = _rotate(inverse_q[None],
                                        (points[point_id] - cap_pose[:3])[None])[0]
                        first_contact_points.append({
                            "body": body_name, "normal_force_n": force_n,
                            "separation_m": float(distances.reshape(-1)[point_id]),
                            "point_cap_local_m": local.detach().cpu().tolist(),
                            "normal_w": normals[point_id].detach().cpu().tolist(),
                        })
            if peak_right_cap_force > 0.01:
                contact_made = True
    positions = wp.to_torch(art.get_dof_positions()).reshape(args.num_envs, -1)
    pitch_error = positions[:, slide_id] - cfg.thread_pitch * positions[:, spin_id] / (2 * math.pi)
    bp = bottle.data.root_pose_w.torch
    cp = cap.data.root_pose_w.torch
    if bp.shape != (args.num_envs, 7) or cp.shape != (args.num_envs, 7):
        raise AssertionError("Existing Bottle/Cap RigidObject link data shapes changed")
    if not torch.isfinite(torch.cat((bp, cp), dim=-1)).all():
        raise AssertionError("Existing Bottle/Cap RigidObject link data are nonfinite")
    local_axis = _rotate(bp[:, 3:], torch.tensor(
        [0.0, 0.0, 1.0], device=args.device).expand(args.num_envs, 3))
    physical_axial = ((cp[:, :3] - bp[:, :3]) * local_axis).sum(-1) - cfg.cap_initial_z
    physical_pitch_error = physical_axial - positions[:, slide_id]
    if float(pitch_error.abs().max()) > 0.00005:
        raise AssertionError("Mimic joint no longer enforces signed pitch")
    if float(physical_pitch_error.abs().max()) > 0.00005:
        raise AssertionError("Cap link no longer follows bottle-local screw axis")
    if args.num_envs > 1 and float(positions[1:].abs().max()) > 1e-5:
        raise AssertionError("Load on env 0 leaked into other environments")
    max_unexpected_hand_force = 0.0
    if contacts is not None:
        for obj in ("Bottle", "Cap"):
            for side in ("left", "right"):
                measured_force = contacts.force_magnitudes(obj, side)
                if not torch.isfinite(measured_force).all():
                    raise AssertionError(f"Nonfinite hand contact force on {obj}/{side}")
                max_unexpected_hand_force = max(max_unexpected_hand_force,
                                                float(measured_force.max()))
        if pregrasp is None and max_unexpected_hand_force > 0.01:
            raise AssertionError("Distant hands reported unexpected object contact")
        if pregrasp is not None and peak_right_cap_force <= 0.01:
            bodies = right.data.body_link_pose_w.torch[0, :, :3]
            radial = torch.linalg.vector_norm(bodies[:, :2] - cp[0, None, :2], dim=-1)
            nearest = torch.topk(radial, k=8, largest=False)
            print("MIMIC_RIGHT_CONTACT_DEBUG", {
                "peak_right_cap_force_n": peak_right_cap_force,
                "cap_pos": cp[0, :3].detach().cpu().tolist(),
                "right_root_pos": right.data.root_pose_w.torch[0, :3].detach().cpu().tolist(),
                "right_joint_pos": right.data.joint_pos.torch[0].detach().cpu().tolist(),
                "nearest_body_origins": [(right.body_names[int(i)], float(radial[i]),
                                          bodies[int(i)].detach().cpu().tolist())
                                         for i in nearest.indices],
            }, flush=True)
            raise AssertionError("Scripted right pregrasp approach made no cap contact")
        if pregrasp is not None:
            if not first_contact_points:
                raise AssertionError("Contact force had no matching PhysX manifold point")
            if min(point["separation_m"] for point in first_contact_points) < -0.002:
                raise AssertionError("Right pregrasp contact penetrated the cap by >2 mm")
            if not any(
                0.014 <= math.hypot(*point["point_cap_local_m"][:2]) <= 0.020
                and abs(point["point_cap_local_m"][2]) <= cfg.cap_height / 2
                for point in first_contact_points
            ):
                raise AssertionError("Right pregrasp contact was not on the cap sidewall")
    result = {"num_envs": args.num_envs, "steps": args.steps,
              "implicit_running_drive_enabled": drive is not None,
              "drive_sliding_count": (int(drive.sliding.sum()) if drive is not None else None),
              "drive_breakaway_seen_count": (int(drive.breakaway_seen.sum()) if drive is not None else None),
              "dof_names": names, "link_names": links,
              "bottle_pose_w": bp.detach().cpu().tolist(),
              "cap_pose_w": cp.detach().cpu().tolist(),
              "spin_rad": positions[:, spin_id].detach().cpu().tolist(),
              "axial_m": positions[:, slide_id].detach().cpu().tolist(),
              "max_pitch_error_m": float(pitch_error.abs().max()),
              "max_physical_pitch_error_m": float(physical_pitch_error.abs().max()),
              "tilt_deg": args.tilt_deg, "initial_angle_rad": args.initial_angle_rad,
              "reset_repetitions": args.reset_repetitions,
              "reset_unique_envs": reset_unique_envs,
              "max_peer_reset_change": max_peer_reset_change,
              "mean_single_env_reset_ms": mean_reset_ms,
              "mean_batched_read_ms": mean_read_ms,
              "max_unexpected_hand_force_n": max_unexpected_hand_force,
              "peak_right_cap_force_n": peak_right_cap_force,
              "first_contact_points": first_contact_points,
              "approach_offset_m": approach_offset,
              "diagnostic_right_pregrasp_contact": pregrasp is not None}
    print("BIMANUAL_MIMIC_SCENE", json.dumps({
        "num_envs": result["num_envs"], "steps": result["steps"],
        "implicit_running_drive_enabled": result["implicit_running_drive_enabled"],
        "drive_sliding_count": result["drive_sliding_count"],
        "spin_rad_first": result["spin_rad"][0],
        "max_pitch_error_m": result["max_pitch_error_m"],
        "max_physical_pitch_error_m": result["max_physical_pitch_error_m"],
        "tilt_deg": result["tilt_deg"],
        "initial_angle_rad": result["initial_angle_rad"],
        "reset_repetitions": result["reset_repetitions"],
        "reset_unique_envs": result["reset_unique_envs"],
        "max_peer_reset_change": result["max_peer_reset_change"],
        "mean_single_env_reset_ms": result["mean_single_env_reset_ms"],
        "mean_batched_read_ms": result["mean_batched_read_ms"],
        "max_unexpected_hand_force_n": result["max_unexpected_hand_force_n"],
        "peak_right_cap_force_n": result["peak_right_cap_force_n"],
        "first_contact_min_separation_m": min(
            (point["separation_m"] for point in first_contact_points), default=None),
        "approach_offset_m": result["approach_offset_m"],
        "diagnostic_right_pregrasp_contact": result[
            "diagnostic_right_pregrasp_contact"]}),
          flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    app.close()


if __name__ == "__main__":
    main()
