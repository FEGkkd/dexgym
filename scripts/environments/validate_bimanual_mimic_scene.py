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
import tempfile

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--tilt-deg", type=float, default=0.0)
parser.add_argument("--initial-angle-rad", type=float, default=0.0)
parser.add_argument("--contact-audit", action="store_true")
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
from dexteroushands_lab.assets.bottle.batched_thread import _rotate
from dexteroushands_lab.assets.wuji_hand2_beta import load_hand2_spec
from dexteroushands_lab.scenes.batched_contacts import BatchedContactMonitor
from dexteroushands_lab.scenes.bimanual_bottle import make_bimanual_bottle_scene_cfg


def main():
    cfg = BottleConfig()
    tilt = math.radians(args.tilt_deg)
    if not math.isfinite(tilt) or abs(tilt) > math.pi / 2:
        raise ValueError("tilt-deg must be finite and within +/-90 degrees")
    if not math.isfinite(args.initial_angle_rad) or abs(args.initial_angle_rad) > 2 * math.pi:
        raise ValueError("initial-angle-rad must be finite and within +/-one turn")
    orientation = (0.0, math.sin(tilt / 2), 0.0, math.cos(tilt / 2))
    asset_dir = Path(tempfile.mkdtemp(prefix="dexgym-bimanual-mimic-scene-"))
    build_bottle_usd(cfg, asset_dir)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.002, device=args.device))
    scene = InteractiveScene(make_bimanual_bottle_scene_cfg(
        num_envs=args.num_envs, asset_dir=asset_dir, bottle=cfg,
        bottle_center_z=0.5, bottle_tilt_radians=tilt,
        wrist_control_mode="pose_command_wrist"))
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
    if args.contact_audit:
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
    if len(left.joint_names) != 20 or len(right.joint_names) != 20:
        raise AssertionError("Official hand active finger joints changed")
    # Exercise the exact link-write API used by the current RL Reset path.
    # An articulation link may be readable through RigidObject while writes
    # are rejected or silently ignored; that difference determines whether
    # the hard-pitch assembly can be integrated without a Reset adapter.
    first_id = torch.tensor([0], dtype=torch.long, device=args.device)
    initial_bottle = bottle.data.root_pose_w.torch.clone()
    initial_cap = cap.data.root_pose_w.torch.clone()
    moved_bottle = initial_bottle[0:1].clone()
    moved_cap = initial_cap[0:1].clone()
    moved_bottle[:, 0] += 0.02
    moved_cap[:, 0] += 0.02
    bottle.write_root_pose_to_sim_index(root_pose=moved_bottle, env_ids=first_id)
    cap.write_root_pose_to_sim_index(root_pose=moved_cap, env_ids=first_id)
    physics_view.update_articulations_kinematic()
    bottle_after_write = bottle.data.root_pose_w.torch
    cap_after_write = cap.data.root_pose_w.torch
    if not torch.allclose(bottle_after_write[0, 0], moved_bottle[0, 0], atol=1e-4):
        raise AssertionError("Existing Bottle RigidObject write cannot move articulation root")
    if not torch.allclose(cap_after_write[0, 0], moved_cap[0, 0], atol=1e-4):
        raise AssertionError("Existing Cap RigidObject write cannot move articulation link")
    if args.num_envs > 1 and (not torch.allclose(bottle_after_write[1:], initial_bottle[1:], atol=1e-5)
                             or not torch.allclose(cap_after_write[1:], initial_cap[1:], atol=1e-5)):
        raise AssertionError("Selective link write altered a peer environment")
    if args.initial_angle_rad:
        q = wp.to_torch(art.get_dof_positions()).reshape(args.num_envs, -1).clone()
        q[0, spin_id] = args.initial_angle_rad
        q[0, slide_id] = cfg.thread_sign * cfg.thread_pitch * args.initial_angle_rad / (2 * math.pi)
        art.set_dof_positions(wp.from_torch(q), wp.from_torch(first_id.to(dtype=torch.int32)))
        target = wp.to_torch(art.get_dof_position_targets()).reshape(args.num_envs, -1).clone()
        target[0, spin_id] = args.initial_angle_rad
        art.set_dof_position_targets(wp.from_torch(target),
                                     wp.from_torch(first_id.to(dtype=torch.int32)))
        physics_view.update_articulations_kinematic()
        measured = wp.to_torch(art.get_dof_positions()).reshape(args.num_envs, -1)
        if not torch.allclose(measured[0], q[0], atol=1e-5):
            raise AssertionError("Selective screw DOF Reset did not restore angle and travel")
        if args.num_envs > 1 and float(measured[1:].abs().max()) > 1e-5:
            raise AssertionError("Selective screw DOF Reset altered peer environments")
    force = torch.zeros((args.num_envs, 3, 3), device=args.device)
    force[0, cap_id, 0] = 0.5 * math.sin(tilt)
    force[0, cap_id, 2] = 0.5 * math.cos(tilt)
    ids = wp.from_torch(torch.arange(args.num_envs, dtype=torch.int32, device=args.device))
    for _ in range(args.steps):
        art.apply_forces_and_torques_at_position(wp.from_torch(force), None, None, ids, True)
        sim.step()
        scene.update(0.002)
        if contacts is not None:
            contacts.update(0.002)
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
        if max_unexpected_hand_force > 0.01:
            raise AssertionError("Distant hands reported unexpected object contact")
    result = {"num_envs": args.num_envs, "steps": args.steps,
              "dof_names": names, "link_names": links,
              "bottle_pose_w": bp.detach().cpu().tolist(),
              "cap_pose_w": cp.detach().cpu().tolist(),
              "spin_rad": positions[:, spin_id].detach().cpu().tolist(),
              "axial_m": positions[:, slide_id].detach().cpu().tolist(),
              "max_pitch_error_m": float(pitch_error.abs().max()),
              "max_physical_pitch_error_m": float(physical_pitch_error.abs().max()),
              "tilt_deg": args.tilt_deg, "initial_angle_rad": args.initial_angle_rad,
              "max_unexpected_hand_force_n": max_unexpected_hand_force}
    print("BIMANUAL_MIMIC_SCENE", json.dumps({
        "num_envs": result["num_envs"], "steps": result["steps"],
        "spin_rad_first": result["spin_rad"][0],
        "max_pitch_error_m": result["max_pitch_error_m"],
        "max_physical_pitch_error_m": result["max_physical_pitch_error_m"],
        "tilt_deg": result["tilt_deg"],
        "initial_angle_rad": result["initial_angle_rad"],
        "max_unexpected_hand_force_n": result["max_unexpected_hand_force_n"]}),
          flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    app.close()


if __name__ == "__main__":
    main()
