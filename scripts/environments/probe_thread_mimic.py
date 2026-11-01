"""Isolated PhysX GPU mimic-joint screw feasibility probe.

The fixture has a free bottle root, a collisionless prismatic carriage, and
a dynamic cap on a revolute joint. It deliberately has no hands, no RL task,
and no scripted object poses. Test forces act on cap 0 only. This is a
research probe; passing it does not qualify the production bottle assembly.

Sources:
https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html#articulation-mimic-joints
https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.2/dev_guide/schemas/physxschema.html
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
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--torque", type=float, default=1e-5, help="Cap torque in Nm, env 0 only")
parser.add_argument("--axial-force", type=float, default=0.5, help="Cap local-axis force in N, env 0 only")
parser.add_argument("--tilt-deg", type=float, default=0.0)
parser.add_argument("--slide-anchor-sign", type=float, choices=(-1.0, 1.0), default=1.0,
                    help="Diagnostic sign of bottle-local joint anchor")
parser.add_argument("--slide-anchor-body", choices=("body0", "body1"), default="body0",
                    help="Diagnostic link that owns the nonzero prismatic joint anchor")
parser.add_argument("--no-mimic", action="store_true")
parser.add_argument("--real-geometry", action="store_true",
                    help="Use generated bottle/cap native USD instead of plain diagnostic cylinders")
parser.add_argument("--no-self-collision", action="store_true",
                    help="Disable bottle-cap collision within the mimic articulation")
parser.add_argument("--contact-proxy", action="store_true",
                    help="Push a separate dynamic sphere into cap 0 to verify external contact")
parser.add_argument("--selective-reset", action="store_true",
                    help="Reset env 0 at half time without advancing or changing peer environments")
parser.add_argument("--assert-physics", action="store_true",
                    help="Exit nonzero unless local-axis pitch, collision and isolation checks pass")
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.steps < 1:
    parser.error("num-envs and steps must be positive")
if args.contact_proxy and not args.real_geometry:
    parser.error("contact-proxy requires real-geometry")
app = AppLauncher(args).app

import torch
import warp as wp
import isaaclab.sim as sim_utils
from isaaclab_physx.physics import PhysxManager
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics
from dexteroushands_lab.assets.bottle import BottleConfig
from dexteroushands_lab.assets.bottle.generate_usd import build_bottle_usd


BOTTLE_CONFIG = BottleConfig()
PITCH_M = BOTTLE_CONFIG.thread_pitch
CAP_OFFSET_M = BOTTLE_CONFIG.cap_initial_z
DT_S = 0.002


def _body(stage, path: str, world_pos: Gf.Vec3f, q: Gf.Quatf,
          mass: float, radius: float | None, height: float | None):
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(world_pos)
    xform.AddOrientOp().Set(q)
    prim = xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)
    if radius is None:
        # The carriage is a real articulation link without a collider.
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-7, 1e-7, 1e-7))
    else:
        collider = UsdGeom.Cylinder.Define(stage, path + "/Collision")
        collider.CreateRadiusAttr(radius)
        collider.CreateHeightAttr(height)
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    return prim


def _assembly(stage, env: int, geometry_paths=None):
    root = f"/World/envs/env_{env}"
    UsdGeom.Xform.Define(stage, root)
    tilt = Gf.Rotation(Gf.Vec3d(0, 1, 0), args.tilt_deg)
    q = Gf.Quatf(tilt.GetQuat())
    if env == 0:
        print("MIMIC_AUTHORED_QUAT_WXYZ", q.GetReal(), tuple(q.GetImaginary()), flush=True)
    b = Gf.Vec3f(env * 1.0, 0.0, 0.5)
    offset = tilt.TransformDir(Gf.Vec3d(0, 0, CAP_OFFSET_M))
    c = Gf.Vec3f(b[0] + offset[0], b[1] + offset[1], b[2] + offset[2])
    if geometry_paths is None:
        bottle = _body(stage, root + "/Bottle", b, q, 0.03, 0.035, 0.2)
    else:
        # Isaac Lab 3.0 spawn_from_usd takes xyzw; USD Gf.Quatf stores wxyz.
        orientation_xyzw = (*q.GetImaginary(), q.GetReal())
        bottle = sim_utils.spawn_from_usd(
            root + "/Bottle", sim_utils.UsdFileCfg(usd_path=str(geometry_paths[0])),
            translation=tuple(b), orientation=orientation_xyzw)
    _body(stage, root + "/Carriage", c, q, 0.0001, None, None)
    if geometry_paths is None:
        cap = _body(stage, root + "/Cap", c, q, 0.006, 0.017, 0.025)
    else:
        cap = sim_utils.spawn_from_usd(
            root + "/Cap", sim_utils.UsdFileCfg(usd_path=str(geometry_paths[1])),
            translation=tuple(c), orientation=orientation_xyzw)
    if args.real_geometry:
        PhysxSchema.PhysxContactReportAPI.Apply(cap)
    if args.contact_proxy and env == 0:
        # Separate dynamic body, initially 5 mm outside the cap sidewall.
        radial = tilt.TransformDir(Gf.Vec3d(1, 0, 0))
        proxy_pos = Gf.Vec3f(*(c[i] + 0.027 * radial[i] for i in range(3)))
        proxy = _body(stage, root + "/Puck", proxy_pos, q, 0.02, None, None)
        sphere = UsdGeom.Sphere.Define(stage, root + "/Puck/Collision")
        sphere.CreateRadiusAttr(0.005)
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(bottle)
    PhysxSchema.PhysxArticulationAPI.Apply(bottle).CreateEnabledSelfCollisionsAttr(
        not args.no_self_collision)

    slide = UsdPhysics.PrismaticJoint.Define(stage, root + "/SlideJoint")
    slide.CreateBody0Rel().SetTargets([Sdf.Path(root + "/Bottle")])
    slide.CreateBody1Rel().SetTargets([Sdf.Path(root + "/Carriage")])
    slide.CreateAxisAttr(UsdPhysics.Tokens.z)
    slide.CreateLocalPos0Attr(Gf.Vec3f(
        0, 0, args.slide_anchor_sign * CAP_OFFSET_M if args.slide_anchor_body == "body0" else 0))
    slide.CreateLocalPos1Attr(Gf.Vec3f(
        0, 0, args.slide_anchor_sign * CAP_OFFSET_M if args.slide_anchor_body == "body1" else 0))
    slide.CreateLowerLimitAttr(-0.009)
    slide.CreateUpperLimitAttr(0.009)

    spin = UsdPhysics.RevoluteJoint.Define(stage, root + "/SpinJoint")
    spin.CreateBody0Rel().SetTargets([Sdf.Path(root + "/Carriage")])
    spin.CreateBody1Rel().SetTargets([Sdf.Path(root + "/Cap")])
    spin.CreateAxisAttr(UsdPhysics.Tokens.z)
    spin.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    spin.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    spin.CreateLowerLimitAttr(-1080.0)
    spin.CreateUpperLimitAttr(1080.0)
    if not args.no_mimic:
        # USD rotational positions are degrees; one turn advances one pitch.
        # The supported instance token is rotX even on a prismatic follower.
        mimic = PhysxSchema.PhysxMimicJointAPI.Apply(
            slide.GetPrim(), UsdPhysics.Tokens.rotX)
        mimic.GetReferenceJointRel().SetTargets([spin.GetPath()])
        mimic.GetReferenceJointAxisAttr().Set(UsdPhysics.Tokens.rotZ)
        mimic.GetGearingAttr().Set(-PITCH_M / 360.0)
        mimic.GetOffsetAttr().Set(0.0)


def main():
    geometry_paths = None
    if args.real_geometry:
        asset_dir = Path(tempfile.mkdtemp(prefix="dexgym-mimic-geometry-"))
        geometry_paths = build_bottle_usd(BOTTLE_CONFIG, asset_dir)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT_S, device=args.device))
    stage = sim.stage
    UsdGeom.Xform.Define(stage, "/World/envs")
    for env in range(args.num_envs):
        _assembly(stage, env, geometry_paths)
    print("MIMIC_STAGE authored", flush=True)
    sim.reset()
    physics_view = PhysxManager.get_physics_sim_view()
    view = physics_view.create_articulation_view("/World/envs/env_*/Bottle")
    if view is None or view.count != args.num_envs:
        raise RuntimeError(f"Expected {args.num_envs} floating articulations, got {view}")
    names = list(view.shared_metatype.dof_names)
    links = list(view.shared_metatype.link_names)
    if set(names) != {"SlideJoint", "SpinJoint"} or set(links) != {"Bottle", "Carriage", "Cap"}:
        raise RuntimeError(f"Unexpected articulation topology: DOF {names}, links {links}")
    if view.shared_metatype.fixed_base:
        raise RuntimeError("Bottle is unexpectedly world-fixed")
    print("MIMIC_STAGE topology", names, links, flush=True)
    initial_link_positions = (wp.to_torch(view.get_link_transforms())
                              .reshape(args.num_envs, len(links), 7)[:, :, :3]
                              .detach().cpu().tolist())
    initial_link_transforms = (wp.to_torch(view.get_link_transforms())
                               .reshape(args.num_envs, len(links), 7)
                               .detach().cpu().tolist())
    contact_views = []
    if args.real_geometry:
        sim_view = PhysxManager.get_physics_sim_view()
        for env in range(args.num_envs):
            root = f"/World/envs/env_{env}"
            contact_views.append(sim_view.create_rigid_contact_view(
                root + "/Cap", filter_patterns=[root + "/Bottle"],
                max_contact_data_count=128))
    proxy_contact_view = None
    proxy_body_view = None
    proxy_force = None
    if args.contact_proxy:
        sim_view = PhysxManager.get_physics_sim_view()
        proxy_contact_view = sim_view.create_rigid_contact_view(
            "/World/envs/env_0/Cap", filter_patterns=["/World/envs/env_0/Puck"],
            max_contact_data_count=128)
        proxy_body_view = sim_view.create_rigid_body_view("/World/envs/env_0/Puck")
        proxy_force = torch.zeros((1, 3), device=args.device)
        proxy_force[0] = torch.tensor((-math.cos(math.radians(args.tilt_deg)),
                                       0.0, math.sin(math.radians(args.tilt_deg))),
                                      device=args.device) * 0.2
    slide_id, spin_id, cap_id = names.index("SlideJoint"), names.index("SpinJoint"), links.index("Cap")
    forces = torch.zeros((args.num_envs, len(links), 3), device=args.device)
    torques = torch.zeros_like(forces)
    tilt = Gf.Rotation(Gf.Vec3d(0, 1, 0), args.tilt_deg)
    axis = tuple(float(x) for x in tilt.TransformDir(Gf.Vec3d(0, 0, 1)))
    radial = tuple(float(x) for x in tilt.TransformDir(Gf.Vec3d(1, 0, 0)))
    if proxy_force is not None:
        proxy_force[0] = torch.tensor([-x for x in radial], device=args.device) * 0.2
    forces[0, cap_id] = torch.tensor(axis, device=args.device) * args.axial_force
    torques[0, cap_id] = torch.tensor(axis, device=args.device) * args.torque
    ids = wp.from_torch(torch.arange(args.num_envs, dtype=torch.int32, device=args.device))
    max_error = 0.0
    max_peer_motion = 0.0
    max_physical_pitch_error = 0.0
    max_lateral_offset = 0.0
    max_internal_contact_force = 0.0
    min_internal_separation = None
    first_contact = None
    max_proxy_contact_force = 0.0
    first_proxy_contact = None
    selective_reset_result = None
    traces = []
    for step in range(args.steps):
        view.apply_forces_and_torques_at_position(
            wp.from_torch(forces), wp.from_torch(torques), None, ids, True)
        if proxy_body_view is not None:
            proxy_body_view.apply_forces_and_torques_at_position(
                wp.from_torch(proxy_force), None, None,
                wp.from_torch(torch.tensor([0], dtype=torch.int32, device=args.device)), True)
        sim.step()
        if args.selective_reset and step + 1 == args.steps // 2:
            old_root = wp.to_torch(view.get_root_transforms()).reshape(args.num_envs, 7).clone()
            old_root_vel = wp.to_torch(view.get_root_velocities()).reshape(args.num_envs, 6).clone()
            old_q = wp.to_torch(view.get_dof_positions()).reshape(args.num_envs, -1).clone()
            old_qd = wp.to_torch(view.get_dof_velocities()).reshape(args.num_envs, -1).clone()
            reset_id = wp.from_torch(torch.tensor([0], dtype=torch.int32, device=args.device))
            initial_root = torch.tensor(initial_link_transforms[0][0], device=args.device)
            # The installed tensor backend requires full-size input even when
            # indices selects only one articulation.
            root_data, root_vel_data = old_root.clone(), old_root_vel.clone()
            q_data, qd_data = old_q.clone(), old_qd.clone()
            root_data[0], root_vel_data[0] = initial_root, 0
            q_data[0], qd_data[0] = 0, 0
            view.set_root_transforms(wp.from_torch(root_data), reset_id)
            view.set_dof_positions(wp.from_torch(q_data), reset_id)
            view.set_root_velocities(wp.from_torch(root_vel_data), reset_id)
            view.set_dof_velocities(wp.from_torch(qd_data), reset_id)
            physics_view.update_articulations_kinematic()
            new_root = wp.to_torch(view.get_root_transforms()).reshape(args.num_envs, 7)
            new_root_vel = wp.to_torch(view.get_root_velocities()).reshape(args.num_envs, 6)
            new_q = wp.to_torch(view.get_dof_positions()).reshape(args.num_envs, -1)
            new_qd = wp.to_torch(view.get_dof_velocities()).reshape(args.num_envs, -1)
            peer_error = max(float((new_root[1:] - old_root[1:]).abs().max()) if args.num_envs > 1 else 0.0,
                             float((new_root_vel[1:] - old_root_vel[1:]).abs().max()) if args.num_envs > 1 else 0.0,
                             float((new_q[1:] - old_q[1:]).abs().max()) if args.num_envs > 1 else 0.0,
                             float((new_qd[1:] - old_qd[1:]).abs().max()) if args.num_envs > 1 else 0.0)
            selective_reset_result = {
                "step": step + 1,
                "max_immediate_peer_state_change": peer_error,
                "env0_dof_error": float(new_q[0].abs().max()),
                "env0_root_pose_error": float((new_root[0] - initial_root).abs().max()),
                "env0_root_velocity_error": float(new_root_vel[0].abs().max()),
                "env0_dof_velocity_error": float(new_qd[0].abs().max()),
            }
        if proxy_contact_view is not None:
            normal_forces, _, _, distances, counts, _ = proxy_contact_view.get_contact_data(dt=DT_S)
            count = int(wp.to_torch(counts).reshape(-1)[0].item())
            if count:
                force = float(wp.to_torch(normal_forces)[:count].sum().item())
                max_proxy_contact_force = max(max_proxy_contact_force, force)
                if first_proxy_contact is None:
                    first_proxy_contact = {"step": step + 1, "count": count,
                                           "force_n": force,
                                           "min_separation_m": float(
                                               wp.to_torch(distances)[:count].min().item())}
        if contact_views and (step < 10 or step % 10 == 9 or step == args.steps - 1):
            for env, contact_view in enumerate(contact_views):
                normal_forces, _, _, distances, counts, _ = contact_view.get_contact_data(dt=DT_S)
                count = int(wp.to_torch(counts).reshape(-1)[0].item())
                if count:
                    force = float(wp.to_torch(normal_forces)[:count].sum().item())
                    separation = float(wp.to_torch(distances)[:count].min().item())
                    max_internal_contact_force = max(max_internal_contact_force, force)
                    min_internal_separation = (separation if min_internal_separation is None
                                               else min(min_internal_separation, separation))
                    if first_contact is None:
                        first_contact = {"step": step + 1, "env": env,
                                         "count": count, "force_n": force,
                                         "min_separation_m": separation}
        if step == 0 or step % 10 == 9 or step == args.steps - 1:
            positions = wp.to_torch(view.get_dof_positions()).reshape(args.num_envs, -1)
            root_vel = wp.to_torch(view.get_root_velocities()).reshape(args.num_envs, 6)
            link_transforms = wp.to_torch(view.get_link_transforms()).reshape(
                args.num_envs, len(links), 7)
            axial = positions[:, slide_id]
            theta = positions[:, spin_id]
            error = axial - PITCH_M * theta / (2 * math.pi)
            rel = link_transforms[:, cap_id, :3] - link_transforms[:, links.index("Bottle"), :3]
            root_q = link_transforms[:, links.index("Bottle"), 3:7]
            qx, qy, qz, qw = root_q.unbind(dim=-1)
            current_axis = torch.stack((2 * (qx * qz + qw * qy),
                                        2 * (qy * qz - qw * qx),
                                        1 - 2 * (qx.square() + qy.square())), dim=-1)
            physical_axial = (rel * current_axis).sum(dim=-1) - CAP_OFFSET_M
            lateral = (rel - (physical_axial + CAP_OFFSET_M).unsqueeze(-1)
                       * current_axis).norm(dim=-1)
            physical_pitch_error = physical_axial - PITCH_M * theta / (2 * math.pi)
            max_error = max(max_error, float(error.abs().max()))
            max_physical_pitch_error = max(max_physical_pitch_error,
                                           float(physical_pitch_error.abs().max()))
            max_lateral_offset = max(max_lateral_offset, float(lateral.max()))
            if args.num_envs > 1:
                max_peer_motion = max(max_peer_motion,
                                      float(positions[1:].abs().max()))
            traces.append({"step": step + 1, "theta_rad": theta.detach().cpu().tolist(),
                           "axial_m": axial.detach().cpu().tolist(),
                           "pitch_error_m": error.detach().cpu().tolist(),
                           "physical_axial_m": physical_axial.detach().cpu().tolist(),
                           "physical_pitch_error_m": physical_pitch_error.detach().cpu().tolist(),
                           "lateral_offset_m": lateral.detach().cpu().tolist(),
                           "root_angular_velocity_rad_s": root_vel[:, 3:].detach().cpu().tolist(),
                           "cap_position_w_m": (wp.to_torch(view.get_link_transforms())
                                                  .reshape(args.num_envs, len(links), 7)[:, cap_id, :3]
                                                  .detach().cpu().tolist()
                                                  if args.contact_proxy else None),
                           "all_link_positions_w_m": (wp.to_torch(view.get_link_transforms())
                                                        .reshape(args.num_envs, len(links), 7)[:, :, :3]
                                                        .detach().cpu().tolist()
                                                        if args.contact_proxy else None),
                           "proxy_position_w_m": (wp.to_torch(proxy_body_view.get_transforms())
                                                    [:, :3].detach().cpu().tolist()
                                                    if args.contact_proxy else None)})
    result = {"backend": "Isaac Sim 6.0.1 PhysX GPU floating articulation mimic probe",
              "mimic_enabled": not args.no_mimic,
              "real_geometry": args.real_geometry,
              "self_collision_enabled": not args.no_self_collision,
              "contact_proxy_enabled": args.contact_proxy,
              "num_envs": args.num_envs, "steps": args.steps, "physics_dt_s": DT_S,
              "pitch_m": PITCH_M, "tilt_deg": args.tilt_deg,
              "bottle_local_axis_w": axis,
              "slide_anchor_sign": args.slide_anchor_sign,
              "slide_anchor_body": args.slide_anchor_body,
              "test_torque_nm": args.torque, "test_axial_force_n": args.axial_force,
              "dof_names": names, "link_names": links,
              "initial_link_positions_w_m": initial_link_positions,
              "initial_link_transforms_xyzw": initial_link_transforms,
              "max_abs_pitch_error_m": max_error,
              "max_abs_physical_pitch_error_m": max_physical_pitch_error,
              "max_lateral_offset_m": max_lateral_offset,
              "max_peer_joint_motion": max_peer_motion,
              "max_internal_contact_force_n": max_internal_contact_force,
              "min_internal_separation_m": min_internal_separation,
              "first_internal_contact": first_contact,
              "max_proxy_contact_force_n": max_proxy_contact_force,
              "first_proxy_contact": first_proxy_contact,
              "selective_reset": selective_reset_result,
              "last": traces[-1], "trace": traces}
    if args.assert_physics:
        failures = []
        measured = (max_error, max_physical_pitch_error, max_lateral_offset,
                    max_peer_motion, max_internal_contact_force, max_proxy_contact_force)
        if not all(math.isfinite(value) for value in measured):
            failures.append("non-finite physics measurement")
        if max_physical_pitch_error > 5e-5:
            failures.append(f"physical pitch error {max_physical_pitch_error:.6g} m > 50 um")
        if max_lateral_offset > 5e-5:
            failures.append(f"lateral offset {max_lateral_offset:.6g} m > 50 um")
        if max_internal_contact_force > 1.0:
            failures.append(f"initial bottle-cap collision force {max_internal_contact_force:.6g} N > 1 N")
        if max_peer_motion > 1e-5:
            failures.append(f"unforced peer joint motion {max_peer_motion:.6g} > 1e-5")
        if args.contact_proxy and max_proxy_contact_force < 0.1:
            failures.append("external cap proxy contact was not observed")
        if args.selective_reset and (selective_reset_result is None or
                                     not all(math.isfinite(value) for value in selective_reset_result.values()) or
                                     selective_reset_result["max_immediate_peer_state_change"] > 1e-7 or
                                     selective_reset_result["env0_root_pose_error"] > 1e-6 or
                                     selective_reset_result["env0_dof_error"] > 1e-6):
            failures.append("selective reset state or peer isolation failed")
        result["physics_assertions"] = {"passed": not failures, "failures": failures}
    print("MIMIC_PROBE", json.dumps({key: result[key] for key in (
        "num_envs", "steps", "real_geometry", "mimic_enabled", "self_collision_enabled",
        "max_abs_pitch_error_m", "max_abs_physical_pitch_error_m",
        "max_lateral_offset_m", "max_peer_joint_motion",
        "max_internal_contact_force_n", "max_proxy_contact_force_n",
        "selective_reset")}), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    app.close()
    if args.assert_physics and failures:
        raise RuntimeError("Mimic physics assertions failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
