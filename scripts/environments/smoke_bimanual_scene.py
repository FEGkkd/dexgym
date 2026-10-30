"""Standalone Isaac Lab bimanual bottle fixture; no RL environment or policy.

Example: scripts/run_isaaclab.sh scripts/environments/smoke_bimanual_scene.py
    --headless --num_envs 2 --steps 300 --test_torque 0.04
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--variant", choices=("beta1", "beta2"), default="beta1")
parser.add_argument("--bottle_profile", choices=("engineering_example", "pco1881_reference"),
                    default="engineering_example")
parser.add_argument("--left_xyz", type=float, nargs=3, default=(0.0, 0.0, 0.0))
parser.add_argument("--right_xyz", type=float, nargs=3, default=(0.0, 0.0, 0.0))
parser.add_argument("--left_rot", type=float, nargs=3, default=(0.0, 0.0, 0.0))
parser.add_argument("--right_rot", type=float, nargs=3, default=(0.0, 0.0, 0.0))
parser.add_argument("--left_grip", type=float, default=0.0)
parser.add_argument("--right_grip", type=float, default=0.0)
parser.add_argument("--left_fingers", type=float, nargs=20, default=None,
                    help="20 canonical left finger target positions in radians")
parser.add_argument("--right_fingers", type=float, nargs=20, default=None,
                    help="20 canonical right finger target positions in radians")
parser.add_argument("--test_torque", type=float, default=0.0, help="N m about bottle-local +Z, env 0 only")
parser.add_argument("--torque_cycle_steps", type=int, default=0,
                    help="Reverse scripted test torque every N physics steps; 0 keeps the torque constant")
parser.add_argument("--allow_detachment", action="store_true",
                    help="Release the D6 joint after the configured thread travel is complete")
parser.add_argument("--detach_probe_force", type=float, default=0.0,
                    help="Optional world-X force in N on the released cap for a separation fixture")
parser.add_argument("--fixture_hold_bottle", action="store_true",
                    help="Hold bottle with a bounded PD wrench for isolated opening-torque tests")
parser.add_argument("--bottle_tilt_deg", type=float, default=0.0)
parser.add_argument("--bottle_center_z", type=float, default=0.5,
                    help="Bottle center height in metres; 0.1 rests the sample bottle on ground")
parser.add_argument("--contact_debug", action="store_true")
parser.add_argument("--contact_probe_side", choices=("left", "right"),
                    help="Place bottle beside one fingertip once at fixture initialization")
parser.add_argument("--trace", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
kit_markers_available = importlib.util.find_spec("isaaclab_visualizers") is not None
if args.contact_debug and not kit_markers_available:
    print("BIMANUAL_GUI_NOTE isaaclab_visualizers is unavailable; contact debug markers disabled", flush=True)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.markers import VisualizationMarkers  # noqa: E402
from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

from dexteroushands_lab.assets.bottle import BottleConfig, ThreadCoupler  # noqa: E402
from dexteroushands_lab.assets.bottle.generate_usd import build_bottle_usd, author_thread_joint  # noqa: E402
from dexteroushands_lab.assets.wrist_6dof import Wrist6DoFController, WristLimits  # noqa: E402
from dexteroushands_lab.assets.wuji_hand2_beta import load_hand2_spec  # noqa: E402
from dexteroushands_lab.scenes.bimanual_bottle import make_bimanual_bottle_scene_cfg  # noqa: E402


def _rows(values):
    return values.detach().cpu().tolist()


def _grip_targets(robot, spec, grip: float, explicit=None):
    if not 0.0 <= grip <= 1.0:
        raise ValueError("grip must be in [0, 1]")
    targets = robot.data.default_joint_pos.torch.clone()
    for canonical_index, joint in enumerate(spec.joints):
        index = robot.joint_names.index(joint.name)
        if explicit is not None:
            value = explicit[canonical_index]
            if not joint.lower <= value <= joint.upper:
                raise ValueError(f"{joint.name} target outside official limits")
            targets[:, index] = value
        elif not joint.name.endswith("_abd"):
            targets[:, index] = min(joint.upper, max(joint.lower, joint.upper * grip))
    return targets


def _apply_wrench(asset, forces, torques, *, body_id=None):
    composer = asset.permanent_wrench_composer
    composer.reset()
    kwargs = {}
    if body_id is not None:
        kwargs["body_ids"] = torch.tensor([body_id], device=asset.device, dtype=torch.long)
    composer.add_forces_and_torques_index(
        forces=forces[:, None, :], torques=torques[:, None, :],
        is_global=True, **kwargs,
    )


def main():
    print("BIMANUAL_STAGE starting", flush=True)
    if (args.num_envs < 1 or args.steps < 1 or args.torque_cycle_steps < 0
            or args.detach_probe_force < 0):
        raise ValueError("environment count, steps and fixture force must be valid")
    base_config = (BottleConfig.pco1881_reference()
                   if args.bottle_profile == "pco1881_reference" else BottleConfig())
    config = replace(base_config, allow_detachment=args.allow_detachment)
    if args.bottle_center_z < config.bottle_height / 2:
        raise ValueError("bottle center must keep the sample body above the ground")
    asset_dir = REPO_ROOT / "source/dexteroushands_lab/assets/bottle/generated"
    if args.bottle_profile != "engineering_example":
        asset_dir /= args.bottle_profile
    build_bottle_usd(config, asset_dir)
    print("BIMANUAL_STAGE bottle_usd_generated", flush=True)
    left_spec = load_hand2_spec(variant=args.variant, side="left")
    right_spec = load_hand2_spec(variant=args.variant, side="right")
    print("BIMANUAL_STAGE official_assets_verified", flush=True)

    physics_dt = 0.0001 if config.source_profile == "pco1881_reference" else 0.002
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=physics_dt, device=args.device))
    print("BIMANUAL_STAGE simulation_context_created", flush=True)
    focus_z = min(0.55, max(0.22, args.bottle_center_z + 0.12))
    sim.set_camera_view((0.45, -0.70, focus_z + 0.30), (0.0, 0.0, focus_z))
    print("BIMANUAL_STAGE camera_configured", flush=True)
    scene = InteractiveScene(make_bimanual_bottle_scene_cfg(
        num_envs=args.num_envs, asset_dir=asset_dir, bottle=config,
        variant=args.variant, contact_debug=args.contact_debug and kit_markers_available,
        bottle_tilt_radians=math.radians(args.bottle_tilt_deg),
        bottle_center_z=args.bottle_center_z,
    ))
    print("BIMANUAL_STAGE interactive_scene_created", flush=True)
    thread_joints = [
        author_thread_joint(sim.stage, f"/World/envs/env_{env_index}", config)
        for env_index in range(args.num_envs)
    ]
    detachment_steps = [None] * args.num_envs
    # The official native USD is fixed-base. Verify the Isaac Lab override
    # disabled its world joint before PhysX creates the articulations.
    for name in ("LeftHand", "RightHand"):
        path = f"/World/envs/env_0/{name}/root_joint"
        joint = UsdPhysics.Joint(sim.stage.GetPrimAtPath(path))
        if not joint or joint.GetJointEnabledAttr().Get() is not False:
            raise RuntimeError(f"Floating-base override did not disable {path}")
    sim.reset()
    print("BIMANUAL_STAGE simulation_reset", flush=True)
    left = scene["left_hand"]
    right = scene["right_hand"]
    bottle = scene["bottle_body"]
    cap = scene["bottle_cap"]
    sensor = scene["cap_contact"]
    left_sensor = scene["left_hand_contact"]
    right_sensor = scene["right_hand_contact"]
    for robot, spec in ((left, left_spec), (right, right_spec)):
        if set(robot.joint_names) != set(spec.joint_names):
            raise RuntimeError(f"USD/Isaac joint mismatch: {spec.side}: {robot.joint_names}")
        if not set(spec.fingertip_bodies).issubset(robot.body_names):
            raise RuntimeError(f"USD/Isaac fingertip mismatch: {spec.side}: {robot.body_names}")

    if args.contact_probe_side:
        robot, spec = ((left, left_spec) if args.contact_probe_side == "left"
                       else (right, right_spec))
        tip_index = robot.body_names.index(spec.fingertip_bodies[1])
        tip_pos = robot.data.body_link_pose_w.torch[0, tip_index, :3].clone()
        shift = tip_pos - bottle.data.root_pos_w.torch[0]
        shift[0] += config.bottle_radius * (0.72 if args.contact_probe_side == "left" else -0.72)
        env_id = torch.tensor([0], device=bottle.device, dtype=torch.long)
        for asset in (bottle, cap):
            pose = asset.data.root_pose_w.torch[0:1].clone()
            pose[:, :3] += shift
            asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_id)
        print(f"BIMANUAL_CONTACT_PROBE side={args.contact_probe_side} tip={tip_pos.tolist()}",
              flush=True)

    left_pose = _rows(left.data.root_pose_w.torch)
    right_pose = _rows(right.data.root_pose_w.torch)
    left_ctrl = [Wrist6DoFController(tuple(p[:3]), tuple(p[3:7])) for p in left_pose]
    right_ctrl = [Wrist6DoFController(tuple(p[:3]), tuple(p[3:7])) for p in right_pose]
    bottle_hold_limits = WristLimits(
        max_force=100.0, max_torque=3.0,
        kp_position=300.0, kd_position=30.0,
        kp_rotation=10.0, kd_rotation=0.1,
    )
    bottle_ctrl = ([Wrist6DoFController(tuple(p[:3]), tuple(p[3:7]), bottle_hold_limits)
                    for p in _rows(bottle.data.root_pose_w.torch)]
                   if args.fixture_hold_bottle else None)
    for ctrl in left_ctrl:
        ctrl.command(tuple(args.left_xyz), tuple(args.left_rot))
    for ctrl in right_ctrl:
        ctrl.command(tuple(args.right_xyz), tuple(args.right_rot))
    couplers = [ThreadCoupler(config) for _ in range(args.num_envs)]
    for i, (bpose, cpose) in enumerate(zip(_rows(bottle.data.root_pose_w.torch),
                                            _rows(cap.data.root_pose_w.torch))):
        couplers[i].observe(bpose[:3], bpose[3:7], cpose[:3], cpose[3:7])
        if args.trace:
            print(f"BIMANUAL_INITIAL env={i} bottle={bpose} cap={cpose}", flush=True)
    left_target = _grip_targets(left, left_spec, args.left_grip, args.left_fingers)
    right_target = _grip_targets(right, right_spec, args.right_grip, args.right_fingers)
    left_id = left.body_names.index(left_spec.root_body)
    right_id = right.body_names.index(right_spec.root_body)

    markers = None
    thread_label = None
    if not args.headless:
        import omni.ui as ui

        if kit_markers_available:
            marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/WristAndThreadFrames")
            marker_cfg.markers["frame"].scale = (0.045, 0.045, 0.045)
            markers = VisualizationMarkers(marker_cfg)
        thread_window = ui.Window("DexGym bottle thread", width=450, height=125)
        thread_window.position_x = 850
        thread_window.position_y = 95
        with thread_window.frame:
            with ui.VStack(spacing=5):
                ui.Label("Thread state (env 0, bottle-local +Z)")
                thread_label = ui.Label("waiting for physics")

    device = left.device
    max_contact = 0.0
    max_left_contact = 0.0
    max_right_contact = 0.0
    max_speed = 0.0
    max_error = 0.0
    max_breakaway_resistance = 0.0
    max_running_resistance = 0.0
    max_end_stop_resistance = 0.0
    max_applied_thread_torque = 0.0
    max_wrist_speed = 0.0
    max_wrist_angular_speed = 0.0
    max_wrist_force = 0.0
    max_wrist_torque = 0.0
    final_states = []
    final_breakaway_limits = []
    final_running_limits = []
    for step in range(args.steps):
        left_pose = _rows(left.data.root_pose_w.torch)
        right_pose = _rows(right.data.root_pose_w.torch)
        bottle_pose = _rows(bottle.data.root_pose_w.torch)
        cap_pose = _rows(cap.data.root_pose_w.torch)
        left_vel = _rows(left.data.root_vel_w.torch)
        right_vel = _rows(right.data.root_vel_w.torch)
        bottle_vel = _rows(bottle.data.root_vel_w.torch)
        cap_vel = _rows(cap.data.root_vel_w.torch)
        lw, rw, bw, cw = [], [], [], []
        final_states = []
        final_breakaway_limits = []
        final_running_limits = []
        for i in range(args.num_envs):
            lw.append(left_ctrl[i].wrench(left_pose[i][:3], left_pose[i][3:7],
                                          left_vel[i][:3], left_vel[i][3:6]))
            rw.append(right_ctrl[i].wrench(right_pose[i][:3], right_pose[i][3:7],
                                           right_vel[i][:3], right_vel[i][3:6]))
            for velocity, wrench in ((left_vel[i], lw[-1]), (right_vel[i], rw[-1])):
                max_wrist_speed = max(max_wrist_speed, math.dist(velocity[:3], (0, 0, 0)))
                max_wrist_angular_speed = max(max_wrist_angular_speed, math.dist(velocity[3:6], (0, 0, 0)))
                max_wrist_force = max(max_wrist_force, math.dist(wrench[0], (0, 0, 0)))
                max_wrist_torque = max(max_wrist_torque, math.dist(wrench[1], (0, 0, 0)))
            if args.trace and i == 0 and (step < 20 or step % 100 == 0):
                print(
                    f"WRIST_STEP {step} left_pos={left_pose[i][:3]} "
                    f"left_v={left_vel[i][:3]} left_force={lw[-1][0]} "
                    f"right_pos={right_pose[i][:3]} right_force={rw[-1][0]}",
                    flush=True,
                )
            result = couplers[i].wrenches(
                bottle_pose[i][:3], bottle_pose[i][3:7], bottle_vel[i][:3], bottle_vel[i][3:6],
                cap_pose[i][:3], cap_pose[i][3:7], cap_vel[i][:3], cap_vel[i][3:6],
            )
            cycle_sign = (-1 if args.torque_cycle_steps
                          and (step // args.torque_cycle_steps) % 2 else 1)
            if result.state.detached and detachment_steps[i] is None:
                thread_joints[i].GetJointEnabledAttr().Set(False)
                thread_joints[i].CreateCollisionEnabledAttr(True)
                detachment_steps[i] = step
                print(f"THREAD_DETACHED env={i} step={step}", flush=True)
            applied = args.test_torque * cycle_sign if i == 0 and not result.state.detached else 0.0
            cap_torque = tuple(result.cap_torque[j] + applied * result.state.axis_world[j]
                               for j in range(3))
            cap_force = result.cap_force
            if result.state.detached and i == 0 and args.detach_probe_force:
                cap_force = (cap_force[0] + args.detach_probe_force,
                             cap_force[1], cap_force[2])
            bottle_force = result.bottle_force
            bottle_torque = result.bottle_torque
            if bottle_ctrl is not None:
                hold_force, hold_torque = bottle_ctrl[i].wrench(
                    bottle_pose[i][:3], bottle_pose[i][3:7],
                    bottle_vel[i][:3], bottle_vel[i][3:6],
                )
                bottle_force = tuple(bottle_force[j] + hold_force[j] for j in range(3))
                bottle_torque = tuple(bottle_torque[j] + hold_torque[j] for j in range(3))
            bw.append((bottle_force, bottle_torque))
            cw.append((cap_force, cap_torque))
            final_states.append(result.state)
            final_breakaway_limits.append(result.breakaway_limit_nm)
            final_running_limits.append(result.running_limit_nm)
            max_breakaway_resistance = max(max_breakaway_resistance,
                                           abs(result.breakaway_resistance_torque))
            max_running_resistance = max(max_running_resistance,
                                         abs(result.resistance_torque))
            max_end_stop_resistance = max(max_end_stop_resistance,
                                          abs(result.end_stop_resistance_torque))
            max_applied_thread_torque = max(max_applied_thread_torque,
                                            result.applied_coupling_torque_nm)
            if args.trace and i == 0 and step < 20:
                print(
                    f"BIMANUAL_STEP {step} theta={result.state.theta_unwrapped:.6f} "
                    f"axial={result.state.axial:.6f} error={result.state.constraint_error:.6f} "
                    f"cap_v={cap_vel[i][:3]} force={result.cap_force} "
                    f"contact={max_contact:.3f}", flush=True,
                )
            max_error = max(max_error, abs(result.state.constraint_error))
            max_speed = max(max_speed, math.dist(cap_vel[i][:3], (0, 0, 0)))
        for asset, wrenches, body_id in (
            (left, lw, left_id), (right, rw, right_id),
            (bottle, bw, None), (cap, cw, None),
        ):
            forces = torch.tensor([pair[0] for pair in wrenches], device=device, dtype=torch.float32)
            torques = torch.tensor([pair[1] for pair in wrenches], device=device, dtype=torch.float32)
            _apply_wrench(asset, forces, torques, body_id=body_id)
        left.set_joint_position_target_index(target=left_target)
        right.set_joint_position_target_index(target=right_target)
        scene.write_data_to_sim()
        if args.trace and step in (0, 19, 100):
            print(
                f"WRIST_APPLIED {step} root_id={left_id} "
                f"body_names={left.body_names[:3]} "
                f"force_body={left.permanent_wrench_composer.out_force_b.torch[0, left_id].tolist()}",
                flush=True,
            )
        sim.step()
        scene.update(sim.get_physics_dt())
        for asset in (left, right, bottle, cap):
            if not torch.isfinite(asset.data.root_pose_w.torch).all().item():
                raise RuntimeError(f"Nonfinite root pose at step {step}")
            if not torch.isfinite(asset.data.root_vel_w.torch).all().item():
                raise RuntimeError(f"Nonfinite root velocity at step {step}")
        if sensor.data.net_forces_w is not None:
            max_contact = max(max_contact, torch.linalg.norm(
                sensor.data.net_forces_w.torch, dim=-1,
            ).amax().item())
        for hand_sensor, side in ((left_sensor, "left"), (right_sensor, "right")):
            if hand_sensor.data.net_forces_w is not None:
                measured = torch.linalg.norm(hand_sensor.data.net_forces_w.torch, dim=-1).amax().item()
                if side == "left":
                    max_left_contact = max(max_left_contact, measured)
                else:
                    max_right_contact = max(max_right_contact, measured)
        if markers is not None and step % 10 == 0:
            markers.visualize(
                torch.cat((left.data.root_pose_w.torch[:, :3],
                           right.data.root_pose_w.torch[:, :3],
                           bottle.data.root_pose_w.torch[:, :3]), dim=0),
                torch.cat((left.data.root_pose_w.torch[:, 3:7],
                           right.data.root_pose_w.torch[:, 3:7],
                           bottle.data.root_pose_w.torch[:, 3:7]), dim=0),
            )
        if thread_label is not None and step % 10 == 0:
            state = final_states[0]
            thread_label.text = (
                f"angle {state.theta_unwrapped:+.3f} rad    "
                f"axial {state.axial * 1000:.2f} mm    "
                f"error {state.constraint_error * 1000:+.3f} mm\n"
                f"thread overlap {state.engagement_fraction * 100:.0f}%    "
                f"breakaway limit {final_breakaway_limits[0]:.4f} N m"
                + ("    DETACHED" if state.detached else "")
            )
    print("BIMANUAL_SCENE_RESULT " + json.dumps({
        "status": "ran_without_nonfinite_state",
        "num_envs": args.num_envs,
        "steps": args.steps,
        "physics_dt_s": physics_dt,
        "simulated_time_s": args.steps * physics_dt,
        "asset_variant": args.variant,
        "bottle_profile": args.bottle_profile,
        "test_torque_nm": args.test_torque,
        "fixture_hold_bottle": args.fixture_hold_bottle,
        "torque_cycle_steps": args.torque_cycle_steps,
        "allow_detachment": args.allow_detachment,
        "detachment_steps": detachment_steps,
        "joint_enabled": [joint.GetJointEnabledAttr().Get() for joint in thread_joints],
        "thread": [
            {"theta_rad": s.theta_unwrapped, "axial_m": s.axial,
             "expected_axial_m": s.desired_axial, "constraint_error_m": s.constraint_error,
             "engagement_fraction": s.engagement_fraction,
             "detached": s.detached, "lateral_error_m": s.lateral_error,
             "breakaway_limit_nm": final_breakaway_limits[i],
             "running_limit_nm": final_running_limits[i]}
            for i, s in enumerate(final_states)
        ],
        "max_constraint_error_m": max_error,
        "max_breakaway_resistance_nm": max_breakaway_resistance,
        "max_running_resistance_nm": max_running_resistance,
        "max_end_stop_requested_nm": max_end_stop_resistance,
        "max_applied_thread_torque_nm": max_applied_thread_torque,
        "max_contact_force_n": max_contact,
        "max_left_hand_contact_force_n": max_left_contact,
        "max_right_hand_contact_force_n": max_right_contact,
        "max_cap_speed_m_s": max_speed,
        "max_wrist_speed_m_s": max_wrist_speed,
        "max_wrist_angular_speed_rad_s": max_wrist_angular_speed,
        "max_wrist_force_n": max_wrist_force,
        "max_wrist_torque_nm": max_wrist_torque,
        "left_joint_count": len(left.joint_names),
        "right_joint_count": len(right.joint_names),
        "left_joint_names": list(left.joint_names),
        "right_joint_names": list(right.joint_names),
        "left_final_joints": _rows(left.data.joint_pos.torch),
        "right_final_joints": _rows(right.data.joint_pos.torch),
        "left_initial_pose": [tuple(p) for p in _rows(left.data.default_root_pose.torch)],
        "right_initial_pose": [tuple(p) for p in _rows(right.data.default_root_pose.torch)],
        "left_final_pose": _rows(left.data.root_pose_w.torch),
        "right_final_pose": _rows(right.data.root_pose_w.torch),
        "bottle_final_pose": _rows(bottle.data.root_pose_w.torch),
        "bottle_final_velocity": _rows(bottle.data.root_vel_w.torch),
        "cap_final_pose": _rows(cap.data.root_pose_w.torch),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        import traceback
        print(f"BIMANUAL_ERROR {type(exc).__name__}: {exc!r}", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
