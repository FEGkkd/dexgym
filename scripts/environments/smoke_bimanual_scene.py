"""Standalone Isaac Lab bimanual bottle fixture; no RL environment or policy.

Example: scripts/run_isaaclab.sh scripts/environments/smoke_bimanual_scene.py
    --headless --num_envs 2 --steps 300 --test_torque 0.04
"""

from __future__ import annotations

import argparse
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
parser.add_argument("--bottle_tilt_deg", type=float, default=0.0)
parser.add_argument("--bottle_center_z", type=float, default=0.5,
                    help="Bottle center height in metres; 0.1 rests the sample bottle on ground")
parser.add_argument("--contact_debug", action="store_true")
parser.add_argument("--contact_probe_side", choices=("left", "right"),
                    help="Place bottle beside one fingertip once at fixture initialization")
parser.add_argument("--trace", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

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
from dexteroushands_lab.assets.wrist_6dof import Wrist6DoFController  # noqa: E402
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
    if args.num_envs < 1 or args.steps < 1:
        raise ValueError("num_envs and steps must be positive")
    config = BottleConfig()
    if args.bottle_center_z < config.bottle_height / 2:
        raise ValueError("bottle center must keep the sample body above the ground")
    asset_dir = REPO_ROOT / "source/dexteroushands_lab/assets/bottle/generated"
    build_bottle_usd(config, asset_dir)
    print("BIMANUAL_STAGE bottle_usd_generated", flush=True)
    left_spec = load_hand2_spec(variant=args.variant, side="left")
    right_spec = load_hand2_spec(variant=args.variant, side="right")
    print("BIMANUAL_STAGE official_assets_verified", flush=True)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.002, device=args.device))
    print("BIMANUAL_STAGE simulation_context_created", flush=True)
    sim.set_camera_view((0.7, -0.9, 0.95), (0.0, 0.0, 0.55))
    print("BIMANUAL_STAGE camera_configured", flush=True)
    scene = InteractiveScene(make_bimanual_bottle_scene_cfg(
        num_envs=args.num_envs, asset_dir=asset_dir, bottle=config,
        variant=args.variant, contact_debug=args.contact_debug,
        bottle_tilt_radians=math.radians(args.bottle_tilt_deg),
        bottle_center_z=args.bottle_center_z,
    ))
    print("BIMANUAL_STAGE interactive_scene_created", flush=True)
    for env_index in range(args.num_envs):
        author_thread_joint(sim.stage, f"/World/envs/env_{env_index}", config)
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

        marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/WristAndThreadFrames")
        marker_cfg.markers["frame"].scale = (0.045, 0.045, 0.045)
        markers = VisualizationMarkers(marker_cfg)
        thread_window = ui.Window("DexGym bottle thread", width=370, height=95)
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
            applied = args.test_torque if i == 0 else 0.0
            cap_torque = tuple(result.cap_torque[j] + applied * result.state.axis_world[j]
                               for j in range(3))
            bw.append((result.bottle_force, result.bottle_torque))
            cw.append((result.cap_force, cap_torque))
            final_states.append(result.state)
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
            if thread_label is not None:
                state = final_states[0]
                thread_label.text = (
                    f"angle {state.theta_unwrapped:+.3f} rad    "
                    f"axial {state.axial * 1000:.2f} mm    "
                    f"error {state.constraint_error * 1000:+.3f} mm"
                )
    print("BIMANUAL_SCENE_RESULT " + json.dumps({
        "status": "ran_without_nonfinite_state",
        "num_envs": args.num_envs,
        "steps": args.steps,
        "asset_variant": args.variant,
        "test_torque_nm": args.test_torque,
        "thread": [
            {"theta_rad": s.theta_unwrapped, "axial_m": s.axial,
             "expected_axial_m": s.desired_axial, "constraint_error_m": s.constraint_error}
            for s in final_states
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
