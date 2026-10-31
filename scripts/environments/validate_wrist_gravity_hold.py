"""Isolated gravity-on wrist pose hold with no bottle or hand contact."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--max_torque", type=float, default=2.0)
parser.add_argument("--kp_rotation", type=float, default=4.0)
parser.add_argument("--kd_rotation", type=float, default=0.05)
parser.add_argument("--strong_wrist", action="store_true")
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402
from dexteroushands_lab.assets.wrist_6dof import WristLimits, _orientation_error  # noqa: E402
import warp as wp  # noqa: E402


def main():
    limits = (WristLimits(max_force=80.0, kp_position=4000.0,
                          kd_position=120.0, max_torque=8.0,
                          kp_rotation=30.0, kd_rotation=6.0)
              if args.strong_wrist else WristLimits(max_torque=args.max_torque,
                                                      kp_rotation=args.kp_rotation,
                                                      kd_rotation=args.kd_rotation))
    f = BimanualPhase0Fixture(num_envs=1, device=args.device, include_hand_hand=False,
                              wrist_limits=limits)
    sample = f.initializer.generate(mode="near_contact", seed=0)
    left_pose = (-.45, -.05, .5, *sample.left.root_pose[3:7])
    right_pose = (.45, .05, .5, *sample.right.root_pose[3:7])
    sample = replace(sample,
                     left=replace(sample.left, root_pose=left_pose,
                                  wrist_target_pose=left_pose),
                     right=replace(sample.right, root_pose=right_pose,
                                   wrist_target_pose=right_pose))
    f.reset_manager.reset([0], [sample])
    start = f.left.data.root_pose_w.torch[0].detach().cpu().tolist()
    masses = wp.to_torch(f.left.root_view.get_masses()).detach().cpu().tolist()
    right_masses = wp.to_torch(f.right.root_view.get_masses()).detach().cpu().tolist()
    print("WRIST_HOLD_MASS " + json.dumps({"masses": masses,
          "total_mass_kg": sum(masses[0]),
          "right_total_mass_kg": sum(right_masses[0])}), flush=True)
    final_pose = None
    final_velocity = None
    previous_pose = None
    for i in range(args.steps):
        f.step()
        current_pose = f.left.data.root_pose_w.torch[0].detach().cpu().tolist()
        link_velocity = f.left.data.root_link_vel_w.torch[0].detach().cpu().tolist()
        if previous_pose is not None:
            qa, qb = previous_pose[3:7], current_pose[3:7]
            delta = min(1.0, abs(sum(a*b for a,b in zip(qa,qb))))
            finite_difference_angular_speed = 2.0*math.acos(delta)/f.dt
        else:
            finite_difference_angular_speed = None
        previous_pose = current_pose
        if i % 50 == 49 or i == args.steps - 1:
            pose = current_pose
            velocity = f.left.data.root_vel_w.torch[0].detach().cpu().tolist()
            final_pose, final_velocity = pose, velocity
            wrench = f.controllers[0][0].wrench(pose[:3],pose[3:7],velocity[:3],velocity[3:6])
            joints = f.left.data.joint_pos.torch[0].detach().cpu().tolist()
            print("WRIST_HOLD_STEP " + json.dumps({"t_s": (i+1)*f.dt,
                  "pose": pose, "velocity": velocity,
                  "root_link_velocity": link_velocity,
                  "pose_difference_angular_speed_rad_s": finite_difference_angular_speed,
                  "commanded_wrench": wrench,
                  "z_drift_m": pose[2]-start[2],
                  "max_joint_abs_rad": max(abs(x) for x in joints),
                  "joints_rad": joints}), flush=True)
    if final_pose is None:
        raise ValueError("At least one physics step is required")
    rotation = math.sqrt(sum(x*x for x in _orientation_error(start[3:7],
                                                              final_pose[3:7])))
    displacement_z = abs(final_pose[2] - start[2])
    result = {"duration_s": args.steps*f.dt, "total_hand_mass_kg": sum(masses[0]),
              "vertical_drift_m": displacement_z,
              "orientation_error_rad": rotation,
              "angular_speed_rad_s": math.sqrt(sum(v*v for v in final_velocity[3:6])),
              "vertical_pass_2cm": displacement_z < 0.02,
              "orientation_pass_10deg": rotation < math.radians(10)}
    print("WRIST_GRAVITY_HOLD_RESULT " + json.dumps(result), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.steps*f.dt >= 1.0 and not (result["vertical_pass_2cm"]
                                        and result["orientation_pass_10deg"]):
        raise AssertionError("Isolated gravity-on wrist pose hold failed")


try:
    main()
finally:
    app.close()
