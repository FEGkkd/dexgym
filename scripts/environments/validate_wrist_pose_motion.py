"""GPU check for independent six-axis, world-anchored wrist trajectories.

This tests root motion and finger articulation. Contact acceptance is checked
separately by ``validate_wrist_pose_contact.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from dexteroushands_lab.assets.wrist_6dof import _orientation_error  # noqa: E402
from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402


def main():
    if args.num_envs < 1:
        raise ValueError("At least one environment is required")
    fixture = BimanualPhase0Fixture(num_envs=args.num_envs, device=args.device,
                                    control_mode="pose_command_wrist")
    left = fixture.controllers[0][0]
    right = fixture.controllers[1][0]
    left_fingers = [0.0]*20
    right_fingers = [0.0]*20
    left_fingers[fixture.specs[0].joint_names.index("l_index_finger_pip")] = .2
    right_fingers[fixture.specs[1].joint_names.index("r_thumb_ip")] = .2
    fixture.command_action52([0], [[.01, .008, .006, .08, -.07, .06,
                                    *left_fingers,
                                    -.009, -.007, .005, -.06, .07, -.08,
                                    *right_fingers]])
    if args.num_envs > 1:
        fixture.controllers[0][1].command_relative((0.0, 0.0, 0.008),
                                                    (0.06, 0.0, 0.0))
    initial = tuple(asset.data.root_pose_w.torch.detach().cpu().tolist()
                    for asset in fixture.assets[:2])
    initial_fingers = tuple(asset.data.joint_pos.torch[0].detach().cpu().tolist()
                            for asset in fixture.assets[:2])
    max_speed = [0.0, 0.0]
    max_angular_speed = [0.0, 0.0]
    for _ in range(500):
        fixture.step()
        for side, controller in enumerate((left, right)):
            max_speed[side] = max(max_speed[side], math.dist(controller.current_velocity[:3],
                                                            (0.0, 0.0, 0.0)))
            max_angular_speed[side] = max(max_angular_speed[side], math.dist(
                controller.current_velocity[3:], (0.0, 0.0, 0.0)))
    records = []
    for side, asset in enumerate(fixture.assets[:2]):
        for env in range(args.num_envs):
            controller = fixture.controllers[side][env]
            actual = asset.data.root_pose_w.torch[env].detach().cpu().tolist()
            expected = controller.current_pose
            angular_error = math.dist(_orientation_error(expected[3:7], actual[3:7]),
                                      (0.0, 0.0, 0.0))
            records.append({"side": ("left", "right")[side], "env_id": env,
                            "initial_pose_w": initial[side][env],
                            "target_pose_w": controller.target_pose,
                            "actual_pose_w": actual,
                            "position_error_m": math.dist(actual[:3], expected[:3]),
                            "angular_error_rad": angular_error,
                            "at_target": controller.at_target()})
    result = {"control_mode": fixture.control_mode,
              "control_interface": "52D: per side world Δxyz, local Δrotation, 20 canonical finger targets",
              "root_joint_type": "official_world_fixed_joint",
              "asset_commit": fixture.specs[0].source_commit,
              "left_usd_sha256": fixture.specs[0].usd_sha256,
              "right_usd_sha256": fixture.specs[1].usd_sha256,
              "num_envs": args.num_envs,
              "max_linear_command_speed_m_s": max_speed,
              "max_angular_command_speed_rad_s": max_angular_speed,
              "hand_joint_counts": [len(fixture.left.joint_names),
                                    len(fixture.right.joint_names)],
              "driven_finger_motion_rad": [
                  float(asset.data.joint_pos.torch[0, asset.joint_names.index(name)]) -
                  initial_fingers[side][asset.joint_names.index(name)]
                  for side, (asset, name) in enumerate((
                      (fixture.left, "l_index_finger_pip"),
                      (fixture.right, "r_thumb_ip")))],
              "bottle_z_m": fixture.bottle.data.root_pose_w.torch[:, 2].detach().cpu().tolist(),
              "records": records}
    result["passed"] = (all(record["position_error_m"] < 2e-4 and
                            record["angular_error_rad"] < 2e-3 and
                            record["at_target"] for record in records)
                        and all(count == 20 for count in result["hand_joint_counts"])
                        and all(abs(value) > .05 for value in result["driven_finger_motion_rad"])
                        and all(speed <= 0.150001 for speed in max_speed)
                        and all(speed <= 1.000001 for speed in max_angular_speed))
    print("WRIST_POSE_MOTION_RESULT " + json.dumps(result), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise RuntimeError("Ideal wrist motion acceptance failed")


try:
    main()
finally:
    app.close()
