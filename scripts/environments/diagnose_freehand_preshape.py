"""Check a grasp preshape's unforced finger motion with the bottle displaced away.

This separates hand self-contact, joint-limit and drive transients from
hand-bottle contact before committing a candidate to grasp synthesis.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reach_scan", type=Path, required=True)
parser.add_argument("--proposal_index", type=int, required=True)
parser.add_argument("--joints_field", choices=("pregrasp_joints_rad", "joints_rad"),
                    default="pregrasp_joints_rad")
parser.add_argument("--steps", type=int, default=25)
parser.add_argument("--disable_hand_self_collisions", action="store_true",
                    help="Diagnostic only: isolate hand self collision from drive motion")
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402
from isaaclab_physx.physics import PhysxManager  # noqa: E402
import warp as wp  # noqa: E402


def main():
    scan = json.loads(args.reach_scan.read_text())
    proposal = next(item for item in scan["best"] if item["index"] == args.proposal_index)
    q = proposal["wrist_quaternion_wxyz"]
    wrist = (*proposal["wrist_position_w_m"], q[1], q[2], q[3], q[0])
    fixture = BimanualPhase0Fixture(num_envs=1, device=args.device,
                                    control_mode="fixed_wrist", left_initial_pose=wrist,
                                    hand_self_collisions=not args.disable_hand_self_collisions)
    template = fixture.initializer.generate(mode="near_contact", seed=args.proposal_index)
    names = fixture.specs[0].joint_names
    preshape = proposal[args.joints_field]
    joints = tuple(preshape[name] for name in names)
    right_pose = tuple(float(value) for value in fixture.right.data.root_pose_w.torch[0])
    def shift(body):
        return replace(body, root_pose=(body.root_pose[0] + 1.0, *body.root_pose[1:]))
    sample = replace(template,
                     left=replace(template.left, root_pose=wrist,
                                  wrist_target_pose=wrist,
                                  joint_position=joints, joint_target=joints),
                     right=replace(template.right, root_pose=right_pose,
                                   wrist_target_pose=right_pose),
                     bottle=shift(template.bottle), cap=shift(template.cap))
    fixture.reset_manager.reset([0], [sample])
    self_views = []
    if not args.disable_hand_self_collisions:
        base = "/World/envs/env_0/LeftHand"
        bodies = fixture.specs[0].body_names
        sim_view = PhysxManager.get_physics_sim_view()
        for body in bodies:
            others = [f"{base}/{name}" for name in bodies if name != body]
            view = sim_view.create_rigid_contact_view(
                f"{base}/{body}", filter_patterns=others,
                max_contact_data_count=128)
            self_views.append((body, others, view))
    trace = []
    for step in range(args.steps):
        fixture.step()
        hand = fixture.left.data
        position = hand.joint_pos.torch[0].detach().cpu().tolist()
        velocity = hand.joint_vel.torch[0].detach().cpu().tolist()
        target = hand.joint_pos_target.torch[0].detach().cpu().tolist()
        torque = hand.applied_torque.torch[0].detach().cpu().tolist()
        collisions = []
        for body, others, view in self_views:
            forces, _, _, distances, counts, starts = view.get_contact_data(dt=fixture.dt)
            forces, distances, counts, starts = map(
                wp.to_torch, (forces, distances, counts, starts))
            for index, other in enumerate(others):
                count = int(counts.reshape(-1)[index])
                start = int(starts.reshape(-1)[index])
                if count:
                    collisions.append({"source": body,
                                       "partner": other.rsplit("/", 1)[-1],
                                       "minimum_separation_m": float(
                                           distances.reshape(-1)[start:start+count].min()),
                                       "maximum_normal_force_n": float(
                                           forces[start:start+count].abs().max())})
        trace.append({"t_s": (step+1)*fixture.dt,
                      "max_joint_speed_rad_s": max(abs(value) for value in velocity),
                      "max_target_error_rad": max(abs(a-b) for a,b in zip(target,position)),
                      "max_applied_torque_nm": max(abs(value) for value in torque),
                      "joint_positions_rad": position,
                      "joint_speeds_rad_s": velocity,
                      "joint_torques_nm": torque,
                      "contact_categories": list(fixture.monitor.summary(0)),
                      "hand_self_contacts": collisions})
    report = {"proposal_index": args.proposal_index,
              "joints_field": args.joints_field,
              "control_mode": "fixed_wrist", "bottle_shift_x_m": 1.0,
              "left_usd_sha256": fixture.specs[0].usd_sha256,
              "joint_order": fixture.left.joint_names,
              "initial_joint_position_rad": [joints[names.index(name)]
                                             for name in fixture.left.joint_names],
              "hand_self_collisions": not args.disable_hand_self_collisions,
              "max_freehand_joint_speed_rad_s": max(
                  row["max_joint_speed_rad_s"] for row in trace),
              "final_body_link_pose_w": {
                  name: fixture.left.data.body_link_pose_w.torch[
                      0, fixture.left.body_names.index(name)].detach().cpu().tolist()
                  for name in fixture.left.body_names if name.endswith(("_tip", "_distal"))},
              "trace": trace}
    print("FREEHAND_PRESHAPE_RESULT " + json.dumps({
        key: value for key, value in report.items() if key != "trace"}), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


try:
    main()
finally:
    app.close()
