"""PhysX staged left-hand grasp synthesis and support-free hold diagnostic.

Reach-scan proposals are never accepted from MJCF alone. This runner first
uses native USD collisions and normal drives, then removes the bounded bottle
support before measuring the suspended hold. It stops at left-only failure.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reach_scan", type=Path, required=True)
parser.add_argument("--candidate_count", type=int, default=3)
parser.add_argument("--proposal_index", type=int,
                    help="Run one reproducible proposal by its original index")
parser.add_argument("--control_mode", choices=("floating_wrench_pd", "fixed_wrist",
                                               "pose_command_wrist"),
                    default="floating_wrench_pd")
parser.add_argument("--radius", type=float, default=None,
                    help="Select one radius from a multi-radius reach scan")
parser.add_argument("--fixed_support", action="store_true",
                    help="Use a removable world fixed joint only during synthesis")
parser.add_argument("--ramp_fixed_support", action="store_true",
                    help="Release the fixed joint before settle and taper bounded wrench support")
parser.add_argument("--start_at_goal", action="store_true",
                    help="Use a collision-checked preshape at the goal wrist pose for contact comparison")
parser.add_argument("--visualize_grasp", action="store_true",
                    help="GUI markers for live fingertip sites, bottle axis, contacts and normals")
parser.add_argument("--require_power_grasp", action="store_true",
                    help="Require sustained four-finger sidewall contact opposite the thumb")
parser.add_argument("--strong_wrist", action="store_true",
                    help="Use bounded 80 N / 8 Nm wrist pose gains for contact search")
parser.add_argument("--wrist_torque", type=float,
                    help="Diagnostic wrist torque limit in Nm")
parser.add_argument("--wrist_kp_rotation", type=float,
                    help="Diagnostic rotation proportional gain in Nm/rad")
parser.add_argument("--wrist_kd_rotation", type=float,
                    help="Diagnostic rotation damping in Nm s/rad")
parser.add_argument("--settle_s", type=float, default=0.2)
parser.add_argument("--close_s", type=float, default=0.45)
parser.add_argument("--pose_compensation", action="store_true",
                    help="Use bounded outer-loop wrist target correction while supported")
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from dexteroushands_lab.scenes.grasp_synthesis import (  # noqa: E402
    GraspSynthesisFixture, SynthesisLimits)
from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402


def main():
    if args.ramp_fixed_support and not args.fixed_support:
        raise ValueError("--ramp_fixed_support requires --fixed_support")
    if args.control_mode != "floating_wrench_pd" and (
        args.strong_wrist or args.wrist_torque is not None or
        args.wrist_kp_rotation is not None or args.wrist_kd_rotation is not None or
        args.pose_compensation
    ):
        raise ValueError("Wrench PD gain options are only valid in floating_wrench_pd mode")
    proposals = json.loads(args.reach_scan.read_text())
    if isinstance(proposals, list):
        matching = [item for item in proposals if args.radius is None or
                    abs(item["radius_m"] - args.radius) < 1e-9]
        if not matching:
            raise ValueError("Requested radius is absent from reach scan")
        proposals = matching[0]
    selected = proposals["best"]
    if args.proposal_index is not None:
        selected = [item for item in selected if item["index"] == args.proposal_index]
        if len(selected) != 1:
            raise ValueError("Requested proposal index is absent or not unique")
    selected = selected[:args.candidate_count]
    if (args.control_mode == "fixed_wrist" or args.start_at_goal) and len(selected) > 1:
        anchor = (selected[0]["wrist_position_w_m"],
                  selected[0]["wrist_quaternion_wxyz"])
        if any((item["wrist_position_w_m"],
                item["wrist_quaternion_wxyz"]) != anchor for item in selected[1:]):
            raise ValueError("Candidates in one anchored fixture need the same wrist pose")
    from dexteroushands_lab.assets.bottle import BottleConfig
    from dexteroushands_lab.assets.wrist_6dof import WristLimits
    config = BottleConfig(bottle_radius=proposals["radius_m"])
    wrist_limits = (WristLimits(max_force=80.0, kp_position=4000.0,
                                 kd_position=120.0, max_torque=8.0,
                                 kp_rotation=30.0, kd_rotation=6.0)
                    if args.strong_wrist else None)
    if any(value is not None for value in (args.wrist_torque,
                                            args.wrist_kp_rotation,
                                            args.wrist_kd_rotation)):
        if args.strong_wrist:
            raise ValueError("Choose strong_wrist or explicit wrist gains")
        defaults = WristLimits()
        wrist_limits = WristLimits(
            max_torque=(args.wrist_torque if args.wrist_torque is not None
                        else defaults.max_torque),
            kp_rotation=(args.wrist_kp_rotation if args.wrist_kp_rotation is not None
                         else defaults.kp_rotation),
            kd_rotation=(args.wrist_kd_rotation if args.wrist_kd_rotation is not None
                         else defaults.kd_rotation),
        )
    fixed_pose = None
    if args.control_mode == "fixed_wrist" or args.start_at_goal:
        q = selected[0]["wrist_quaternion_wxyz"]
        fixed_pose = (*selected[0]["wrist_position_w_m"], q[1], q[2], q[3], q[0])
    fixture = BimanualPhase0Fixture(num_envs=1, device=args.device, config=config,
                                    include_hand_hand=False,
                                    wrist_limits=wrist_limits,
                                    temporary_bottle_support=args.fixed_support,
                                    control_mode=args.control_mode,
                                    left_initial_pose=fixed_pose)
    if args.visualize_grasp and args.headless:
        raise ValueError("Grasp markers require a GUI run")
    visualizer = None
    if args.visualize_grasp:
        from dexteroushands_lab.scenes.grasp_visualization import GraspDebugMarkers
        visualizer = GraspDebugMarkers()
    synth = GraspSynthesisFixture(fixture, SynthesisLimits(
        close_s=args.close_s, settle_s=args.settle_s),
                                  pose_compensation=args.pose_compensation,
                                  visualizer=visualizer,
                                  require_power_grasp=args.require_power_grasp,
                                  ramp_fixed_support=args.ramp_fixed_support)
    records = []
    for proposal in selected:
        q = proposal["wrist_quaternion_wxyz"]
        wrist = (*proposal["wrist_position_w_m"], q[1], q[2], q[3], q[0])
        joints = tuple(proposal["joints_rad"][name]
                       for name in fixture.specs[0].joint_names)
        preshape = proposal.get("pregrasp_joints_rad")
        preshape = (tuple(preshape[name] for name in fixture.specs[0].joint_names)
                    if preshape is not None else None)
        waypoints = proposal.get("joint_waypoints_rad")
        waypoints = (tuple(tuple(point[name] for name in fixture.specs[0].joint_names)
                           for point in waypoints) if waypoints is not None else None)
        result = synth.run_left(wrist_goal=wrist, joint_goal=joints,
                                seed=proposal["index"], pregrasp_joints=preshape,
                                start_at_goal=args.start_at_goal,
                                joint_waypoints=waypoints)
        record = {**result, "candidate": asdict(result["candidate"])
                  if result["candidate"] is not None else None,
                  "reach_proposal_index": proposal["index"],
                  "reach_proposal_distance_m": proposal["minimum_mjcf_convex_distance_m"]}
        records.append(record)
        print("SYNTHESIS_RESULT " + json.dumps({
            "proposal": proposal["index"], "aborted": result["aborted"],
            "min_separation_m": result["minimum_separation_m"],
            "max_force_n": result["maximum_contact_force_n"],
            "bottle_displacement_m": result["bottle_displacement_m"],
            "release_passed": result["release_passed"],
            "power_grasp_ratio": result["release_power_grasp"]["valid_topology_ratio"],
            "reset_replay_passed": result["reset_replay"] is not None and
                                   result["reset_replay"]["passed"],
            "left_contact_ratio": result["contact_summary"].get(
                "left_bottle", {}).get("contact_ratio", 0.0)}), flush=True)
    output = {"asset_commit": fixture.specs[0].source_commit,
              "left_usd_sha256": fixture.specs[0].usd_sha256,
              "right_usd_sha256": fixture.specs[1].usd_sha256,
              "bottle_radius_m": config.bottle_radius, "dt_s": fixture.dt,
              "fixture": "bounded_support_then_gravity_on_release",
              "support_kind": "fixed_then_bounded_wrench" if args.ramp_fixed_support
                              else "temporary_world_fixed_joint" if args.fixed_support
                              else "bounded_bottle_wrench",
              "wrist_control_mode": args.control_mode,
              "wrist_limits": vars(wrist_limits) if wrist_limits else "default",
              "pose_compensation": args.pose_compensation,
              "required_grasp_topology": (
                  "cylindrical_power_grasp_v1" if args.require_power_grasp else None),
              "results": records}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")


try:
    main()
except Exception:
    # Isaac Kit can exit from app.close() before Python prints the pending
    # exception; preserve the actual failure in headless experiment logs.
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
