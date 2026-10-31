"""Compare static and commanded wrist contact from the same PhysX state.

Uses a collision-checked Phase 0B proposal to form contact with a temporary
bottle support. It then removes the support and performs two short, independently
reset free-bottle trials. The one-second hold result is reported separately;
contact or wrist tracking alone never certifies a grasp.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reach_scan", type=Path, required=True)
parser.add_argument("--proposal_index", type=int, default=0)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from dexteroushands_lab.assets.bottle import BottleConfig  # noqa: E402
from dexteroushands_lab.scenes.grasp_synthesis import GraspSynthesisFixture  # noqa: E402
from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402


def main():
    print("WRIST_POSE_CONTACT_BEGIN", flush=True)
    scan = json.loads(args.reach_scan.read_text())
    proposal = next(item for item in scan["best"] if item["index"] == args.proposal_index)
    q = proposal["wrist_quaternion_wxyz"]
    wrist = (*proposal["wrist_position_w_m"], q[1], q[2], q[3], q[0])
    fixture = BimanualPhase0Fixture(
        num_envs=1, device=args.device,
        config=BottleConfig(bottle_radius=scan["radius_m"]),
        control_mode="pose_command_wrist", left_initial_pose=wrist,
        temporary_bottle_support=True, include_hand_hand=False,
    )
    print("WRIST_POSE_CONTACT_FIXTURE_READY", flush=True)
    synth = GraspSynthesisFixture(fixture)
    print("WRIST_POSE_CONTACT_SYNTH_READY", flush=True)
    names = fixture.specs[0].joint_names
    goal = tuple(proposal["joints_rad"][name] for name in names)
    preshape = proposal.get("pregrasp_joints_rad")
    opened = tuple(preshape[name] for name in names) if preshape else None
    formed = synth.run_left(wrist_goal=wrist, joint_goal=goal,
                            pregrasp_joints=opened, seed=args.proposal_index,
                            start_at_goal=True)
    print("WRIST_POSE_CONTACT_FORMED " + json.dumps({
        "aborted": formed["aborted"],
        "release_passed": formed["release_passed"],
        "release_displacement_m": formed["bottle_displacement_m"],
    }), flush=True)
    if formed["candidate"] is None:
        raise RuntimeError("No safe physical contact state was formed")
    sample = formed["candidate"]
    joint = fixture.temporary_support_joint
    joint.GetJointEnabledAttr().Set(False)

    def trial(command_translation):
        fixture.reset_manager.reset([0], [sample])
        controller = fixture.controllers[0][0]
        controller.command_relative(command_translation, (0.0, 0.0, 0.0))
        start = fixture.bottle.data.root_pose_w.torch[0].detach().cpu().tolist()
        rows = []
        for step in range(25):
            fixture.step()
            pose = fixture.bottle.data.root_pose_w.torch[0].detach().cpu().tolist()
            actual = fixture.left.data.root_pose_w.torch[0].detach().cpu().tolist()
            points = [item for item in fixture.monitor.raw_contact_points(0, fixture.dt)
                      if item["category"] == "left_bottle"]
            rows.append({"t_s": (step+1)*fixture.dt,
                         "bottle_pose_w": pose,
                         "wrist_pose_w": actual,
                         "wrist_position_error_m": math.dist(actual[:3],
                                                               controller.current_pos),
                         "contact_count": len(points),
                         "contact_peak_n": max((point["normal_force_n"]
                                                for point in points), default=0.0),
                         "minimum_separation_m": min((point["separation_m"]
                                                      for point in points), default=None),
                         "contact_points": points[:8]})
        final = rows[-1]["bottle_pose_w"]
        return {"command_translation_w_m": command_translation,
                "initial_bottle_pose_w": start,
                "final_bottle_pose_w": final,
                "bottle_displacement_w_m": [final[i]-start[i] for i in range(3)],
                "max_wrist_position_error_m": max(x["wrist_position_error_m"]
                                                  for x in rows),
                "max_contact_force_n": max(x["contact_peak_n"] for x in rows),
                "minimum_separation_m": min((x["minimum_separation_m"] for x in rows
                                            if x["minimum_separation_m"] is not None),
                                            default=None),
                "contact_step_ratio": sum(x["contact_count"] > 0 for x in rows)/len(rows),
                "trace": rows}

    static = trial((0.0, 0.0, 0.0))
    print("WRIST_POSE_CONTACT_STATIC_DONE", flush=True)
    moving = trial((0.005, 0.0, 0.0))
    print("WRIST_POSE_CONTACT_MOVING_DONE", flush=True)
    deviation = math.dist(static["bottle_displacement_w_m"],
                          moving["bottle_displacement_w_m"])
    result = {"control_mode": "pose_command_wrist",
              "bottle_radius_m": fixture.config.bottle_radius,
              "asset_commit": fixture.specs[0].source_commit,
              "left_usd_sha256": fixture.specs[0].usd_sha256,
              "proposal_index": args.proposal_index,
              "support_removed_during_trials": not joint.GetJointEnabledAttr().Get(),
              "one_second_hold_passed": formed["release_passed"],
              "one_second_hold_displacement_m": formed["bottle_displacement_m"],
              "formation_aborted": formed["aborted"],
              "static": static, "moving": moving,
              "bottle_motion_difference_m": deviation}
    result["contact_trial_passed"] = (
        result["support_removed_during_trials"]
        and formed["aborted"] is None
        and static["contact_step_ratio"] > 0
        and moving["contact_step_ratio"] > 0
        and moving["max_wrist_position_error_m"] < 2e-4
        and moving["minimum_separation_m"] is not None
        and moving["minimum_separation_m"] >= -0.002
        and moving["max_contact_force_n"] < 100.0)
    print("WRIST_POSE_CONTACT_RESULT " + json.dumps({
        key: value for key, value in result.items() if key not in ("static", "moving")
    }), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["contact_trial_passed"]:
        raise RuntimeError("Pose-command contact trial did not pass")


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
