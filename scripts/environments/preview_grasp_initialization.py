"""Preview a pinned bimanual initial state; candidates are labelled unverified."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("stable_pregrasp", "weak_contact", "near_contact"),
                    default="stable_pregrasp")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--bank", type=Path)
parser.add_argument("--synthesis_report", type=Path,
                    help="Preview an unvalidated support-release candidate from a PhysX report")
parser.add_argument("--candidate", type=Path,
                    help="Preview one unvalidated GraspInitSample JSON")
parser.add_argument("--geometry_proposal", type=Path,
                    help="Preview a still unvalidated official-model pose proposal")
parser.add_argument("--sample_index", type=int, default=0)
parser.add_argument("--steps", type=int, default=500,
                    help="Maximum physics steps in headless or GUI preview")
parser.add_argument("--contact_debug", action="store_true",
                    help="Display Isaac Lab contact markers when the visualizer is installed")
parser.add_argument("--interactive_static", action="store_true",
                    help="Render the initialized pose until the GUI closes, without advancing physics")
parser.add_argument("--focus_left", action="store_true",
                    help="Hide the right hand in the viewer while leaving physics unchanged")
parser.add_argument("--camera_eye", nargs=3, type=float,
                    default=(0.35, -0.65, 0.78), metavar=("X", "Y", "Z"))
parser.add_argument("--camera_target", nargs=3, type=float,
                    default=(0.0, 0.0, 0.55), metavar=("X", "Y", "Z"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from dexteroushands_lab.scenes.grasp_init import (  # noqa: E402
    GraspInitBank, grasp_sample_from_dict)
from dexteroushands_lab.assets.bottle import BottleConfig  # noqa: E402
from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402


def main():
    if sum(bool(x) for x in (args.bank, args.synthesis_report, args.candidate,
                            args.geometry_proposal)) > 1:
        raise ValueError("Choose one bank, report, candidate, or geometry proposal")
    if args.bank:
        samples = GraspInitBank(args.bank).load()
        if not samples:
            raise RuntimeError(f"No validated initial states in {args.bank}")
        sample = samples[args.sample_index]
    elif args.synthesis_report:
        records = json.loads(args.synthesis_report.read_text())["results"]
        candidate = records[args.sample_index]["candidate"]
        if candidate is None:
            raise ValueError("Selected synthesis attempt was rejected before support release")
        sample = grasp_sample_from_dict(candidate)
    elif args.candidate:
        sample = grasp_sample_from_dict(json.loads(args.candidate.read_text()))
    elif args.geometry_proposal:
        source = json.loads(args.geometry_proposal.read_text())
        proposal = source["best"][args.sample_index]
        q = proposal["wrist_quaternion_wxyz"]
        left_pose = (*proposal["wrist_position_w_m"], q[1], q[2], q[3], q[0])
        fixture = BimanualPhase0Fixture(
            num_envs=1, device=args.device,
            config=BottleConfig(bottle_radius=source["radius_m"]),
            control_mode="fixed_wrist", left_initial_pose=left_pose,
            contact_debug=args.contact_debug)
        sample = fixture.initializer.generate(
            mode="stable_pregrasp", seed=proposal["index"])
        finger = tuple(proposal["joints_rad"][name]
                       for name in fixture.specs[0].joint_names)
        right_pose = tuple(float(x) for x in
                           fixture.right.data.root_pose_w.torch[0].detach().cpu())
        sample = replace(
            sample,
            left=replace(sample.left, root_pose=left_pose,
                         wrist_target_pose=left_pose,
                         joint_position=finger, joint_target=finger),
            right=replace(sample.right, root_pose=right_pose,
                          wrist_target_pose=right_pose),
            source="official_MJCF_geometry_proposal_unvalidated",
            metadata={**sample.metadata, "wrist_control_mode": "fixed_wrist",
                      "geometry_proposal": str(args.geometry_proposal)})
    else:
        fixture = BimanualPhase0Fixture(num_envs=1, device=args.device,
                                        contact_debug=args.contact_debug)
        sample = fixture.initializer.generate(mode=args.mode, seed=args.seed)
    if args.bank or args.synthesis_report or args.candidate:
        wrist_mode = sample.metadata.get("wrist_control_mode", "floating_wrench_pd")
        if wrist_mode not in ("floating_wrench_pd", "fixed_wrist",
                              "pose_command_wrist"):
            raise ValueError(f"Unknown sample wrist control mode: {wrist_mode}")
        fixture = BimanualPhase0Fixture(
            num_envs=1, device=args.device,
            config=BottleConfig(**sample.metadata["bottle_parameters"]),
            control_mode=wrist_mode,
            left_initial_pose=sample.left.root_pose,
            right_initial_pose=sample.right.root_pose,
            include_hand_hand=sample.validation is not None and
                              sample.validation.get("scope") == "bimanual",
            contact_debug=args.contact_debug)
    fixture.reset_manager.reset([0], [sample])
    if args.focus_left:
        from pxr import UsdGeom
        prim = fixture.sim.stage.GetPrimAtPath("/World/envs/env_0/RightHand")
        UsdGeom.Imageable(prim).MakeInvisible()
    if not args.headless:
        fixture.sim.set_camera_view(args.camera_eye, args.camera_target)
    print("GRASP_PREVIEW_INITIAL " + json.dumps({"mode": sample.mode,
          "seed": sample.seed, "validated": sample.validated,
          "source": sample.source, "asset_variant": sample.metadata["asset_variant"],
          "wrist_control_mode": fixture.control_mode}),
          flush=True)
    if args.interactive_static:
        if args.headless:
            raise ValueError("--interactive_static requires a GUI")
        fixture.step()  # Refresh live contact points once, then hold this visual state.
        print("GRASP_PREVIEW_STATIC_GUI_READY", flush=True)
        while app.is_running():
            fixture.sim.render()
        return
    for step in range(args.steps):
        if not app.is_running():
            break
        fixture.step()
        if step % 100 == 0 or step == args.steps - 1:
            pose = fixture.bottle.data.root_pose_w.torch[0].detach().cpu().tolist()
            contacts = fixture.monitor.summary(0)
            print("GRASP_PREVIEW_FRAME " + json.dumps({"step": step + 1,
                  "bottle_pose_xyzw": pose,
                  "contact_categories": list(contacts),
                  "contact_ratios": {name: data["contact_ratio"]
                                     for name, data in contacts.items()}}), flush=True)


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
