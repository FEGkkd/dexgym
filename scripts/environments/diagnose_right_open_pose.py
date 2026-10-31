"""Measure cap/bottle drift with a stationary right-hand open preshape.

This is an unsupported PhysX control experiment. No hand or object pose is
written after Reset; both hands retain their normal finger drives. The
optional proposal only changes the initial right-hand Reset state.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--left_bank", type=Path, required=True)
parser.add_argument("--left_sample_index", type=int, default=0)
parser.add_argument("--right_preshape", type=Path)
parser.add_argument("--duration_s", type=float, default=0.2)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from dexteroushands_lab.assets.bottle import BottleConfig  # noqa: E402
from dexteroushands_lab.scenes.grasp_init import GraspInitBank, HandInit, _rotate  # noqa: E402
from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402


def main():
    sample = GraspInitBank(args.left_bank).load()[args.left_sample_index]
    if sample.validation.get("scope") != "left_only":
        raise ValueError("A validated left-only sample is required")
    proposal = None
    right_pose = sample.right.root_pose
    if args.right_preshape:
        proposal = json.loads(args.right_preshape.read_text())["best"][0]
        q = proposal["wrist_quaternion_wxyz"]
        right_pose = (*proposal["wrist_position_w_m"], *q[1:], q[0])
    f = BimanualPhase0Fixture(
        num_envs=1, device=args.device,
        config=BottleConfig(**sample.metadata["bottle_parameters"]),
        control_mode="pose_command_wrist",
        left_initial_pose=sample.left.root_pose,
        right_initial_pose=right_pose,
        include_hand_hand=True,
    )
    if proposal:
        joints = tuple(proposal["pregrasp_joints_rad"][name]
                       for name in f.specs[1].joint_names)
        right = HandInit(right_pose, (0.0,)*6, joints, (0.0,)*20,
                         right_pose, joints)
        sample = replace(sample, right=right, validated=False, validation=None)
    f.reset_manager.reset([0], [sample])
    from isaaclab_physx.physics import PhysxManager
    import warp as wp

    root = "/World/envs/env_0"
    cap_bottle = PhysxManager.get_physics_sim_view().create_rigid_contact_view(
        f"{root}/Cap", filter_patterns=[f"{root}/Bottle"],
        max_contact_data_count=128)
    trace = []
    max_lateral = max_native_force = 0.0
    steps = math.ceil(args.duration_s/f.dt)
    for step in range(steps):
        f.step()
        bottle = f.bottle.data.root_pose_w.torch[0].detach().cpu().tolist()
        cap = f.cap.data.root_pose_w.torch[0].detach().cpu().tolist()
        axis = _rotate(bottle[3:7], (0.0, 0.0, 1.0))
        delta = tuple(a-b for a,b in zip(cap[:3], bottle[:3]))
        axial = sum(a*b for a,b in zip(delta, axis))
        lateral = math.sqrt(sum((d-axial*a)**2 for d,a in zip(delta, axis)))
        max_lateral = max(max_lateral, lateral)
        forces, _, _, distances, counts, _ = cap_bottle.get_contact_data(dt=f.dt)
        count = int(wp.to_torch(counts).reshape(-1)[0].item())
        force = float(wp.to_torch(forces)[:count].sum().item()) if count else 0.0
        penetration = (float(wp.to_torch(distances)[:count].min().item())
                       if count else None)
        max_native_force = max(max_native_force, force)
        if step % max(1, round(.01/f.dt)) == 0 or step == steps-1:
            summary = f.monitor.summary(0)
            trace.append({"t_s": (step+1)*f.dt,
                          "thread_lateral_error_m": lateral,
                          "bottle_position_w_m": bottle[:3],
                          "cap_position_w_m": cap[:3],
                          "cap_bottle_native_contact_force_n": force,
                          "cap_bottle_native_min_separation_m": penetration,
                          "right_cap_contact_ratio": summary.get("right_cap", {}).get("contact_ratio", 0),
                          "right_bottle_contact_ratio": summary.get("right_bottle", {}).get("contact_ratio", 0),
                          "hand_hand_contact_ratio": summary.get("hand_hand", {}).get("contact_ratio", 0)})
    output = {"right_preshape": str(args.right_preshape) if proposal else None,
              "right_proposal_index": proposal["index"] if proposal else None,
              "left_seed": sample.seed,
              "duration_s": steps*f.dt,
              "max_thread_lateral_error_m": max_lateral,
              "max_cap_bottle_native_contact_force_n": max_native_force,
              "contact_summary": f.monitor.summary(0), "trace": trace}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print("RIGHT_OPEN_POSE_DIAG " + json.dumps({
        k: output[k] for k in ("right_proposal_index", "duration_s",
                             "max_thread_lateral_error_m",
                             "max_cap_bottle_native_contact_force_n")}), flush=True)


try:
    main()
finally:
    app.close()
