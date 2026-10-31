"""Check selective Phase 1 reset isolation without advancing physics."""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--resets", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 2:
    parser.error("Selective reset requires at least two environments")
app = AppLauncher(args).app

import torch
from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env import BimanualBottleDirectEnv
from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env_cfg import GraspHoldDirectEnvCfg

cfg = GraspHoldDirectEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
env = BimanualBottleDirectEnv(cfg)
print("RESET_STAGE environment_ready", flush=True)
try:
    env.reset()
    print("RESET_STAGE reset_ready", flush=True)
    for _ in range(3):
        env.step(torch.zeros(args.num_envs, 52, device=env.device))
    print("RESET_STAGE physics_ready", flush=True)

    def capture():
        tensors = [asset.data.root_pose_w.torch.clone() for asset in
                   (env.left, env.right, env.bottle, env.cap)]
        tensors += [asset.data.root_vel_w.torch.clone() for asset in
                    (env.left, env.right, env.bottle, env.cap)]
        tensors += [asset.data.joint_pos.torch.clone() for asset in (env.left, env.right)]
        tensors += [wrist.current_pose.clone() for wrist in env.wrists]
        tensors += [wrist.target_pose.clone() for wrist in env.wrists]
        tensors += [env.thread.theta.clone(), env.thread.stick_reference.clone(),
                    env.thread.sliding.clone(), env.contacts.steps.clone(),
                    env._finger_targets[0].clone(), env._finger_targets[1].clone()]
        tensors += [getattr(env.thread, name).clone() for name in
                    ("last_wrapped", "initial_theta", "initial_axial", "initialized")]
        for wrist in env.wrists:
            tensors += [wrist.initial_pos.clone(), wrist.current_velocity.clone(),
                        wrist.workspace_clip_count.clone()]
        tensors += [getattr(env, name).clone() for name in
                    ("_stable_steps", "_success_steps", "_breakaway_seen",
                     "_maximum_opening", "_source_stable", "_reset_sample_index",
                     "_theta_at_reset", "_axial_at_reset", "_bottle_at_reset",
                     "_previous_open", "_previous_proximity", "_proximity_fresh",
                     "_last_action", "_last_raw_action", "_action_clip_count",
                     "_joint_clip_count")]
        tensors += [value.clone() for value in env._reward_sums.values()]
        tensors += [value.clone() for value in env.contacts.latest_ground_force.values()]
        return tensors

    max_error = 0.0
    for iteration in range(args.resets):
        if iteration % 20 == 0:
            print("RESET_STAGE iteration", iteration, flush=True)
        chosen = iteration % args.num_envs
        ids = torch.tensor([chosen], device=env.device)
        before = capture()
        if iteration % 20 == 0:
            env.thread_joints[chosen].GetJointEnabledAttr().Set(False)
            env.thread_joints[chosen].CreateCollisionEnabledAttr(True)
        env._reset_idx(ids)
        after = capture()
        if not env.thread_joints[chosen].GetJointEnabledAttr().Get():
            raise AssertionError("Selected D6 joint did not re-engage")
        if env.thread_joints[chosen].GetCollisionEnabledAttr().Get():
            raise AssertionError("Selected bottle/cap collision exclusion was not restored")
        other = torch.arange(args.num_envs, device=env.device) != chosen
        for field_index, (old, new) in enumerate(zip(before, after)):
            delta = (old[other].float() - new[other].float()).abs().max().item()
            max_error = max(max_error, delta)
            if delta != 0:
                raise AssertionError(f"Selective reset changed peer env at iteration {iteration} "
                                     f"field {field_index}: {delta}")
        if int(env.contacts.steps[chosen]) != 0:
            raise AssertionError("Selected contact history was not cleared")
        if not torch.isfinite(torch.cat([env.bottle.data.root_pose_w.torch,
                                         env.cap.data.root_pose_w.torch], dim=-1)).all():
            raise AssertionError("Nonfinite reset body state")
    print("PHASE1_RESET_ISOLATION", json.dumps({"num_envs": args.num_envs,
                                                "selective_resets": args.resets,
                                                "max_peer_state_error": max_error}), flush=True)
except BaseException as exc:
    print("RESET_STAGE exception", type(exc).__name__, repr(exc), flush=True)
    raise
finally:
    env.close()
    app.close()
