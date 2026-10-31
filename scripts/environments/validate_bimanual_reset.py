"""GPU validation of indexed Phase 0 resets and episode-local state isolation."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--iterations", type=int, default=100)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.iterations < 1:
    parser.error("--num_envs and --iterations must be positive")
app = AppLauncher(args).app

import torch  # noqa: E402
from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402


def snapshot(fixture, env):
    assets = []
    for asset in fixture.assets:
        row = [asset.data.root_pose_w.torch[env].clone(),
               asset.data.root_vel_w.torch[env].clone()]
        if hasattr(asset.data, "joint_pos"):
            row.extend((asset.data.joint_pos.torch[env].clone(),
                        asset.data.joint_vel.torch[env].clone(),
                        asset.data.joint_pos_target.torch[env].clone()))
        assets.append(row)
    controllers = [(controller[env].initial_pos, controller[env].initial_quat,
                    controller[env].target_pos, controller[env].target_quat)
                   for controller in fixture.controllers]
    coupler = fixture.couplers[env]
    coupler_state = tuple(getattr(coupler, name) for name in (
        "_last_wrapped", "_theta", "_initial_theta", "_initial_axial",
        "_stick_reference", "_sliding", "_detached"))
    return assets, controllers, coupler_state, fixture.monitor.history.summary(env)


def unchanged(a, b):
    return (all(torch.equal(x, y) for rows_a, rows_b in zip(a[0], b[0])
                for x, y in zip(rows_a, rows_b)) and a[1:] == b[1:])


def main():
    fixture = BimanualPhase0Fixture(num_envs=args.num_envs, device=args.device,
                                    include_hand_hand=False)
    samples = [fixture.initializer.generate(mode="near_contact", seed=i)
               for i in range(args.num_envs)]
    fixture.reset_manager.reset(range(args.num_envs), samples)
    fixture.step()
    fixture.couplers[0]._detached = True
    fixture.joints[0].GetJointEnabledAttr().Set(False)
    fixture.joints[0].CreateCollisionEnabledAttr(True)
    fixture.reset_manager.reset([0], [samples[0]])
    if (fixture.couplers[0].detached or not fixture.joints[0].GetJointEnabledAttr().Get()
            or fixture.joints[0].GetCollisionEnabledAttr().Get()):
        raise AssertionError("Detached thread state did not recover on reset")
    rng = random.Random(2201)
    max_pose_error = 0.0
    max_root_error = 0.0
    max_joint_error = 0.0
    max_initial_wrist_force = 0.0
    for iteration in range(args.iterations):
        selected = rng.randrange(args.num_envs)
        others = {env: snapshot(fixture, env) for env in range(args.num_envs)
                  if env != selected}
        sample = fixture.initializer.generate(mode="near_contact", seed=10000 + iteration)
        fixture.reset_manager.reset([selected], [sample])
        for env, before in others.items():
            if not unchanged(before, snapshot(fixture, env)):
                raise AssertionError(f"Reset {selected} changed env {env} without a physics step")
        try:
            fixture.monitor.summary(selected)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Contact history remained fresh after reset")
        origin = fixture.scene.env_origins[selected].detach().cpu()
        expected = torch.tensor(sample.bottle.root_pose[:3]) + origin
        actual = fixture.bottle.data.root_pose_w.torch[selected, :3].detach().cpu()
        max_pose_error = max(max_pose_error, float(torch.linalg.vector_norm(expected - actual)))
        for asset, state in zip(fixture.assets,
                                (sample.left, sample.right, sample.bottle, sample.cap)):
            expected_pose = torch.tensor(state.root_pose, device=asset.device)
            expected_pose[:3] += origin.to(asset.device)
            expected_velocity = torch.tensor(state.root_velocity, device=asset.device)
            pose_error = float((asset.data.root_pose_w.torch[selected] - expected_pose).abs().max())
            velocity_error = float((asset.data.root_vel_w.torch[selected] - expected_velocity).abs().max())
            max_root_error = max(max_root_error, pose_error, velocity_error)
            if max(pose_error, velocity_error) > 1e-5:
                raise AssertionError("Root pose/velocity was not fully restored")
        for hand, spec, state in ((fixture.left, fixture.specs[0], sample.left),
                                  (fixture.right, fixture.specs[1], sample.right)):
            canonical = spec.joint_names
            for values, actual_values in (
                (state.joint_position, hand.data.joint_pos.torch[selected]),
                (state.joint_velocity, hand.data.joint_vel.torch[selected]),
                (state.joint_target, hand.data.joint_pos_target.torch[selected]),
            ):
                expected_values = torch.tensor([values[canonical.index(name)]
                                                for name in hand.joint_names],
                                               device=hand.device)
                joint_error = float((actual_values - expected_values).abs().max())
                max_joint_error = max(max_joint_error, joint_error)
                if joint_error > 1e-5:
                    raise AssertionError("Finger state/target not restored in canonical order")
        for side, state in enumerate((sample.left, sample.right)):
            controller = fixture.controllers[side][selected]
            expected_target = tuple(state.wrist_target_pose[i] + float(origin[i])
                                    for i in range(3))
            if (any(not math.isclose(a, b, abs_tol=1e-6)
                    for a, b in zip(controller.target_pos, expected_target))
                    or any(not math.isclose(a, b, abs_tol=1e-6)
                           for a, b in zip(controller.target_quat,
                                           state.wrist_target_pose[3:7]))):
                raise AssertionError("Wrist PD target was not restored")
            hand = fixture.left if side == 0 else fixture.right
            pose = hand.data.root_pose_w.torch[selected].detach().cpu().tolist()
            velocity = hand.data.root_vel_w.torch[selected].detach().cpu().tolist()
            force, torque = controller.wrench(pose[:3], pose[3:7],
                                               velocity[:3], velocity[3:6])
            force_norm = math.sqrt(sum(x * x for x in force))
            torque_norm = math.sqrt(sum(x * x for x in torque))
            max_initial_wrist_force = max(max_initial_wrist_force, force_norm)
            if force_norm > 0.001 or torque_norm > 0.001:
                raise AssertionError("Reset introduced an initial wrist PD impulse")
        coupler = fixture.couplers[selected]
        if (coupler.detached or coupler._sliding
                or abs(coupler._theta - sample.thread_initial_angle_rad) > 1e-6):
            raise AssertionError("Thread angle or episode-local state not restored")
        if not fixture.joints[selected].GetJointEnabledAttr().Get():
            raise AssertionError("Thread joint disabled after reset")
        if fixture.joints[selected].GetCollisionEnabledAttr().Get():
            raise AssertionError("Thread joint collision flag not restored")
        if (fixture.reset_manager.init_valid[selected] != sample.validated
                or fixture.reset_manager.sample_seed[selected] != sample.seed):
            raise AssertionError("Episode validity or provenance did not reset")
        fixture.step()
    result = {"num_envs": args.num_envs, "iterations": args.iterations,
              "other_env_unchanged": True, "max_bottle_position_reset_error_m": max_pose_error,
              "post_reset_finite": True, "detached_recovered": True,
              "max_root_pose_velocity_error": max_root_error,
              "max_joint_state_target_error": max_joint_error,
              "max_initial_wrist_force_n": max_initial_wrist_force,
              "finger_targets_restored": True, "contact_history_invalidated": True,
              "random_reset_seed": 2201, "asset_variant": fixture.specs[0].variant,
              "asset_commit": fixture.specs[0].source_commit,
              "left_usd_sha256": fixture.specs[0].usd_sha256,
              "right_usd_sha256": fixture.specs[1].usd_sha256}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("BIMANUAL_RESET_RESULT " + json.dumps(result, sort_keys=True), flush=True)


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
