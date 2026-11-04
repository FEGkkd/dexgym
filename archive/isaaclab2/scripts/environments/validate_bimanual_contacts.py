"""Exercise exact PhysX per-object/body contact filters in two environments."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1:
    parser.error("--steps must be positive")
app = AppLauncher(args).app

from dexteroushands_lab.scenes.grasp_init import BodyInit  # noqa: E402
from dexteroushands_lab.scenes.phase0_fixture import BimanualPhase0Fixture  # noqa: E402
import torch  # noqa: E402


def moved(sample, shift):
    def body(original):
        p = original.root_pose
        return BodyInit(tuple(p[i] + shift[i] for i in range(3)) + p[3:7],
                        original.root_velocity)
    return replace(sample, bottle=body(sample.bottle), cap=body(sample.cap))


def main():
    print("CONTACT_STAGE construct", flush=True)
    fixture = BimanualPhase0Fixture(num_envs=2, device=args.device,
                                    include_hand_hand=True)
    print("CONTACT_STAGE constructed", flush=True)
    base = fixture.initializer.generate(mode="near_contact", seed=73)
    # Keep the hands at the distant scene display poses to isolate object pairs.
    left_pose = tuple(fixture.left.data.root_pose_w.torch[0].detach().cpu().tolist())
    right_pose = tuple(fixture.right.data.root_pose_w.torch[0].detach().cpu().tolist())
    left = replace(base.left, root_pose=left_pose, wrist_target_pose=left_pose)
    right = replace(base.right, root_pose=right_pose, wrist_target_pose=right_pose)
    base = replace(base, left=left, right=right)
    fixture.reset_manager.reset([0, 1], [base, base])
    print("CONTACT_STAGE base_reset", flush=True)
    for _ in range(2):
        fixture.step()
    no_contact = [fixture.monitor.summary(env) for env in (0, 1)]
    print("CONTACT_STAGE no_contact " + json.dumps(no_contact), flush=True)
    if any(k in summary for summary in no_contact
           for k in ("left_bottle", "right_cap", "left_cap", "right_bottle")):
        raise AssertionError(f"False positive in no-contact state: {no_contact}")

    cases = [
        ("left_bottle", fixture.left, fixture.specs[0].fingertip_bodies[1],
         fixture.bottle, (fixture.config.bottle_radius * 0.72, 0, 0)),
        ("right_cap", fixture.right, fixture.specs[1].fingertip_bodies[1],
         fixture.cap, (-fixture.config.cap_outer_radius * 0.72, 0, 0)),
        ("left_cap", fixture.left, fixture.specs[0].fingertip_bodies[1],
         fixture.cap, (fixture.config.cap_outer_radius * 0.72, 0, 0)),
        ("right_bottle", fixture.right, fixture.specs[1].fingertip_bodies[1],
         fixture.bottle, (-fixture.config.bottle_radius * 0.72, 0, 0)),
    ]
    result = {"no_contact": no_contact, "cases": {},
              "detection_force_floor_n": fixture.monitor.config.detection_force_n,
              "asset_commit": fixture.specs[0].source_commit,
              "asset_variant": fixture.specs[0].variant,
              "left_usd_sha256": fixture.specs[0].usd_sha256,
              "right_usd_sha256": fixture.specs[1].usd_sha256,
              "contact_placement": "episode_boundary_object_pair_translation_for_classification_only"}
    for category, hand, tip_body, obj, offset in cases:
        print("CONTACT_STAGE " + category, flush=True)
        fixture.reset_manager.reset([0, 1], [base, base])
        fixture.step()
        tip = hand.data.body_link_pose_w.torch[0, hand.body_names.index(tip_body), :3]
        object_pos = obj.data.root_pose_w.torch[0, :3]
        shift = tuple((tip - object_pos).detach().cpu().tolist()[i] + offset[i]
                      for i in range(3))
        print("CONTACT_SHIFT " + category + " " + str(shift), flush=True)
        sample = moved(base, shift)
        fixture.reset_manager.reset([0, 1], [sample, base])
        print("CONTACT_RESET " + category, flush=True)
        for _ in range(args.steps):
            fixture.step()
        print("CONTACT_STEPPED " + category, flush=True)
        env0 = fixture.monitor.summary(0)
        env1 = fixture.monitor.summary(1)
        result["cases"][category] = {"env0": env0.get(category),
                                      "env1": env1.get(category),
                                      "all_env0_categories": list(env0),
                                      "other_env_categories": list(env1)}
        if category not in env0:
            raise AssertionError(f"Missing {category} contact; measured {env0}")
        if category in env1:
            raise AssertionError(f"{category} leaked into env 1")
    # Align the two index distal links once at the episode boundary. This is
    # intentionally a classification test, not a valid grasp initialization.
    fixture.reset_manager.reset([0, 1], [base, base])
    fixture.step()
    left_tip = fixture.left.data.body_link_pose_w.torch[
        0, fixture.left.body_names.index(fixture.specs[0].fingertip_bodies[1]), :3]
    right_tip = fixture.right.data.body_link_pose_w.torch[
        0, fixture.right.body_names.index(fixture.specs[1].fingertip_bodies[1]), :3]
    shift = (left_tip - right_tip).detach().cpu().tolist()
    shift[0] += 0.03  # distal surfaces approach; avoid centre-on-centre overlap
    old_pose = base.right.root_pose
    new_pose = tuple(old_pose[i] + shift[i] for i in range(3)) + old_pose[3:7]
    overlap = replace(base, right=replace(base.right, root_pose=new_pose,
                                          wrist_target_pose=new_pose))
    fixture.reset_manager.reset([0, 1], [overlap, base])
    for _ in range(args.steps):
        fixture.step()
    hand0, hand1 = fixture.monitor.summary(0), fixture.monitor.summary(1)
    result["cases"]["hand_hand"] = {"env0": hand0.get("hand_hand"),
                                      "env1": hand1.get("hand_hand")}
    if "hand_hand" not in hand0 or "hand_hand" in hand1:
        raise AssertionError("Hand-hand contact was missed or leaked into env 1")
    # Detach only in this classification fixture to make the cap touch ground
    # without the bottle's D6 constraint pulling it back to the neck.
    fixture.reset_manager.reset([0, 1], [base, base])
    fixture.couplers[0]._detached = True
    fixture.joints[0].GetJointEnabledAttr().Set(False)
    fixture.joints[0].CreateCollisionEnabledAttr(True)
    cap_pose = fixture.cap.data.root_pose_w.torch[0:1].clone()
    cap_pose[:, 2] = fixture.config.cap_height * 0.5 - 0.001
    fixture.cap.write_root_pose_to_sim_index(
        root_pose=cap_pose,
        env_ids=torch.tensor([0], device=fixture.cap.device, dtype=torch.long),
    )
    fixture.cap.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((1, 6), device=fixture.cap.device),
        env_ids=torch.tensor([0], device=fixture.cap.device, dtype=torch.long),
    )
    for _ in range(args.steps):
        fixture.step()
    cap0, cap1 = fixture.monitor.summary(0), fixture.monitor.summary(1)
    result["cases"]["cap_ground"] = {"env0": cap0.get("cap_ground"),
                                       "env1": cap1.get("cap_ground")}
    if "cap_ground" not in cap0 or "cap_ground" in cap1:
        raise AssertionError("Cap-ground contact was missed or leaked into env 1")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("BIMANUAL_CONTACT_RESULT " + json.dumps(result, sort_keys=True), flush=True)


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
