"""Verify the no-hand-contact USD retains the complete robot physics contract."""

import argparse
import hashlib
import json
from pathlib import Path

from dexgym.assets.marvin_wuji.robot_cfg import NO_HAND_CONTACT_USD, PACKAGED_USD
from pxr import Usd, UsdGeom, UsdPhysics


def hand_path(path: str) -> bool:
    return "l_mount" in path.split("/") or "r_mount" in path.split("/")


def prim_signature(prim: Usd.Prim) -> dict:
    """Record all physics attributes/relations on a rigid body or joint."""
    return {
        "type": prim.GetTypeName(),
        "attributes": {attr.GetName(): str(attr.Get()) for attr in prim.GetAttributes()},
        "relationships": {rel.GetName(): [str(path) for path in rel.GetTargets()] for rel in prim.GetRelationships()},
    }


def inspect(full_usd: Path, no_hand_usd: Path) -> dict:
    full = Usd.Stage.Open(str(full_usd.resolve()))
    no_hand = Usd.Stage.Open(str(no_hand_usd.resolve()))
    if not full or not no_hand or not full.GetDefaultPrim().IsValid() or not no_hand.GetDefaultPrim().IsValid():
        raise ValueError("Both USD stages must open with a default prim")
    if full.GetDefaultPrim().GetPath() != no_hand.GetDefaultPrim().GetPath():
        raise ValueError("The articulation root changed")
    if (UsdGeom.GetStageUpAxis(full), UsdGeom.GetStageMetersPerUnit(full)) != (
        UsdGeom.GetStageUpAxis(no_hand),
        UsdGeom.GetStageMetersPerUnit(no_hand),
    ):
        raise ValueError("The stage coordinate convention changed")
    full_physics = {
        str(prim.GetPath()): prim_signature(prim)
        for prim in full.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint)
    }
    no_hand_physics = {
        str(prim.GetPath()): prim_signature(prim)
        for prim in no_hand.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint)
    }
    if full_physics != no_hand_physics:
        changed = [
            path
            for path in full_physics.keys() | no_hand_physics.keys()
            if full_physics.get(path) != no_hand_physics.get(path)
        ]
        raise ValueError(f"Rigid body or joint physics changed: {changed[:10]}")
    full_hand_shapes = [
        prim for prim in full.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI) and hand_path(str(prim.GetPath()))
    ]
    no_hand_collisions = [prim for prim in no_hand.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
    hand_collisions = [prim for prim in no_hand_collisions if hand_path(str(prim.GetPath()))]
    arm_collisions = [prim for prim in no_hand_collisions if not hand_path(str(prim.GetPath()))]
    if len(full_hand_shapes) != 74 or len(hand_collisions) != 74 or len(arm_collisions) != 53:
        raise ValueError("The expected 74 hand and 53 arm/visual collision geometries were not preserved")
    if any(prim.GetAttribute("physics:collisionEnabled").Get() is not False for prim in hand_collisions):
        raise ValueError("A hand collision mesh is still enabled")
    for prim in arm_collisions:
        original = full.GetPrimAtPath(prim.GetPath())
        if (
            prim.GetAttribute("physics:collisionEnabled").Get()
            != original.GetAttribute("physics:collisionEnabled").Get()
        ):
            raise ValueError("An arm/support collision flag changed")
    enabled_arms = sum(prim.GetAttribute("physics:collisionEnabled").Get() is not False for prim in arm_collisions)
    if enabled_arms != 33:
        raise ValueError("Expected 33 enabled arm/support collision shapes")
    rigid_count = sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in no_hand.Traverse())
    revolute_count = sum(prim.IsA(UsdPhysics.RevoluteJoint) for prim in no_hand.Traverse())
    fixed_count = sum(prim.IsA(UsdPhysics.FixedJoint) for prim in no_hand.Traverse())
    if (rigid_count, revolute_count, fixed_count) != (74, 54, 20):
        raise ValueError("Robot bodies, DoFs, or fixed mounts changed")
    return {
        "full_usd_sha256": hashlib.sha256(full_usd.read_bytes()).hexdigest(),
        "no_hand_usd_sha256": hashlib.sha256(no_hand_usd.read_bytes()).hexdigest(),
        "rigid_bodies": rigid_count,
        "revolute_joints": revolute_count,
        "fixed_joints": fixed_count,
        "hand_collision_shapes_disabled": len(hand_collisions),
        "arm_collision_shapes_enabled": enabled_arms,
        "rigid_mass_inertia_and_joint_properties_identical": True,
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-usd", type=Path, default=PACKAGED_USD)
    parser.add_argument("--no-hand-usd", type=Path, default=NO_HAND_CONTACT_USD)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.full_usd, args.no_hand_usd)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
