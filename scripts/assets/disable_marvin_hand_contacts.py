"""Create a USD layer that disables hand collisions without changing the robot model."""

import argparse
import os
from pathlib import Path

from dexgym.assets.marvin_wuji.robot_cfg import PACKAGED_USD
from pxr import Usd, UsdGeom, UsdPhysics


def is_hand_path(path: str) -> bool:
    parts = path.split("/")
    return "l_mount" in parts or "r_mount" in parts


def collision_inventory(stage: Usd.Stage) -> tuple[list, list, list]:
    """Return hand instance paths, direct arm collision prims and unexpected shapes."""
    hand_instances = []
    arm_collisions = []
    unexpected = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            (unexpected if is_hand_path(path) else arm_collisions).append(prim.GetPath())
        if prim.IsInstance():
            collision_children = [
                child for child in prim.GetPrototype().GetChildren() if child.HasAPI(UsdPhysics.CollisionAPI)
            ]
            if collision_children:
                (hand_instances if is_hand_path(path) and len(collision_children) == 1 else unexpected).append(
                    prim.GetPath()
                )
    return hand_instances, arm_collisions, unexpected


def create_overlay(base_usd: Path, output: Path) -> dict:
    base_usd = base_usd.expanduser().resolve()
    output = output.expanduser().resolve()
    if not base_usd.is_file() or output == base_usd:
        raise ValueError("Existing base USD and a separate output path are required")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing USD: {output}")
    source = Usd.Stage.Open(str(base_usd))
    if not source or not source.GetDefaultPrim().IsValid():
        raise ValueError(f"Invalid base USD: {base_usd}")
    hand_instances, arm_collisions, unexpected = collision_inventory(source)
    left = sum("l_mount" in str(path).split("/") for path in hand_instances)
    right = sum("r_mount" in str(path).split("/") for path in hand_instances)
    if (left, right, len(arm_collisions), len(unexpected)) != (27, 27, 33, 0):
        raise ValueError(
            f"Unexpected collision inventory: left={left}, right={right}, arm={len(arm_collisions)}, "
            f"other={len(unexpected)}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    stage.GetRootLayer().subLayerPaths.append(os.path.relpath(base_usd, output.parent))
    stage.SetDefaultPrim(stage.GetPrimAtPath(source.GetDefaultPrim().GetPath()))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source))
    stage.SetMetadata("kilogramsPerUnit", source.GetMetadata("kilogramsPerUnit"))
    for path in hand_instances:
        instance = stage.GetPrimAtPath(path)
        instance.SetInstanceable(False)
        collisions = [child for child in instance.GetChildren() if child.HasAPI(UsdPhysics.CollisionAPI)]
        if len(collisions) != 1:
            raise ValueError(f"Hand collision instance did not deinstance cleanly: {path}")
        UsdPhysics.CollisionAPI(collisions[0]).CreateCollisionEnabledAttr(False)
    stage.GetRootLayer().Save()
    return {"hand_collision_shapes_disabled": 54, "arm_collision_shapes_retained": 33, "usd": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-usd", type=Path, default=PACKAGED_USD)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(create_overlay(args.base_usd, args.output))


if __name__ == "__main__":
    main()
