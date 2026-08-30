"""Inspect the prim and physics structure of a native Wuji USD asset."""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402


def main() -> None:
    stage = Usd.Stage.Open(os.path.abspath(args.usd))
    if stage is None:
        raise RuntimeError(f"Unable to open {args.usd}")
    stage.Load()
    print(f"USD default={stage.GetDefaultPrim().GetPath()}", flush=True)
    mesh = stage.GetPrimAtPath("/colliders/right_finger2_tip_link/right_finger2_tip_link/mesh")
    if mesh.IsValid() and mesh.IsA(UsdGeom.Mesh):
        points = list(UsdGeom.Mesh(mesh).GetPointsAttr().Get())
        point_min = tuple(min(point[index] for point in points) for index in range(3))
        point_max = tuple(max(point[index] for point in points) for index in range(3))
        matrix = UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default)
        translation = matrix.ExtractTranslation()
        print(
            "MESH probe=distal "
            f"points_min={point_min} points_max={point_max} "
            f"translation=({translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f})",
            flush=True,
        )
    for probe_path in (
        "/wujihand_right/right_finger2_tip_link",
        "/wujihand_right/right_finger2_tip_link/collisions",
        "/colliders/right_finger2_tip_link/right_finger2_tip_link/mesh",
    ):
        probe = stage.GetPrimAtPath(probe_path)
        print(
            f"PROBE path={probe_path} valid={probe.IsValid()} type={probe.GetTypeName()} "
            f"instance={probe.IsInstance()} proxy={probe.IsInstanceProxy()} "
            f"children={len(probe.GetAllChildren())}",
            flush=True,
        )
        if probe.IsValid():
            print(
                f"PROBE_STACK path={probe_path} "
                f"stack={[str(item.layer.identifier) + ':' + str(item.path) for item in probe.GetPrimStack()]}",
                flush=True,
            )
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.ArticulationRootAPI) or "collision" in prim.GetPath().pathString.lower() or prim.IsA(UsdGeom.Gprim):
            print(
                f"PHYS path={prim.GetPath()} type={prim.GetTypeName()} "
                f"apis={prim.GetAppliedSchemas()} children={len(prim.GetChildren())}",
                flush=True,
            )

        if "collision" in prim.GetPath().pathString.lower() and prim.IsA(UsdGeom.Xformable):
            translation = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default
            ).ExtractTranslation()
            print(
                f"COLLISION path={prim.GetPath()} type={prim.GetTypeName()} "
                f"translation=({translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f})",
                flush=True,
            )


try:
    main()
finally:
    app.close()
