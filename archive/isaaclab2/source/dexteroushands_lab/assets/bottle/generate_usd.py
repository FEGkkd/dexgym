"""Reproducibly author compact OpenUSD bottle/cap assets with primitive collision.

Run with Isaac Sim's Python. The helical ridge is made from short capsule
colliders, not one expensive triangle-mesh collider.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

from .config import BottleConfig, sample_helix


def _material(stage, root: str, cfg: BottleConfig):
    from pxr import UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, root + "/PlasticContact")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(cfg.static_friction)
    api.CreateDynamicFrictionAttr(cfg.dynamic_friction)
    api.CreateRestitutionAttr(0.0)
    return material


def _bind_collision(prim, material):
    from pxr import UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")


def _cylinder(stage, path, radius, height, z, color, material, collision=True,
              visible=True):
    from pxr import Gf, UsdGeom

    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    cylinder.AddTranslateOp().Set(Gf.Vec3d(0, 0, z))
    cylinder.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collision:
        _bind_collision(cylinder.GetPrim(), material)
    if not visible:
        cylinder.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    return cylinder


def _round_mesh(stage, path, outer_radius, inner_radius, height, z, color,
                segments):
    """Author a round visual shell; physics uses separate primitive colliders."""
    from pxr import Gf, UsdGeom

    if not 0 <= inner_radius < outer_radius:
        raise ValueError("Visual shell radii are invalid")
    mesh = UsdGeom.Mesh.Define(stage, path)
    rings = (outer_radius,) if inner_radius == 0 else (outer_radius, inner_radius)
    points = []
    for radius in rings:
        for height_sign in (-1, 1):
            for i in range(segments):
                angle = 2 * math.pi * i / segments
                points.append(Gf.Vec3f(
                    radius * math.cos(angle), radius * math.sin(angle),
                    z + height_sign * height / 2,
                ))
    indices = []
    counts = []

    def quad(a, b, c, d):
        counts.append(4)
        indices.extend((a, b, c, d))

    for i in range(segments):
        j = (i + 1) % segments
        quad(i, j, segments + j, segments + i)
        if inner_radius:
            inner = 2 * segments
            quad(inner + j, inner + i, inner + segments + i,
                 inner + segments + j)
            quad(segments + i, segments + j,
                 inner + segments + j, inner + segments + i)
            quad(j, i, inner + i, inner + j)
    if inner_radius == 0:
        bottom_center = len(points)
        points.append(Gf.Vec3f(0, 0, z - height / 2))
        top_center = len(points)
        points.append(Gf.Vec3f(0, 0, z + height / 2))
        for i in range(segments):
            j = (i + 1) % segments
            counts.extend((3, 3))
            indices.extend((bottom_center, j, i,
                            top_center, segments + i, segments + j))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return mesh


def _thread_capsules(stage, root, points, radius, material, color):
    from pxr import Gf, UsdGeom

    for i, (start, end) in enumerate(zip(points, points[1:])):
        delta = Gf.Vec3d(*(b - a for a, b in zip(start, end)))
        midpoint = Gf.Vec3d(*(0.5 * (a + b) for a, b in zip(start, end)))
        length = delta.GetLength()
        capsule = UsdGeom.Capsule.Define(stage, f"{root}/ridge_{i:02d}")
        capsule.CreateRadiusAttr(radius)
        capsule.CreateHeightAttr(max(0.0, length - 2 * radius))
        capsule.AddTranslateOp().Set(midpoint)
        capsule.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Rotation(Gf.Vec3d(0, 0, 1), delta).GetQuat()
        )
        capsule.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        _bind_collision(capsule.GetPrim(), material)


def _cap_shell(stage, root, cfg: BottleConfig, material):
    from pxr import Gf, UsdGeom

    radial_center = (cfg.cap_inner_radius + cfg.cap_outer_radius) / 2
    wall_thickness = cfg.cap_outer_radius - cfg.cap_inner_radius
    segments = 12
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        wall = UsdGeom.Cube.Define(stage, f"{root}/wall_{i:02d}")
        wall.CreateSizeAttr(1.0)
        wall.AddTranslateOp().Set(Gf.Vec3d(
            radial_center * math.cos(angle), radial_center * math.sin(angle), 0,
        ))
        wall.AddRotateZOp().Set(math.degrees(angle) + 90.0)
        wall.AddScaleOp().Set(Gf.Vec3f(
            2 * cfg.cap_outer_radius * math.sin(math.pi / segments) * 1.05,
            wall_thickness, cfg.cap_height,
        ))
        wall.CreateDisplayColorAttr([Gf.Vec3f(0.92, 0.35, 0.10)])
        _bind_collision(wall.GetPrim(), material)
        wall.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    _cylinder(
        stage, root + "/top", cfg.cap_outer_radius, 0.002,
        cfg.cap_height / 2 - 0.001, (0.92, 0.35, 0.10), material,
        visible=False,
    )
    _round_mesh(stage, root + "/shell_visual", cfg.cap_outer_radius,
                cfg.cap_inner_radius, cfg.cap_height, 0.0,
                (0.92, 0.35, 0.10), cfg.visual_radial_segments)
    _round_mesh(stage, root + "/top_visual", cfg.cap_outer_radius,
                0.0, 0.002, cfg.cap_height / 2 - 0.001,
                (0.92, 0.35, 0.10), cfg.visual_radial_segments)


def _body(stage, path, mass, inertia):
    from pxr import Gf, UsdGeom, UsdPhysics

    root = UsdGeom.Xform.Define(stage, path)
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateRigidBodyEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(mass)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    return root


def _stage(path: Path, root: str):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, root).GetPrim())
    return stage


def build_bottle_usd(config: BottleConfig, output_dir: Path) -> tuple[Path, Path]:
    """Generate two separate dynamic rigid-body USD files and a parameter manifest.

    Source: https://openusd.org/release/api/usd_physics_page_front.html
    This creates geometry and collision, while ``ThreadCoupler`` supplies the
    physical screw relation with bounded, reciprocal wrenches at runtime.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bottle_path = output_dir / "bottle.usda"
    cap_path = output_dir / "cap.usda"

    bottle = _stage(bottle_path, "/Bottle")
    bottle_inertia_xy = config.bottle_mass * (
        3 * config.bottle_radius**2 + config.bottle_height**2
    ) / 12
    bottle_inertia_z = 0.5 * config.bottle_mass * config.bottle_radius**2
    _body(bottle, "/Bottle", config.bottle_mass,
          (bottle_inertia_xy, bottle_inertia_xy, bottle_inertia_z))
    bottle_material = _material(bottle, "/Bottle", config)
    _cylinder(
        bottle, "/Bottle/body", config.bottle_radius, config.bottle_height,
        0.0, (0.10, 0.50, 0.72), bottle_material, visible=False,
    )
    _round_mesh(bottle, "/Bottle/body_visual", config.bottle_radius,
                0.0, config.bottle_height, 0.0, (0.10, 0.50, 0.72),
                config.visual_radial_segments)
    _cylinder(
        bottle, "/Bottle/neck", config.neck_outer_radius, config.neck_height,
        config.bottle_height / 2 + config.neck_height / 2,
        (0.18, 0.65, 0.80), bottle_material, visible=False,
    )
    _round_mesh(bottle, "/Bottle/neck_visual", config.neck_outer_radius,
                0.0, config.neck_height,
                config.bottle_height / 2 + config.neck_height / 2,
                (0.18, 0.65, 0.80), config.visual_radial_segments)
    thread_start = config.bottle_height / 2 + config.neck_height
    if config.thread_sign > 0:
        thread_start -= config.thread_pitch * config.thread_turns
    external = sample_helix(
        config, radius=config.neck_outer_radius + config.thread_height / 2,
        start_z=thread_start,
    )
    _thread_capsules(
        bottle, "/Bottle/external_thread", external,
        config.thread_height / 2, bottle_material, (0.03, 0.32, 0.60),
    )
    bottle.GetRootLayer().Save()
    bottle_path.write_text(bottle_path.read_text(encoding="utf-8").rstrip() + "\n",
                           encoding="utf-8")

    cap = _stage(cap_path, "/Cap")
    cap_r2 = (config.cap_inner_radius**2 + config.cap_outer_radius**2) / 2
    cap_inertia_z = config.cap_mass * cap_r2
    cap_inertia_xy = config.cap_mass * (3 * cap_r2 + config.cap_height**2) / 12
    _body(cap, "/Cap", config.cap_mass,
          (cap_inertia_xy, cap_inertia_xy, cap_inertia_z))
    cap_material = _material(cap, "/Cap", config)
    _cap_shell(cap, "/Cap", config, cap_material)
    # Half-pitch phase offset places internal ridges between external ridges.
    internal = sample_helix(
        config, radius=config.cap_inner_radius - config.thread_height / 2,
        start_z=max(-config.cap_height / 2 + config.thread_height / 2,
                    thread_start - config.cap_initial_z
                    + config.thread_sign * config.thread_pitch / 2),
    )
    _thread_capsules(
        cap, "/Cap/internal_thread", internal,
        config.thread_height / 2, cap_material, (0.55, 0.13, 0.04),
    )
    cap.GetRootLayer().Save()
    cap_path.write_text(cap_path.read_text(encoding="utf-8").rstrip() + "\n",
                        encoding="utf-8")
    (output_dir / "parameters.json").write_text(
        json.dumps({"schema": "dexgym-bottle-example-v1", "units": "SI", "parameters": asdict(config)},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bottle_path, cap_path


def author_thread_joint(stage, env_path: str, config: BottleConfig):
    """Create a PhysX D6 joint between a free bottle and free cap.

    The joint locks lateral and swing motion while leaving axial translation
    and axial twist free. ``ThreadCoupler`` supplies pitch coupling and
    finite breakaway resistance through equal-and-opposite bounded wrenches.

    Source: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/joints.html#d6-joint
    """
    from pxr import Gf, Sdf, UsdPhysics

    bottle_path = env_path + "/Bottle"
    cap_path = env_path + "/Cap"
    for path in (bottle_path, cap_path):
        if not stage.GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Thread joint endpoint is not a rigid body: {path}")
    joint = UsdPhysics.Joint.Define(stage, env_path + "/BottleThreadJoint")
    joint.CreateJointEnabledAttr(True)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(bottle_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(cap_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, config.cap_initial_z))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    # The D6 twist axis is local X. Rotate X into bottle-local +Z.
    axis_frame = Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 1, 0), -90.0).GetQuat())
    joint.CreateLocalRot0Attr(axis_frame)
    joint.CreateLocalRot1Attr(axis_frame)
    for axis in (
        UsdPhysics.Tokens.transY, UsdPhysics.Tokens.transZ,
        UsdPhysics.Tokens.rotY, UsdPhysics.Tokens.rotZ,
    ):
        limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
        limit.CreateLowAttr(1.0)
        limit.CreateHighAttr(-1.0)
    travel = config.thread_pitch * config.thread_turns
    axial_limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), UsdPhysics.Tokens.transX)
    axial_limit.CreateLowAttr(-travel)
    axial_limit.CreateHighAttr(travel)
    return joint
