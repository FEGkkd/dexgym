"""Author a floating, hard-pitch PhysX bottle articulation in an existing scene.

The bottle and cap are already separate dynamic USD bodies. This authoring
adds a lightweight prismatic carriage, a revolute cap joint, and a PhysX
mimic relation. No object is attached to the world frame. The caller must
exclude old D6 joints and explicit screw wrenches for this assembly.
"""

from __future__ import annotations

import math
from typing import Sequence

from .config import BottleConfig


def author_mimic_thread(stage, env_path: str, config: BottleConfig,
                        *, cap_local_pose_xyzw: Sequence[float]):
    """Turn one existing Bottle/Cap pair into a floating screw articulation.

    ``cap_local_pose_xyzw`` is the cap's initial pose relative to its
    ``env_path`` parent. Isaac Lab scene spawners accept XYZW, while the USD
    orientation op requires a Gf quaternion with scalar first. This function
    does not move either already spawned physical body.
    """
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    if (len(cap_local_pose_xyzw) != 7 or
            not all(math.isfinite(float(value)) for value in cap_local_pose_xyzw)):
        raise ValueError("cap_local_pose_xyzw must be finite xyz+xyzw")
    bottle_path, cap_path = f"{env_path}/Bottle", f"{env_path}/Cap"
    for path in (bottle_path, cap_path):
        if not stage.GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Mimic joint endpoint is not a dynamic rigid body: {path}")
    if stage.GetPrimAtPath(f"{env_path}/BottleThreadJoint"):
        raise ValueError("Cannot install mimic screw alongside the legacy D6 joint")

    bottle = stage.GetPrimAtPath(bottle_path)
    UsdPhysics.ArticulationRootAPI.Apply(bottle)
    # The generated example's internal/external ridge capsules overlap at
    # initial engagement. The screw constraint replaces their internal
    # normal coupling; external hand/object collision remains active.
    PhysxSchema.PhysxArticulationAPI.Apply(bottle).CreateEnabledSelfCollisionsAttr(False)
    carriage_path = f"{env_path}/Carriage"
    carriage = UsdGeom.Xform.Define(stage, carriage_path)
    carriage.AddTranslateOp().Set(Gf.Vec3f(*cap_local_pose_xyzw[:3]))
    qx, qy, qz, qw = cap_local_pose_xyzw[3:]
    carriage.AddOrientOp().Set(Gf.Quatf(qw, qx, qy, qz))
    carriage_prim = carriage.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(carriage_prim).CreateRigidBodyEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(carriage_prim)
    mass.CreateMassAttr(0.0001)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-7, 1e-7, 1e-7))

    slide = UsdPhysics.PrismaticJoint.Define(stage, f"{env_path}/SlideJoint")
    slide.CreateBody0Rel().SetTargets([Sdf.Path(bottle_path)])
    slide.CreateBody1Rel().SetTargets([Sdf.Path(carriage_path)])
    slide.CreateAxisAttr(UsdPhysics.Tokens.z)
    slide.CreateLocalPos0Attr(Gf.Vec3f(0, 0, config.cap_initial_z))
    slide.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    travel = config.thread_pitch * config.thread_turns
    slide.CreateLowerLimitAttr(-travel)
    slide.CreateUpperLimitAttr(travel)
    slide.CreateCollisionEnabledAttr(False)

    spin = UsdPhysics.RevoluteJoint.Define(stage, f"{env_path}/SpinJoint")
    spin.CreateBody0Rel().SetTargets([Sdf.Path(carriage_path)])
    spin.CreateBody1Rel().SetTargets([Sdf.Path(cap_path)])
    spin.CreateAxisAttr(UsdPhysics.Tokens.z)
    spin.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    spin.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    spin.CreateLowerLimitAttr(-360 * config.thread_turns)
    spin.CreateUpperLimitAttr(360 * config.thread_turns)
    spin.CreateCollisionEnabledAttr(False)
    mimic = PhysxSchema.PhysxMimicJointAPI.Apply(slide.GetPrim(), UsdPhysics.Tokens.rotX)
    mimic.GetReferenceJointRel().SetTargets([spin.GetPath()])
    mimic.GetReferenceJointAxisAttr().Set(UsdPhysics.Tokens.rotZ)
    mimic.GetGearingAttr().Set(-config.thread_sign * config.thread_pitch / 360)
    mimic.GetOffsetAttr().Set(0.0)

    drive = UsdPhysics.DriveAPI.Apply(spin.GetPrim(), UsdPhysics.Tokens.angular)
    drive.CreateTypeAttr(UsdPhysics.Tokens.force)
    # OpenUSD angular drives use degrees; BottleConfig gains use radians.
    drive.CreateStiffnessAttr(config.breakaway_stiffness * math.pi / 180)
    drive.CreateDampingAttr(config.breakaway_damping * math.pi / 180)
    drive.CreateMaxForceAttr(config.breakaway_torque)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    return slide, spin
