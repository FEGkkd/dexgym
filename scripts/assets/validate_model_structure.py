"""Check external MuJoCo and converted USD structure without running physics."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from dexgym.assets.marvin_wuji.joint_mapping import GROUPS, inspect_joint_contract
from dexgym.assets.marvin_wuji.robot_cfg import SOURCE_SHA256, resolve_source_mjcf
from pxr import Usd, UsdGeom, UsdPhysics


def inspect(mjcf: Path, usd: Path) -> dict:
    source_hash = hashlib.sha256(mjcf.read_bytes()).hexdigest()
    if source_hash != SOURCE_SHA256:
        raise ValueError(f"Source SHA256 mismatch: {source_hash}")
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    contract = inspect_joint_contract(model)
    stage = Usd.Stage.Open(str(usd))
    if not stage:
        raise ValueError(f"Cannot open USD: {usd}")
    joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)]
    roots = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    fixed = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.FixedJoint)]
    rigid = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    plain_collisions = sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in stage.Traverse())
    instanced_collisions = sum(
        any(child.HasAPI(UsdPhysics.CollisionAPI) for child in prim.GetPrototype().GetChildren())
        for prim in stage.Traverse()
        if prim.IsInstance()
    )
    usd_collisions = plain_collisions + instanced_collisions
    source_collisions = int(np.count_nonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
    filtered_pairs = sum(
        len(UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI)
    )
    names = [prim.GetName() for prim in joints]
    if len(joints) != 54 or len(set(names)) != 54 or set(names) != set(contract.names):
        raise ValueError("USD does not retain the 54 unique source joints")
    if len(roots) != 1 or len(rigid) < 54 or not fixed:
        raise ValueError("USD articulation root, fixed base, or rigid body structure is missing")
    root_self_collision = roots[0].GetAttribute("newton:selfCollisionEnabled").Get()
    if root_self_collision is not True:
        raise ValueError(f"Newton whole-body self collision must be enabled: {root_self_collision}")
    if usd_collisions != source_collisions or filtered_pairs != model.nexclude:
        raise ValueError(
            f"Collision count/filter mismatch: USD {usd_collisions}/{filtered_pairs}, "
            f"MuJoCo {source_collisions}/{model.nexclude}"
        )
    if UsdGeom.GetStageUpAxis(stage) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError("USD must use Z-up and meter units")
    variant = stage.GetDefaultPrim().GetVariantSets().GetVariantSet("Physics")
    if variant.GetVariantSelection() != "physics":
        raise ValueError("USD Physics variant is not active")
    invalid_mass = []
    invalid_inertia = []
    invalid_drive = []
    limit_errors = []
    parameter_errors = []
    massless_fixed_markers = []
    for prim in rigid:
        mass_api = UsdPhysics.MassAPI(prim)
        mass = mass_api.GetMassAttr().Get()
        inertia = mass_api.GetDiagonalInertiaAttr().Get()
        if mass is None or inertia is None:
            if prim.GetName() not in {"center_mount", "TCP_Link_L", "TCP_Link_R"}:
                invalid_mass.append(prim.GetName())
            else:
                massless_fixed_markers.append(prim.GetName())
        elif mass <= 0 or min(inertia) <= 0:
            invalid_mass.append(prim.GetName())
    for prim in joints:
        name = prim.GetName()
        joint = UsdPhysics.RevoluteJoint(prim)
        targets = joint.GetBody1Rel().GetTargets()
        body = stage.GetPrimAtPath(targets[0]) if targets else None
        mass_api = UsdPhysics.MassAPI(body) if body else None
        mass = mass_api.GetMassAttr().Get() if mass_api else None
        inertia = mass_api.GetDiagonalInertiaAttr().Get() if mass_api else None
        if mass is None or mass <= 0:
            invalid_mass.append(name)
        if inertia is None or min(inertia) <= 0:
            invalid_inertia.append(name)
        source_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.GetName())
        if source_body_id < 0 or not np.isclose(mass, model.body_mass[source_body_id], rtol=1e-5):
            parameter_errors.append(f"{name}: mass")
        if (
            inertia is None
            or source_body_id < 0
            or not np.allclose(inertia, model.body_inertia[source_body_id], rtol=1e-5, atol=1e-10)
        ):
            parameter_errors.append(f"{name}: inertia")
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive or (drive.GetStiffnessAttr().Get() or 0) <= 0 or (drive.GetMaxForceAttr().Get() or 0) <= 0:
            invalid_drive.append(name)
        actuator_id = contract.ctrl_by_name[name]
        imported_drive = [
            drive.GetStiffnessAttr().Get(),
            drive.GetDampingAttr().Get(),
            drive.GetMaxForceAttr().Get(),
        ]
        source_drive = [
            model.actuator_gainprm[actuator_id, 0],
            -model.actuator_biasprm[actuator_id, 2],
            model.actuator_forcerange[actuator_id, 1],
        ]
        if not np.allclose(imported_drive, source_drive, rtol=1e-5, atol=1e-5):
            parameter_errors.append(f"{name}: drive")
        source_lower, source_upper = contract.limits_by_name[name]
        if not np.allclose(
            [joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()],
            np.rad2deg([source_lower, source_upper]),
            atol=0.02,
        ):
            limit_errors.append(name)
    if invalid_mass or invalid_inertia or invalid_drive or limit_errors or parameter_errors:
        raise ValueError(
            f"USD invalid mass={invalid_mass}, inertia={invalid_inertia}, drive={invalid_drive}, "
            f"limits={limit_errors}, parameters={parameter_errors}"
        )
    return {
        "source_mjcf_sha256": source_hash,
        "mujoco_nq": model.nq,
        "mujoco_nu": model.nu,
        "mujoco_bodies": model.nbody - 1,
        "mujoco_meshes": model.nmesh,
        "usd_revolute_joints": len(joints),
        "usd_rigid_bodies": len(rigid),
        "usd_fixed_joints": len(fixed),
        "massless_fixed_markers": massless_fixed_markers,
        "source_collision_geometries": source_collisions,
        "usd_collision_geometries": usd_collisions,
        "usd_instanced_collision_meshes": instanced_collisions,
        "source_excluded_pairs": model.nexclude,
        "usd_filtered_pairs": filtered_pairs,
        "usd_articulation_root": str(roots[0].GetPath()),
        "newton_self_collision_enabled": root_self_collision,
        "control_groups": {group: list(contract.groups[group]) for group in GROUPS},
        "qpos_ctrl_order_equal": all(contract.qpos_by_name[n] == contract.ctrl_by_name[n] for n in contract.names),
        "usd_up_axis": "Z",
        "usd_meters_per_unit": 1.0,
        "invalid_mass": invalid_mass,
        "invalid_inertia": invalid_inertia,
        "invalid_drive": invalid_drive,
        "limit_errors": limit_errors,
        "parameter_errors": parameter_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf")
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(resolve_source_mjcf(args.mjcf), args.usd.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
