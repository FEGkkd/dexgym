"""Audit official WujiHand2 Beta MJCF mappings against each native Isaac USD.

Run with Isaac Sim Python; this does not start Kit or load a GPU scene.
"""

from __future__ import annotations

import json
from hashlib import sha256
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source"))

from pxr import Usd, UsdPhysics  # noqa: E402
from dexteroushands_lab.assets.wuji_hand2_beta import load_hand2_spec  # noqa: E402


def inspect(variant: str, side: str):
    spec = load_hand2_spec(variant=variant, side=side)
    stage = Usd.Stage.Open(str(spec.usd_path))
    root = stage.GetDefaultPrim()
    actual_joints = {
        prim.GetName(): UsdPhysics.RevoluteJoint(prim)
        for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)
    }
    if set(actual_joints) != set(spec.joint_names):
        raise AssertionError(f"USD joint names differ from MJCF: {variant}/{side}")
    for joint in spec.joints:
        usd_joint = actual_joints[joint.name]
        lower = math.radians(usd_joint.GetLowerLimitAttr().Get())
        upper = math.radians(usd_joint.GetUpperLimitAttr().Get())
        if max(abs(lower - joint.lower), abs(upper - joint.upper)) > 0.002:
            raise AssertionError(f"USD/MJCF limits differ: {joint.name}")
        drive = UsdPhysics.DriveAPI.Get(usd_joint.GetPrim(), "angular")
        if abs(drive.GetMaxForceAttr().Get() - joint.usd_drive_max_force) > 1e-5:
            raise AssertionError(f"USD drive max force changed: {joint.name}")
    if not stage.GetPrimAtPath(str(root.GetPath()) + "/root_joint").HasAPI(
        UsdPhysics.ArticulationRootAPI
    ):
        raise AssertionError("Official root_joint structure changed")
    for name in (spec.root_body, *spec.fingertip_bodies):
        prim = stage.GetPrimAtPath(str(root.GetPath()) + "/" + name)
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise AssertionError(f"USD rigid body missing: {name}")
    return {
        "variant": variant, "side": side, "usd": str(spec.usd_path),
        "source_commit": spec.source_commit, "commit_verified": spec.commit_verified,
        "usd_sha256": spec.usd_sha256,
        "asset_tree_sha256": spec.asset_tree_sha256,
        "joint_count": len(spec.joints), "body_count_mjcf": len(spec.body_names),
        "fingertips": spec.fingertip_bodies,
        "collision_exclusion_count_mjcf": len(spec.collision_exclusions),
        "usd_drive_max_force_nm": sorted(set(j.usd_drive_max_force for j in spec.joints)),
        "mjcf_effort_limits_nm": sorted(set(j.mjcf_effort_limit for j in spec.joints)),
        "joint_order": spec.joint_names,
        "actuator_to_joint": {j.actuator: j.name for j in spec.joints},
    }


def compare_beta_mjcf(root: Path, side: str):
    """Report factual mechanical differences without inferring hardware revision."""
    snapshots = {}
    for variant in ("beta1", "beta2"):
        path = root / "hand2" / f"hand2_{variant}" / "body/mjcf" / f"{side}.xml"
        tree = ET.parse(path)
        body_mass = {}
        for body in tree.findall(".//body[@name]"):
            inertial = body.find("inertial")
            if inertial is not None:
                body_mass[body.attrib["name"]] = float(inertial.attrib["mass"])
        joints = {
            joint.attrib["name"]: {
                "axis": joint.attrib.get("axis"),
                "range": joint.attrib.get("range"),
                "actuatorfrcrange": joint.attrib.get("actuatorfrcrange"),
            }
            for joint in tree.findall(".//joint[@name]")
        }
        snapshots[variant] = {
            "body_mass": body_mass,
            "joints": joints,
            "collision_geoms": {
                body.attrib["name"]: sorted(
                    (geom.attrib.get("type"), geom.attrib.get("mesh"))
                    for geom in body.findall("geom")
                    if geom.attrib.get("group") == "2"
                )
                for body in tree.findall(".//body[@name]")
            },
            "mesh_hashes": {
                mesh.name: sha256(mesh.read_bytes()).hexdigest()
                for mesh in (root / "hand2" / f"hand2_{variant}" / "body/meshes" / side).glob("*.STL")
            },
            "collision_exclusions": sorted(
                (e.attrib["body1"], e.attrib["body2"])
                for e in tree.findall("./contact/exclude")
            ),
        }
    one, two = snapshots["beta1"], snapshots["beta2"]
    common_bodies = one["body_mass"].keys() & two["body_mass"].keys()
    common_meshes = one["mesh_hashes"].keys() & two["mesh_hashes"].keys()
    mount_masses = {}
    for variant in ("beta1", "beta2"):
        mount_tree = ET.parse(
            root / "hand2" / f"hand2_{variant}" / "body/mjcf" / f"{side}_with_mount.xml"
        )
        mount = mount_tree.find(f".//body[@name='{side[0]}_mount']/inertial")
        mount_masses[variant] = None if mount is None else float(mount.attrib["mass"])
    return {
        "side": side,
        "beta1_body_count": len(one["body_mass"]),
        "beta2_body_count": len(two["body_mass"]),
        "added_bodies_beta2": sorted(two["body_mass"].keys() - one["body_mass"].keys()),
        "removed_bodies_beta2": sorted(one["body_mass"].keys() - two["body_mass"].keys()),
        "changed_common_body_masses": {
            name: [one["body_mass"][name], two["body_mass"][name]]
            for name in sorted(common_bodies)
            if abs(one["body_mass"][name] - two["body_mass"][name]) > 1e-10
        },
        "changed_joint_limits_axes_efforts": {
            name: [one["joints"].get(name), two["joints"].get(name)]
            for name in sorted(one["joints"].keys() | two["joints"].keys())
            if one["joints"].get(name) != two["joints"].get(name)
        },
        "collision_exclusions_equal": one["collision_exclusions"] == two["collision_exclusions"],
        "changed_collision_owners": sorted(
            name for name in one["collision_geoms"].keys() & two["collision_geoms"].keys()
            if one["collision_geoms"][name] != two["collision_geoms"][name]
        ),
        "mesh_count_beta1": len(one["mesh_hashes"]),
        "mesh_count_beta2": len(two["mesh_hashes"]),
        "added_meshes_beta2": sorted(two["mesh_hashes"].keys() - one["mesh_hashes"].keys()),
        "removed_meshes_beta2": sorted(one["mesh_hashes"].keys() - two["mesh_hashes"].keys()),
        "changed_common_meshes": sorted(
            name for name in common_meshes
            if one["mesh_hashes"][name] != two["mesh_hashes"][name]
        ),
        "with_mount_mass_kg": mount_masses,
        "beta1_total_mass_kg": sum(one["body_mass"].values()),
        "beta2_total_mass_kg": sum(two["body_mass"].values()),
    }


def main():
    import os

    root = Path(os.environ["WUJI_DESCRIPTION_ROOT"])
    output = {
        "assets": [inspect(v, s) for v in ("beta1", "beta2") for s in ("left", "right")],
        "beta_comparison": [compare_beta_mjcf(root, s) for s in ("left", "right")],
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
