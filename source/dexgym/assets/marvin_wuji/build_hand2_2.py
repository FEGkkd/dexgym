"""Rebuild Marvin with Cube Hand2.2 physics and official chiral left geometry."""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from dexgym.assets.wuji_hand2_cube import ROOT as CUBE_ROOT
from dexgym.assets.wuji_hand2_cube.rig import (
    HAND2_REORIENT_CAPSULES,
    apply_hand2_reorient_actuator_gains,
    apply_hand2_reorient_collision_overlay,
    apply_hand2_reorient_force_limits,
    apply_hand2_reorient_physics,
)
from dexgym.assets.wuji_hand2_cube.source_model import WUJI_HAND2_TASK_NAMES, _get_spec

ROOT = Path(__file__).parent
OUT = ROOT / "hand2_2/generated"
LEGACY = ROOT / "source_model/dual_arm_marvin.xml"


def numbers(v):
    return " ".join(format(float(x), ".17g") for x in v)


def hand_spec(side):
    if side == "right":
        return _get_spec(CUBE_ROOT / "official/mjcf/right.xml")
    path = ROOT / "hand2_2/official/mjcf/left.xml"
    xml = ET.parse(path).getroot()
    xml.set("model", "wujihand2_2-right")
    xml.find("compiler").set("meshdir", str(path.parent.parent / "meshes/left"))
    # Reuse the strict name adapter while retaining official left geometry/axes.
    for node in xml.iter():
        for key, value in list(node.attrib.items()):
            if key != "file" and value.startswith("l_"):
                node.set(key, "r_" + value[2:])
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "right.xml"
        ET.ElementTree(xml).write(scratch, encoding="unicode")
        return _get_spec(scratch)


def prepared_hands():
    hands = {side: hand_spec(side) for side in ("right", "left")}
    models = {side: spec.compile() for side, spec in hands.items()}
    data = {side: mujoco.MjData(model) for side, model in models.items()}
    for side in hands:
        mujoco.mj_forward(models[side], data[side])
    for side, spec in hands.items():
        apply_hand2_reorient_collision_overlay(spec)
        if side == "left":
            reflection = np.diag([1.0, -1.0, 1.0])
            for name, (endpoints, _) in HAND2_REORIENT_CAPSULES.items():
                right_id, left_id = models["right"].body(name).id, models["left"].body(name).id
                rd, ld = data["right"], data["left"]
                points = np.asarray(endpoints).reshape(2, 3)
                world = points @ rd.xmat[right_id].reshape(3, 3).T + rd.xpos[right_id]
                local = (world @ reflection - ld.xpos[left_id]) @ ld.xmat[left_id].reshape(3, 3)
                spec.geom(name + "_col").fromto = local.reshape(-1)
        apply_hand2_reorient_physics(spec)
        apply_hand2_reorient_actuator_gains(spec)
        apply_hand2_reorient_force_limits(spec)
    return hands


def build_xml():
    OUT.mkdir(parents=True, exist_ok=True)
    meshes = OUT / "meshes"
    meshes.mkdir(exist_ok=True)
    root = ET.parse(LEGACY).getroot()
    root.set("model", "marvin_m6_dual_arm_wujihand2_2")
    root.find("compiler").set("meshdir", "meshes")
    assets = root.find("asset")
    for mesh in list(assets.findall("mesh")):
        if mesh.get("name").startswith(("l_", "r_")):
            assets.remove(mesh)
        else:
            src = LEGACY.parent / "assets" / mesh.get("file")
            filename = mesh.get("name") + ".STL"
            shutil.copyfile(src, meshes / filename)
            mesh.set("file", filename)
    elem = root.find("sensor")
    if elem is not None:
        root.remove(elem)
    for section, attrs in [("actuator", ("joint",)), ("contact", ("body1", "body2"))]:
        elem = root.find(section)
        for node in list(elem):
            if any(node.get(a, "").startswith(("l_", "r_")) for a in attrs):
                elem.remove(node)
    mapping = {v: k for kind in ("body", "joint", "site", "actuator") for k, v in WUJI_HAND2_TASK_NAMES[kind].items()}
    for side, spec in prepared_hands().items():
        short = "r" if side == "right" else "l"
        suffix = short.upper()
        for idx, geom in enumerate(spec.geoms):
            if not geom.name:
                geom.name = f"right_visual_{idx}"
        model = spec.compile()
        xml = ET.fromstring(spec.to_xml())
        # Materialize defaults so the arm XML's defaults cannot leak into hands.
        for node in xml.findall(".//body/joint"):
            j = model.joint(node.get("name")).id
            d = model.jnt_dofadr[j]
            for key, value in [
                ("armature", model.dof_armature[d]),
                ("damping", model.dof_damping[d]),
                ("frictionloss", model.dof_frictionloss[d]),
            ]:
                node.set(key, str(value))
        for node in xml.findall(".//body/geom"):
            g = model.geom(node.get("name")).id
            for key, value in [
                ("friction", model.geom_friction[g]),
                ("solref", model.geom_solref[g]),
                ("solimp", model.geom_solimp[g]),
            ]:
                node.set(key, numbers(value))
            for key, value in [
                ("condim", model.geom_condim[g]),
                ("priority", model.geom_priority[g]),
                ("contype", model.geom_contype[g]),
                ("conaffinity", model.geom_conaffinity[g]),
                ("group", model.geom_group[g]),
            ]:
                node.set(key, str(int(value)))
        # Invert palm-rooting and retain the original TCP-to-mount transform.
        palm = xml.find("worldbody/body")
        mount = palm.find("body[@name='right_mount']")
        p = np.fromstring(mount.get("pos", "0 0 0"), sep=" ")
        q = np.fromstring(mount.get("quat", "1 0 0 0"), sep=" ")
        ip, iq = np.zeros(3), np.zeros(4)
        mujoco.mju_negPose(ip, iq, p, q / np.linalg.norm(q))
        palm.remove(mount)
        mount.append(palm)
        palm.set("pos", numbers(ip))
        palm.set("quat", numbers(iq))
        tcp = root.find(f".//body[@name='TCP_Link_{suffix}']")
        old = tcp.find(f"body[@name='{short}_mount']")
        mount.set("pos", old.get("pos", "0 0 0"))
        mount.set("quat", old.get("quat", "1 0 0 0"))
        tcp.remove(old)

        def rename(value):
            value = mapping.get(value, value)
            if value.startswith("right_"):
                value = "r_" + value[6:]
            if side == "left" and value.startswith("r_"):
                value = "l_" + value[2:]
            if value in ("matte_black", "metal_silver", "sensor_pad"):
                value = short + "_" + value
            return value

        for section in [mount, xml.find("asset"), xml.find("actuator"), xml.find("contact")]:
            if section is None:
                continue
            for node in section.iter():
                for key, value in list(node.attrib.items()):
                    if key in ("name", "mesh", "material", "joint", "body1", "body2", "site"):
                        node.set(key, rename(value))
        for mesh in xml.findall("asset/mesh"):
            old_file = Path(mesh.get("file")).name
            payload = next(v for k, v in spec.assets.items() if Path(k).name == old_file)
            filename = mesh.get("name") + ".STL"
            (meshes / filename).write_bytes(payload)
            mesh.set("file", filename)
        assets.extend(list(xml.find("asset")))
        tcp.append(mount)
        root.find("actuator").extend(list(xml.find("actuator")))
        root.find("contact").extend(list(xml.find("contact")))
    # Preserve ctrl order (arms L/R, hands L/R), independent of qpos order.
    acts = root.find("actuator")

    def order(n):
        j = n.get("joint")
        return (
            0
            if j.startswith("Joint") and j.endswith("_L")
            else 1
            if j.startswith("Joint")
            else 2
            if j.startswith("l_")
            else 3
        )

    acts[:] = sorted(list(acts), key=order)
    ET.indent(root)
    dest = OUT / "dual_arm_marvin.xml"
    ET.ElementTree(root).write(dest, encoding="unicode")
    m = mujoco.MjModel.from_xml_path(str(dest))
    # Preserve inferred fixed-base/support inertias too: otherwise mesh-based
    # inference by the USD importer changes their mass distribution.
    for node in root.findall(".//body"):
        b = m.body(node.get("name")).id
        if node.find("inertial") is None and m.body_mass[b] > 0:
            node.insert(
                0,
                ET.Element(
                    "inertial",
                    {
                        "mass": str(m.body_mass[b]),
                        "pos": numbers(m.body_ipos[b]),
                        "quat": numbers(m.body_iquat[b]),
                        "diaginertia": numbers(m.body_inertia[b]),
                    },
                ),
            )
    ET.indent(root)
    ET.ElementTree(root).write(dest, encoding="unicode")
    m = mujoco.MjModel.from_xml_path(str(dest))
    assert (m.nq, m.nv, m.nu) == (54, 54, 54)
    meta = {
        "source_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "joint_order": [m.joint(i).name for i in range(m.njnt)],
        "keyframes": {m.key(i).name: m.key_qpos[i].tolist() for i in range(m.nkey)},
        "drives": {
            m.joint(int(m.actuator_trnid[a, 0])).name: {
                "kp": float(m.actuator_gainprm[a, 0]),
                "kd": float(-m.actuator_biasprm[a, 2]),
                "effort": float(m.actuator_forcerange[a, 1]),
            }
            for a in range(m.nu)
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return dest


def export_usd(path, importer_root):
    if importer_root:
        namespace = types.ModuleType("isaacsim")
        namespace.__path__ = [str(importer_root)]
        sys.modules["isaacsim"] = namespace
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    with tempfile.TemporaryDirectory(prefix="dual_hand22_") as tmp:
        result = MjcfConverter(
            MjcfConverterCfg(
                asset_path=str(path.resolve()),
                usd_dir=tmp,
                fix_base=True,
                self_collision=True,
                force_usd_conversion=True,
            )
        )
        stage = Usd.Stage.Open(result.usd_path)
        stage.GetDefaultPrim().GetVariantSet("Physics").SetVariantSelection("mujoco")
        for p in list(stage.Traverse()):
            if p.IsInstance():
                p.SetInstanceable(False)
        for p in stage.Traverse():
            if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI):
                p.AddAppliedSchema("NewtonMeshCollisionAPI")
                p.CreateAttribute("newton:maxHullVertices", Sdf.ValueTypeNames.Int).Set(32768)
        stage.GetRootLayer().Save()
        target = OUT / "usd"
        target.mkdir(exist_ok=True)
        shutil.copytree(Path(result.usd_path).parent, target, dirs_exist_ok=True)
    base = OUT / "usd/dual_arm_marvin.usda"
    overlay = OUT / "usd/dual_arm_marvin_no_hand_contacts.usda"
    if overlay.exists():
        overlay.unlink()
    stage = Usd.Stage.CreateNew(str(overlay))
    stage.GetRootLayer().subLayerPaths = ["dual_arm_marvin.usda"]
    stage.SetDefaultPrim(stage.GetPrimAtPath(Usd.Stage.Open(str(base)).GetDefaultPrim().GetPath()))
    original = Usd.Stage.Open(str(base))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(original))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(original))
    stage.SetMetadata("kilogramsPerUnit", original.GetMetadata("kilogramsPerUnit"))
    disabled = 0
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.CollisionAPI) and any(x in str(p.GetPath()).split("/") for x in ("l_mount", "r_mount")):
            UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
            disabled += 1
    stage.GetRootLayer().Save()
    print(json.dumps({"usd": str(base), "disabled_in_optional_overlay": disabled}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--importer-root", type=Path)
    parser.add_argument("--xml-only", action="store_true")
    args = parser.parse_args()
    path = build_xml()
    if not args.xml_only:
        export_usd(path, args.importer_root)
    print(path)


if __name__ == "__main__":
    main()
