"""Rebuild bundled MJCF/USD assets with the kitless Isaac Lab converter.

Run inside the Isaac Lab runtime. --importer-root is optional and only needed
when an Isaac Sim installation shadows a separately installed kitless importer.
"""

import argparse
import json
import shutil
import sys
import tempfile
import types
import xml.etree.ElementTree as ET
from pathlib import Path

from . import ROOT
from .rig import get_hand2_reorient_spec
from .source_model import _get_spec, wuji_hand2_xml_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importer-root", type=Path, help="Directory containing the standalone isaacsim namespace")
    args = parser.parse_args()
    spec = get_hand2_reorient_spec(lambda: _get_spec(wuji_hand2_xml_path()))
    out = ROOT / "generated"
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    for key, data in spec.assets.items():
        (out / "meshes" / Path(key).name).write_bytes(data)
    xml = ET.fromstring(spec.to_xml())
    xml.find("compiler").set("meshdir", "meshes")
    for mesh in xml.findall("asset/mesh"):
        mesh.set("file", Path(mesh.get("file")).name)
    ET.ElementTree(xml).write(out / "hand2_cube_rig.xml", encoding="unicode")
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(out / "hand2_cube_rig.xml"))
    metadata = {
        "nq": m.nq,
        "nu": m.nu,
        "joint_names": [m.joint(i).name for i in range(m.njnt)],
        "joint_range": m.jnt_range.tolist(),
        "kp": m.actuator_gainprm[:, 0].tolist(),
        "kd": (-m.actuator_biasprm[:, 2]).tolist(),
        "force_limits": m.actuator_forcerange.tolist(),
    }
    (out / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    if args.importer_root:
        namespace = types.ModuleType("isaacsim")
        namespace.__path__ = [str(args.importer_root)]
        sys.modules["isaacsim"] = namespace
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    staging = tempfile.TemporaryDirectory(prefix="cube_export_")
    build = Path(staging.name)
    hand = MjcfConverter(
        MjcfConverterCfg(
            asset_path=str(out / "hand2_cube_rig.xml"),
            usd_dir=str(build / "usd"),
            fix_base=True,
            self_collision=True,
            force_usd_conversion=True,
        )
    )
    cube = MjcfConverter(
        MjcfConverterCfg(
            asset_path=str(ROOT / "cube/xmls/cube.xml"),
            usd_dir=str(build / "cube_usd"),
            fix_base=False,
            self_collision=False,
            force_usd_conversion=True,
        )
    )
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    # A standalone rigid cube must not carry an articulation root API.
    layer = Path(cube.usd_path).parent / "payloads/Physics/physics.usda"
    stage = Usd.Stage.Open(str(layer))
    prim = stage.GetDefaultPrim()
    prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    prim.RemoveAppliedSchema("NewtonArticulationRootAPI")
    stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(hand.usd_path)
    stage.GetDefaultPrim().GetVariantSet("Physics").SetVariantSelection("mujoco")
    for prim in list(stage.Traverse()):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema("NewtonMeshCollisionAPI")
            # Avoid Newton's default 64-vertex approximation of the source hulls.
            prim.CreateAttribute("newton:maxHullVertices", Sdf.ValueTypeNames.Int).Set(32768)
    stage.GetRootLayer().Save()
    # Build in an empty directory so the converter cannot append numeric suffixes.
    for name in ("usd", "cube_usd"):
        target = out / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(build / name, target)
    staging.cleanup()
    print(
        json.dumps(
            {
                "hand": str(out / "usd/hand2_cube_rig/hand2_cube_rig.usda"),
                "cube": str(out / "cube_usd/cube/cube.usda"),
                "joint_count": m.nq,
            }
        )
    )


if __name__ == "__main__":
    main()
