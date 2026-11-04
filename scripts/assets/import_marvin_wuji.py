"""Convert the external, pinned Marvin/Wuji MJCF with Isaac Lab 3.0's importer."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from xml.etree import ElementTree

import mujoco
from dexgym.assets.marvin_wuji.robot_cfg import SOURCE_COMMIT, SOURCE_SHA256, resolve_source_mjcf
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mesh_dependencies(mjcf: Path) -> list[Path]:
    root = ElementTree.parse(mjcf).getroot()
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    dependencies = [mjcf.parent / meshdir / node.attrib["file"] for node in root.findall("./asset/mesh")]
    missing = [str(path) for path in dependencies if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} MJCF mesh resources: {missing[:5]}")
    return dependencies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", help="External dual_arm_marvin.xml; or set DEXGYM_MARVIN_MJCF")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/marvin_wuji/generated"))
    args = parser.parse_args()
    mjcf = resolve_source_mjcf(args.mjcf)
    observed_hash = sha256(mjcf)
    if observed_hash != SOURCE_SHA256:
        raise ValueError(f"Unpinned source MJCF SHA256 {observed_hash}; expected {SOURCE_SHA256}")
    meshes = mesh_dependencies(mjcf)
    output_dir = (args.output_dir / f"{observed_hash[:8]}-self-collision").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = MjcfConverterCfg(
        asset_path=str(mjcf),
        usd_dir=str(output_dir),
        force_usd_conversion=True,
        fix_base=True,
        merge_mesh=False,
        self_collision=True,
        run_asset_transformer=True,
        run_multi_physics_conversion=True,
    )
    usd = Path(MjcfConverter(cfg).usd_path)
    output_files = [p for p in usd.parent.rglob("*") if p.is_file()]
    source_model = mujoco.MjModel.from_xml_path(str(mjcf))
    keyframes = {
        mujoco.mj_id2name(source_model, mujoco.mjtObj.mjOBJ_KEY, i): list(map(float, source_model.key_qpos[i]))
        for i in range(source_model.nkey)
    }
    keyframe_file = output_dir / "source_keyframes.json"
    keyframe_file.write_text(
        json.dumps(
            {
                "source_sha256": observed_hash,
                "joint_order": [
                    mujoco.mj_id2name(source_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(source_model.njnt)
                ],
                "keyframes": keyframes,
            },
            indent=2,
        )
        + "\n"
    )
    manifest = {
        "source_repository": "FEGkkd/whole-body-teleop",
        "source_commit": SOURCE_COMMIT,
        "source_mjcf_sha256": observed_hash,
        "source_mesh_count": len(meshes),
        "source_keyframes_sha256": sha256(keyframe_file),
        "source_mesh_sha256": {str(path.relative_to(mjcf.parent)): sha256(path) for path in meshes},
        "isaaclab_version": importlib.metadata.version("isaaclab"),
        "importer_versions": {
            package: importlib.metadata.version(package)
            for package in ("isaacsim-asset-isolated", "mujoco-usd-converter", "usd-exchange", "newton")
        },
        "converter": "isaaclab.sim.converters.MjcfConverter",
        "converter_options": {
            "fix_base": True,
            "merge_mesh": False,
            "self_collision": True,
            "run_asset_transformer": True,
            "run_multi_physics_conversion": True,
        },
        "usd_path": str(usd),
        "generated_sha256": {str(path.relative_to(usd.parent)): sha256(path) for path in output_files},
    }
    report = output_dir / "import_manifest.json"
    report.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"USD={usd}\nMANIFEST={report}\nMESHES={len(meshes)}")


if __name__ == "__main__":
    main()
