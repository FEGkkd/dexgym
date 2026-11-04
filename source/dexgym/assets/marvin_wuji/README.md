# Marvin M6 + WujiHand2 Beta 2 asset

This module defines the 54 DoF whole-body articulation contract and loads a
**locally converted** USD. The authoritative input is `dual_arm_marvin.xml` from
`FEGkkd/whole-body-teleop` commit `5d4c345842f197bffeae7d7e08c4c3718230f9e0`,
SHA256 `6e0d569ec28c8ac26a84bfc9a48652b7a11a2f113193c2d3ea4191621280b3ce`.
The model retains the Marvin J6/J7 axes and TCP frame conventions. The raw
MJCF, mesh files and generated USD are external/local artifacts because their
publication rights have not been established. They are not package data.

Install the standalone importer with `bash scripts/assets/setup_mjcf_importer.sh`
after `uv sync --locked --group dev`. Set `DEXGYM_MARVIN_MJCF` to the external XML and run
`scripts/assets/import_marvin_wuji.py`. The importer verifies the source hash
and 70 mesh dependencies, converts with Isaac Lab 3.0's `MjcfConverter`, fixes
the base, and enables self collisions while preserving the 38 excluded pairs.
Its output is under `artifacts/marvin_wuji/generated/` by default and is ignored
by Git. The generated `import_manifest.json` records mesh and USD hashes.

`robot_cfg.make_robot_cfg(usd_path)` returns an Isaac Lab `ArticulationCfg`.
Every joint uses the imported USD position drive; the public joint order is the original MuJoCo `qpos` order and joints are addressed by name.
`joint_mapping.inspect_joint_contract` verifies the source's 7+7+20+20 grouping,
one actuator per joint, and legal limits; `qpos_to_newton` remaps by name.
`frames.wxyz_to_xyzw` explicitly converts MuJoCo quaternion order to Isaac Lab
3.0's XYZW order.

See [the validation report](../../../../docs/robots/marvin_wuji_validation.md)
for exact commands, measured FK/control/dynamics results, and limitations.
