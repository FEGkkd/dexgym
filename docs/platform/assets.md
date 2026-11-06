# Asset inventory and provenance

The Cartpole Newton tasks use the **official Isaac Lab v3.0.0 Cartpole** via
`isaaclab_assets.robots.cartpole.CARTPOLE_CFG`. Its USD is supplied through
Isaac Lab's asset source. The task changes only its actuator configuration in
memory; it does not edit the USD. Marvin SuperMimic tasks additionally use
the packaged 54 DoF Marvin/Wuji source MJCF, meshes and no-hand-contact USD.
Their physical gates are recorded in [SuperMimic validation](../supermimic/core_architecture.md).

`source/dexgym/assets/manifest.json` records local file hashes and external
source contracts. The local bottle and cap USDA layers, generated parameters,
and PCO1881 reference layers remain in
`archive/isaaclab2/source/dexteroushands_lab/assets/bottle/generated/`. All original asset
mapping modules remain at `archive/isaaclab2/source/dexteroushands_lab/assets/`, including
Allegro Fly, Wuji Hand, WujiHand2 Beta, Wrist 6DoF, bottle thread and contact
configurations. These are historical Isaac Lab 2.x / PhysX adapters, outside the
installed DexGym 2.0 package and **not Newton certified**. Their source files
and SHA256 declarations are preserved exactly where available. The research
evidence in `docs/assets/` and `artifacts/` remains in the repository.

| Asset | Storage and source | Version/path contract | Newton status |
| --- | --- | --- | --- |
| Cartpole | Isaac Lab official asset | Isaac Lab v3.0.0, `Robots/Classic/Cartpole/cartpole.usd` | Active task, verify per host |
| Shadow Hand | Isaac Lab official asset | Official `isaaclab_assets`, `Robots_Multiphysics/ShadowRobot/ShadowHandMultiPhysics_v0/right_hand/right_hand.usda` | Official Newton variant exists; DexGym unverified |
| Allegro Fly | External description | version not pinned; `ALLEGRO_DESCRIPTION_ROOT`, optional `ALLEGRO_ASSET_PATH` | Unverified |
| Wuji Hand | External `wuji-description` | version not pinned; `WUJI_DESCRIPTION_ROOT`, `hand/body/usd/right/wujihand.usd` | Unverified |
| WujiHand2 Beta 1/2 | External `wuji-description` | tag `v2026.9.11`, commit `c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95`; USD and tree hashes in `wuji_hand2_beta.py` | Unverified |
| Marvin M6 + WujiHand2 Beta 2 whole body | Pinned `FEGkkd/whole-body-teleop` commit `5d4c345842f197bffeae7d7e08c4c3718230f9e0`; MJCF, 70 meshes and converted USD packaged | Package-relative default; `DEXGYM_MARVIN_MJCF` selects an explicit source for asset tools; SHA256 and conversion in [robot validation](../robots/marvin_wuji_validation.md) | 54 DoF Newton; [no-hand-contact mode](../robots/marvin_wuji_no_hand_contacts.md) is default; full-contact USD retained |
| Wrist 6DoF | Repository Python controller | `archive/isaaclab2/source/dexteroushands_lab/assets/wrist_6dof.py` | PhysX only; unverified |
| Bottle/cap and PCO1881 reference | Repository USDA and JSON | local SHA256 in manifest | Unverified |

The pinned runtime Marvin source was copied into this repository with its
[component notices](../../source/dexgym/assets/marvin_wuji/source_model/NOTICE.md).
External files are not copied automatically by setup. The Wuji
asset contract checks the official checkout commit and both per-USD and asset
tree SHA256 values. Neither those checks nor prior PhysX evidence establish
Newton compatibility.
