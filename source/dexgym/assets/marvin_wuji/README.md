# Marvin M6 arms + Wuji Hand2.2 hands

The default `make_robot_cfg()` now loads the composed Hand2.2 asset under
`hand2_2/generated/`, with hand contacts enabled and explicit SI position-drive
gains/effort limits for all 54 joints. `hand_contacts=False` is an explicit
collision-free diagnostic option. Joint names and ordering remain compatible
with the previous 7+7+20+20 interface.

The right hand reuses the successful Cube task's official Hand2.2 model,
inertias, joint properties, collision overlay, gains and force limits. The left
hand uses the matching official Hand2.2 left asset (including its own axes and
meshes); it receives the corresponding Cube gains and a palm-reflected capsule
overlay. Those gains have not been identified independently on left hardware.
The Cube jig/tag and fixed-palm pose are task fixtures, so they are not mounted
to the arms. The original arm geometry, dynamics, joint axes and TCP-to-mount
transforms are retained. Inferred support inertias are explicitly materialized
before USD conversion.

`hand2_2/provenance.json` records upstream commits and input hashes.
`hand2_2/generated/metadata.json` records the composed source hash, joint order,
keyframes and SI drive parameters. The old `source_model/` is retained as the
pinned arm input; the old `usd/` and `controller.py` describe the previous Beta2
experiments and are not selected by the default configuration.

Rebuild inside the Isaac Lab runtime:

```bash
bash scripts/supermimic/host.sh -m dexgym.assets.marvin_wuji.build_hand2_2 \
  --importer-root /path/to/standalone/isaacsim
```

The importer override is host-dependent and optional. The generated USD uses
its `mujoco` physics variant. Explicit SI actuator config avoids the old USD
angular-gain scale error; no high-gain legacy controller profile is applied.

See [Hand2.2 validation](../../../../docs/robots/marvin_hand22_upgrade_20261111.md)
for measured results and the distinction between simulator consistency and
real hardware calibration. Existing policies trained on the old full robot
must not be assumed compatible with these changed hand dynamics.
