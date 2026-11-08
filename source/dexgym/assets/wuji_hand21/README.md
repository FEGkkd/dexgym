# Official Wuji Hand 2.1 right hand

The independent Newton articulation loads `official/usd/right/wujihand2_1.usd`
directly. This is the **official, unmodified, layered USD** from
[`wuji-description`](https://github.com/wuji-technology/wuji-description),
release `v2026.10.26`, commit `5c4dfa16917355d073e2e1d7e3d03897b6cf9508`.
At the time of import, the release commit also matched the upstream `HEAD`.
The official MIT license is retained in `official/LICENSE`.

The asset has one fixed `r_mount` root, 20 actuated right-hand joints, five
fingertip frames, self-collision enabled, 22 instanced collision meshes, and a
`Physics=PhysX` variant. `robot_cfg.py` selects this variant and sets the
published `config.yaml` stiffness/damping in SI units plus the official MJCF
joint force limits. These explicit runtime gains matter: the USD's authored
fallback angular drives are `50/2` for every joint, while its conversion
config specifies per-joint-group gains. The USD files themselves are untouched.

`official/mjcf/right.xml` and the corresponding right-hand STL meshes are only
used for offline fingertip FK and self-collision auditing. They are **not** the
Newton simulation asset. The old Marvin dual-arm USD is not referenced here.
`scene_cfg.py` supplies a reusable one-hand `InteractiveSceneCfg` and Newton
simulation configuration without registering or changing an RL task.
The Newton configuration selects `implicitfast` integration because the
default Euler integrator destabilizes the low-inertia finger joints under
the published PD gains at 500 Hz.

Run the independent model check with:

```bash
PYTHONPATH=source python scripts/assets/validate_wuji_hand21.py \
  --manifest data/processed/hand21_right_geometry_feasible/manifest.json
```
