> Historical Beta2 report. The current default asset is Hand2.2; see [the 2026-11-11 upgrade](marvin_hand22_upgrade_20261111.md). These controller/asset measurements do not describe the new default.

# Marvin/Wuji: Newton asset without hand contacts

**2026-11-04.** `make_robot_cfg()` now selects a packaged USD override that
disables contact collision for both WujiHand2 hands. It leaves all 40 hand
joints, hand masses and inertias, visual geometry, wrist mounts and fixed
installation relationships intact. The original full-contact USD is retained
and available through `make_robot_cfg(hand_contacts=True)`. No source MJCF,
mesh, real robot setting or original USD payload was edited. The
[machine-readable result](marvin_wuji_no_hand_contacts.json) records hashes
and all measured values.

## Asset construction

The original asset is `usd/dual_arm_marvin.usda`, SHA256
`4643b25b3ddc1412f83ef2f089bfc4fe6dd90ad2f36ef76efd9d48112ecd1ceb`.
The new `usd/dual_arm_marvin_no_hand_contacts.usda` is a small USD layer over
that asset, SHA256
`8ed2cb9a14a813022149aaa8a101159af86dab676aec7dfdde4a00a8d9698fa6`.
It deinstances the **54 hand collision mesh instances** (27 per hand) and
authors `physics:collisionEnabled=false` on their mesh prims. The meshes and
their visual references remain. The **33 arm/support collision shapes stay
enabled**. No rigid-body, joint, drive, mass or inertial attribute is authored
in the override.

The override can be regenerated without overwriting the packaged file:

```bash
uv run --no-sync python scripts/assets/disable_marvin_hand_contacts.py \
  --output artifacts/marvin_wuji/dual_arm_marvin_no_hand_contacts.usda
```

The generator rejects an unexpected collision inventory and refuses to
overwrite an existing output. For normal Newton use:

```python
from dexgym.assets.marvin_wuji.robot_cfg import make_robot_cfg

robot_cfg = make_robot_cfg()                    # no hand contact, default
full_contact_cfg = make_robot_cfg(hand_contacts=True)  # original asset
```

## Actual validation

The USD structure comparison found **74 identical rigid bodies, 54 identical
revolute joints and 20 identical fixed joints**. Every rigid-body and joint
physics attribute and relation, including authored mass, inertia and fixed
mounts, compared equal. Loading the two assets into Newton MJWarp also gave
**exactly zero difference** across all 74 effective body masses and inertia
tensors, as well as joint stiffness, damping, armature, friction, effort
limits and position limits. This checks the solver's effective properties,
not just USD authoring.

The Newton contact test used legal, deterministic source-model poses:

| Pose | MuJoCo source contacts | Newton contacts, no-hand asset | Newton hand contacts | Newton arm-to-arm contacts |
| --- | ---: | ---: | ---: | ---: |
| Home | 0 | 0 | 0 | 0 |
| Hand against support | 83 | 0 | 0 | 0 |
| Crossed hands/arms | 103 | 0 | 0 | 0 |
| Arm links crossing | 61 | 20 | 0 | **4** |

Thus hand contact disappears in the actual MJWarp solver, while left/right
arm collision still produces contacts. The original MuJoCo source model was
used only to find and label legal test poses; its collisions were not changed.

The collision-free asset passed **1,000-pose FK parity** in 16 Newton
environments: maximum TCP position error 0.000950 mm, maximum TCP orientation
error 0.0000367°, failed TCP comparisons 0. Its default imported controller
passed **16 environments × 3 resets × 200 steps** with finite state and
maximum joint jump 0.01446 rad/step. All **54 named control channels**
responded in the 16-environment opposed-command test; minimum separation was
0.0570 rad, failed channels 0. The optional arm tracking profile also passed
the same 16-environment smoke; its fixed-TCP wrist test finished at
0.558/0.566 mm and 0.065/0.067° (left/right), without arm drive clipping or
joint-limit violation.

To repeat the structural, solver and motion checks, set
`DEXGYM_MARVIN_MJCF` to the pinned source `dual_arm_marvin.xml`, then run:

```bash
USD=source/dexgym/assets/marvin_wuji/usd/dual_arm_marvin_no_hand_contacts.usda
uv run --no-sync python scripts/assets/validate_marvin_hand_contacts.py
uv run --no-sync python scripts/assets/validate_marvin_runtime_mass.py
uv run --no-sync python scripts/assets/validate_no_hand_newton_contacts.py
uv run --no-sync python scripts/assets/validate_fk_parity.py --usd "$USD" --samples 1000 --num-envs 16
uv run --no-sync python scripts/assets/validate_dynamics.py --usd "$USD" \
  --mode smoke --num-envs 16 --steps 200 --resets 3
uv run --no-sync python scripts/assets/validate_dynamics.py --usd "$USD" \
  --mode channels --num-envs 16 --steps 150
```

`validate_no_hand_newton_contacts.py` reads MJWarp's internal per-pair
contact diagnostic in Newton 1.6.1; its exact accessor is version-specific.
The hand geometry remains visible and massive but cannot collide with the
robot, support or objects in this mode. Any future grasp/contact/tactile task
must select the original full-contact asset and revalidate its contact model.
No Arm S0 task or PPO training was added here.
