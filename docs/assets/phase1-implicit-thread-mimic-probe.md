# Phase 1 implicit screw joint probe (2026-11-01)

This is a **standalone PhysX GPU research fixture**, not the current RL bottle
scene. The production `BimanualBottleDirectEnv` still uses its D6 joint and
explicit `BatchedThreadCoupler`. No PPO checkpoint or CapTurn success score in
the existing Phase 1 reports was produced with the mimic fixture.

## Mechanism and source

`scripts/environments/probe_thread_mimic.py` authors one floating PhysX
articulation per environment: a dynamic bottle root, a lightweight
collisionless prismatic carriage, and a dynamic cap linked by a revolute
joint. A `PhysxMimicJointAPI` on the prismatic joint references the revolute
joint with gearing `-pitch/360` metre/degree. The joint constraint, rather
than an object pose write or an explicit axial spring, enforces
`z_rel = pitch * theta / (2*pi)` in the bottle's current local frame.
External forces in this fixture are test loads on env 0, never an RL action.

The installed Isaac Sim 6.0.1 / Isaac Lab 3.0 tensor API accepted the
floating articulation and GPU-batched DOF, link, force and selective Reset
operations. [Omniverse Physics articulations](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html#articulation-mimic-joints)
documents mimic joints and their GPU path;
[the PhysX schema](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.2/dev_guide/schemas/physxschema.html)
defines the API, axis tokens and gearing.

## Assembly and collision diagnosis

An early draft passed a USD `wxyz` quaternion to Isaac Lab 3.0
`spawn_from_usd`, whose local implementation expects `xyzw`. The generated
bottle and cap then had the wrong orientation. Trying to reverse the joint
anchor could make the *upright* case appear aligned, but the 30-degree tilt
test exposed centimetre-scale physical-axis error. The fixture now converts
the quaternion explicitly and retains the standard bottle-local joint
anchor. It checks the cap and bottle **link positions**, not only joint
coordinates, after rotation and load.

With the corrected orientation, the engineering-example internal and
external thread capsule colliders still overlap at initial engagement. A
zero-load, 30-degree tilted test with articulation self-collision enabled
reported 53 bottle-cap contacts on the first step, about **5.29 kN** summed
normal force, and -0.5 mm minimum separation. This contact impulse is
unacceptable. The tested mimic configuration disables **within-assembly**
self-collision and relies on the joint for screw coupling. It does not
disable collision with external objects: a separate dynamic 20 g sphere
contacted the cap at step 16 with 1.12 N measured force and moved the free
assembly, while physical pitch error stayed below 0.11 micrometre.
The internal thread ridges remain visual/collision geometry in the USD but
do not contribute contact forces in this tested configuration. Their
clearance and phase must be repaired before self-collision can be enabled.

## Measured checks

All data below is from the installed GPU PhysX backend, 2 ms steps. Full
machine-readable traces are in ignored `artifacts/phase1/`.

| Fixture | Result |
|---|---|
| 30° tilt, 2 env, env 0 loaded with 0.5 N local axial force and 10 µN m torque, 200 steps | Max joint pitch error 0.70 nm; max **physical link** pitch error 0.11 µm; max lateral offset 0.05 µm; unforced env 1 joint motion 1.84e-7. |
| 30° tilt, 16 env, load env 0 only, Reset env 0 at step 50 without physics advance | Immediate peer state change 0; env 0 DOF, root pose and velocity Reset errors at numerical zero. `update_articulations_kinematic()` was required after the indexed tensor writes. Max physical pitch error 0.16 µm; max lateral offset 0.13 µm. |
| Reversed load, 30° tilt, 2 env | Env 0 final angle -3.181 rad and bottle-local axial travel -1.519 mm; max physical pitch error 0.095 µm. |
| No mimic control, otherwise matched 30° 2-env load | Max physical pitch error **9.0 mm**; prismatic joint reached its 9 mm limit independently of the spin. This confirms the mimic constraint causes the pitch coupling. |
| External dynamic sphere, self-collision disabled | Cap contact measured at 1.12 N; physical pitch error below 0.11 µm. |

Commands for the fail-on-physics checks:

```bash
export ISAACLAB_PYTHON=/home/ubuntu/miniconda3/envs/allo_track/bin/python
scripts/run_isaaclab.sh scripts/environments/probe_thread_mimic.py \
  --real-geometry --no-self-collision --num-envs 16 --selective-reset \
  --steps 100 --tilt-deg 30 --torque 0.00001 --axial-force 0.5 \
  --assert-physics --output artifacts/phase1/thread_mimic_16env_asserted.json --headless
scripts/run_isaaclab.sh scripts/environments/probe_thread_mimic.py \
  --real-geometry --no-self-collision --contact-proxy --steps 100 \
  --torque 0 --axial-force 0 --assert-physics \
  --output artifacts/phase1/thread_mimic_proxy_asserted.json --headless
scripts/run_isaaclab.sh scripts/environments/probe_thread_mimic.py \
  --real-geometry --no-self-collision --num-envs 2 --steps 100 \
  --tilt-deg 30 --torque -0.00001 --axial-force -0.5 \
  --assert-physics --output artifacts/phase1/thread_mimic_reverse_asserted.json --headless
```

## Open work before RL integration

The fixture has **no breakaway or running thread resistance**. Its low
cap/bottle mass and ideal screw constraint let 0.5 N drive several radians
quickly; that is a deliberately strong diagnostic load, not calibrated
bottle behavior. It also has no fingered hands or 1024-env training test.
The next controlled implementation must add bounded, bidirectional
resistance to the articulation spin DOF, verify low/high torque and contact
reaction, repair the internal collider overlap or retain a documented
internal collision exclusion, and integrate the articulation with the
existing contact monitor, Reset banks and RL observation/reward path.
Only then should prior policies be reevaluated or new PPO training begin.
