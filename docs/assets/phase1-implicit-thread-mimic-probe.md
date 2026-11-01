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

An optional `--resistance` probe applies a GPU-batched version of the
existing engineering-example stick/slip and running torque law to the
revolute DOF. **It is not a validated production parameter set.** The
original example's explicit damping of 0.003 N m s/rad is unstable with
this 6 g cap, reduced-coordinate joint and 2 ms step: with *zero external
load* a microscopic spin velocity flips sign and grows by about 4.5 times
per step, reaching 26.8 rad/s at step 15. This is a discrete explicit
feedback instability, not evidence of spontaneous bottle torque. A
controlled one-parameter ablation to 0.0001 N m s/rad removed the zero-load
drift for 100 steps (final spin angle about 7e-12 rad). With that diagnostic
override, 0.005 N m over 0.4 s stayed in the sticking state (1.33° elastic
rotation), whereas 0.04 N m entered sliding at step 3 and reached the
three-turn joint stop by step 80. The latter is very fast because this
fixture has no hand contact, and no actual bottle torque calibration exists.
The model and two CPU state tests are in `mimic_resistance.py` and
`test_mimic_resistance.py`; their presence does not qualify it for PPO.

A second controlled arm, `--implicit-drive`, uses `UsdPhysics.DriveAPI` on
the articulation spin joint with the *original* 0.003 N m s/rad damping,
0.2 N m/rad spring and 0.02 N m force limit. The [USD Physics schema](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/schemas/usdphysics.html)
states angular drive targets and gains use **degrees**. The fixture converts
the engineering-example radian gains by `pi/180` when authoring USD; without
this, the low-torque spring is about 57 times too stiff. On the corrected
fixture, zero load stayed within 2e-13 rad over 100 steps. At 0.005 N m,
the 0.4 s final angle was 0.02235 rad, with env 1 within 1.5e-11 rad. At
0.04 N m it reached 14.72 rad and 7.03 mm signed axial travel in 0.4 s.
The physical screw relation stayed within the earlier micrometre-scale
limits. This proves a **numerically stable bounded spring comparator**; it
does not prove calibrated running friction, finger contact, or RL success.

The installed PhysX tensor interface accepts a nonzero 0.5 rad initial spin
with a matching 0.239 mm follower displacement and matching drive target.
Across 16 tilted environments, Reset of env 0 at step 50 restored its root,
both DOFs, velocity and drive target while immediate peer state change was
exactly zero. The final angle remained 0.5 rad; maximum physical pitch
error was 0.16 µm. The Reset code refreshes articulation link kinematics
after indexed tensor writes and before contact is read.

The same complete USD articulation also loaded and stepped in **1024
environments** on the GPU. A 200-step zero-load run, with per-env contact
views omitted for this throughput measurement, took 0.357 s inside the
physics loop: about 574,000 environment physics steps/s. GPU memory used
at the end of that process was about 5.0 GB, including Isaac Sim. This is
an isolated three-link bottle fixture, **not** a 1024-env bimanual RL
throughput or contact benchmark. A separate 1024-env, 30-degree tilt test
restored one 0.5 rad engaged bottle at step 50: immediate peer state change
zero, max physical pitch error 0.48 µm and lateral error 0.93 µm. The
fixture uses a 32×32 origin grid; a preliminary one-kilometre row produced
5.8 µm float32 world-coordinate subtraction error and was discarded for
physical accuracy claims.

```bash
scripts/run_isaaclab.sh scripts/environments/probe_thread_mimic.py \
  --real-geometry --no-self-collision --resistance --resistance-damping 0.0001 \
  --num-envs 2 --steps 200 --tilt-deg 30 --torque 0.005 --axial-force 0 \
  --assert-physics --max-abs-angle 0.05 --headless
scripts/run_isaaclab.sh scripts/environments/probe_thread_mimic.py \
  --real-geometry --no-self-collision --resistance --resistance-damping 0.0001 \
  --num-envs 2 --steps 200 --tilt-deg 30 --torque 0.04 --axial-force 0 \
  --assert-physics --min-abs-angle 0.5236 --expect-slide --headless
scripts/run_isaaclab.sh scripts/environments/probe_thread_mimic.py \
  --real-geometry --no-self-collision --implicit-drive --num-envs 16 \
  --selective-reset --steps 100 --tilt-deg 30 --torque 0 --axial-force 0 \
  --initial-angle-rad 0.5 --assert-physics --max-angle-drift 0.01 --headless
scripts/run_isaaclab.sh scripts/environments/probe_thread_mimic.py \
  --real-geometry --no-self-collision --skip-contact-audit --implicit-drive \
  --num-envs 1024 --selective-reset --steps 100 --tilt-deg 30 \
  --torque 0 --axial-force 0 --initial-angle-rad 0.5 \
  --assert-physics --max-angle-drift 0.01 --headless
```

### Reset-bank compatibility

`audit_mimic_bank.py` checks each cap's bottle-local axial offset against
its continuous thread angle before admission to a hard-pitch articulation.
It also reports lateral offset and cap twist. With a 50 µm pitch tolerance,
**0/4** committed `phase0e_bimanual_grasp_init_bank.json` stable states
qualify: their pitch errors are -1.203, -1.232, -1.256 and -1.203 mm. The
single ignored local composite stable state used by prior PPO also fails
at -0.457 mm. Their measured cap twist agrees with the stored angle, so
the primary mismatch is axial. In contrast, **32/32** committed geometric
`phase0_pregrasp_reset_bank.json` states pass. This says nothing about
their ability to grasp or hold the bottle. The audit is read-only, and raw
results are in ignored `artifacts/phase1/`.

```bash
scripts/run_isaaclab.sh scripts/environments/audit_mimic_bank.py \
  docs/assets/phase0e_bimanual_grasp_init_bank.json \
  --output artifacts/phase1/mimic_bank_audit_phase0e.json
scripts/run_isaaclab.sh scripts/environments/audit_mimic_bank.py \
  docs/assets/phase0_pregrasp_reset_bank.json --strict \
  --output artifacts/phase1/mimic_bank_audit_pregrasp.json
```

The fixture still has no fingered hands or 1024-env **training** test. The next
physical task is to make the implicit-drive breakaway law transition to the
configured lower running resistance during slip. Holding the 0.02 N m
force-limited spring throughout a full turn is not the configured 0.006 N m
running regime. Batched stiffness, damping, force-limit and target APIs are
available in the installed tensor backend, but their per-environment dynamic
switching and contact behavior have not yet been validated. Repair the
internal collider overlap or retain a documented internal collision
exclusion, then integrate the articulation with the existing contact monitor,
Reset banks and RL observation/reward path. The four legacy stable states
require physical resynthesis and validation under the hard-pitch constraint;
their `validated=true` flag refers only to the former soft-coupler scene.
Only then should prior policies be reevaluated or new PPO training begin.
