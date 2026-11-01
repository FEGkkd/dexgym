# Phase 1 hard-pitch screw integration probe (2026-11-01)

This is a measured **assembly probe**, not a production CapTurn30 training
backend or a learned opening policy. It uses the pinned WujiHand2 Beta 1 native
USD at `wuji-description` commit
`c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95` and the example bottle
parameters in `BottleConfig`. The example geometry and torque are not real
bottle calibration.

`assets/bottle/mimic_thread.py` adds a collisionless carriage and a floating
Bottle–Carriage prismatic joint plus Carriage–Cap revolute joint to the
existing two-hand Isaac Lab scene. A PhysX mimic joint couples slide and spin:
one 360° turn advances one signed `thread_pitch` along the Bottle **local**
axis. The Bottle is the free articulation root, not a world anchor. An
implicit angular force drive uses the example breakaway stiffness and damping;
its maximum force is the example breakaway torque. The two generated USD
models and both hands remain otherwise unchanged. Articulation self-collision
is disabled because the example bottle's internal/external ridge capsules
overlap at initial engagement. External hand/object collision remains
enabled, but this probe does not yet establish contact-mediated cap turning.

The integration script creates the normal left and right hand `InteractiveScene`
with 20 active finger joints each. It verifies the existing `RigidObject`
Bottle/Cap link pose readers, a selective 2 cm **Bottle root** translation on
env 0, a
nonzero screw-DOF Reset with consistent axial travel, a 0.5 N cap axial load
on env 0, screw pitch, finite poses, and no measured peer movement. The load
is a diagnostic external force, **not** a policy action. The diagnostic is
intentionally not registered as a Gym task.

| Test | Result |
|---|---|
| 1 env, 50 physics steps | Pass; first spin 0.0009543 rad; mimic error 5.44e-11 m. |
| 2 env, 100 steps | Pass; env 1 spin about -7.12e-12 rad. |
| 16 env, 100 steps | Pass; no measured peer motion. |
| 64 env, 100 steps | Pass; no measured peer motion. |
| 1024 env, 20 steps | Pass; full two-hand scene and finite link data. This short smoke is not a throughput benchmark or PPO run. |
| 16 env, bottle tilt 45°, 100 steps | Pass; maximum physical local-axis pitch error 5.97e-8 m. |
| 16 env, 45° tilt, env 0 Reset to 0.5 rad, 100 steps | Pass; final spin 0.46637 rad; maximum physical local-axis pitch error 1.60e-7 m; peer DOFs stayed at zero within 1e-5. |
| 2 env, 50 steps, exact contact audit | Pass; the production `BatchedContactMonitor` created Bottle/Cap link filter views and measured zero force against the deliberately distant hands, with no false positive. This does not test positive hand contact. |
| 1 env, 30 steps, natural right pregrasp contact probe | Pass; a scripted wrist approach at ≤0.05 m/s stopped after 0.2 mm and measured 0.237 N on `r_middle_finger_distal` against the cap sidewall. First manifold point had +0.065 mm separation, cap-local radial position 17.71 mm and axial position 1.54 mm. The cap followed the screw with maximum physical pitch error 6.77e-8 m. This is one-digit positive contact, not a stable three-digit grip or a learned turn. |

The script uses `articulation_view.set_dof_positions` and matching spin drive
targets for nonzero screw-angle Reset. The Bottle `RigidObject` root write
moves the PhysX articulation; the Cap `RigidObject` wrapper can **read** its
link pose after a physics update but cannot directly set that non-root link's
transform. An early probe version tried to write both root and cap. PhysX
logged `RigidBodyView contains non-root articulation links whose transforms
cannot be set directly`; the wrapper's cached Cap pose also remained stale
immediately after the root write even though the articulation link tensor had
already moved. That probe's claimed direct Cap write compatibility was wrong
and was withdrawn. The corrected script writes only Bottle root plus screw
DOFs and compares `articulation_view.get_link_transforms()` at the no-step
Reset boundary. It passed 2, 16 and 1024 environment reruns without the
non-root-write warning. The reusable low-level `MimicThreadView` now restores
the floating articulation root, both screw DOFs, targets and velocities for
selected environments. **It is not yet connected to the DirectRLEnv's full
episode Reset**, which must also restore both hands, contacts and curriculum
state. Writing an arbitrary
legacy Bottle pose and Cap pose independently would conflict with the hard
pitch constraint.

An offline bank audit showed that **0/4** committed stable grasp samples and
**0/1** local composite PPO sample satisfy the hard-pitch axial relation at
their saved cap angles. Their initial axial discrepancies are about 1.20–1.26
mm and 0.457 mm respectively. The **32/32** geometry-qualified natural
pregrasp samples are screw-compatible, but earlier policy evaluation from
natural pregrasp was 0/200. None of the old stable samples may be silently
used as validated Reset states for this backend. The DirectRLEnv retains
legacy D6 as its default; the experimental Mimic switch is described below.
The new authorer rejects an existing legacy D6 on the same assembly.
The four committed stable rows are variants of one seed-204 grasp family;
their measured lateral cap offset is only 2–9 µm, so it is specifically the
axial mismatch that fails this gate. Forcing such poses into the joint would
introduce a millimetre-scale solver correction at Reset, so no dynamic hold
pass is inferred from them.

The initial fixed implicit drive was bounded in the isolated low-torque
probe. A dynamic breakaway-to-running implementation and its later tests are
described below. Its torque values are not measured from a real bottle. The
isolated probe's explicit viscous resistance at the original damping was
numerically unstable; that path must not be installed as the production
backend. One gentle right-middle-finger sidewall contact was measured in the
full scene, but bilateral grasp, contact-driven 30° rotation and learned
opening remain **not tested**. A new
stable grasp bank compatible with the constrained cap pose is required
before meaningful stable-start CapTurn PPO can resume.

Run the integration probe on the target Isaac Lab 3.0 workstation:

```bash
export WUJI_DESCRIPTION_ROOT=/tmp/wuji-description-v2026.9.11
export ISAACLAB_PYTHON=/home/ubuntu/miniconda3/envs/allo_track/bin/python
scripts/run_isaaclab.sh scripts/environments/validate_bimanual_mimic_scene.py \
  --num-envs 16 --steps 100 --tilt-deg 45 \
  --initial-angle-rad 0.5 --headless \
  --output artifacts/phase1/mimic_scene_tilt45_reset05.json
scripts/run_isaaclab.sh scripts/environments/validate_bimanual_mimic_scene.py \
  --num-envs 1 --steps 30 --right-pregrasp-contact --headless \
  --output artifacts/phase1/mimic_right_pregrasp_contact.json
```

The local runner currently returns shell status 0 even when the Python script
raises an exception. Check its output for `Traceback` and require a
`BIMANUAL_MIMIC_SCENE` result before recording a pass.

The focused bottle/thread/Phase 1 numeric tests returned **39 passed**. The
full offline suite with `WUJI_DESCRIPTION_ROOT` set returned **161 passed,
1 skipped, 2 failed**. The two failures are in the existing user-owned
Allegro asset fixture and MANO keypoint-order test; neither is caused by this
three-file additive probe. An initial full-suite invocation without the
required model-root variable stopped during collection and is not counted as
a test result.

The initial integration probe led to the stable-grasp bridge experiment below.
Selective articulation Reset in the DirectRLEnv and a topology-qualified bank
are still required before this backend can be connected to PPO.

## Legacy stable-grasp bridge probe

`scripts/environments/probe_mimic_stable_grasp.py` now runs a controlled
one-family migration test. It reads one of the four committed legacy stable
rows, computes the signed axial discrepancy, shifts the **right wrist only**
along the bottle-local axis by the corresponding 1.20–1.26 mm, and restores
the Bottle root plus slide/spin DOFs. PhysX determines the Cap link pose; the
script never writes that link or applies a support force or test cap torque.
Both wrists are fixed, 40 finger joints retain their normal physical drives,
and gravity remains on. The script records full 1 s contact and bottle motion,
first-step manifold penetration, point force and per-digit contact ratios.

All four rows passed a **basic physical hold** check in separate fresh Isaac
Sim processes. Correct object contact was present for all 500 physics steps
on both sides; bottle displacement was 0.037–0.068 mm, rotation
0.0012–0.0020 rad, first-step minimum separation -0.015 to -0.032 mm,
first-step maximum point normal force 5.3–6.7 N, and maximum screw error
5–12 nm. No ground contact was measured. Row 0 also passed a left-only
ablation with the right hand moved 0.35 m away: 1 s left-bottle contact,
0.486 mm bottle displacement, 0.0467 rad rotation and no right-cap contact.
These are strong fixture results for a free bottle and a hard-pitch cap.

**Zero of four rows passed the Phase 1 target contact topology.** In row 0,
left thumb and middle finger contact ratios were 1.000 and 0.998, but left
index was 0.044 and ring/pinky were zero. Right thumb and middle were both
1.000, while right index was zero. The other three rows showed the same
two-digit dominant family: left index 0.008–0.028, right index zero. The
left-only ablation gained left index contact on 48.2% of steps, still short of
a sustained multi-finger cylindrical grasp. The four rows are variations of
the same seed-204 family, so even a future pass would not prove cross-family
generalization. These rows are **not** a new topology-qualified or Reset
replayed hard-pitch bank, and no new PPO success is inferred.

```bash
scripts/run_isaaclab.sh scripts/environments/probe_mimic_stable_grasp.py \
  --bank-index 0 --duration 1.0 --headless \
  --output artifacts/phase1/mimic_stable_grasp_probe_0.json
scripts/run_isaaclab.sh scripts/environments/probe_mimic_stable_grasp.py \
  --bank-index 0 --duration 1.0 --left-only --headless \
  --output artifacts/phase1/mimic_left_only_probe_0.json
```

Any new sample must show distributed target-object contact, normal-force
balance, shallow penetration and an independent 1 s replay before entering a
validated bank. A contact-topology-verified bank and a batched articulation
Reset are prerequisites for meaningful hard-pitch PPO.

An initial controlled right-index ablation already narrowed that next step.
With row 0, the index tip site was about 52.0 mm from the cap axis and
30.8 mm above the cap centre after 0.25 s; the example cap sidewall radius is
about 18 mm. Ramping both right index MCP flex and PIP targets by +0.2 rad
over 0.2 s moved the real joints to their new targets and brought the tip to
43.5 mm radial distance and 11.6 mm axial height, but produced **zero** index
contact during a separate 1 s run. The -0.2 rad control moved the tip farther
away, to 56.8 mm radial distance and 51.6 mm axial height, again with zero
contact. Thus the lack of index contact is a measured geometric placement
problem in this legacy grasp family, not a joint drive failure. Continuing to
increase the same flex target without checking other collision bodies and
wrist/index abduction geometry would be blind search. The next experiment
should start from the existing geometrically qualified tripod pregrasp family
or explicitly optimize the right wrist/index abduction around the cap while
retaining thumb and middle contact. No new bank row was marked validated.

## Batched screw state and selective Reset (later 2026-11-01)

`assets/bottle/mimic_thread_view.py` binds the PhysX articulation tensor view
and reads Bottle root, Cap link, continuous spin, axial travel, local-axis
physical pitch error and reset-relative opening directly on GPU. Its
`reset(env_ids, root_pose_w, root_velocity_w, theta_rad, spin_velocity_rad_s)`
sets the root pose/velocity, coupled slide/spin position/velocity and implicit
spin drive target for selected environments. It does not call `sim.step()` or
write the Cap link independently. The installed backend requires full-size
input tensors even for selected IDs; the adapter retains peer rows from the
current PhysX state before each indexed write.
The Cap link velocity is determined by the root and joint velocities. An
arbitrary legacy saved Cap velocity cannot be restored independently; a full
episode migration must project it onto this constrained state and record the
projection error.

The full two-hand integration script called this adapter 100 times without
advancing physics. At 16 environments, deterministic random selection visited
all 16; at 1024 environments, it visited 93 distinct IDs. After every write,
the maximum immediate change in all unselected root poses, velocities, Cap
link poses, screw angles, axial positions and opening references was **0**.
The final full restore matched its starting state. Two 1024-env probe runs
measured 0.74–0.97 ms per single-env Reset averaged over 100 calls, including
Python validation, full-size PhysX buffer reads/writes and GPU work. A later
run measured 0.249 ms per batched `read()` over 100 calls. These are
diagnostic microbenchmarks, not sustained RL throughput. The 16-env tilted 45° and
0.5-rad reset still had maximum physical local-axis pitch error 0.060 µm after
100 physics steps. A right pregrasp contact probe after this Reset measured
0.273 N on the cap sidewall with +0.065 mm first contact separation.

The one-family unassisted-hold fixture now uses `MimicThreadView.reset`
instead of manually setting articulation tensors. Row 0 again held 1 s with
both correct object contacts at 100%, bottle displacement 0.063 mm, maximum
screw error 14 nm and -0.016 mm initial minimum separation. Its target
contact topology remained **false**: the right index had zero contact, so
this is not a new validated training bank.

```bash
scripts/run_isaaclab.sh scripts/environments/validate_bimanual_mimic_scene.py \
  --num-envs 1024 --steps 20 --reset-repetitions 100 \
  --random-reset-ids --headless
scripts/run_isaaclab.sh scripts/environments/validate_bimanual_mimic_scene.py \
  --num-envs 16 --steps 100 --tilt-deg 45 --initial-angle-rad 0.5 \
  --reset-repetitions 100 --random-reset-ids --headless
```

## Implicit breakaway and running drive probe (2026-11-01)

`assets/bottle/mimic_drive.py` now changes the PhysX spin joint's implicit
drive from a capped stick spring to a lower-force running damper when the
measured angular deflection and velocity exceed the configured slip criteria.
The body remains free, the cap receives no scripted pose writes, and the
hard-pitch mimic joint continues to enforce axial travel. The resistance
numbers are engineering examples, **not measured closure parameters**.

The installed Isaac Sim 6.0.1 tensor backend requires CPU indices for
`set_dof_stiffnesses`, `set_dof_dampings` and `set_dof_max_forces`, but CUDA
indices for `set_dof_position_targets`. Its runtime angular gain tensors use
radian units even though USD angular drive attributes use degrees. An initial
run mistakenly converted the runtime gains a second time: 0.005 N m then
produced 1.569 rad of slip in 0.4 s. After removing that extra conversion,
the same low-torque probe ended at 0.0233 rad without sliding. The failed
measurement is recorded here; the current JSON result is from corrected code.

| Isolated PhysX test | Measured result |
|---|---|
| Zero torque, 0.4 s | Spin approximately 0; no slide |
| +0.005 N m, 0.4 s | 0.0233 rad; no slide; pitch error 0.112 µm |
| +0.025 N m, 0.06 s | 5.45 rad; slide observed; pitch error 0.084 µm |
| −0.025 N m, 0.04 s | −3.27 rad, −1.56 mm local axial travel; slide observed |
| 0.5 rad initial angle, zero torque, selective Reset | 0.5 rad final; immediate peer change 0 |
| 1024 isolated articulations, 20 steps | 129,637 environment physics steps/s; no assertion failures |

The 0.025 N m isolated torque accelerates the ungrasped light cap to about
105 rad/s within 0.06 s; 0.05 N m reached the three-turn joint stop. These
are **not** realistic hand-controlled opening trajectories. The fixture
applies a constant external diagnostic torque and has no grasping hand.
For RL, wrist command speed and real finger contact must be tested with this
drive before its motion or success metrics are accepted.

The full two-hand scene also ran with the drive updated every physics step:
16 environments and 100 random selective Resets visited all 16 with zero
immediate peer change; 1024 environments completed 20 steps with maximum
physical pitch error 0.069 µm; and a right-hand contact probe measured
0.273 N on the cap sidewall at +0.065 mm first separation. These are scene
integration checks, not PPO or grasp success. Raw JSON and logs are in the
ignored `artifacts/phase1/mimic_running_*.json` and
`artifacts/phase1/mimic_scene_drive_*.json` files.

```bash
scripts/run_isaaclab.sh scripts/environments/probe_thread_mimic.py \
  --real-geometry --no-self-collision --implicit-running-drive \
  --num-envs 2 --steps 200 --torque 0.005 --axial-force 0.5 \
  --assert-physics --max-abs-angle 0.05 --headless
scripts/run_isaaclab.sh scripts/environments/validate_bimanual_mimic_scene.py \
  --num-envs 1024 --steps 20 --implicit-running-drive --headless
```

## Experimental DirectRLEnv integration (2026-11-01)

The shared GraspHold/CapTurn30 DirectRLEnv now accepts
`thread_backend="mimic"` through its configuration. The default remains
`legacy_d6` for prior checkpoints. `MimicThreadRuntime` reads the floating
Bottle articulation, updates the implicit spin drive every physics substep,
and exposes physical screw state to the existing reward and observation
schema. It applies no explicit equal/opposite screw wrenches or Cap root
writes. The full episode Reset restores both hands, finger drives, wrists,
Bottle root and screw DOFs, drive mode, contact history and task progress.
The Cap pose is produced by PhysX from Bottle root and screw angle.

The four legacy stable rows are intentionally **rejected** for this backend:
their 1.2–1.26 mm axial mismatch would violate the hard-pitch joint. Mimic
training and evaluation currently require `stable_reset_probability=0`, using
the 32/32 screw-compatible geometric pregrasp rows. This is an experimental
pregrasp-only setting, not a validated stable-start CapTurn curriculum.

Measured on this workstation with Isaac Sim 6.0.1 / Isaac Lab 3.0:

| DirectRLEnv check | Result |
|---|---|
| 1 env Reset/5 steps | Finite 277/421 Actor/Critic observations and reward; Cap link cache error 0 |
| 2 env, 5 selective Resets | Maximum immediate peer state change 0 |
| 16 env, 100 selective Resets | Maximum immediate peer state change 0 |
| 1024 env, 3 policy steps | Finite observations/rewards; Cap link cache error 0 |
| 1024 env, 100 selective Resets | Maximum immediate peer state change 0 |
| 1024 env, 100 policy steps, 10 warmup | 10,271 environment steps/s; 102,707 physics environment steps/s; 8,684 MiB used GPU memory |
| Default legacy D6, 2 env/5 steps | Smoke passed |

These checks establish a running vectorized environment, **not** a learned
grasp or contact-driven cap turn. The natural pregrasp source has no
demonstrated stable hold in the hard-pitch backend. Independent policy
evaluation, contact topology, PPO learning and cross-family generalization
remain open. The previous strict CapTurn success rates refer to the legacy
D6 backend only and are not transferred to Mimic.

```bash
scripts/run_isaaclab.sh scripts/environments/smoke_bimanual_rl.py \
  --num-envs 1024 --steps 3 --thread-backend mimic --headless
scripts/run_isaaclab.sh scripts/environments/validate_bimanual_rl_reset.py \
  --num-envs 1024 --resets 100 --thread-backend mimic --headless
scripts/run_isaaclab.sh scripts/environments/benchmark_bimanual_rl.py \
  --num-envs 1024 --steps 100 --thread-backend mimic --headless
```
