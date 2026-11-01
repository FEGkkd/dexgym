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
non-root-write warning. **A formal RL Reset adapter has not been implemented**.
It must restore the floating articulation root, both screw DOFs, targets and
velocities atomically, then audit contacts and peer state. Writing an arbitrary
legacy Bottle pose and Cap pose independently would conflict with the hard
pitch constraint.

An offline bank audit showed that **0/4** committed stable grasp samples and
**0/1** local composite PPO sample satisfy the hard-pitch axial relation at
their saved cap angles. Their initial axial discrepancies are about 1.20–1.26
mm and 0.457 mm respectively. The **32/32** geometry-qualified natural
pregrasp samples are screw-compatible, but earlier policy evaluation from
natural pregrasp was 0/200. None of the old stable samples may be silently
used as validated Reset states for this backend. The existing DirectRLEnv
continues to use the legacy D6 plus explicit `BatchedThreadCoupler`; the new
authorer rejects an existing legacy D6 on the same assembly.
The four committed stable rows are variants of one seed-204 grasp family;
their measured lateral cap offset is only 2–9 µm, so it is specifically the
axial mismatch that fails this gate. Forcing such poses into the joint would
introduce a millimetre-scale solver correction at Reset, so no dynamic hold
pass is inferred from them.

The implicit drive is bounded and stable in the prior isolated probe under
low and high diagnostic torques. It does not yet implement the example
breakaway-to-running-resistance transition, and its current max-force value
is not a measured real-bottle torque. The isolated probe's explicit viscous
resistance at the original damping was numerically unstable; that path must
not be installed as the production backend. One gentle right-middle-finger
sidewall contact was measured in the full scene, but bilateral grasp,
contact-driven 30° rotation, full Reset, sustained 1024-env training
throughput, and learned opening remain **not tested**. A new
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

The next small physics experiment should adjust left index/ring and right
index joint targets around this stable state, with the wrists and hard-pitch
assembly held fixed. It should require distributed target-object contact,
normal-force balance, shallow penetration and an independent 1 s replay
before adding any row to a new bank. A contact-topology-verified bank and a
batched articulation Reset are prerequisites for meaningful hard-pitch PPO.
