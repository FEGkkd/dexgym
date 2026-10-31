# Phase 0D: left-only suspended grasp and Reset

Status: **four fixed-wrist left grasps passed** a gravity-on, unsupported 1 s
hold, an indexed `BimanualResetManager` replay in the same simulation, and a
second replay after loading the persisted bank in a fresh Isaac Sim process.
These are engineering-sample bottle results, not measured real-bottle or
real-hardware validation. The right hand is parked and does not support the
bottle. No RL task or reward was added.

## Runtime and provenance

- Isaac Sim 6.0.1, Isaac Lab 3.0, PhysX GPU, 2 ms physics step.
- Official WujiHand2 Beta 1 left/right USD from `wuji-description` commit
  `c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95`. Left USD SHA-256:
  `8d1b83a6130b64fd759ea3c0720d6b255da8c1e76abe5a4226788183058dbfda`.
  Beta 1 remains a **candidate** for the user's non-tactile hardware; its
  hardware identity has not been verified.
- Engineering bottle: radius 35 mm, height 200 mm, bottle mass 30 g, cap mass
  6 g, bottle static/dynamic friction 0.6/0.4. The native USD hand material
  uses 0.7/0.7. These values are simulation settings, not bottle measurements.
- The left root uses the official world fixed joint. All twenty finger links,
  collisions, inertias, official USD position drives and 2 N·m drive limits
  remain active. Bottle and cap are dynamic, gravity is on, and the threaded
  cap is not world anchored. The temporary bottle support joint is disabled
  before the measured hold and before every Reset replay.

## What caused the earlier failures

The Phase 0B half-closed initial state put hand colliders about 4 cm into the
70 mm bottle. The Phase 0C directed preshapes fixed hand-bottle clearance but
introduced **hand self-collision**. With the bottle moved one metre away, the
old proposal 6 still reached 18.89 rad/s on the first steps. Its middle and
ring finger colliders overlapped by up to 11.13 mm, with reported first-step
normal impulses over 1 kN. Disabling self-collision for a diagnostic-only
comparison reduced the maximum speed to 0.026 rad/s. Formal runs keep hand
self-collision enabled.

`prepare_collision_free_preshape.py` now constrains cross-finger convex
distances as well as hand-bottle distance. The corrected preshapes had no
native PhysX hand self-contact in a 25-step free-hand test, and maximum joint
speeds of about 0.02–0.03 rad/s. The optimizer is a *proposal generator*;
native USD and PhysX decide acceptance.

An additional geometry error in the search objective was to count a thumb
touch at the cylinder's top rim as a sidewall pinch. In one failed candidate
the thumb contact was at `z=0.600 m`, exactly the 200 mm cylinder top, so the
thumb lost contact when the bottle began to fall. The optimizer now records
contact height and penalizes candidates within a configured margin of either
rim. We also confirmed that the existing live USD `*_tip` marker positions
agree with official MJCF sites within 0.11–0.24 mm when compared at the **same
joint state**; a previous comparison against stale proposal coordinates was
invalid. No fingertip-tracker change was needed.

## Physical synthesis and acceptance

For each bank entry, a geometry-checked open preshape starts about 3–5 mm
clear of the bottle. Normal PhysX finger position drives close over 1.2 s.
The bottle support is present only during formation and a 0.5 s settling
period. Finger targets may lie slightly behind the nominal bottle surface to
request a bounded grip preload; actual PhysX contact penetration remained
below 0.07 mm. No finger or bottle pose is written during physics steps.

The original angular-coverage metric required contact patches to span more
than 180° at every step. That excluded a real frictional pinch: the bottle
remained suspended for 1 s, with thumb contact throughout and middle-finger
contact for 0.93–0.98 s, although the strict 180° coverage ratio was zero.
The analyzer now reports both full angular coverage and the configured
friction-cone pair condition. The latter is intermittent (about 58% for the
first sample) and is retained as a diagnostic, **not** claimed as a continuous
force-closure proof. Bank admission requires the directly observed 1 s
gravity-on hold, under 20 mm displacement and 10° rotation, contact ratio
over 80%, sustained thumb and another digit, a third supporting digit,
no ground contact, under 2 mm penetration, under 100 N formation peak, and
a separately reset 1 s replay passing the same physical checks.

| Seed | Wrist change from seed 204 | Hold displacement | Rotation | Formation peak | Hold contact | Reset replay displacement | Fresh-process replay |
|---|---:|---:|---:|---:|---:|---:|---:|
| 204 | reference | 7.50 mm | 5.54° | 56.6 N | 100% | 8.43 mm | 7.69 mm, pass |
| 235 | X −0.5 mm | 7.27 mm | 3.96° | 53.8 N | 100% | 7.07 mm | 7.05 mm, pass |
| 245 | X +0.5 mm | 7.98 mm | 5.61° | 57.4 N | 100% | 7.79 mm | 7.96 mm, pass |
| 304 | Z −10 mm | 6.58 mm | 6.13° | 50.7 N | 100% | 6.56 mm | 6.54 mm, pass |

The maximum measured penetration across these formation/hold runs was under
0.07 mm. Unsupported hold peak forces were 8.2–10.3 N; formation peaks are
shown above because they are the anomaly guard. The fixed wrist had exactly
zero measured pose drift during all four contact holds. The bottle centre,
orientation, velocity, per-link contact position/force and joint torque
saturation traces are in [the telemetry JSON](phase0d_left_grasp_telemetry.json).
Complete raw PhysX result JSON is retained as `phase0d_physx_<seed>.json.gz`;
decompressing each file reproduces the SHA-256 in its bank entry.

The Z +10 mm diagnostic (seed 305) **failed**: 94.1 mm displacement, 18.3°
rotation and no Reset replay. It was not put in the bank. This shows the bank
does not represent a large certified grasp region. The four passing samples
form a local cluster, with one verified 10 mm height variation.

## Reproduce and use

Set `WUJI_DESCRIPTION_ROOT` to the pinned official checkout and
`ISAACLAB_PYTHON` to the Isaac Sim Python environment. For example:

```bash
export WUJI_DESCRIPTION_ROOT=/tmp/wuji-description-v2026.9.11
export ISAACLAB_PYTHON=/home/ubuntu/miniconda3/envs/allo_track/bin/python
./scripts/run_isaaclab.sh scripts/environments/generate_grasp_synthesis.py \
  --reach_scan docs/assets/phase0d_grasp_preshape_204.json \
  --proposal_index 204 --control_mode fixed_wrist --fixed_support \
  --close_s 1.2 --settle_s 0.5 --headless --output /tmp/grasp204.json
./scripts/run_isaaclab.sh scripts/environments/validate_left_grasp_bank.py \
  --bank docs/assets/phase0d_left_only_grasp_init_bank.json \
  --sample_index 0 --headless --output /tmp/replay204.json
```

The same commands apply to seeds 235, 245 and 304 and bank indices 1, 2 and
3. Omit `--headless` and add `--visualize_grasp` to the synthesis command for
GUI fingertip, bottle-axis and contact markers. The validated sample bank
stores canonical 20+20 joint state and targets, both root poses and velocities,
wrist targets, bottle/cap poses and velocities, ThreadCoupler snapshot,
engagement, seed, official asset SHA and control mode. The bank's `scope` is
`left_only`; it does not assert right-hand cap grasp. Use
`BimanualResetManager.reset([env_id], [sample])` with a fixed-wrist scene whose
root anchor is authored at the sample's wrist pose.

The offline curation command rechecks the evidence and refuses failures:

```bash
PYTHONPATH=source "$ISAACLAB_PYTHON" scripts/environments/curate_left_grasp_bank.py \
  /tmp/grasp204.json --bank /tmp/rechecked_left_bank.json
```

The independent indexed Reset regression used `validate_bimanual_reset.py`
with `--control_mode pose_command_wrist --iterations 100 --num_envs` set to
1, 2 and 16. All three runs recovered a detached test cap, restored finger
targets and thread state, invalidated contact history, and left other envs
unchanged at the no-physics-step boundary. Maximum root state error was
2.38e-7 at 16 envs. Raw summaries are `phase0d_reset_{1,2,16}.json`.

## Gates and remaining limits

| Phase 0C gate | Status | Evidence / limit |
|---|---|---|
| A: official hand joints, mass, collisions | Pass | Pinned USD checks; native self-collision was kept and measured. |
| B: fixed wrist with dynamic fingers | Pass | Zero wrist drift and finite joint torques/contact forces. |
| C: live fingertip sites | Pass | Live USD/MJCF agreement within 0.24 mm at matched joint state. |
| D: opposing contact and forces measured | Pass | Per-body points, normals, friction data and torque traces saved; continuous force closure was not established. |
| E: unsupported left-only hold | Pass | Four 1 s samples, gravity on, no external support. |
| F: pose-command wrist contact | Not Tested in this phase | Earlier Phase 0C evidence remains; this phase changed no wrist trajectory code. |
| G: pose-command wrist grasp | Not Tested | Validated bank uses fixed wrist only. |
| H: independent Reset replay | Pass | Four same-scene and four fresh-process replays. |
| I: 1/2/16 env Reset and old tasks | Fail overall | Indexed Reset passed 100 iterations at each size; full offline suite had 100 pass, 1 skip, 2 failures in pre-existing user-dirty Allegro/MANO work. |

The full offline command was `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
./scripts/run_isaaclab.sh -m pytest -q tests`. The two failures were
`test_allegro_config_declares_contract_and_resolves_reference` (missing
Allegro Fly asset in its temporary fixture) and
`test_build_21_keypoints_uses_mano_order_and_fingertips` (MANO mapping
expectation). Those files were already modified/untracked before this work
and were not changed here. Focused grasp/contact tests passed 18/18.

The next phase still needs right-hand cap contact, bimanual hold, pose-command
wrist grasp, longer holds and perturbation tests. The bottle geometry,
friction and mass are engineering examples; real-bottle calibration and
hardware model identification remain open. No PPO, reward or observation
contract was added.
