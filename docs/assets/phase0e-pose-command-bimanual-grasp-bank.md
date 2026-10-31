# Phase 0E: pose-command wrists and reproducible bimanual grasp states

**Measured result:** a free engineering bottle and threaded cap were held by
the left and right official WujiHand2 Beta 1 hands for 1 s with gravity on,
without a ground contact, bottle support, world-anchored cap, or bottle pose
updates. Four full states passed an unsupported hold, a same-process indexed
Reset replay, and a fresh-process Reset replay. They belong to **one relative
grasp family**: the release state, two later physical snapshots (0.2/0.4 s),
and a 5° world-yaw transform. This is a reproducible *local* bank, not four
independent grasp geometries or a certified robust grasp region.

No opening policy, reward, PPO, or final observation was added.

## Hardware and physics boundary

- Native left/right Beta 1 USD: `wuji-description` tag `v2026.9.11`, commit
  `c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95`; SHA-256 left
  `8d1b83a6130b64fd759ea3c0720d6b255da8c1e76abe5a4226788183058dbfda`,
  right `22600d3f0aeb185552bb6eeb454b5b7199966a60fc844d4a4b5e4791ec77f0ed`.
  Both use 20 active official finger joints. Beta 1 has **not** been proven to
  match the user's physical hand. The earlier 1 g positive-mass compatibility
  override for each official zero-mass fingertip marker remains in place.
- Isaac Sim 6.0.1, Isaac Lab 3.0, GPU PhysX, 2 ms step. The ideal
  `pose_command_wrist` uses a world-fixed articulation root anchor and a
  speed-limited root pose/velocity trajectory at each physics substep. Contact
  does not integrate into wrist displacement. The fingers keep normal dynamic
  articulation position drives and the official 2 N·m drive limit. The
  floating-wrench controller remains an explicit legacy comparison mode.
- Bottle radius 35 mm, body mass 30 g, cap mass 6 g; materials and thread
  values are the repository's **engineering examples**, with no real bottle
  calibration. Bottle/cap remain dynamic and use the existing D6 joint and
  `ThreadCoupler`. Gravity and hand self-collision are enabled. No new
  kinematic bottle following is used.
- The hand action contract remains 26 per side / 52 total: world translation
  delta (m), wrist-local rotation vector delta (rad), then 20 canonical finger
  targets (rad). Stored pose quaternions use `xyzw`; the official MJCF geometry
  proposer uses `wxyz` and is converted explicitly. The right proposal is
  geometry-only; USD/PhysX determines admission.

## Formation and acceptance

The fixed-wrist left bank from [Phase 0D](phase0d-left-only-grasp-reset.md)
was re-synthesized with pose-command wrists, retaining official USD collision,
finger drives and a temporary bottle support **only during the original left
formation**. Seeds 204 and 245 passed their own unsupported 1 s hold and
fresh-process replay, with worst displacement 7.82 and 7.83 mm respectively.
Seed 235 failed Reset replay; seed 304 failed fresh-process replay. The
[pose-left bank](phase0e_pose_left_grasp_init_bank.json) therefore has exactly
two validated states. In a separate 2 s left-only diagnostic, both fell about
0.476 m after contact was lost. The left-only claim is strictly **1 s**. The
failed 2 s raw traces are
[seed 204](phase0e_pose_left_2s_seed204.json.gz) and
[seed 245](phase0e_pose_left_2s_seed245.json.gz).

The right-hand sidewall search uses the official right Beta 1 MJCF only to
propose collision-cleared wrist and independent thumb/middle finger targets.
The [reach scan](phase0e_right_cap_sidewall_reach.json),
[contact refinement](phase0e_right_cap_sidewall_refined45066.json), and
[open preshape](phase0e_right_cap_sidewall_preshape45066.json) record the
proposal. Right formation starts from a proven pose-left sample and uses no
additional bottle support. Finger targets close over 0.35 s. Before the first
effective cap contact, the right ideal wrist may follow bounded **XY** cap
translation; it then latches the target. Tracking downward cap movement after
top pressing was rejected because it fed the contact-induced motion back into
the wrist command. No object pose is written during a physics step.

Admission requires left-bottle and right-cap contact over 80% for 1 s; thumb
and another digit on each side over 80%; right thumb/other **opposed radial
contact normals** over 80%; bottle displacement below 20 mm and rotation below
10°; no ground contact; no significant cross-hand or hand-hand contact; cap
motion constrained to the bottle-local thread axis; initial and worst
penetration under 2 mm; and force peaks below 100 N. Both same-process and
fresh-process Reset replay must independently satisfy those guards. The
100 N/2 mm guards detect anomalies, not measured hardware limits. A cap-rim
point only counts as radial contact when its normal is mostly transverse to
the cap axis; pure axial top pressing is rejected. The measured successful
patch is near the **upper sidewall/rim**, not evidence of a deep cylindrical
sidewall pinch or certified force closure.

| Bank index | State | Worst displacement across 3 holds | Right opposed-radial ratio | Worst penetration | Peak force, including formation | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | seed 204 release | 0.078 mm | 100% | 0.156 mm | 8.87 N | Pass |
| 1 | same grasp, +0.2 s PhysX snapshot | 0.049 mm | 100% | 0.036 mm | 2.77 N | Pass |
| 2 | same grasp, +0.4 s PhysX snapshot | 0.020 mm | 100% | 0.043 mm | 2.95 N | Pass |
| 3 | same relative grasp, +5° world yaw | 0.068 mm | 100% | 0.038 mm | 2.68 N | Pass |

For the original release hold, left/right correct contact ratios were both
100%; right thumb and middle contact each lasted 1.0 s. The smallest opposed
radial normal dot product was about -0.999. The largest bottle-local cap
lateral error across its three holds was 17.5 µm. Cross-contact and ground
ratios were zero. Its 1 s bottle displacement was 0.069 mm, and the
fresh-process replay was 0.069 mm. The right applied-finger torque tensor
reached, but did not exceed, the official 2 N·m drive limit; maximum sampled
right finger speed was 0.82 rad/s during formation. Full per-body contact
points, normals, force samples, torque vectors, cap/thread motion, and root
trajectories are in the compressed raw results. The earlier
[left telemetry](phase0d_left_grasp_telemetry.json) records left joint and
contact details. PhysX exposes pair-level friction data; the current backend
does not justify attributing that vector to an individual contact point.

The 0.2/0.4 s samples are real states captured after dynamic PhysX stepping,
with measured rigid-body and joint velocities. They differ by only tens of
micrometres in bottle position and share the same relative grasp geometry.
The +5° sample rotates all roots, velocities, wrist targets and objects
together, then passes its own independent physics checks. A +30° transform
failed opposed-radial coverage (0.2%) although contact persisted. A 0.3 mm
right-wrist Y change and using pose-left seed 245 also had 0% opposed-radial
coverage. These three rejected cases are not in the bank. This sensitivity
limits its use as a broad RL reset distribution.

## Files and reproducibility

- [Bimanual bank](phase0e_bimanual_grasp_init_bank.json) stores complete
  canonical 20+20 finger state/targets, two wrist roots and targets,
  bottle/cap poses and velocities, thread/engagement snapshot, seed, asset
  hashes, control mode and metric/evidence SHA-256. The
  [numeric manifest](phase0e_physx_results.json) names the 11 compressed
  passing/failing raw JSON files and their **uncompressed** SHA-256 hashes.
  Decompress with `gzip -dc FILE.json.gz > /tmp/FILE.json` before replay.
- `BimanualGraspSynthesisFixture` forms the right contact and measures
  per-object contact, radial normals, penetration, forces and bottle-local
  thread drift. `curate_bimanual_grasp_bank.py` refuses a candidate without
  all three passing no-support runs and an exact raw-result SHA match.
- `preview_grasp_initialization.py` now creates the scene in the stored
  sample's wrist mode and initial root frames, so a pose-command bank can be
  inspected through the same GUI/headless entry point. A headless 20-step
  preview loaded index 0 and reported left-bottle/right-cap contact on every
  sampled step. GUI readability itself was not formally scored.

On a GPU workstation with the pinned official asset checkout:

```bash
export WUJI_DESCRIPTION_ROOT=/tmp/wuji-description-v2026.9.11
export ISAACLAB_PYTHON=/home/ubuntu/miniconda3/envs/allo_track/bin/python
./scripts/run_isaaclab.sh scripts/environments/generate_bimanual_grasp_bank.py \
  --headless --left_bank docs/assets/phase0e_pose_left_grasp_init_bank.json \
  --left_sample_index 0 \
  --right_preshape docs/assets/phase0e_right_cap_sidewall_preshape45066.json \
  --right_proposal_index 45066 --close_s 0.35 --settle_s 0.05 \
  --follow_cap --hand_hand --output /tmp/bimanual_204.json
./scripts/run_isaaclab.sh scripts/environments/validate_bimanual_grasp_candidate.py \
  --headless --result /tmp/bimanual_204.json \
  --output /tmp/bimanual_204_fresh.json
PYTHONPATH=source "$ISAACLAB_PYTHON" scripts/environments/curate_bimanual_grasp_bank.py \
  --formation /tmp/bimanual_204.json \
  --fresh_replay /tmp/bimanual_204_fresh.json \
  --left_bank docs/assets/phase0e_pose_left_grasp_init_bank.json \
  --bank /tmp/rechecked_bimanual_bank.json
./scripts/run_isaaclab.sh scripts/environments/preview_grasp_initialization.py \
  --bank docs/assets/phase0e_bimanual_grasp_init_bank.json \
  --sample_index 0 --contact_debug --steps 500
```

Add `--headless` to the preview command for CI or remove it for a GUI. For a
later physical snapshot use `derive_bimanual_grasp_snapshot.py` with
`--parent_result`, `--parent_fresh_replay`, `--offset_s 0.2` and `--output`;
for the 5° orientation experiment use `validate_bimanual_world_yaw.py` with
the same parent inputs plus `--yaw_deg 5`. Both still need a fresh-process
replay and curation. `GraspInitBank.load()` returns validated samples usable
by `BimanualResetManager.reset(env_ids, samples)` in a scene using
`pose_command_wrist` and matching asset/config hashes.

## Gates, tests and limits

| Phase 0C gate | Status | Evidence / limit |
| --- | --- | --- |
| A: official Beta 1 hands, mass, collision | Pass | Pinned USD, 20+20 joints and marker-mass fix retained; native collision/torque data measured. |
| B: fixed wrist and physical fingers | Pass from Phase 0D | Four fixed-wrist unsupported 1 s holds; no fixed-wrist implementation changed here. |
| C: dynamic five fingertip sites | Pass from Phase 0C/0D | Live PhysX markers and USD/MJCF matched-state comparison; no site tracker changed here. |
| D: opposing contact and forces | Pass for measured pairs | Right radial opposition, exact points/normals, pair friction and joint torque recorded; no formal force-closure proof. |
| E: fixed-wrist left 1 s hold | Pass from Phase 0D | Four persisted and freshly replayed states. |
| F: pose-command 6-axis trajectory and contact | Pass from Phase 0C | Existing two-env 52D motion and static/moving contact comparison; controller unchanged here. |
| G: pose-command grasp | Pass for static 1 s | Two left-only and four bimanual static-target states pass; 2 s left-only hold fails and commanded moving-grasp hold is not tested. |
| H: independent Reset replay | Pass for 1 env | Four bimanual states pass same-process and fresh-process 1 s replays. |
| I: 1/2/16 env Reset and old-task regression | Mixed | 100 indexed resets at each size pass; full pytest is 105 pass, 1 skip, 2 failures in pre-existing user-worktree Allegro/MANO tests. |

The focused grasp/asset/contact/reset/thread suite was **51 passed**. The
complete Isaac Lab wrapper run used
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 WUJI_DESCRIPTION_ROOT=... ./scripts/run_isaaclab.sh -m pytest -q tests`
and yielded **105 passed, 1 skipped, 2 failed**. Failures are
`test_allegro_config_declares_contract_and_resolves_reference` (missing
Allegro Fly URDF in that user's temporary fixture) and
`test_build_21_keypoints_uses_mano_order_and_fingertips` (MANO order
assertion). Both were already present in the user-dirty worktree and were
reported in Phase 0D. Neither subsystem was modified in this phase.
The `pose_command_wrist` indexed Reset validator passed 100 selective resets
for each of 1, 2 and 16 environments, with unselected state unchanged at the
no-physics-step boundary; worst root restore error was `2.38e-7`.
[Reset 1](phase0e_reset_1.json), [Reset 2](phase0e_reset_2.json), and
[Reset 16](phase0e_reset_16.json) retain the measured summaries.

**Next experiment:** obtain a second relative cap-grasp geometry with loaded
radial opposition over a larger wrist/finger neighborhood. Then test
commanded wrist motion while holding, longer holds, randomized resets and
real-bottle mass/friction/torque calibration. The current 4-state bank is
valuable for deterministic scene initialization and contact/debug tests but
should not be treated as a robust RL start distribution.
