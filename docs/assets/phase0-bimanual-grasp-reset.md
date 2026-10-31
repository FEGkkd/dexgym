# Phase 0: suspended bimanual grasp initialization, contact, and Reset

## Scope and status

This phase adds reusable initial-state, PhysX contact, and selective Reset code to
the existing standalone Isaac Lab scene. It does not register an RL task or add
observations, rewards, or training. The pinned **WujiHand2 Beta 1** USD remains
an unconfirmed candidate for the user's untactilized hardware. Official source:
`wuji-technology/wuji-description` tag `v2026.9.11`, commit
`c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95`. Left USD SHA-256:
`8d1b83a6130b64fd759ea3c0720d6b255da8c1e76abe5a4226788183058dbfda`;
right USD SHA-256:
`22600d3f0aeb185552bb6eeb454b5b7199966a60fc844d4a4b5e4791ec77f0ed`.

**No stable suspended grasp has passed.** The `stable_pregrasp` generator emits
reproducible *candidates* with `validated=false`. The bank is intentionally
absent until a 1-second gravity-on, unsupported hold and left-only ablation both
pass. A candidate's geometric contact is never promoted into a grasp claim.

## Interfaces

- `BimanualGraspInitializer.generate(mode, seed, bottle_pose)` returns a complete
  `GraspInitSample`: bottle/cap and two floating roots, velocities, 40 canonical
  finger positions/velocities/targets, two wrist PD targets, initial thread
  engagement/angle, source commit, hashes, scene parameters, units, and seed.
  The modes are `stable_pregrasp`, `weak_contact`, and `near_contact`; only the
  first is searched physically in this phase. All stored poses are in the env-0
  world frame, metres and **xyzw** quaternions. Hand poses are composed from
  bottle/cap-local transforms so a tilted bottle keeps the grasp geometry.
- `BimanualContactMonitor` creates exact per-environment native PhysX
  `RigidContactView` filters. Bottle and cap are each a single sensor source
  against individual left/right hand links and ground. Left links are sensor
  sources against right links for hand-hand detection. Force matrices provide
  normal-force vectors; `get_friction_data` and `get_contact_data` supply
  friction and position when available. Histories store body names, force
  peak/mean, active time/ratio, and latest contact point. `0.01 N` is only a
  numerical event floor observed to give no false positives in the isolated
  no-contact fixture; it is **not** a grasp-strength threshold. No reliable
  relative slip metric is exposed yet. A Beta 1 distal-body collision does
  not prove contact at the physical fingertip site.
- `BimanualResetManager.reset(env_ids, samples)` restores indexed root state,
  canonical-to-Isaac finger state and targets, wrist PD anchors/targets, thread
  angle/stick/sliding/detachment state, D6 enabled/collision flags, and contact
  history. It adds `scene.env_origins` only when writing to PhysX. It never
  calls `sim.step()`. Contact summaries are invalid until the next physics
  update. `ThreadCoupler.reset_from_pose` checks the requested unwrapped angle
  against cap orientation modulo one turn and re-engages a detached coupler.
- `BimanualPhase0Fixture` applies normal finger position drives, bounded wrist
  root wrenches, and equal/opposite thread wrenches. The bottle and cap remain
  free rigid bodies under gravity. It generates **runtime** USD derivatives in
  a temporary directory to avoid rewriting locally edited model files; the
  generator source and canonical assets stay in this repository.

## Reproduce

From the repository root, with official assets and Isaac Lab installed:

```bash
export WUJI_DESCRIPTION_ROOT=/path/to/wuji-description-v2026.9.11
export ISAACLAB_ROOT=/path/to/IsaacLab
export ISAACLAB_PYTHON=/path/to/isaacsim/bin/python
./scripts/run_isaaclab.sh scripts/environments/validate_bimanual_contacts.py --headless --output docs/assets/phase0_contact_results.json
./scripts/run_isaaclab.sh scripts/environments/validate_bimanual_reset.py --headless --num_envs 1 --iterations 100 --output docs/assets/phase0_reset_1.json
./scripts/run_isaaclab.sh scripts/environments/validate_bimanual_reset.py --headless --num_envs 2 --iterations 100 --output docs/assets/phase0_reset_2.json
./scripts/run_isaaclab.sh scripts/environments/validate_bimanual_reset.py --headless --num_envs 16 --iterations 100 --output docs/assets/phase0_reset_16.json
./scripts/run_isaaclab.sh scripts/environments/validate_suspended_grasp.py --headless --seeds 0 1 2 --output docs/assets/phase0_suspended_grasp_results.json --bank docs/assets/grasp_init_bank.json
./scripts/run_isaaclab.sh scripts/environments/generate_grasp_init_bank.py --headless --report docs/assets/phase0_grasp_search_results.json --bank docs/assets/grasp_init_bank.json
./scripts/run_isaaclab.sh scripts/environments/generate_grasp_init_bank.py --headless --left_only_probe --short_duration 0.1 --report docs/assets/phase0_left_only_search_results.json
./scripts/run_isaaclab.sh scripts/environments/preview_grasp_initialization.py --headless --steps 500
./scripts/run_isaaclab.sh scripts/environments/preview_grasp_initialization.py --viz kit --contact_debug --steps 500
./scripts/run_isaaclab.sh scripts/environments/smoke_task.py --task DexterousHands-Wuji-Hand-Reorientation-Direct-v0 --num_envs 1 --steps 3 --headless
```

The legacy Wuji smoke on this workstation used
`WUJI_DESCRIPTION_ROOT=/home/ubuntu/wuji-description`; the Phase 0 scripts
used the pinned Beta checkout at `/tmp/wuji-description-v2026.9.11`.

`--bank` writes **only** passing samples. Until one exists, omit `--bank` for
preview; this shows a labelled unverified candidate. When a bank exists, pass
`--bank docs/assets/grasp_init_bank.json --sample_index 0` to preview it.

The contact classification fixture translates the bottle/cap assembly once
at an episode boundary to exercise each pair. Its strong contact forces are
not grasp validation. Reset and hold loops never teleport objects or hands.

## Measured results on this workstation

Isaac Sim 6.0.1, Isaac Lab 3.0, RTX 5090, 2 ms step, engineering-example
bottle (unmeasured 30 g shell and 6 g cap). Raw machine-readable results are
in `phase0_contact_results.json`, `phase0_reset_{1,2,16}.json`,
`phase0_suspended_grasp_results.json`, and `phase0_grasp_search_results.json`.
`phase0_left_only_search_results.json` records an additional 45-candidate
unsupported left-hand probe.

| Gate | Status | Evidence and limit |
| --- | --- | --- |
| Compatibility | Pass for tested paths | Legacy Wuji Direct task GPU smoke: `SMOKE_OK`, observation 207, critic 413, action 20. Full suite: 81 passed, 2 failed, 1 skipped; the same Allegro config and untracked checkpoint-helper tests failed in the recorded baseline (72 passed, 2 failed, 1 skipped). |
| Complete initial-state representation | Pass | Offline contracts cover all three modes, bottle-local transforms, joint/target counts, provenance, and refusal to bank an unvalidated sample. |
| Contact classification | Pass for deliberate contact fixtures | No-contact envs produced empty histories; left-bottle, right-cap, left-cap, right-bottle, hand-hand and detached-cap-to-ground were detected with per-body names, finite normal/friction force and contact positions. Env 1 remained empty while only env 0 was touched. Forced hand overlap peaked near 338 N and is not a valid physical grasp state. Bottle-ground appeared in the falling hold cases. |
| Indexed Reset | Pass for 1/2/16 environments | 100 selective resets per count, immediate nonselected-state comparison, targets and detached-to-engaged recovery, no global physics step inside Reset, finite post-reset steps. Maximum selected-root pose/velocity error `2.38e-7`; joint/target error `0`; initial wrist PD force below `0.0002 N`. See JSON results. |
| Suspended stable grasp | **Fail** | Three seeds held for 1 s with gravity and no support: bottle displacement `0.489`, `0.529`, `0.569 m`; correct left/right contact ratios were at most `0.014`/`0.026`. Left-only ablation also failed. No validated bank entry. |
| Free-fall control | Pass | No-contact bottle moved `0.0500 m` in 0.1 s versus ideal `0.0491 m`, proving gravity and absence of hidden support. |
| GUI preview | Pass for launch; human inspection Not Tested | `--viz kit --contact_debug --steps 5` launched and emitted live pose/contact frames; headless five-step run also passed. These exits do not establish visual grasp quality. |

The bimanual geometry search tried nine deterministic finger-MCP/wrist
candidates and found zero passing samples. Three one-second candidates began
with PhysX contact separation `-0.038` to `-0.042 m`, so their apparent initial
contact reflects severe overlap. The left-only probe tried 45 candidates for
0.1 s: eight (17.8%) had no measured first-step penetration deeper than 5 mm
(a contact-distance proxy, not a full geometry clearance proof), but none of
those sustained meaningful contact or arrested free fall. No candidate
qualified for a long left-only trial. Some forced contacts exceeded hundreds
of newtons; they must not be used as RL initial states.
The next physical step is a supported **candidate-generation-only** scripted
closing/pose search, followed by release and fresh gravity-on acceptance.
Actual finger geometry, friction, bottle mass and cap torque must be measured
before treating any grip threshold as representative of hardware.

## Pending before an RL task

1. Find at least three independent seeds that pass 1-second unsupported hold,
   left-only ablation, both contact ratios, motion/tilt and force guards, then
   store their full states in the bank.
2. Measure contact threshold false-positive/false-negative behaviour over more
   gentle approaches and compare distal collision points with fingertip sites.
3. Quantify relative slip using body-local contact points, longer hold horizons,
   and reset under a physically separated cap with D6 re-engagement.
4. Repeat with the user's measured bottle and verified hand revision. Until
   those gates pass, a contact-based RL opening task should not be registered.
