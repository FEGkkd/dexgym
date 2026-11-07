# Layered validation of the Hand-only Newton position drive

**Date:** 2026-11-08. **Scope:** three diagnostic stages, no RL training. The
shared 54-DoF USD, task registration, PPO configuration, GRAB bank, and 10 mm
success definition were not changed.

## 1. Isolated USD angular-drive correction

`scripts/assets/build_hand_only_drive_overlay.py` authors separate USD layers
that sublayer the original packaged robot. The original MJCF and USD remain
untouched. USD angular-drive stiffness and damping are authored as the source
MuJoCo `Kp`/`Kd` multiplied by `π/180`, because Newton's runtime buffers
multiply those USD values by `180/π`. There are two contact variants for each
of two diagnostic profiles:

- `source`: all 54 runtime joint gains match the source MJCF.
- `hand_x2`: all 54 gains first match the source; only the 20 right-hand Kp and
  Kd values are then doubled in the isolated USD layer.

`HandOnlyDriveExperimentCfg` explicitly selects these assets. The registered
RL task continues to use its existing robot configuration. Runtime readback in
Newton found maximum all-joint gain error **3.8e-8** for `source`, and **7.6e-8**
relative to the expected hand-scaled values for `hand_x2`. Effort-limit error
was below **2.4e-8 N·m**. The source-parity index MCP had `Kp=0.373522` and
`Kd=0.018823`; the tuned hand profile had `Kp=0.747044` and `Kd=0.037645`.

## 2. Timestep, collision, solver, and effective torque

For a Home-pose index-MCP +0.1 rad step, hold the target for 1.5 s and sample
each physics step. The corrected `source` profile uses the same gain and effort
limit across all six conditions. Home produced **no active hand contacts**;
collision on/off therefore tests the asset switch but does not excite contact.

| Physics dt | Hand contacts | Max index-MCP speed in last 0.2 s | Final index-MCP error | Drive saturated? |
| --- | --- | ---: | ---: | --- |
| 1/120 s | On / off | 3.35 / 3.23 rad/s | 0.066 / 0.073 rad | No |
| 1/240 s | On / off | 2.43e-5 / 2.43e-5 rad/s | 0.0883 rad | No |
| 1/500 s | On / off | 2.68e-5 / 2.68e-5 rad/s | 0.0883 rad | No |

This isolates a **physics-step numerical stability problem** at 1/120 s with
source-matched gains. The index joint did not exceed its limit and its actuator
did not reach the 2 N·m force limit. At 1/240 and 1/500 s the joint settles to
the same approximately 0.1883 rad equilibrium observed in source MuJoCo,
leaving a **static 0.0883 rad offset** from the 0.1 rad command. Timestep
reduction fixes the dynamic oscillation; it does not remove this source-model
steady-state bias.

The same 324 GRAB start/goal pairs were then run with the corrected `source`
profile across the three timesteps and both collision settings. Define
"settled" as the maximum absolute speed of any right-hand joint during the
last 0.2 s being below **0.05 rad/s**. Define pass as settled **and** all five
fingertips below **10 mm** at 1.5 s.

| Physics dt | Contacts on: settled / pass | Contacts off: settled / pass | Failure type |
| --- | ---: | ---: | --- |
| 1/120 s | 0 / 0 | 0 / 0 | All 324 have dynamic error |
| 1/240 s | 324 / 286 | 324 / 290 | Stable, remaining error is static |
| 1/500 s | 324 / 286 | 324 / 290 | Stable, remaining error is static |

Across the stable source-gain runs, joint-limit violation was at most about
**0.0013 rad**. The full-contact source profile's worst fingertip error at
1/240 s was **11.29 mm**, and its mean absolute joint error was **0.0203 rad**.
The four additional full-contact failures relative to no-contact occur on
identical sampled GRAB frames.

## 3. Stable controller on 324 fixed GRAB targets

The isolated `hand_x2` profile reduces static right-hand bias. At 1/240 s it
still leaves 16 dynamic failures with contacts on; at 1/500 s it gives the
following 1.5 s result. The target remains fixed throughout; this test does
not use policy actions or sequence advancement.

| `hand_x2`, 1/500 s | Settled | All five <10 mm and settled | Dynamic failures | Static failures | Worst final fingertip error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full hand contacts | **323/324** | **321/324** | 1 | 2 | 10.78 mm |
| Hand contacts off | **324/324** | **324/324** | 0 | 0 | 5.69 mm |

With hand contacts off, the maximum right-hand speed in the final 0.2 s was
**6.1e-5 rad/s**, so the fixed-target acceptance criterion is met for all 324
sampled poses in that diagnostic geometry. With full contacts, the maximum
tail speed was **0.077 rad/s** in the one dynamic failure. The three full-contact
failures are samples 126, 138, and 159; **none saturated its actuator**.
The largest observed right-hand limit violation was below **0.001 rad**.

A source-MJCF forward-contact audit of the exact 324 goal joint poses found
hand geometry penetration in 170 poses. The three full-contact failures have
deepest signed contact distances of approximately **-14.04, -9.17, and
-11.07 mm**, mainly between middle and ring finger links. They all pass when
hand collisions are disabled. This strongly associates the remaining three
failures with contact-incompatible reference poses, while not proving that no
alternative collision-aware pose could satisfy their fingertip goals.

## Interpretation

The isolated USD correction passed its runtime-gain check. Newton's widespread
residual oscillation was caused by the 1/120 s step under this solver/drive
setup, not by torque saturation, joint limits, or active contacts in the Home
probe. At stable timesteps, low source gains leave a static position offset;
raising only right-hand gains in the isolated profile reduces that offset.
The full-contact 324-pose acceptance target is **not yet met**: three sampled
targets remain contact-sensitive. Keeping full hand collisions requires a
collision-aware retargeting or geometry/constraint decision before treating
these GRAB goals as reachable under the 10 mm task definition.

## Reproduction and local artifacts

- Build overlays: `PYTHONPATH=source python
  scripts/assets/build_hand_only_drive_overlay.py`.
- Runtime gain and timestep grid: `scripts/probe_hand_only_drive_grid.py`
  with `--dt-denominator {120,240,500}` and optional `--no-hand-contacts`.
- 324-pose check: `scripts/probe_right_hand_position_hold.py --corrected-drive
  --drive-profile hand_x2 --dt-denominator 500 --starts-per-motion 2`.
- All generated metrics, trajectories, and the source contact audit are ignored
  local files under `artifacts/hand_only_drive_grid/` and
  `artifacts/hand_only_corrected_grab/`. The latter includes
  `final_hand_x2_contacts_on/`, `final_hand_x2_contacts_off/`, and
  `source_target_contact_audit.csv` / `.json`. The combined
  `layered_validation_summary.json` and `layered_validation.svg` summarize
  the grid and acceptance results.

This host requires the clean native-library overlay noted in the earlier
[4096-environment PPO report](hand_only_4096_1000_s42.md). No GRAB frames,
training logs, or checkpoints are committed.
