# Direct right-hand position-target hold probe

**Date:** 2026-11-08. **Task:** `DexGym-Wuji-Right-Hand-Tracking-Direct-v0`.
This checks the position-controlled hand before changing PPO or the 10 mm task
definition.

## Protocol

- Validation bank: all 162 s8 GRAB motions, two seeded random start frames per
  motion, 324 independent environments. Seed 20261108.
- Initialize the 20 right-hand joints from a reference frame. Send the reference
  joint angles four frames later directly through
  `robot.actuators.target_command.set_position_index`, bypassing the RL action
  integrator. Hold that same command for 45 control steps = 1.5 s at 30 Hz.
  Newton MJWarp uses 1/120 s physics steps, four per control step. Arms and left
  hand target Home. Hand contacts are enabled, matching the training task.
- Record absolute joint error and five fingertip Euclidean errors in the wrist
  frame at time 0 and every 1/30 s through 1.5 s. The per-pose fingertip
  metric is the **maximum across the five fingers**.
- Before the dynamic run, teleport to each target joint pose and call forward
  kinematics to measure any mismatch between the stored fingertip reference
  and the current USD model. Restore the start pose before stepping physics.
- Controls: hold the start joint pose with full contacts; repeat the four-frame
  goal with hand contacts disabled. Both use the same 324 motion/start samples.

## Results at 1.5 s

| Command and geometry | Mean joint absolute error | Five-finger maximum error, median / p95 / worst | All five <10 mm at final step |
| --- | ---: | ---: | ---: |
| Four-frame target, full contacts | **0.115 rad** | **36.05 / 57.14 / 72.78 mm** | **1 / 324** |
| Hold start pose, full contacts | 0.115 rad | 36.54 / 55.43 / 71.31 mm | 1 / 324 |
| Four-frame target, hand contacts off | 0.116 rad | 37.07 / 55.90 / 82.43 mm | 0 / 324 |

For the main four-frame target, the largest absolute error among the 20 joints
reached 0.698 rad. The per-sample largest joint error was 0.381 rad on average.
At the final step, individual fingertip error maxima over all 324 samples were
31.30 mm (thumb), 64.90 mm (index), 59.84 mm (middle), 69.57 mm (ring), and
72.78 mm (pinky). The mean per-finger errors were 10.49, 21.46, 21.19, 23.69,
and 22.72 mm, respectively. The four non-thumb MCP flexion joints had the
largest mean joint errors (0.202–0.229 rad).

The command was verified in the actuator target tensor. The sampled reference
joint angles were within the task joint limits. With exact reference joint
angles set kinematically, the reference fingertip mismatch was tiny: maximum
0.0066 mm across all target samples. Thus the measured 1.5 s error is not
explained by a stored fingertip reference / model FK mismatch.

The target fingertip displacement from the start pose was modest: median
6.66 mm, p95 19.11 mm. At time 0, 221/324 samples already had all five fingers
within 10 mm of the upcoming target; after 1/30 s of dynamics, only 2/324 did.
Holding the exact start joint pose also drifted, and disabling hand contacts
did not restore tracking. These observations place the immediate limitation in
the position-driven dynamics or their configuration. They do not yet separate
actuator gains, effort limits, coupling, solver behavior, or other constraints.

## Reproduction and artifacts

The reusable script is `scripts/probe_right_hand_position_hold.py`. Example:

```bash
PYTHONPATH=source python scripts/probe_right_hand_position_hold.py \
  --split validation --starts-per-motion 2 --hold-seconds 1.5 \
  --output-dir artifacts/hand_only_position_hold
```

This host also requires the clean native-library overlay described in the
[4096-environment PPO report](hand_only_4096_1000_s42.md).

The ignored local `artifacts/hand_only_position_hold/` directory contains
`summary.json`, `timeseries.csv`, `per_sample.csv`, `errors.npz`, and
`position_hold_comparison.svg`. The two control runs are in sibling directories
`hand_only_position_hold_start_contact/` and
`hand_only_position_hold_no_contact/`. No GRAB frames or generated result arrays
are committed to Git.
