# Three position-control diagnostics for right-hand tracking

**Date:** 2026-11-08. **Question:** Can the current position-controlled
simulation reach and hold a legal fixed right-hand target accurately enough
for the 10 mm, five-finger task? The task code, assets, and PPO configuration
were not changed. The source MJCF and its converted USD were compared in
diagnostic processes only.

## Experiment 1 — zero-gravity position hold

Repeat the previous 1.5 s four-frame-reference hold with full hand contacts
for the **same 324 validation start/goal pairs** (all 162 s8 motions, two
starts each). Newton's gravity was changed from `(0, 0, -9.81)` to `(0, 0, 0)`;
the position targets, 1/120 s physics step, 30 Hz control rate, and asset stayed
the same. The script checked the gravity vector read back from Newton's
physics manager. Each condition was run three times, preserving all raw
trajectories.

| Gravity | Mean absolute joint error at 1.5 s | Median / p95 of each pose's worst fingertip | All five <10 mm at 1.5 s |
| --- | ---: | ---: | ---: |
| Normal, three-run range | 0.111–0.115 rad | 33.59–36.05 / 55.55–57.14 mm | 1/324 in each run |
| Zero, three-run range | 0.123–0.127 rad | 40.59–42.70 / 64.28–66.45 mm | 0/324 in each run |

The paired zero-gravity minus normal-gravity change in worst-fingertip error
averaged **+5.00 to +8.40 mm** across the three runs. Removing gravity did not resolve
the drift. This argues against gravity load alone as the primary cause in
Newton; it does not exclude an effect of gravity on a weak source controller.

## Experiment 2 — one-joint response

Initialize the robot at its legal Home keyframe. Only the commanded target of
`r_index_finger_mcp_flex` changes, from 0 to **+0.1 rad**; all other 53
joint targets remain at Home. Hold for 1.5 s with normal gravity and full
contacts. The exact position command reached the actuator target tensor.
Record 30 Hz position, target, velocity, pre-limit computed drive, applied
drive, and effort limit in `response.csv`.

With the imported Newton gains, the index MCP moves in the **positive initial
direction**, then oscillates: sampled position spans **-0.331 to +0.664 rad**,
peak speed is **29.6 rad/s**, and computed drive reaches **37.38 N·m**. Applied
drive clips at the **2 N·m** limit in **41 of 45** control steps. The last ten
30 Hz samples span **0.943 rad**, so this is not a stable hold. The final
position alone is not a reliable performance measure of this oscillation.

## Experiment 3 — source MuJoCo versus Newton

Run the same Home pose, +0.1 rad index MCP target, full robot, 1/120 s physics
step, normal gravity, and 1.5 s hold in both the packaged source MJCF and its
full-contact converted USD. Joint and actuator indices are mapped by validated
names. Fingertip target FK differs by at most **0.00037 mm** between backends.

The **imported drive parameters are not equal**: source MuJoCo index MCP
`Kp=0.37352`, `Kd=0.01882`; Newton exposes `Kp=21.40125`, `Kd=1.07846`, each
about **57.2958×** the source value (`180/π`). Both use a **2 N·m** effort
limit. The repository's existing controller code documents this USD angular
gain scaling, but this diagnostic does not establish whether that scaling is
appropriate for the actual Newton execution path.

For a true same-drive comparison, a second Newton run sets **all 54** Kp and
Kd buffers in memory to the source MuJoCo values. Readback verified maximum
all-joint gain differences below `1.1e-8`; effort limits also matched within
`2.4e-8 N·m`. This did not modify the packaged USD or training task.

| Backend / drive | Final index MCP angle; target 0.1 rad | Final five-finger max error to target pose | Max absolute velocity in last 10 samples | Steps at 2 N·m limit |
| --- | ---: | ---: | ---: | ---: |
| Newton, imported USD gains | -0.279 rad | 50.93 mm | 20.70 rad/s | 41/45 |
| Newton, source-matched gains | 0.162–0.173 rad | 9.88–10.16 mm | 2.52–3.39 rad/s | 0/45 |
| MuJoCo, source gains | 0.188 rad | 11.58 mm | 0.000017 rad/s | 0/45 |

The source-matched Newton row gives the range across three runs with identical
settings; imported Newton and MuJoCo values repeated identically in those
three runs. Matching gains removes Newton's saturation and most of its large oscillation,
but it **still has substantial residual velocity** at 1.5 s and therefore
has not demonstrated a stable hold. MuJoCo settles in this single-joint test,
but its steady angle is about **0.088 rad** from the command and its maximum
fingertip error is **above 10 mm**. Neither tested path yet meets the stated
acceptance criterion of stably holding all five fingertips within 10 mm.

## Interpretation and limits

The zero-gravity result directs attention away from gravity as the sole Newton
failure mechanism. The single-joint response demonstrates that the command is
received and the direction is initially correct; the imported Newton drive is
overactive and saturates. The source-versus-imported gain difference is a
concrete MJCF → USD / Newton drive-semantic boundary to inspect. Gain parity
eliminates saturation but does not eliminate Newton's residual oscillation,
so it is not yet a complete explanation. MuJoCo's stable but biased source
controller also fails the 10 mm criterion in this pose.

This comparison uses one Home-pose joint step for the backend pair; it does
not claim that every GRAB pose behaves the same. Backend solver and contact
implementations inherently differ. No production gain, solver, reward, or
threshold changes were made.

## Artifacts

- `scripts/probe_right_hand_position_hold.py --zero-gravity` runs experiment 1.
- `scripts/probe_right_index_mcp_backends.py` runs the source-versus-imported
  single-joint comparison; `--match-source-gains` enables the same-drive check.
- Ignored local artifact directories:
  `artifacts/hand_only_zero_gravity/`, `artifacts/hand_only_position_hold_rep1`–`rep3`,
  `artifacts/hand_only_zero_gravity_rep1`–`rep3`, `artifacts/right_index_mcp_backends/`,
  `artifacts/right_index_mcp_backends_matched/`, their `_rep1`–`_rep3`
  repeat directories, and
  `artifacts/hand_only_backend_controls/` (combined summary and SVG plot).

The source script and this report contain no GRAB frames or model weights.
