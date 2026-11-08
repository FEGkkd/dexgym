# Wuji Hand 2.1 standalone model and GRAB rebuild (2026-11-09)

## Model and retarget provenance

- Robot: official [`wuji-description`](https://github.com/wuji-technology/wuji-description)
  `hand2/hand2_1/body/usd/right/wujihand2_1.usd`, tag `v2026.10.26`, commit
  `5c4dfa16917355d073e2e1d7e3d03897b6cf9508` (also upstream HEAD at import).
- Retarget: official [`wuji-sdk`](https://github.com/wuji-technology/wuji-sdk)
  Python `RetargetSession.for_hand(HandModel.WujiHand2, side=Handedness.Right)`,
  package `2026.10.26`, repository commit `b6be39b1c923b822f627ac8564dd91ceb38b303b`.
- Newton simulation model: the official layered **right-hand USD**, with `Physics=PhysX`.
  Offline FK and collision geometry: the matching official right-hand MJCF.
  The old dual-arm Marvin USD and Beta 2 MJCF are excluded from this path.
- The official asset contains 20 revolute joints, one articulation root, 22
  instanced collision meshes, and enabled self-collision. Its authored USD
  fallback drives are 50/2 for all joints. The isolated `robot_cfg.py` sets
  the per-group SI gains from official `config.yaml` and force limits from
  official MJCF at runtime. The Newton scene uses the `implicitfast`
  integrator at 500 Hz for fixed-pose validation and 480 Hz for 30 Hz replay.

## Local dataset

The ten downloaded subjects contain 1,335 GRAB right-hand motions. The newly
extracted subjects are s6 (148), s7 (120), s9 (125), and s10 (145) motions.
The original archives and `.npz` files are unchanged and excluded from Git.

| Stage | Motions or segments | Frames | Comment |
| --- | ---: | ---: | --- |
| Official SDK retarget + Hand 2.1 FK | 1,335 motions | 406,264 | No format, nonfinite, or joint-limit rejection |
| Hand 2.1 self-collision post-optimization | 1,335 motions | 406,264 | 17 fallback clips retained for frame-level audit |
| Severe collision frame filter | 1,404 segments from all 1,335 motions | 404,311 | 1,953 frames (0.48%) removed; no whole motion removed |

The final filter uses **5 mm residual self-penetration** in the official
Hand 2.1 MJCF. It retains all contiguous valid segments, including short ones.
It does not use the former every-frame 10 mm position replay rule. The maximum
recorded penetration of a retained segment is 4.976 mm. All source-to-segment
mappings and removed-frame counts are in local `collision_audit.json`.
The 5 mm cutoff is above the 95th percentile (3.875 mm) of each clip's maximum
residual penetration after post-optimization. This is an operational severity
criterion, not proof that every rejected frame is mathematically irreparable.
Training metadata assigns s1–s7 to train and s8–s10 to validation; no RL task
was modified as part of this data/model import. The later
[RL model switch](hand21_rl_model_switch_20261109.md) connects the Clean
task family to this library.

Two clips initially hit infeasible *local optimizer step bounds* rather than a
data-quality failure. The optimizer now retains such frames for final collision
screening. Both clips were retried and included in the 1,335-motion bank.

The official Python SDK and old C SDK give identical joint commands for three
checked overlapping clips at the same `2026.10.26` version. Hand 2.1 FK
changes fingertip positions by up to 5.7 mm in those clips, so the old
Beta 2 `tip_ref` cannot be reused.

## Validation

- USD composition: pass. `validate_usd_static.json` records 20 joints,
  one articulation root, 22 collision instances, the `PhysX` variant, and
  enabled self-collision.
- Saved FK against official Hand 2.1 MJCF: pass on 100 sampled poses across
  all ten subjects; maximum discrepancy **0.0000154 mm**.
- Official Hand 2.1 MuJoCo position control: 100/100 sampled four-frame
  target holds had all five final fingertip errors under 10 mm after 1.5 s.
  Worst finger error was 7.86 mm; worst final joint speed was 0.000426 rad/s.
  This is a MuJoCo reference, not a Newton result.
- Newton USD FK: maximum error against official MJCF FK was **0.00044 mm**
  across 100 sampled GRAB postures. Runtime Kp, Kd, and effort limits matched
  the independent configuration to within floating-point precision.
- Newton position hold: 100/100 sampled four-frame target holds across s1–s10
  had all five final fingertip errors under 10 mm after 1.5 s, with full hand
  collision and gravity enabled. Worst finger error was **2.84 mm**; worst
  final joint speed was **0.00021 rad/s**. Peak effort reached its configured
  limit in at least one sample. This validates sampled fixed targets, not the
  entire bank or RL task.
- The initial Newton `euler` integration diverged even for a zero-gravity,
  collision-free fixed pose: finger DIP speed exceeded 700 rad/s within six
  steps. With `implicitfast`, that same pose held to numerical precision.
  The failure was a time-integration and drive interaction, not a bad FK
  mapping or an initial contact. The official USD files were not modified.

## Direct-control video

The selected s9 `spheresmall_inspect_1.npz` segment has 170 frames (5.67 s).
`replay_wuji_hand21_newton.py` drives the official Hand 2.1 USD with the 30 Hz
retargeted joint sequence and records **actual Newton joint states**. The
render script draws those states with the matching official Hand 2.1 MJCF
visual meshes on the left, with the SDK joint reference on the right. It
overlays the maximum instantaneous fingertip error. The video is local at
`artifacts/hand21/hand21_grab_direct_replay_newton.mp4` (1280×520, 30 fps).
Across this continuous Newton replay, the mean finger error was **1.55 mm**,
95th percentile **3.68 mm**, and maximum **6.56 mm**; all 170 frames had all
five fingers under 10 mm. This is one selected clip, not a full-bank dynamic
tracking result. A matching MuJoCo comparison video is also available at
`artifacts/hand21/hand21_grab_direct_replay_mujoco.mp4`; its mean, 95th
percentile, and maximum finger errors were 6.30, 12.66, and 22.70 mm.

The video re-renders recorded Newton states through the official MJCF visual
meshes; it is not a USD viewport capture. The simulated movement and error
statistics come from Newton, with no RL policy involved.
