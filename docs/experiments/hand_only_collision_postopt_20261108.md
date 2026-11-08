# GRAB right-hand self-collision post-optimization

**Date:** 2026-11-08. **Scope:** data preparation and fixed-target Newton
verification; no PPO training or change to the registered task.

## Method

The pinned Wuji SDK remains the first retargeting stage. A separate pass reads
its 30 Hz `q_ref[T,20]`, keeps GRAB and the SDK output unchanged, and writes a
new MotionBank-compatible directory. The optimized `tip_ref[T,5,3]` is
recomputed from the right-hand MJCF fingertip sites via FK.

For each contacting frame, bounded L-BFGS-B minimizes weighted squared
fingertip displacement from the SDK pose, joint displacement from the SDK
pose, negative signed self-contact distances, and local velocity/acceleration
energy. Fingertip, joint, and collision residuals are normalized by 10 mm,
0.1 rad, and 2 mm respectively. Default dimensionless weights are 20, 0.1,
1, and 0.1. MuJoCo's collision meshes from the pinned right-hand MJCF supply
contact distances and analytic contact Jacobians; the paper uses attached
collision spheres. Two local temporal sweeps and a per-motion joint-step bound
limit discontinuities. The optimization is an approximation to the joint
sequence objective, not an exact global minimizer.
If the deepest penetration becomes more than 0.1 mm worse than the SDK
motion, or if any fingertip moves more than 12 mm from its SDK reference, the
whole motion falls back to the SDK trajectory. The manifest records the
reason; these fallback motions remain available but are explicitly not
collision-refined.

The generated manifest stores all weights, provenance hash, and per-motion
metrics. The original source data and shared 54-DoF USD are untouched.

## Verification

The initial pilot used the three motions whose fixed targets failed with full
hand collisions in the previous [layered drive check](hand_only_drive_layered_validation_20261108.md):
`s8/flashlight_on_1`, `s8/fryingpan_cook_2`, and `s8/hammer_use_1`.
It reused the exact six start/goal pairs from that check, the independent
`hand_x2` gain overlay, a 1/500 s Newton step, full hand collisions, and a
1.5 s fixed target. Success requires the maximum right-hand joint speed during
the last 0.2 s to be below 0.05 rad/s and all five final fingertip errors
to be below 10 mm.

The final library was rebuilt in one pass with the original split assignments.
The validation set was checked against the same 324 start/goal pairs as before.

### Complete library audit

The separate output contains **797 motions / 252,679 frames** in the same
source order and split as the SDK bank: 635 train motions / 200,010 frames,
and 162 validation motions / 52,669 frames. All files have finite joint and
tip arrays, 30 Hz metadata, and valid joint bounds; all 797 were accepted by
`MotionBank`. A reproducible random one-frame-per-motion FK audit found a
maximum output-tip mismatch of **1.6e-8 m**. The input manifest SHA-256 matches
the provenance hash in the output manifest.

Across the full library, **89,967 frames** initially exceeded 0.5 mm of
mesh penetration; **37,863** do afterward, a **57.9% reduction**. The median
per-motion deepest penetration fell from **5.40 to 1.13 mm**. This soft
objective reduces penetration but does not eliminate all contacts: 98,442
frames still have some negative signed contact distance, and the worst
non-fallback optimized motion retains **11.04 mm** penetration. Maximum
fingertip displacement from the SDK target is **11.40 mm** across accepted
motions. **10 train motions** triggered the conservative fallback (none in
validation), leaving their original SDK poses and identifying them in the
manifest. Among training motions, 163 still have a frame deeper than 2 mm;
these should be reviewed or filtered before large-scale training.
The temporal penalty and step bound constrain local changes, but 109 motions
still have a larger maximum single-joint frame step than their SDK source.
Temporal quality therefore also needs review before using the entire bank for
high-speed tracking.

### Fixed-target Newton check

The exact 324 motion/start/goal triples from the earlier drive check were
verified equal after replacing the library. With full hand collision,
`hand_x2` gains, and a 1/500 s step:

| Library | Settled, tail max speed <0.05 rad/s | Settled and all five tips <10 mm | Worst final tip error | Worst tail joint speed |
| --- | ---: | ---: | ---: | ---: |
| SDK original | 323/324 | 321/324 | 10.78 mm | 0.0767 rad/s |
| Collision-aware | **324/324** | **324/324** | **5.69 mm** | **0.0150 rad/s** |

The optimized library's maximum FK mismatch between the stored tip target
and Newton's teleported joint target was **0.0068 mm**. Six of 324 samples
briefly reached an actuator effort limit; all met the final acceptance test.

### Direct 30 Hz sequence replay

To separate fixed-target reachability from continuous tracking, three
previously contact-sensitive motions were replayed at 30 Hz by sending their
`q_ref` directly to the same Newton position drive after a 1.5 s start-pose
hold. This is a feedforward controller diagnostic, not a policy evaluation.

| Motion | SDK frames with all tips <10 mm | Optimized frames with all tips <10 mm | Mean worst-finger error, SDK → optimized |
| --- | ---: | ---: | ---: |
| `s8/flashlight_on_1` | 285/334 | **330/334** | 6.51 → 4.02 mm |
| `s8/fryingpan_cook_2` | 161/239 | **223/239** | 8.70 → 5.61 mm |
| `s8/hammer_use_1` | 280/317 | **317/317** | 7.10 → 3.81 mm |

The flashlight and fryingpan clips still contain a few frames above 10 mm
under direct sequence replay; the largest optimized replay errors were 10.82
and 17.54 mm respectively. Thus the fixed-target acceptance result does not
establish full-trajectory success or PPO convergence.

### Strict clean library

Following the direct-replay finding, every optimized motion is screened at
30 Hz in the same full-contact `hand_x2`, 1/500 s Newton controller. The
clean library includes a motion only when **every frame** has all five
fingertip errors below 10 mm, and the post-optimizer did not fall back to
the SDK trajectory. Excluded clips are absent from the clean directory and
listed with reasons in `excluded.json`. The original GRAB recordings, SDK
bank, and unfiltered post-optimization bank remain unchanged for provenance.
Each replay result records the SHA-256 of its input `.npz`; the clean-bank
builder checks these hashes again before copying and after copying so changed
motion contents cannot reuse stale passing evidence. Replay evidence also
binds to a fingerprint of the replay code, task physics configuration,
robot USD assets, Newton package sources, and package versions. Changing
the controller or solver invalidates the cached replay results.
The screen measures a direct position controller under one fixed simulation
configuration; passing it is a data-quality gate, not a PPO success claim.

The complete replay screen found **51/635 train** and **89/162 validation**
motions passing the strict every-frame criterion. The clean bank therefore
contains **140 motions / 37,768 frames** (train: 51 / 11,788; validation:
89 / 25,980). The other **657** motion files are absent, with reasons in
`excluded.json`. The two user-identified clips, `s8/flashlight_on_1` and
`s8/fryingpan_cook_2`, are excluded; `s8/hammer_use_1` remains. The full
unfiltered bank had 797 motions / 252,679 frames, so the clean bank is much
smaller. This strict direct-controller test also discards clips that a future
policy might track; that is an intentional conservative data selection.
It screens tracking error and SDK fallback, not residual mesh penetration or
peak joint step: **22/140** retained motions still have a frame deeper than
2 mm, and **18/140** have a maximum joint step more than 0.015 rad above
the SDK motion. The clean bank should be described as replay-clean under the
tested controller, not collision-free or globally time-optimal.

The same fixed-target probe restricted to the 89 retained validation motions
passed **178/178** sampled start/goal pairs in full-contact Newton. All 178
settled; the worst final fingertip error was **5.63 mm** and the worst tail
right-hand joint speed was **0.00383 rad/s**.

## Reproduction

```bash
PYTHONPATH=source python scripts/data/postoptimize_grab_right_hand_library.py \
  --input data/processed/hand_only_right \
  --output data/processed/hand_only_right_collision_aware --workers 8

PYTHONPATH=source python scripts/probe_right_hand_position_hold.py \
  --corrected-drive --drive-profile hand_x2 --dt-denominator 500 \
  --library-path data/processed/hand_only_right_collision_aware/manifest.json \
  --sample-pairs-csv artifacts/hand_only_corrected_grab/final_hand_x2_contacts_on/per_sample.csv \
  --output-dir artifacts/hand_only_postopt/final_hold

PYTHONPATH=source python scripts/probe_right_hand_sequence_replay.py \
  --library-path data/processed/hand_only_right_collision_aware/manifest.json \
  --motion s8/flashlight_on_1.npz --motion s8/fryingpan_cook_2.npz \
  --motion s8/hammer_use_1.npz \
  --output artifacts/hand_only_postopt/sequence_replay_postopt.json

PYTHONPATH=source python scripts/data/screen_grab_right_hand_library.py \
  --library-path data/processed/hand_only_right_collision_aware/manifest.json \
  --split validation --batch-size 64 \
  --output-dir artifacts/hand_only_postopt/screen_validation
PYTHONPATH=source python scripts/data/screen_grab_right_hand_library.py \
  --library-path data/processed/hand_only_right_collision_aware/manifest.json \
  --split train --batch-size 64 \
  --output-dir artifacts/hand_only_postopt/screen_train
PYTHONPATH=source python scripts/data/build_clean_grab_right_hand_library.py \
  --input data/processed/hand_only_right_collision_aware \
  --train-screen artifacts/hand_only_postopt/screen_train/screen.json \
  --validation-screen artifacts/hand_only_postopt/screen_validation/screen.json \
  --output data/processed/hand_only_right_collision_clean
```

The generated `.npz` motion data and diagnostic artifacts remain local and
ignored by Git. The registered PPO task still points to the original SDK
library and original physics configuration; this experiment does not silently
replace its training targets or controller.
