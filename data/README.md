# Data layout for right-hand fingertip tracking

## Wuji Hand 2.1 rebuild (current)

The current independent right-hand model is the official Wuji Hand 2.1 USD
under `source/dexgym/assets/wuji_hand21/`. Local GRAB recordings now cover all
ten subjects, `s1`–`s10`. The additional `grab__s6.zip`, `grab__s7.zip`,
`grab__s9.zip`, and `grab__s10.zip` were extracted into `data/grab/` without
overwriting earlier recordings. Raw GRAB files and generated libraries remain
ignored by Git.

Retarget with the official Python `wuji-sdk==2026.10.26`
`RetargetSession.for_hand(HandModel.WujiHand2, side=Handedness.Right)`. Retarget
output is 20 joint angles in firmware order. Hand 2.1 FK then generates all
five `tip_ref` positions in the right-wrist frame. The subjects `s1`–`s7` are
the train split; `s8`–`s10` are held out by subject.
Install `wuji-sdk==2026.10.26` in the data-processing Python environment
alongside NumPy, SciPy, and MuJoCo before running these commands.

```bash
PYTHONPATH=source python scripts/data/build_grab_hand21_library.py \
  --output data/processed/hand21_right_sdk
PYTHONPATH=source python scripts/data/postoptimize_grab_right_hand_library.py \
  --input data/processed/hand21_right_sdk \
  --output data/processed/hand21_right_collision_aware \
  --right-xml source/dexgym/assets/wuji_hand21/official/mjcf/right.xml --workers 24
PYTHONPATH=source python scripts/data/filter_hand21_severe_collisions.py \
  --input data/processed/hand21_right_collision_aware \
  --output data/processed/hand21_right_geometry_feasible
```

The last stage removes residual self-penetration greater than 5 mm at the
**frame** level, then saves all contiguous valid segments, including short ones. It
records every removed frame and split in `collision_audit.json`. It does not
apply the old whole-motion, every-frame 10 mm position replay rule. The three
libraries stay separate so optimization and filtering can be audited.

The local 2026-11-09 rebuild produced 1,335 SDK motions (406,264 frames).
Geometry filtering removed 1,953 severe-collision frames and retained 1,404
segments (404,311 frames) spanning all 1,335 original motions. Sixteen very
short segments are preserved as data; a later training sampler can choose its
minimum segment length. See
[the model and data report](../docs/experiments/wuji_hand21_model_data_20261109.md).
The independent Newton Hand 2.1 model passed 100 sampled 1.5-second
fixed-pose holds with all five fingertips under 10 mm. One 170-frame GRAB
clip was directly replayed through its position controller and rendered at
`artifacts/hand21/hand21_grab_direct_replay_newton.mp4` (local, not pushed).

## Earlier Beta 2 library and strict replay screen

The GRAB recordings and generated right-hand motion library stay local. Git
tracks this README; those local inputs and outputs are ignored.

Expected local inputs:

```text
data/
├── grab/s1/*.npz ... grab/s8/*.npz
├── models/mano_v1_2/MANO_RIGHT.pkl
└── tools/subject_meshes/.../*_rhand.ply
```

The converter also uses the local WujiHand2 C SDK and pinned right-hand MJCF:

```text
~/teleop_ws/third_party/wuji_c_api/lib/libwuji_sdk_c.so
~/teleop_ws/src/whole_body_mujoco_driver/model/source/wujihand2_beta2/mjcf/right.xml
```

From the repository root, generate the 30 Hz right-hand library with:

```bash
PYTHONPATH=source uv run python scripts/data/build_grab_right_hand_library.py \
  --data-root data --output data/processed/hand_only_right
```

The converter reads GRAB right-hand MANO poses, retargets them through the Wuji
SDK, and saves one `.npz` per motion with `q_ref[T,20]` and robot-FK
`tip_ref[T,5,3]` in the right-wrist frame. The output directory also contains
`manifest.json` with source paths, split assignments, and quality statistics.
Subjects `s1`–`s5` are used for training and `s8` is reserved for validation.
The source recordings are never modified.

To refine the SDK library against Wuji hand self-collision, write a separate
library with:

```bash
PYTHONPATH=source uv run python scripts/data/postoptimize_grab_right_hand_library.py \
  --input data/processed/hand_only_right \
  --output data/processed/hand_only_right_collision_aware --workers 8
```

The optimizer starts from each `q_ref`, penalizes fingertip displacement,
deviation from the SDK angles, MuJoCo hand-mesh penetration, and temporal
roughness. It keeps the raw GRAB recordings and SDK library unchanged. Each
output `tip_ref` is recomputed by Wuji FK from its optimized `q_ref`. The
output manifest records optimization weights and per-motion contact and
fingertip-change metrics. Motions with worse maximum penetration or over
12 mm fingertip shift fall back to the SDK targets and are marked in the
manifest. A separate output directory is required. Use
`--split validation` for a smaller first pass.

For a strict clean bank, screen every optimized motion with full hand
collision, the isolated `hand_x2` controller, a 1/500 s physics step, and
direct 30 Hz joint-reference replay. Then copy only motions whose **every
frame** has all five fingertip errors below 10 mm and that did not fall back
to the SDK result:

```bash
PYTHONPATH=source uv run python scripts/data/screen_grab_right_hand_library.py \
  --library-path data/processed/hand_only_right_collision_aware/manifest.json \
  --split train --batch-size 64 --output-dir artifacts/hand_only_postopt/screen_train
PYTHONPATH=source uv run python scripts/data/screen_grab_right_hand_library.py \
  --library-path data/processed/hand_only_right_collision_aware/manifest.json \
  --split validation --batch-size 64 --output-dir artifacts/hand_only_postopt/screen_validation
PYTHONPATH=source uv run python scripts/data/build_clean_grab_right_hand_library.py \
  --input data/processed/hand_only_right_collision_aware \
  --train-screen artifacts/hand_only_postopt/screen_train/screen.json \
  --validation-screen artifacts/hand_only_postopt/screen_validation/screen.json \
  --output data/processed/hand_only_right_collision_clean
```

The clean directory contains only selected motion files plus `manifest.json`
and `excluded.json`. The original GRAB and both intermediate banks remain
available for provenance. This screen checks direct position-controller
replay; it does not certify PPO tracking. The screen records each input motion
file's SHA-256, and the clean-bank builder rechecks it before copying.

In the current local run, the strict clean bank contains 140 motions and
37,768 frames (51 train motions, 89 validation motions). The two validation
clips `s8/flashlight_on_1.npz` and `s8/fryingpan_cook_2.npz` are excluded.
The exact results and limitations are recorded in the
[post-optimization report](../docs/experiments/hand_only_collision_postopt_20261108.md).

The independent clean tracking task is
`DexGym-Wuji-Right-Hand-Clean-Tracking-Direct-v0`. It loads this clean
manifest, the full-contact `hand_only_drive_corrected_hand_x2.usda`, and runs
Newton at 480 Hz with 16 physics steps per 30 Hz policy action. The old task
ID still selects the original motion bank and drive. Before PPO, verify the
controller with the exact 178-pair validation CSV generated by the fixed-pose
experiment:

```bash
PYTHONPATH=source uv run python scripts/probe_right_hand_position_hold.py \
  --corrected-drive --drive-profile hand_x2 --dt-denominator 480 \
  --library-path data/processed/hand_only_right_collision_clean/manifest.json \
  --sample-pairs-csv artifacts/hand_only_corrected_grab/final_hand_x2_contacts_on/per_sample.csv \
  --output-dir artifacts/hand_only_clean_480/hold
PYTHONPATH=source uv run python scripts/evaluate_right_hand_clean_joint_replay.py \
  --split train --output artifacts/hand_only_clean_480/joint_replay_train.json
PYTHONPATH=source uv run python scripts/evaluate_right_hand_clean_joint_replay.py \
  --split validation --output artifacts/hand_only_clean_480/joint_replay_validation.json
```

The replay checks actual `env.step()` subgoal progression, the 10 mm
all-finger rule, and two-step dwell. The clean task has a 30-second episode cap
so long clips can finish; each subgoal still has a 45-policy-step (1.5-second)
cap.

The 480 Hz controller checks, full-library Joint Replay, and PPO comparison
are in the [clean tracking report](../docs/experiments/hand_only_clean_480_4096_500_s42.md).

Two isolated PPO ablations also use this same clean bank:
`DexGym-Wuji-Right-Hand-Clean-Obs125-Direct-v0` adds the 20D accumulated
position-command error to the observation; and
`DexGym-Wuji-Right-Hand-Clean-Hybrid125-Direct-v0` additionally reduces the
dense reward coefficient and raises the per-step time cost. Neither variant
changes the source GRAB files or the clean-bank contents. Their 4096-env,
500-iteration comparison is in the
[observation and reward ablation report](../docs/experiments/hand_only_obs_reward_ablation_20261108.md).

Action-distribution diagnostic task IDs reuse the same Hybrid125 environment
and clean bank: `DexGym-Wuji-Right-Hand-Clean-Hybrid125-Entropy0-Direct-v0`,
`DexGym-Wuji-Right-Hand-Clean-Hybrid125-TanhGaussian-Direct-v0`, and
`DexGym-Wuji-Right-Hand-Clean-Hybrid125-Beta-Direct-v0`. They vary PPO entropy
and action distribution while keeping the environment and clean bank fixed.
See the [action-distribution report](../docs/experiments/hand_only_action_distribution_20261108.md)
for clipping audits and held-out policy results. These task IDs do not alter
GRAB files or the generated motion banks.

On the current local dataset, conversion produced 797 motions and 252,679
frames with no rejected motions. See the
[pilot report](../docs/experiments/hand_only_right_pilot.md) for the exact audit
and task validation. These counts describe the local input files; another
dataset copy may differ.
