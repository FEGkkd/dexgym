# GRAB right-hand motion data

## Wuji Hand 2.1 rebuild

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
short segments are preserved as data. See
[the model and data report](../docs/experiments/wuji_hand21_model_data_20261109.md).
The independent Newton Hand 2.1 model passed 100 sampled 1.5-second
fixed-pose holds with all five fingertips under 10 mm. One 170-frame GRAB
clip was directly replayed through its position controller and rendered at
`artifacts/hand21/hand21_grab_direct_replay_newton.mp4` (local, not pushed).

For object-relative motion analysis, see the
[GRAB in-hand selection report](../docs/experiments/grab_inhand_motion_selection_20261109.md)
and its [object-focused selection](../docs/experiments/grab_inhand_object_focus_20261109.csv).
It prioritizes camera, toothpaste, game controller, and elephant recordings,
with flashlight, mouse, and stapler as supplements. The metadata indexes
right-hand rotation, translation, and alternating finger-contact candidates
in both the raw recordings and Hand 2.1 MotionBank. The raw recordings and
generated full candidate catalog remain local and ignored by Git.

## Earlier Beta 2 library and strict replay screen (historical)

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

The historical strict replay screen depended on the retired right-hand task.
See the [post-optimization report](../docs/experiments/hand_only_collision_postopt_20261108.md)
and [the previous revision](https://github.com/FEGkkd/dexgym/blob/f2f416e/data/README.md)
for its results and reproduction commands.

On the current local dataset, conversion produced 797 motions and 252,679
frames with no rejected motions. See the
[pilot report](../docs/experiments/hand_only_right_pilot.md) for the exact audit
and task validation. These counts describe the local input files; another
dataset copy may differ.
