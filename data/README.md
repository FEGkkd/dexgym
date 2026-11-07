# Data layout for right-hand fingertip tracking

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

On the current local dataset, conversion produced 797 motions and 252,679
frames with no rejected motions. See the
[pilot report](../docs/experiments/hand_only_right_pilot.md) for the exact audit
and task validation. These counts describe the local input files; another
dataset copy may differ.
