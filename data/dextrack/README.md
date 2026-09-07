# DexTrack GRAB Data

The first Allegro tracking task consumes the processed GRAB archive named
`GRAB_Tracking_PK_reduced_300.zip`. Extract it outside version control so the
directory contains files such as:

```text
<grab-root>/data/passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy
```

Point the task at that extracted directory:

```bash
export DEXTRACK_DATA_ROOT=/path/to/GRAB_Tracking_PK_reduced_300
```

The loader expects each scalar `.npy` dictionary to contain
`robot_delta_states_weights_np`, `object_transl`, and `object_rot_quat`. The
source hand state is `[T, 22]` with six free-root channels and sixteen
Allegro finger channels. Source object quaternions are validated and retained
as `xyzw` until the task boundary.

TACO and LEAP/Franka archives are outside the first migration scope. Archives,
extracted trajectories, and generated object meshes must not be committed to
the repository.
