# CMU 141_16 Wave Hello → Marvin reference

Status: **M4 Reference Ready (experimental simulation profile)**. No policy training was run. The authoritative data and restricted assets remain outside Git at `/home/ubuntu/dexgym_data/marvin_motion_library_v1/`.

## Source and treatment

- CMU repository commit `09a07f54f3bbb58797325f009282d0b2048a2871`; BVH `data/141/141_16.bvh`, SHA256 `d428d2c4fa8873d077537567ad32c95b9687a479a10522d314b203fddea37daf`.
- 300 frames at 0.0083333 s. Frame 0 is the converter's artificial T-Pose and is excluded. All 299 motion frames are used; source duration after removal is 2.4833234 s. M1's 1.7 m stature assumption gives 0.07033711618 m per BVH unit.
- `RightArm` is the anatomical upper arm origin; `RightShoulder` is a clavicle/torso-side node. The final mapping uses RightArm → RightForeArm → RightHand directions. This corrected an earlier candidate's 259 mm average elbow discrepancy. Earlier candidates, reports and videos are preserved in `preview_motions/`, `reports/`, `physics_replays/legacy_tuned/`, and `previews/legacy_tuned/`.
- RightHand local rotation changes substantially only on X. The hand's world orientation is reconstructed through BVH FK, usable for experimental retargeting, and is **not** independently measured high-confidence three-axis wrist orientation. Raw local angular speed P95/max: 1568/2352 °/s. SO(3) quaternion filtering with σ=5 original frames reduces this to 562/642 °/s; >1800 °/s steps fall from 7 to 0. Filter orientation change P95/max: 30.0/41.1°, with position unchanged. Raw channels, raw/filtered local quaternions, quality limitations, and the separate `retarget_orientation_usable` flag are retained.

## Mapping and reference

- BVH XYZ → robot XYZ rotation rows: `[-1,0,0]`, `[0,0,1]`, `[0,1,0]`. Position scale 0.85; fixed offset `[-0.15,-0.10,+0.05]` m. MuJoCo shoulder–elbow and elbow–TCP lengths measured from the pinned MJCF: 0.28756384 and 0.40939591 m.
- `R_tcp = R_alignment R_hand_filtered R_hand_to_tcp`; fixed `R_hand_to_tcp` XYZW = `[0.40617172,-0.69207323,-0.41497978,0.42877845]`, anchored at human time 0.7 s. The authoritative TCP is MJCF `tcp_L`/`tcp_R` site and corresponding frame sensors. `TCP_Link` body → TCP site is a fixed 180° local Z rotation. Nonzero-wrist MuJoCo–Newton FK probes agree to <0.001 mm and <0.001°.
- Full 54DoF model and hands Home; left 7DoF held Home. All seven right joints vary; J5/J6/J7 ranges are 1.031/0.792/0.733 rad. Exact 50 Hz clock: 1357 frames, 27.12 s. The original 2.48 s motion is played at 10.125× duration, with explicit 1 s C2 Home transitions at either end. This is a deliberately slow initial reference and is not natural-speed wave timing.
- Formal NPZ: `/home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/cmu_141_16_wave_right.npz`. It contains 14D/54D q, qd, qdd, both TCP 6D FK poses and velocities, three named links per arm, human raw/filtered poses and channels, raw/adjusted mapping targets, errors, phase labels, source-time mapping, and metadata. Loader clamps the final pose for future 10-frame goals.

## Measured results

| Check | Right position Mean/P95/Max (mm) | Right orientation Mean/P95/Max (°) | Left position Mean/P95/Max (mm) | Left orientation Mean/P95/Max (°) |
| --- | --- | --- | --- | --- |
| Final offline FK vs mapped human target, motion phase | 2.70 / 4.48 / 7.21 | 1.06 / 1.88 / 3.63 | Static Home | Static Home |
| MuJoCo `mj_step`, source gains | 39.27 / 80.59 / 125.27 | 7.74 / 15.43 / 20.41 | 33.02 / 33.17 / 33.17 | 5.02 / 5.05 / 5.05 |
| MuJoCo `mj_step`, runtime comparison profile | 0.638 / 1.064 / 1.679 | 0.100 / 0.183 / 0.235 | 0.596 / 0.596 / 0.872 | 0.090 / 0.090 / 0.131 |
| Newton MJWarp 1 environment | 0.634 / 1.200 / 1.384 | 0.084 / 0.139 / 0.154 | 0.562 / 0.567 / 0.920 | 0.065 / 0.067 / 0.109 |
| Newton MJWarp 16 environments | 0.635 / 1.204 / 1.392 | 0.084 / 0.139 / 0.156 | 0.562 / 0.568 / 0.920 | 0.065 / 0.067 / 0.109 |

The runtime comparison profile scales source arm Kp by 180/π and source Kd by 0.5 in memory; the source XML, imported inertia, effort limits, and joint bounds are unchanged. The source-gain run is retained because its larger gravity/tracking error is a real baseline. Right elbow mapped-target error Mean/P95/Max is 51.2/88.3/91.9 mm, with mean shoulder–elbow direction cosine 0.980. Reference peak speed/acceleration are 1.572 rad/s and 10.827 rad/s²; Newton actual peaks over 16 environments are 1.615 rad/s and 24.604 rad/s². Newton joint-velocity RMSE is about 0.175 rad/s despite sub-millimetre position tracking; retain this limitation when designing velocity rewards.

Every reference frame and midpoint is collision-free in source MJCF. Newton internal contacts are zero at every physics step in 1 and 16 environments. Source geometry recheck of Newton actual states at each 2 ms physics step found no contact. Independent right-hand geometry clearance to the support/left arm is at least 0.300 m over sampled frames. No hard joint-limit violation or torque saturation occurred. The packaged Newton USD disables hand contacts; the independent source-geometry audit is therefore essential.

## Artifacts and verification

- Four actual MP4s: `previews/cmu_141_16_human.mp4` (2.52 s), `previews/cmu_141_16_human_vs_marvin.mp4`, `previews/cmu_141_16_newton_pd.mp4`, `previews/cmu_141_16_mujoco_pd.mp4` (each 27.18 s). Newton and MuJoCo physical states were rendered from recorded simulation states, with distinct labels.
- `reports/cmu_141_16_wave_right_validation.json` records Gate A–D true, FK, contact, limits, videos, and loader checks. `metadata/retarget_config.json` records every mapping, filter, optimizer, and controller parameter. `index.json` registers only the validated candidate.
- `pytest -q tests/assets tests/motion` with the host's working Isaac Lab dependency paths: 21 passed, 3 skipped (optional asset checks). `ruff check` on all new code: passed. The original `141_23` retargeting experiments and whole-body teleoperation code were not changed.

For Newton replay on this host, the existing DexGym `.venv` binary `torch`, `mujoco`, and `pxr` imports caused SIGBUS. This run used known working packages from local conda/Isaac Sim locations through a process-local `PYTHONPATH` override and a `/tmp` pxr plugin symlink; it did not alter the shared virtualenv. The command and full state/report output are in `scripts/motion/replay_marvin_reference_newton.py` and the external library's `physics_replays/` directory.
