# CMU 141_16 Wave Hello: dual arm Marvin reference

Status: **validated for dual arm simulation reference**. No policy training was run. This adds dynamic left arm tracking to the existing right arm wave without replacing the original right arm reference.

## Data and mapping

- Source BVH: `/home/ubuntu/dexgym_data/cmu/cmu-mocap/data/141/141_16.bvh`, commit `09a07f54f3bbb58797325f009282d0b2048a2871`, SHA256 `d428d2c4fa8873d077537567ad32c95b9687a479a10522d314b203fddea37daf`.
- All 299 motion frames after the artificial T-Pose are used. Original duration is 2.4833234 s. The robot reference has 1357 samples at exactly 50 Hz and lasts 27.12 s, including marked Home transitions. The source action is deliberately slowed by 10.125 times.
- The right arm trajectory is exactly the prior validated 7DoF trajectory. The left arm now independently tracks the BVH left shoulder, elbow, forearm, and hand 6D pose with all seven Marvin left joints free. The left arm mapping uses the same BVH→Marvin rotation as the right, scale 0.85, and fixed offset `[+0.15,-0.10,+0.05]` m from the left shoulder mount. Its measured upper arm and elbow→TCP lengths are 0.28756397 and 0.40939594 m.
- Left TCP orientation is `R_alignment R_human_left_hand R_left_hand_to_tcp`; fixed left Hand→TCP XYZW is `[0.33601563,-0.35207712,0.57231655,-0.65999165]`, calibrated at human time 0.7 s. The right calibration remains in the original metadata. Both targets and realized poses use the MJCF `tcp_L`/`tcp_R` site frames; Newton body poses receive the explicit body→site transform.
- Left joint trajectory is smoothed with a 5 frame Gaussian on the final 50 Hz clock. Source human hand poses and channels are preserved. Left local hand channels vary mainly on Z by 6.54°; its world hand pose also changes with the upstream arm. The orientation is BVH FK reconstructed and experimental, not independent measured full wrist rotation.

## Results

| Check | Left TCP position Mean/P95/Max (mm) | Left orientation Mean/P95/Max (°) | Right TCP position Mean/P95/Max (mm) | Right orientation Mean/P95/Max (°) |
| --- | --- | --- | --- | --- |
| Offline FK vs mapped human target, human phase | 0.40 / 1.06 / 5.94 | 0.17 / 0.51 / 2.13 | 2.70 / 4.48 / 7.21 | 1.06 / 1.88 / 3.63 |
| MuJoCo source gains, `mj_step` | 27.22 / 41.82 / 98.93 | 3.81 / 5.47 / 12.69 | 39.27 / 80.59 / 125.27 | 7.74 / 15.43 / 20.41 |
| MuJoCo Newton comparable experimental gains, `mj_step` | 0.48 / 0.63 / 2.10 | 0.068 / 0.083 / 0.313 | 0.64 / 1.06 / 1.68 | 0.100 / 0.183 / 0.235 |
| Newton MJWarp 1 environment, physical step | 0.468 / 0.585 / 1.86 | 0.058 / 0.072 / 0.261 | 0.634 / 1.203 / 1.388 | 0.084 / 0.139 / 0.156 |
| Newton MJWarp 16 environments, physical step | 0.467 / 0.584 / 1.862 | 0.058 / 0.072 / 0.262 | 0.635 / 1.206 / 1.394 | 0.084 / 0.139 / 0.156 |

The left elbow mapped-target error P95 is 21.75 mm; right elbow P95 is 88.29 mm with mean direction cosine 0.980. All seven left joints move by at least 0.315 rad. The maximum reference speed and acceleration are 1.978 rad/s and 10.827 rad/s². Reference frames and midpoints, MuJoCo physical substeps, and Newton actual states rechecked against source MJCF at every 2 ms step have zero forbidden contacts. Joint bounds and effort limits remain unchanged; no hard limit violation or torque saturation occurred. Newton actual joint velocity RMSE is about 0.17 rad/s per side. The packaged USD disables hand contacts, so the source MJCF geometry check remains necessary.

The experimental control profile scales source arm Kp by 180/π and source Kd by 0.5 in memory for MuJoCo, matching the imported Newton profile; the source MJCF and USD were not edited. The source gain replay remains available as a measured baseline and exceeds the initial 25 mm mean position target.

## Artifacts

- Formal NPZ: `/home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/cmu_141_16_wave_bimanual.npz`.
- Validation: `/home/ubuntu/dexgym_data/marvin_motion_library_v1/bimanual/reports/cmu_141_16_wave_bimanual_validation.json`, with Gate A–D true and both sides checked.
- Mapping and control configuration: `/home/ubuntu/dexgym_data/marvin_motion_library_v1/bimanual/metadata/cmu_141_16_wave_bimanual_config.json`.
- Actual MP4s: `bimanual/previews/cmu_141_16_human.mp4`, `cmu_141_16_human_vs_marvin.mp4`, `cmu_141_16_newton_pd.mp4`, and `cmu_141_16_mujoco_pd.mp4`. The robot videos show both left and right TCP target and realized axes; physical videos use recorded stepped states.
- Physical states and metrics: `bimanual/physics_replays/`; motion library index: `index.json`.

The full NPZ contains both 7DoF arm trajectories, full 54DoF Home hand positions, both TCP and three named link FK poses and velocities, both human hand poses, both mapped targets, and per-side position/orientation residuals. `MarvinReference` serves arbitrary times and clamps future 10 frame goals at the end.

Regression run: `tests/assets tests/motion` yielded 23 passed and 3 skipped with the host's working Isaac Lab dependency paths; Ruff passed on the changed Python files.
