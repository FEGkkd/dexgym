# Phase 0B: suspended grasp diagnosis and staged synthesis

Status: **left-only suspended hold failed; bimanual hold and validated bank were not run**. All measurements below used native official Beta 1 USD in Isaac Sim 6.0.1 / Isaac Lab 3.0 / PhysX, gravity enabled. The 70 mm diameter bottle remains the formal engineering sample. Beta 1 is an unverified match to the user's non-tactile hardware.

## Root causes found

1. The old `stable_pregrasp` generator places both hands at half-closed joint positions in a single Reset. Its left wrist is around `(-0.11, 0.03, 0.466)` m relative to the bottle center at `(0, 0, 0.5)` m. Actual USD collisions overlap the 35 mm radius cylinder by 39–42 mm at the first step. The worst pairs include index/middle finger middle and proximal-abduction links and the ring finger proximal-abduction link. The three corrected-mass replays still produce first-step left contact peaks of 220, 151, and 311 N. This is an initialization geometry error, not evidence for insufficient friction. See [measured summary](phase0b-results-summary.json).
2. Five fixed fingertip marker rigid bodies in the official Beta 1 USD declare `mass=0`. PhysX on this workstation interpreted each as 1 kg. Runtime left-hand mass was **5.6207 kg** although the official physical-link masses sum to about **0.6207 kg**. The scene adapter now gives each zero-mass marker a documented 1 g / `1e-6 kg m²` positive proxy, without changing the upstream USD or colliders. Runtime hand mass becomes **0.6257 kg**. An isolated 1 s wrist test then limits vertical drift to **5.1 mm**; before this fix the hand fell roughly 0.325 m and the 40 N controller saturated. This is a PhysX compatibility override, not a hardware mass measurement.
3. The original wrist rotation PD gains (`kp=8 Nm/rad`, `kd=1.2 Nm s/rad`, 2 Nm torque bound) caused a **0.343 rad / 19.6°** orientation error and about **3.6 rad/s** angular speed in a no-contact 1 s gravity-on test. Lowering damping to `0.05 Nm s/rad` and proportional gain to `4 Nm/rad` retains the 2 Nm bound and passes the same test with under 1 mrad orientation error and about 0.009 rad/s angular speed. This identifies a control gain problem in addition to the initial geometry and marker-mass problems. Under bottle contact, even the tuned controller still deviates from the candidate wrist orientation by about 0.36 rad, leaving the thumb tip about 78 mm from the bottle axis. Higher contact authority (`4 Nm`, `kp=8`, `kd=0.05`) improves that error to about 0.27 rad but moves the wrist about 28 mm and still loses the bottle. The expected thumb/finger opposition is absent. Joint position drives reached their 2 Nm official USD limits in earlier slow-close trials. The present evidence does not establish that friction or inherent 70 mm geometry is the sole remaining cause.

The official MJCF fingertip site forward kinematics and native USD tip-frame positions matched to numerical precision at the *same measured wrist and joint state*. Therefore the current site-mapping convention is sound. The failure is in reaching and holding the desired physical state, not simply in reading the wrong site position. Distal link origins remain distinct from the site at the very end of each fingertip.

## Implemented components

- `scenes/grasp_synthesis.py`: per-joint preshape and slow close targets, bounded wrist PD, support by bounded bottle wrench or an explicitly removable world fixed joint, raw contact-point and signed-separation guard, 1 s support-free release, and independent Reset replay only if release passes. Replay rechecks thumb/two-other-finger contact, penetration, force peak, contact duration, and bottle drift. A temporary support joint is disabled before the measured release. No hand or object root is written per physics step.
- `scenes/contact_monitor.py`: `raw_contact_points(env_id, dt)` exposes all exact-pair PhysX points, normals, forces, and signed separations, including zero-force penetration. The contact history retains its configured classification force floor.
- `assets/wuji_hand2_beta.py`: scene-layer Beta 1 fingertip marker mass override. `assets/wrist_6dof.py`: measured lower-damping default rotation PD. Left and right official assets retain their pinned checksums and 20 active joints per side.
- `grasp_init.py`: bank rejects more than 2 mm initial penetration, force guards above 100 N or measured peaks exceeding the configured guard, missing thumb/two-other-finger distribution, samples with support present, and samples without an independent Reset replay. The 100 N cap is an engineering anomaly bound, not a calibrated grasp threshold.
- Standalone diagnostic, geometry search, plotted comparison, GUI preview, and GPU synthesis scripts under `scripts/environments/`. MuJoCo is used only by optional offline MJCF geometry diagnostics. The Isaac runtime and formal asset remain native USD/PhysX.

## Physics results

| Check | Result | Evidence |
| --- | --- | --- |
| Old seeds 0, 1, 2 | **Fail** | First-step separation `-41.6`, `-39.6`, `-41.6` mm; 1 s bottle displacement `0.533`, `0.509`, `0.486` m. |
| 70 mm random geometry, 1,500 offline proposals | Geometric screen only | 23 thumb-plus-two-digit proximity candidates, **zero** also satisfying the 2 mm all-link clearance screen. This finite sample does not prove impossibility. |
| 60 / 50 mm diameter offline screen | Geometric screen only | 22 / 18 proximity candidates; zero met the same clearance screen. These are diagnostic sizes and do not replace 70 mm. |
| 70 mm staged PhysX search, 20 selected proposals | **Fail** | 6 aborted on penetration; other 14 dropped the bottle after support removal. This run predated the release-only metric field, so its contact ratio is for the whole attempt. |
| 50 mm staged PhysX diagnostic, 5 proposals | **Fail** | 2 penetration aborts; 3 support-free releases displaced `0.496–0.547` m with release contact ratios `0.034–0.052`. |
| 70 mm tip-site and convex-distance optimization, 8 PhysX proposals | **Fail** | Optimized MJCF poses were within about 2 mm signed convex distance, but the actual contact-driven wrist pose changed. All releases dropped the bottle; no 1 s hold or Reset replay passed. |
| Temporary fixed support, then disabled | **Fail** | The bottle stayed at `(0,0,0.5)` m during synthesis. After disabling the joint, the release displaced `0.484` m. The trace records the joint as disabled throughout release. |
| Slower 2 s close and 2 s settle | **Fail** | The middle/ring contacts remained one-sided, with a persistent wrist pose error; release displacement `0.508` m. |
| Tuned wrist, 70 mm fixed-support, 8 PhysX proposals | **Fail** | Initial penetration less than 0.14 mm and peaks below 8.5 N, but releases displaced `0.401–0.518` m; none had thumb opposition. |
| Higher contact torque (`2` and `4` Nm) | **Fail** | No-contact wrist hold passed; one supported contact trial per setting still released the bottle, with 0.36 / 0.27 rad wrist orientation error and 0.499 / 0.523 m displacement. |
| No-contact free fall | **Pass** | Measured 0.0500 m in 0.1 s, consistent with `g t²/2`; no hidden bottle suspension. |
| Contact classification | **Pass** | PhysX tests distinguish left-bottle, right-cap, cross contacts, hand-hand, ground, and env 0 from env 1. This is classification, not successful grasp. |
| Selective Reset | **Pass** | 100 resets each at 1, 2, and 16 envs; maximum bottle position and finger target restoration error zero, finite states, no unchanged-env mutation, detached thread restored. |
| Independent wrist orientation hold | **Pass** | Tuned default: position drift 5.1 mm and orientation drift below 1 mrad after 1 s in free space. Contact-loaded wrist tracking remains unproven. |
| Left-only hold / 3 replayed bank samples | **Fail** | 0 valid samples. No `grasp_init_bank` was written. |
| Bimanual hold | **Not Tested** | Right-hand synthesis was intentionally stopped because left-only hold did not pass. |

The force and pose trajectory in [phase0b-results-summary.json](phase0b-results-summary.json) includes bottle/wrist poses, contact points and normals, force peaks, applied finger torque buffer values, and support-joint enabled state. The [raw PhysX evidence bundle](phase0b_raw_physx_evidence.tar.gz) preserves complete legacy, 50 mm diagnostic, tuned-wrist, and high-authority traces plus the proposal inputs (`tar -xzf docs/assets/phase0b_raw_physx_evidence.tar.gz -C /tmp`). The [geometry plot](phase0b_grasp_geometry.png) compares an official MJCF site proposal with the actual USD/PhysX positions. A small signed penetration (for example `-0.3 mm`) is a diagnostic contact, not a stable-grasp pass.

Focused offline contracts: `22 passed` (`test_wrist_6dof.py`, `test_bimanual_grasp_init.py`, `test_bimanual_contact_monitor.py`, `test_bimanual_reset_manager.py`). The earlier complete baseline was `2 failed, 89 passed, 1 skipped`, with failures in unrelated user-worktree Allegro/MANO tests. A subsequent complete Isaac-runtime pytest attempt was interrupted after 140 s during package import (`5 passed` at that point), so it is not reported as a completed suite. The 1/2/16-env Reset and contact classification results in the table are actual post-controller-change GPU runs.

## Reproduce

Use an official checkout pinned at `c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95` and set `WUJI_DESCRIPTION_ROOT` to it. Set `ISAACLAB_ROOT` and `ISAACLAB_PYTHON` for the installed Isaac Lab / Sim. The repository's `scripts/run_isaaclab.sh` launcher supplies the DexGym extension path.

```bash
./scripts/run_isaaclab.sh scripts/environments/analyze_grasp_reach.py --radii 0.035 0.030 0.025 --samples 1500 --output docs/assets/phase0b_reach_scan.json
./scripts/run_isaaclab.sh scripts/environments/optimize_grasp_geometry.py --radius 0.035 --starts 8 --seed 23 --output docs/assets/phase0b_tip_optimized_reach_35mm.json
./scripts/run_isaaclab.sh scripts/environments/generate_grasp_synthesis.py --headless --reach_scan docs/assets/phase0b_tip_optimized_reach_35mm.json --candidate_count 8 --output docs/assets/phase0b_tip_optimized_synthesis_35mm.json
./scripts/run_isaaclab.sh scripts/environments/generate_grasp_synthesis.py --headless --fixed_support --reach_scan docs/assets/phase0b_tip_optimized_reach_35mm.json --candidate_count 8 --output docs/assets/phase0b_tuned_wrist_fixed_support_35mm.json
./scripts/run_isaaclab.sh scripts/environments/validate_wrist_gravity_hold.py --headless --steps 500 --output docs/assets/phase0b_wrist_gravity_hold.json
./scripts/run_isaaclab.sh scripts/environments/validate_bimanual_contacts.py --headless --output docs/assets/phase0b_contact_regression.json
./scripts/run_isaaclab.sh scripts/environments/validate_bimanual_reset.py --headless --num_envs 16 --iterations 100 --output docs/assets/phase0b_reset_16.json
./scripts/run_isaaclab.sh scripts/environments/preview_grasp_initialization.py --mode near_contact --contact_debug
```

The preview is an **unvalidated near-contact state**, and will fall. The captured fixed-support diagnostic state cannot be used as a Reset sample: its measured cap axial coordinate differs from the D6 initial-engagement coordinate by more than the existing 0.1 mm Reset guard. The diagnostic trace remains in the measured summary; it is not advertised as a replayable grasp. The `--fixed_support` option is diagnostic and its joint is disabled before release. The old `--strong_wrist` gains increased no-contact angular speed to about 17 rad/s and caused a 2.37 mm penetration abort; the lower-damping tuning above is the measured alternative. In headless mode add `--headless` to the preview command.

## Next minimum experiment

First measure wrist tracking under a controlled bottle contact load, with angular/linear velocity and force/torque traces, then tune the bounded wrench controller without reintroducing free-space oscillation. Keep the marker mass override and official finger torque limits explicit. Run a fixed-bottle, contact-feedback finger close that brings the **measured** thumb site to one side of the 35 mm radius and at least two other distal contact patches to opposing sides, while logging all-link minimum separation and support reaction. Only after a low-reaction closure should the support be disabled for the independent 1 s left-only hold and Reset replay. A right-hand cap approach and bank promotion are gated by that result. An actual hardware bottle and measured torques would be needed for real-bottle calibration; the current bottle remains an engineering sample.
