# M5 4096-environment Strict-Pure SE(3) pilot

## Status

**Real bilateral SE(3) error improved in three 4096-environment, 40-iteration pilots. Strict Reach & Hold did not pass.** Every independent near-goal 64-Episode Newton evaluation had zero complete successes and zero valid hold steps. The 80-iteration continuation of seed 53 regressed relative to its 40-iteration checkpoint. The 40-iteration checkpoints are retained; none is a qualified Reach & Hold policy.

This is a scale-up of the existing `marvin_bimanual_se3_reach_hold` task, not a new control algorithm. It uses `SchemaReplay`, WM8 masked autoregression, the existing 1/8/32 imagination curriculum, survival-weighted Actor return, TD(lambda) Critic, 14D `arm_joint_delta_v1`, the unchanged 54DoF Newton robot and the existing two-source Model Trust gate. No IK, joint-goal labels, teacher action, old Wave Actor or supervision entered training. The FK goal sampler discards the witness joint configuration after producing each target TCP pose.

## Scale and training contract

The versioned profile is [`m5_se3_near_4096_pilot.json`](../../configs/supermimic/m5_se3_near_4096_pilot.json). It uses 4096 Newton environments, 3072 training and 1024 held-out environments. Within each partition, half execute time-correlated raw-action exploration and half execute the deterministic current Actor. Four 50 Hz control steps produce 16,384 real Transitions per iteration. Sixteen WM8 updates of 2048 windows each retain the previous high-update pilot's nominal **16 model training targets per real Transition**. The 128-step Replay capacity preserves full 32-step imagination starts and uses **1,801,487,360 bytes**; it does not contain an entire 200-step Episode. The trust thresholds and safety limits were not relaxed.

The 4096-environment first-iteration capacity test ran 16,384 physical Transitions and 16 WM8 updates successfully. It did not attempt Actor updates before model warmup. The observed device allocation while training was about **16.4–17.7 GB** from spot `nvidia-smi` readings; these are observations, not a recorded peak. The PyTorch peak in the 40-iteration runs was **2.02 GB**. The full training process, including initialization and Replay checkpoint serialization, took **43.8–44.4 s** per 40-iteration seed, or approximately **14.8–15.0k real Transitions/s** end to end. Isaac/Newton allocations are not included in the PyTorch peak.

| Random seed | Real Transitions | WM8 updates | Actor updates | Critic updates | Final admitted imagination | Noisy held-out normalized WM error 1/8/32 | Nonfinite updates |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 51, 40 iterations | 655,360 | 640 | 36 | 288 | 1 step | 0.66 / 1.12 / 1.86 | 0 |
| 52, 40 iterations | 655,360 | 640 | 34 | 272 | 1 step | 0.66 / 1.11 / 1.83 | 0 |
| 53, 40 iterations | 655,360 | 640 | 36 | 288 | 1 step | 0.63 / 1.05 / 1.74 | 0 |
| 53, 80 iterations | 1,310,720 | 1,280 | 76 | 608 | 8 steps at final gate | 0.60 / 0.99 / 1.50 | 0 |

The 80-iteration continuation briefly admitted 8-step imagination at iterations 67, 75 and 79, producing **six** 8-step Actor updates. The other 34 Actor updates in that continuation remained one-step updates. It made **zero 32-step Actor updates**. An 8-step normalized error near the 1.0 threshold fluctuated above and below it; this does not establish sustained model trust. The 32-step error remained above its 1.1 threshold.

## Independent real-physics evaluation

Each row is an independent Newton process with its own 64 near-goal Episodes of 200 policy steps. Seed 9501 fixes one goal set for direct checkpoint comparison. The second seed 9502 gives a separate goal set. Position is per-TCP Episode mean in millimetres; orientation is per-TCP SO(3) geodesic mean in degrees. Full success requires *both* TCPs at ≤5 mm and ≤2°, joint speed ≤0.15 rad/s during 25 consecutive hold steps, arrival by step 125, and no safety failure.

| Policy | Evaluation goal seed | Mean position L/R, mm | Mean SO(3) L/R, ° | Episode success | Longest valid hold | Contact / illegal joint steps |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| Seed 53 initial Actor, before BPTT | 9501 | 19.91 / 20.08 | 3.81 / 3.91 | 0/64 | 0 | 0 / 0 |
| Seed 51, iteration 40 | 9501 | 11.20 / 11.33 | 2.58 / 2.61 | 0/64 | 0 | 0 / 0 |
| Seed 52, iteration 40 | 9501 | 9.87 / 10.85 | 2.77 / 2.60 | 0/64 | 0 | 0 / 0 |
| Seed 53, iteration 40 | 9501 | 12.06 / 9.74 | 2.70 / 2.57 | 0/64 | 0 | 0 / 0 |
| Seed 53, iteration 80 | 9501 | 26.34 / 17.28 | 3.95 / 3.91 | 0/64 | 0 | 0 / 0 |
| Seed 52, iteration 40 | 9502 | 8.90 / 10.23 | 2.70 / 2.42 | 0/64 | 0 | 0 / 0 |

The three 40-iteration seeds show consistent bilateral physical error reduction relative to the stationary initial Actor evaluated on seed 9501, but each still misses the simultaneous precision and hold criterion. The initial Actor checkpoint was saved for seed 53; per-seed initial evaluations for seeds 51 and 52 were not run, so this is not a paired per-seed baseline claim. Among the 40-iteration checkpoints, seed 52 had the lowest sum of position error divided by 5 mm plus SO(3) error divided by 2° on seed 9501. It is an **interim diagnostic candidate only**, not a selected successful policy. Its separate seed-9502 evaluation also failed the strict criterion.

The 80-iteration seed-53 Actor increased maximum action rate from **0.051 to 0.157** and maximum command jump from **0.00134 to 0.00233 rad** relative to its iteration-40 version, although both stayed collision-free. Real reward fell from **0.05172 to 0.04330**. The later checkpoint is not selected. There was no Medium/Wider policy training after this regression.

## Model accuracy and action-gradient diagnosis

For the seed-53 iteration-40 policy, independent Actor-in-Newton action-conditioned open-loop World Model TCP position error was **3.77/3.93 mm** at 32 steps; separate policy closed-loop Model-vs-Newton error was **5.48/5.59 mm**. For iteration 80 these became **1.87/2.44 mm** and **4.25/2.12 mm**, respectively. These are model errors on each corresponding Actor's real trajectory, not task target errors. The later model predicted its Actor distribution better while its Actor controlled the real task worse.

A 30-environment paired Newton finite-difference audit sampled four independent legal near goals, with two zero-action baselines and 14 positive/negative action pairs at each state. At reset, seed-53 iteration-40/80 model-vs-Newton reward-gradient cosine was **0.898/0.880**, model-to-real gradient norm **2.69×/1.15×**, and joint-velocity action-response cosine **0.194/0.209**. Baseline reward disagreement was ≤6×10⁻⁸. Thus improved state prediction did not prove the learned local action derivative was correct at reset.

An additional teacher-free Newton probe dataset used 4096 environments per physical step, 136 complete independent start groups, raw joint-action perturbations ±0.1 and ±0.5, and one masked physical Transition per probe. Training has **8,192** transitions and a separate **4,096**-transition validation set. Every probe was safety-valid, with paired zero-action reward difference ≤1.6×10⁻⁷. The target joint witness is absent from `TransitionBatch` and Replay. This dataset was used only for a bounded 500-update WM8 diagnostic, alternating four original Replay updates with one masked probe update. No Actor or Critic was updated.

The independent probe validation loss fell **1.761 → 1.043**, but final noisy held-out 8/32-step normalized error was **1.011/1.498**, so only one-step trust remained. Reset reward-gradient cosine improved **0.880 → 0.922**, while gradient norm worsened **1.15× → 1.61×**; after 20 steps of the iteration-80 real Actor the corresponding norm worsened **0.95× → 1.28×**. Joint-velocity response direction at reset stayed near **0.21**. The diagnostic World Model was **not** installed into a training or inference checkpoint. The results do not justify loosening trust or resuming long-horizon BPTT.

## Reproduce and artifacts

Run from the M5 worktree. Keep each output path new; these commands must not overwrite earlier checkpoints.

```bash
export PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source
export ISAAC_PYTHON=/home/ubuntu/isaacsim-6.0.1/bin/python
export M5_DATA=/home/ubuntu/dexgym_data/marvin_supermimic_m5_se3_reach_hold

$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py train_se3_m5 --config configs/supermimic/m5_se3_near_4096_pilot.json --iterations 40 --seed 52 --output "$M5_DATA/my_4096_seed52"
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py evaluate_se3_m5 --checkpoint "$M5_DATA/my_4096_seed52/model.pt" --num-envs 64 --seed 9501 --output "$M5_DATA/my_4096_seed52_eval"
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py validate_se3_model_m5 --checkpoint "$M5_DATA/my_4096_seed52/model.pt" --physics-trace "$M5_DATA/my_4096_seed52_eval/physics_trace.pt" --closed-loop --output "$M5_DATA/my_4096_seed52_eval/model_validation.json"
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py collect_se3_action_probes_m5 --num-envs 4096 --seed 8102 --output "$M5_DATA/my_4096_probes"
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py diagnose_se3_world_model_m5 --checkpoint "$M5_DATA/my_4096_seed52/model.pt" --probe-data "$M5_DATA/my_4096_probes/probes.pt" --updates 500 --batch-size 512 --output "$M5_DATA/my_4096_probe_diagnostic"
```

The actual runs used the identically structured paths `m5_4096_capacity_53`, `m5_4096_pilot_{51,52,53}`, `m5_4096_pilot_53_iter80`, `m5_4096_eval_*`, `m5_4096_probe_contract_8102`, and `m5_4096_probe_wm_diag_53` under `$M5_DATA`. JSON reports, 200-step physical traces, original checkpoints, diagnostic-only World Model and provenance hashes remain there. The older 64-environment M5 runs and M4.1/M4.2/M4.3 assets were not overwritten.

The complete SuperMimic test suite passed (**191 tests**) after the scaling changes. The M4.1 selected Wave model's 1356/1356 Newton regression was already passed on this branch before these M5-only training and diagnostic changes; no legacy Actor, action map, physical asset or controller code changed here.

## Acceptance

- **Passed:** 4096-environment full alternating loop, valid Newton/Replay data, normalizer updates, WM8, 1-step Actor BPTT and Critic updates, no nonfinite updates, three-seed bilateral physical error improvement at 40 iterations, independent goal-set evaluation, physical probe parity, legacy baseline preservation.
- **Not passed:** simultaneous 5 mm / 2° bilateral arrival, 25-step hold, any complete Episode success, sustained 8-step trust, 32-step trust and actual 32-step policy BPTT, stable gains through 80 iterations, Medium/Wider trained-policy performance, universally safe exploratory collection, proof of dynamic feasibility for every FK-sampled target.

**M5 is not a converged Reach & Hold solution.** More environments improved data coverage and early physical tracking but did not by itself solve long-horizon Model Trust or stable policy optimization.
