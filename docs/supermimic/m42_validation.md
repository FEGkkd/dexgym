# M4.2 validation report — 2026-11-06

## Scope and evidence

Branch: `feat/marvin-supermimic-m42-framework`, based on `952dc97fe29099b664f0c40f6d97a911b77ccd3c`. The immutable M4.1 Wave reference and selected Actor have SHA256 `b7af95b289e8aff898df7f5dd06b169f74dba2a506aa8fc5bf589ee3fee8106f` and `fd0054a7ef707d46a30b0235713c9e00cc4d7e69d6687fc3015e52aac2cb773f`. All experiments below are under `/home/ubuntu/dexgym_data/marvin_supermimic_m42/`; none overwrote the M4.1 checkpoint or robot assets. The second reference is `second_task/procedural_bimanual_reach_v1.npz`, SHA256 `ca4741f01734389dd62f9e64c66cdc9c237c90da6370cf76b13bcda89e143af2`. It is a **procedural test**, not validated human motion capture.

The paper and both implementations were checked at the exact versions in [algorithm_parity.md](algorithm_parity.md). The current source was tested with 81 automated tests (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, Isaac Sim 6 Python). Tests include a saved output fixture generated from the untouched M4.1 source and pinned checkpoint; maximum tolerance is 2e-6. The Newton/MuJoCo/Warp pinned import overlay used for all physical runs is preserved at `runtime_overrides/` in the experiment directory. Recorded source SHA in early experiment checkpoints is `4b6328b92a18e94f09f9bd05e83b307b720c50c8`; subsequent code fixes and tests are represented by the final branch SHA, not retroactively by those old experiment files.

## Hard gates

| Gate | Result | Evidence and limit |
| --- | --- | --- |
| A — mathematics | **Pass for implemented 42D tracking contract** | Time alignment, masked 1/2/3/5/8-step Replay, wraparound, partial reset, survival return, TD(λ), finite differences, FK/SO(3), BPTT freeze, normalizers, and exact legacy actions are tested. |
| B — physics/model semantics | **Partial** | Newton pre-reset transitions, command map, 10 substeps, world-ID contact isolation and 1/8/32-step prediction were checked. The learned model remains an approximation and contact/object dynamics are outside its 42D state. |
| C — from-zero training | **Partial** | Three strict-Pure seeds entered 32-step BPTT, completed 200-step independent Newton playback safely, and improved both TCP position errors. Total reward improved in only two of three seeds; Wave from-zero convergence is unproven. |
| D — M4.1 compatibility | **Pass** | Old-source output fixture passes; 1356/1356 Wave Newton steps complete without contacts, illegal joints, or termination. |
| E — second reference | **Pass for reference reuse; broader task claim open** | Procedural reference passed full Newton PD validation, then the identical `train.py` and `SuperMimicTrainer` ran strict-Pure learning and physical evaluation. A different goal encoder or contact-rich Task Adapter has not yet been trained. |
| F — reproducibility | **Partial** | Hashes, versions, complete config, model/optimizer/normalizer/scheduler/RNG states, optional Replay, logs, and actual resume are present. Newton environments restart on resume; physical trajectories are not bitwise continuous. BF16 was not enabled or validated. |

**M4.2 is not Fully Validated.** The result establishes a reusable core for multiple Marvin tracking references, not general cross-robot or contact-task convergence.

## M4.1 Newton regression and second reference

The Legacy loader retains the old identity Actor/Goal/Critic normalization, goal encoder activation, and `legacy_wave_v1` action map. The fixed old-source output fixture matches the new Legacy loader. `wave_regression_cli/evaluation.json` reports 1356/1356 control steps, zero Newton contact steps, zero joint-state violations, left/right mean TCP position error **1.905/4.197 mm**, and orientation error **0.275/0.969°**. Maximum measured arm speed was 2 rad/s. No fine tuning was applied.

The procedural reference has 201 frames at 50 Hz and uses real Marvin FK. `second_task/pd_physical_v3/evaluation.json` reports **200/200** steps, zero contact, joint position RMSE **0.000359 rad**, and maximum arm speed **0.564 rad/s**. This establishes a physically executable second reference, not an optimal learned policy.

The original robot-wide `robot_fixed_v2` action scale caused a **5.50 rad/s** arm-speed failure after two random-Actor control steps; it remains versioned only to interpret early experiments. The `robot_fixed_v3` contract uses the robot-home origin, a 0.25-rad operating envelope clipped by joint range, and a 0.01-rad maximum command increment. The fixed controller gains, 10 physical substeps, MJCF, 14 controlled arm joints, and 40 passive-home fingers were not changed. A 64-env, 3200-transition random physical check of v3 had no termination and maximum arm speed 0.986 rad/s. The Wave Legacy action map was not altered.

## Strict-Pure startup: three seeds

Each seed used 64 Newton environments, 200 iterations, 12,800 real Actor transitions, 800 world-model updates, 175 Actor updates, 1400 Critic updates, 32-step BPTT, 2×4 Critic fitting, no teacher sampling, no reference-action Actor supervision, and no nonfinite optimizer events. All 200-step evaluations used independent full Newton playback of the procedural test. Initial metrics are from a random Actor of the same architecture.

| Seed | Initial reward | Trained reward | Initial TCP L/R mm | Trained TCP L/R mm | Newton completion |
| --- | ---: | ---: | ---: | ---: | ---: |
| 44 | 0.096421 | **0.098226** | 26.96 / 26.50 | **19.94 / 13.61** | 200/200, safe |
| 45 | 0.096421 | **0.097489** | 26.96 / 26.50 | **22.91 / 17.36** | 200/200, safe |
| 46 | 0.096421 | **0.078673** | 26.96 / 26.50 | **22.45 / 16.58** | 200/200, safe |

Thus both TCP position metrics improved in all three seeds, while the full reward regressed in seed 46. Actor update count alone is not treated as convergence. The safe analytic run had no positive real termination labels, so its zero false-positive rate cannot establish failure recall.

## Model prediction and trust

The fixed validation environment IDs are distinct from training IDs. The final logged actor-data held-out action-conditioned open-loop TCP position errors in seed 44 were **1.169/1.092/1.136 mm** at 1/8/32 steps; TCP rotation **0.194/0.180/0.187°** and joint velocity RMSE **0.108/0.126/0.140 rad/s**. Seed 45 position errors were **1.066/1.112/1.207 mm**; seed 46 **1.319/1.179/1.078 mm**. These are held-out Replay windows, not independent Newton actor rollouts. Constant-velocity TCP error in seed 44 was 5.059 mm at one step and 40.988 mm at eight steps.

`validate_dynamics.py` independently started Newton and model from the same physical state for 32 steps. At the final step, action-conditioned open-loop versus Actor closed-loop model TCP error was: seed 44 **1.916/2.092 mm**, seed 45 **1.191/1.291 mm**, seed 46 **1.073/1.410 mm**. The closed-loop error uses the Actor's model states to choose its actions. These are a single starting segment per seed and do not cover all trajectory phases or large off-policy deviations. Trust promotion additionally requires actor-source validation and improvement over constant velocity; teacher-heavy loss alone cannot open the gate.

## Termination modes and resume failure found during validation

Analytic soft termination shares the hard TCP/limit/speed thresholds and keeps gradients. Its safe v3 experiments produced zero real positive labels, so recall and calibration on failures remain unmeasured in this mode.

The split `TaskTerminationHead` uses genuine Newton pre-reset hard labels and the arrival-frame reference. A 150-iteration boundary-probe run had independent seed-51 recall **0.865**, FPR **0.0283**, Brier **0.0296**. An initial resume **without saved Replay** then trained on only negative examples: independent recall collapsed to **0**, Brier **0.0552**. This failure is retained as evidence. The repair optionally saves the entire ring with absolute timestamps, source, failure reason and episode IDs; learned mode defaults to saving it. For older no-Replay learned checkpoints, 32 new Newton boundary-probe steps are collected before Actor updates. New environments receive new episode IDs.

After the repair and 100 additional iterations from the 150-iteration checkpoint, `learned_split_seed50_resume_reprobe/independent_termination_seed51/report.json` evaluated 2993 distinct real Newton boundary transitions, including **170 positives and 2823 negatives**: precision **0.863**, recall **0.665**, FPR **0.00638**, Brier **0.01938**, ECE **0.00937**. All observed positives were TCP deviation failures; speed, limit, and contact failure recall are unvalidated. This is limited calibration, not a general safety predictor. The resulting Actor completed 200/200 physical steps but tracked less accurately than the analytic runs: mean TCP L/R **31.57/27.99 mm**. The saved-Replay checkpoint was then actually restored for five more Newton iterations; Replay advanced from 100 to 105 steps and Actor updates from 75 to 80.

A separate 32-env Newton cosine-LR smoke (`schedule_smoke_seed54`) completed eight iterations, then resumed for four. World-model schedule state advanced from epoch 8 to 12 and LR from 6.5e-5 to its configured 2e-5 floor; Actor/Critic schedules remained at epoch 7 during four resume iterations without valid updates. A first version that stepped every scheduler before warmup optimizer updates emitted a PyTorch ordering warning; it was replaced by optimizer-bound stepping before this successful run. This validates scheduler state recovery, not long-run cosine-LR learning quality.

An earlier M4.2 v1 monolithic-head run (`learned_boundary_seed47_timingfixed`) had independent seed-49 precision **0.816**, recall **0.964**, FPR **0.0151**, Brier **0.01185**, ECE **0.01357**, also only on TCP failures. The pre-time-fix `learned_boundary_seed47` run is invalid and excluded. Split-head learning does not inherit those v1 calibration numbers.

## Critic optimizer ablation

At seed 44 and the same procedural reference, 200 rounds and 175 Actor updates were compared with 1×1, 2×4, and 5×4 Critic configurations. The last 25 Actor updates yielded mean Critic loss **0.03286 / 0.0000868 / 0.0000567** and mean absolute TD error **0.1607 / 0.00584 / 0.00544**. Their independent Newton TCP L/R means were **22.89/28.28**, **19.94/13.61**, and **20.81/12.58 mm**, respectively. Reward was **0.07854 / 0.09823 / 0.07864**. All had zero nonfinite events. Additional Critic epochs reduce fitted TD error; 5×4 did **not** improve the measured total physical reward over 2×4. This single-seed ablation is not a hyperparameter optimum claim.

## RTX 5090 full-loop capacity

Each row ran 60 complete Newton collection → WM → trust → 32-step Actor BPTT → 2×4 Critic iterations, with 4 WM updates per iteration. All reached 50 Actor and 400 Critic updates with zero nonfinite gradients. Device memory is `nvidia-smi` usage at the end of the loop, **not peak total-device memory**; PyTorch peak allocation is separately measured.

| Envs | Full-loop physics transitions/s | WM updates/s | Actor updates/s | End total VRAM / 32,607 MB | Peak PyTorch allocation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 370 | 5.79 | 1.21 | 2,529 MB | 303 MB |
| 1024 | 1,461 | 5.71 | 1.19 | 4,993 MB | 513 MB |
| 2048 | 2,837 | 5.54 | 1.15 | 8,229 MB | 797 MB |
| 4096 | 5,414 | 5.29 | 1.10 | 14,792 MB | 1,376 MB |

4096 fits this measured loop but its fixed 4 WM updates per 4096 new transitions give a much lower update/data ratio than 64 env. **64 env is the only scale with three-seed learning evidence**; choose training scale from physics data and optimizer ratio, not throughput alone. End-device VRAM is not a certified peak.

## Reproduce key checks

From this worktree:

```bash
export PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source
ISAAC=/home/ubuntu/isaacsim-6.0.1/bin/python
M42=scripts/supermimic/isaac_runtime.py
DATA=/home/ubuntu/dexgym_data/marvin_supermimic_m42

$ISAAC "$M42" evaluate --checkpoint /home/ubuntu/dexgym_data/marvin_supermimic_wave_m41/selected_best.pt --reference /home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/cmu_141_16_wave_bimanual.npz --output "$DATA/wave_regression_recheck"
$ISAAC "$M42" evaluate --checkpoint "$DATA/pure_v3_seed44/iteration_000200.pt" --reference "$DATA/second_task/procedural_bimanual_reach_v1.npz" --output "$DATA/seed44_recheck"
$ISAAC "$M42" validate_dynamics --checkpoint "$DATA/pure_v3_seed44/iteration_000200.pt" --reference "$DATA/second_task/procedural_bimanual_reach_v1.npz" --output "$DATA/seed44_dynamics_recheck"
$ISAAC "$M42" validate_termination --checkpoint "$DATA/learned_split_seed50_resume_reprobe/iteration_000250.pt" --reference "$DATA/second_task/procedural_bimanual_reach_v1.npz" --output "$DATA/termination_recheck" --seed 51
$ISAAC "$M42" train --reference "$DATA/second_task/procedural_bimanual_reach_v1.npz" --output "$DATA/new_seed52" --mode pure_model_based --action-schema robot_fixed_v3 --termination-mode analytic --num-envs 64 --iterations 200 --world-updates-per-iteration 4 --world-warmup-updates 100 --world-batch 64 --imag-batch 16 --critic-epochs 2 --critic-minibatches 4 --action-noise 0.005 --reset-perturb-rad 0.002 --no-random-reference-start --seed 52
```

For the full unit suite, use the Isaac interpreter with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, append `/home/ubuntu/dexgym/.venv/lib/python3.12/site-packages` after importing `torch,mujoco,warp`, and run `pytest.main(['-q','tests/supermimic'])`. The normal venv interpreter was not used because local native package files were corrupt; the pinned overlay above is an actual tested runtime workaround, not a change to robot physics.

## Remaining boundaries

The trainer has not yet learned an object/contact task or a different goal tensor schema. `MotionLibrary` handles multiple clip IDs and lengths, but simultaneous mixed-motion training in one environment batch has not received a Newton learning experiment. No full Wave from-zero training, BF16 numerical comparison, wall-time-optimized 4096-env schedule, complete episode-stage closed-loop model sweep, or failure-reason-balanced termination calibration has been established. The resume path restores model/optimizer/scheduler/RNG and optionally Replay, while newly constructed Newton environments necessarily start fresh physical episodes. These items prevent a claim of universal cross-task stability or full paper reproduction.
