# M5 IK-free bimanual SE(3) reach and hold

## Scope and status

This is a new `marvin_bimanual_se3_reach_hold` task on the M4.3.2 SuperMimic Core. It does not use IK, Jacobian inverse, a joint target label, a motion reference, a teacher policy, or PPO. FK is used to create reachable TCP pose targets and to calculate observations and differentiable reward. The generated witness joint configuration is discarded inside `SE3GoalSampler._draw`; the returned sample and Replay contain only the 18D dual TCP pose and difficulty statistics.

**Engineering integration passed. Strict dual-TCP 6D Reach & Hold learning did not pass.** The first three bounded near-goal pilots updated the model, Actor and Critic, but only 1-step imagination passed the original model trust gate. A subsequent dual-distribution pilot correctly blocked Actor updates because held-out model error was still too high. All independently evaluated policies had 0/16 successful full Episodes and zero valid hold steps. Medium and wider levels were physically preflighted but not trained.

## Contracts

| Layer | M5 contract |
| --- | --- |
| Robot | Unchanged 54DoF Marvin Newton MJWarp, 14 controlled arm joints, 40 fingers Home, 500Hz physics and 50Hz policy, 10 linear servo substeps. State is `[q14, qd14, qcmd14]`. |
| Action | New `arm_joint_delta_v1`: `qcmd_next = clamp(qcmd_previous + 0.01*tanh(raw_action), joint_lower, joint_upper)` in rad per 20ms. This is relative to the previous command. Old absolute action schemas remain unchanged. |
| Goal | Fixed `[N,1,18]`: left/right TCP base-frame xyz plus the first two rotation-matrix columns (continuous rotation 6D). Target error is SO(3) geodesic/log, never elementwise 6D rotation L2. |
| Observation | `[N,10,68]`: q14, qd14, qcmd14, previous bounded action14, bilateral TCP position error6, bilateral relative SO(3) log error6. |
| Critic | 60D physical state and target TCP pose. No witness joints. |
| Episode | 200 policy steps. Target does not change. Far initial error does not terminate. Real contacts, illegal joint state and unsafe speed do terminate. |
| Strict success | Both TCPs simultaneously within 5mm and 2 degrees, all arm speeds at most 0.15rad/s for 25 consecutive steps; first complete SE(3) arrival by step125; full 200-step Episode with no safety failure. Arrival does not reset. |

`MarvinSE3ReachHoldEnv` subclasses only `MarvinArmNewtonBaseEnv`. The base robot layer retains the old asset, controller gains, effort limits, hand commands, physical substep interpolation and contact audit. Task Reward and hard termination run through the same Task MDP methods in Newton and imagination. `CoreNewtonCollector` checks the executed command, pre-reset next state, reward and hard termination against the Task MDP for every real transition.

The goal sampler has independent seeded training and evaluation streams. Near, medium and wider candidates use random legal starts and a separate random witness configuration; all sampled points on a five-point joint interpolation are statically collision screened with the pinned MuJoCo asset. Both arms must exceed minimum joint, TCP position and TCP rotation movement. This screen establishes sampled static kinematic feasibility and excludes contacts on the screened interpolation samples. It does **not** prove that an arbitrary learned 4-second trajectory to the pose is collision free. Real Newton reset/episode audits provide the physical observations below.

## Reused SuperMimic Core

The task uses `SchemaReplay`, 8-step masked `StateDeltaWorldModel`, `NetworkFactory`, `ImaginationEngine`, survival-weighted Actor return, TD(lambda) Critic, `SuperMimicTrainer`, `ModelTrustEvaluator`, `CoreCheckpoint` and `PhysicalEvaluator` directly. There is no copied optimization loop. The new dynamics signature differs from `robot_fixed_v3` because relative command integration changes; the factory rejects sharing old World Model weights. Checkpoint inference loading verifies Task, Robot, action and network semantics. Training checkpoints save Replay with `resume_mode="replay"`; a resumed Newton process starts new Episode IDs and revalidates trust.

Strict-pure training uses a seeded AR(1) perturbation of Actor raw actions (`rho=0.95`, `sigma=0.8`). The bounded delta map is applied after the perturbation, and Replay stores the exact raw action, executed command and physical state. There is no teacher source or Actor supervision. The immutable action map is also used by `MarvinCoreRobot.integrate` during imagination.

## Validation evidence

Artifacts: `/home/ubuntu/dexgym_data/marvin_supermimic_m5_se3_reach_hold/`. Checkpoint hashes are in `checkpoint_hashes.json`. The M4.1 actor and Wave reference SHA256 remain `fd0054a7...463` and `b7af95b...106f`, respectively.

The bounded pilot checkpoints were generated while this branch still contained uncommitted M5 source; their embedded `git_commit` is the clean base `68e3ec52618a7c984de9294162c7b3d81911bdb0`, not the M5 source revision. Do not infer that the embedded base SHA alone reproduces the pilots. A clean-source copy of the seed53 model is `m5_pilot_53_clean_source/model.pt` (SHA256 `818227ade630740811e550127b4454623fd694a7ca55385cc9e72785ca88941e`). Its embedded source commit is `0fc9fe6563677104674de2b832e0fc0b890050ac`; Actor, World Model and Critic state tensors, including normalizers, are exactly equal to the original pilot checkpoint. Because it was saved after a fresh-physics resume, its trust status correctly resets to untrusted until new real validation.

- Full SuperMimic Torch suite: **187 passed**. New tests cover the relative action and gradient, 10 substep interpolation, FK goal schema/sampling, separate curriculum streams, no `q_goal` in Task state, bilateral reward symmetry, real Marvin SO(3) finite-difference gradient, strict 200-step success, action compatibility signature, task-specific 32-step BPTT with frozen dynamics parameters, temporal exploration reproducibility, source-aware collection contracts, and dual-distribution trust decisions.
- Newton preflight: near, medium, wider each ran 16 environments × 4 policy steps with zero hard terminations, zero command parity error and valid 18D goals. The near start had nonzero per-joint standard deviation across environments. The independent near evaluation ran 16 complete 200-step Episodes with zero contacts or illegal joint states.
- 16-environment Strict-Pure integration smoke: 768 real transitions, 24 WM updates, 2 Actor updates, 16 Critic updates and no nonfinite updates. It established pipeline execution only.
- M4.1 full Wave regression: **1356/1356** Newton control steps, zero physical contacts, zero illegal joint states. Left/right TCP mean position errors were **2.741/5.075mm**, mean orientation errors **0.371/1.074 degrees**. The original checkpoint and reference hashes were unchanged.
- A second full Wave regression after the data/trust changes again completed **1356/1356** control steps with zero contacts and illegal joint states. Left/right mean position errors were **2.217/4.519mm**, orientation **0.311/1.021 degrees**. This is run-to-run physical variation, with no legacy Actor or controller change.
- Checkpoint fresh-physics resume: saved Replay at step320, resumed at step324; the first new Replay Episode ID was at least one greater than the old next Episode ID in every environment. Trust was invalidated and subsequently re-evaluated. Strict v2 inference loading with a different 16-environment evaluation process and evaluation seed passed without modifying Actor or normalizer state.

### Bounded Strict-Pure near-goal pilot

Each seed used 64 Newton environments, 80 iterations, 4 physics policy steps per iteration, 160 WM updates, 70 Actor updates, 560 Critic gradient steps, 20,480 real transitions and no teacher samples. All three had zero nonfinite updates. The fixed independent evaluation set was 16 Episodes, 200 steps each, seed9501, with exploration disabled. Initial output-zero Actor is identical across seeds; seed53 was also independently evaluated before training.

| Evaluation | Mean reward/step | TCP position L/R mean, mm | TCP rotation L/R mean, deg | Strict success | Longest hold |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial random Actor | 0.04533 | 18.75 / 19.47 | 3.82 / 4.03 | 0/16 | 0 |
| Seed53 iteration40 | 0.04344 | 20.33 / 21.37 | 3.92 / 4.67 | 0/16 | 0 |
| Seed51 iteration80 | 0.04013 | 19.80 / 29.90 | 4.59 / 5.03 | 0/16 | 0 |
| Seed52 iteration80 | 0.03851 | 17.92 / 41.56 | 3.82 / 4.99 | 0/16 | 0 |
| Seed53 iteration80 | 0.04756 | 15.29 / 15.04 | 3.62 / 3.64 | 0/16 | 0 |

Seed53 showed partial mean 6D error improvement; the other two deteriorated. None met 5mm, 2 degrees, or 25-step hold. The best *partial* policy is `/home/ubuntu/dexgym_data/marvin_supermimic_m5_se3_reach_hold/m5_pilot_53/model.pt`; it is **not** a successful Reach & Hold policy.

The existing normalized trust gate required errors below 0.9/1.0/1.1 at horizons 1/8/32 and improvement over a zero-delta baseline. At iteration80 seed53 had normalized errors **0.721/1.705/4.601** (seed51: **0.690/1.670/4.269**), so only 1-step imagination was permitted. The 32-step Core BPTT path passed a task-specific gradient test but was never trusted for pilot policy updates. Held-out, action-conditioned seed53 World Model TCP position errors (mean L/R) were **1.78/1.67mm at 1 step**, **10.48/11.46mm at 8 steps**, and **36.30/41.54mm at 32 steps**. SO(3) errors were 0.0055/0.0051, 0.0358/0.0349 and 0.1136/0.1276rad. These are open-loop model errors on exploratory Actor data, not closed-loop policy errors.

Exploratory Replay used nontrivial actions: mean `abs(tanh(action))≈0.47`, 95th percentile≈0.91; mean executed command change≈0.0047rad, 95th percentile≈0.0091rad. Near target coverage was therefore not limited to tiny zero-action jitter. Seed51/52/53 had 42/40/51 contact-triggered terminations during 20,480 exploratory transitions each; zero joint-limit, speed or nonfinite failures. The independent no-noise policy evaluations had zero contacts. Contact risk under exploration and long-horizon model error remain concrete blockers. We stopped before Medium/Wider policy training and did not relax the trust thresholds.

For seed51, collection took 34.91s (587 real transitions/s); the full alternating loop took 39.47s (519 real transitions/s), averaging 7.86ms per WM update and 46.13ms per Actor/Critic cycle. Peak PyTorch allocation was 144MB, excluding Newton/Isaac device allocations. These are bounded pilot measurements, not a capacity benchmark.

### Held-out distribution correction and stopped learning pilot

The first pilot labelled the noisy Actor transitions as ordinary Actor data. Independent deterministic Actor execution had mean `abs(tanh(raw_action))≈0.048`, versus ≈0.47 in exploratory collection. Its own Newton trace exposed a different model error profile: at 1/8/32 steps, mean TCP position prediction error was L/R **2.6/1.6**, **15.0/13.4**, and **60.2/65.7mm**. The original exploratory held-out numbers therefore did not establish trust on the deployed Actor distribution.

The collector now records noisy Actor transitions as source 3 and deterministic Actor transitions as source 0, in both training and held-out environment lanes. A task-specific trust evaluator requires the noisy action-response model to beat a zero-delta baseline and separately bounds the deterministic Actor's TCP prediction error. The latter does not require beating a nearly perfect zero-delta baseline when the random initialized Actor stays still. This changes trust evaluation and data labels; it does not change WM8, BPTT, the task reward, action map, or the physical controller. The collection/trust protocol is versioned in the checkpoint contract. Old M5 checkpoints remain valid for explicit inference, but their Replay cannot be silently resumed as if collected under the new protocol.

Training resume compares the full collection protocol, including environment lane assignments and trust thresholds. Inference excludes that training-only field while continuing to compare the actuator, action, network and task contracts. The independent evaluator therefore does not recreate training exploration lanes; its report explicitly records `collection_contract: null` as an evaluation override. A failing-first checkpoint test verifies that the same change is accepted for inference but rejected for training resume.

Seed53 was rerun for 80 iterations, 64 Newton environments and 20,480 real transitions under each corrected protocol. In the first split pilot, deterministic model prediction improved but the zero-delta trust rule admitted only 10 one-step Actor updates. Independent Newton evaluation remained **0/16 success**, zero hold, and mean TCP position L/R **21.67/18.78mm**. An additional **500 offline WM8-only** updates lowered training loss but did not consistently beat the stationary zero-delta baseline; no Actor update or new physics transition was made during that diagnostic.

In the final dual-distribution pilot, 24 train environments explored, 24 train environments used deterministic Actor actions, and the 16 held-out environments split 8/8. It ran 160 WM8 updates with no nonfinite failures, but **0 Actor and 0 Critic updates**: the noisy held-out normalized delta RMSE was **0.920/2.291/5.684** at 1/8/32 steps against the unchanged **0.9/1.0/1.1** limits. Deterministic TCP position P95 was **0.54/8.03/33.85mm** against the 5mm model-geometry limit. Independent 16-Episode Newton evaluation reproduced the initial stationary Actor: **0/16 success**, zero hold, mean TCP position L/R **18.75/19.47mm**, mean orientation **3.82/4.03 degrees**, with zero contacts or illegal joint states. The evaluator did not mistake a stationary policy for learning progress.

The final checkpoint was also evaluated on two distinct real-data sources. Its action-conditioned open-loop mean TCP position error L/R at 1/8/32 steps was **2.11/1.86**, **17.35/15.04**, **56.38/46.20mm** on held-out noisy Actor Replay; on an independent deterministic Actor-in-Newton trace it was **0.32/0.41**, **2.49/3.28**, **13.62/15.76mm**. These are model errors under recorded action sequences, not Actor-in-model closed-loop tracking scores. A further bounded 500-update WM-only diagnostic lowered the noisy normalized error to **0.763/1.454/3.332**, while deterministic 32-step TCP position P95 remained **12.07mm**. The 8/32-step trust gates remained closed. This supports stopping before Medium/Wider training; it does not support relaxing the gate to make the pilot appear successful.

All post-pilot reports, checkpoint copies and real Newton traces are preserved in the newly named `m5_v2_*` and `m5_v3_*` subdirectories of the external artifact root. `checkpoint_hashes.json` verifies the new copies and every earlier checkpoint entry. The final v3 pilot checkpoint is `m5_v3_pilot_53/model.pt` (SHA256 `fdbf3eb3cd1213b480cf9325fe18623e8bc406f08c2d743626365f5f24d47c0f`). The archived `m5_v3_eval_53_clean_contract/physics_trace.pt` contains the independent 200-step, 16-environment real transition sequence after the inference contract fix. This rerun again found 0/16 successful episodes and no hold. The v3 checkpoint was created before committing this correction, so its embedded `git_commit=5978fd37e49d7225aaef79e122e6ab78275f2628` identifies its parent revision. The final source revision and configuration SHA are recorded separately in the validation manifest; the embedded commit alone is insufficient provenance for this checkpoint.

The corrected evaluator also loaded an earlier M5 checkpoint into a new single-environment Newton process and completed its 200-step episode. This confirms backward inference compatibility only; it does not make its old Replay valid under the new collection protocol.

## Reproduction

Run from the M5 worktree with the pinned Isaac Sim / Newton overlay:

```bash
export PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source
export ISAAC_PYTHON=/home/ubuntu/isaacsim-6.0.1/bin/python
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py smoke_se3_m5 --num-envs 16 --steps 4 --level near --output /tmp/m5_preflight.json
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py train_se3_m5 --config configs/supermimic/m5_se3_near_pilot.json --num-envs 64 --iterations 80 --seed 53 --output /tmp/m5_pilot_53
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py evaluate_se3_m5 --checkpoint /tmp/m5_pilot_53/model.pt --num-envs 16 --seed 9501 --output /tmp/m5_eval_pilot_53
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py validate_se3_model_m5 --checkpoint /tmp/m5_pilot_53/model.pt --output /tmp/m5_pilot_53/model_validation.json
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py validate_se3_model_m5 --checkpoint /tmp/m5_pilot_53/model.pt --physics-trace /tmp/m5_eval_pilot_53/physics_trace.pt --output /tmp/m5_eval_pilot_53/independent_model_validation.json
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py diagnose_se3_world_model_m5 --checkpoint /tmp/m5_pilot_53/model.pt --updates 500 --output /tmp/m5_wm_diagnostic
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py evaluate --checkpoint /home/ubuntu/dexgym_data/marvin_supermimic_wave_m41/selected_best.pt --reference /home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/cmu_141_16_wave_bimanual.npz --output /tmp/m5_wave_legacy
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source $ISAAC_PYTHON -c "import sys,pytest; sys.path.append('/home/ubuntu/dexgym/.venv/lib/python3.12/site-packages'); raise SystemExit(pytest.main(['-q','tests/supermimic']))"
```

The `train_se3_m5` config fixes WM8, BPTT horizon cap32, TD(lambda), model trust thresholds and strict-pure mode. CLI overrides are explicit. Evaluation uses a separate Newton process and fixed seeded goal set; it records overrides in the report.
The pilot launcher rejects any resume that would exceed 200 total iterations; a 121-iteration request from an 80-iteration checkpoint was tested and failed before new training.

## Remaining gates

- **Passed:** schema/action/FK/reward math tests, 32-step differentiable path unit test, Newton command/Reward/Termination parity, fresh physics resume, independent evaluation protocol, full M4.1 regression.
- **Not passed:** 8/32-step Model Trust, sustained 32-step pilot BPTT, consistent physical improvement across seeds, complete SE(3) arrival, 25-step hold, Medium/Wider trained-policy performance, collision-free exploratory training, dynamic feasibility proof for each sampled target. The final dual-distribution pilot also failed 1-step trust and therefore made no Actor/Critic updates.

The task is a usable, versioned Core integration and a documented failed strict-pure learning pilot. No convergence or generalization claim follows from these results.
