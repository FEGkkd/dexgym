# M5 IK-free bimanual SE(3) reach and hold

## Scope and status

This is a new `marvin_bimanual_se3_reach_hold` task on the M4.3.2 SuperMimic Core. It does not use IK, Jacobian inverse, a joint target label, a motion reference, a teacher policy, or PPO. FK is used to create reachable TCP pose targets and to calculate observations and differentiable reward. The generated witness joint configuration is discarded inside `SE3GoalSampler._draw`; the returned sample and Replay contain only the 18D dual TCP pose and difficulty statistics.

**Engineering integration passed. Strict dual-TCP 6D Reach & Hold learning did not pass.** Three bounded near-goal pilots all updated the model, Actor and Critic, but only 1-step imagination passed the existing model trust gate. All independently evaluated policies had 0/16 successful full Episodes and zero valid hold steps. Medium and wider levels were physically preflighted but not trained.

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

- Full SuperMimic Torch suite: **183 passed**. New tests cover the relative action and gradient, 10 substep interpolation, FK goal schema/sampling, separate curriculum streams, no `q_goal` in Task state, bilateral reward symmetry, real Marvin SO(3) finite-difference gradient, strict 200-step success, action compatibility signature, task-specific 32-step BPTT with frozen dynamics parameters, and temporal exploration reproducibility.
- Newton preflight: near, medium, wider each ran 16 environments × 4 policy steps with zero hard terminations, zero command parity error and valid 18D goals. The near start had nonzero per-joint standard deviation across environments. The independent near evaluation ran 16 complete 200-step Episodes with zero contacts or illegal joint states.
- 16-environment Strict-Pure integration smoke: 768 real transitions, 24 WM updates, 2 Actor updates, 16 Critic updates and no nonfinite updates. It established pipeline execution only.
- M4.1 full Wave regression: **1356/1356** Newton control steps, zero physical contacts, zero illegal joint states. Left/right TCP mean position errors were **2.741/5.075mm**, mean orientation errors **0.371/1.074 degrees**. The original checkpoint and reference hashes were unchanged.
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

## Reproduction

Run from the M5 worktree with the pinned Isaac Sim / Newton overlay:

```bash
export PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source
export ISAAC_PYTHON=/home/ubuntu/isaacsim-6.0.1/bin/python
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py smoke_se3_m5 --num-envs 16 --steps 4 --level near --output /tmp/m5_preflight.json
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py train_se3_m5 --config configs/supermimic/m5_se3_near_pilot.json --num-envs 64 --iterations 80 --seed 53 --output /tmp/m5_pilot_53
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py evaluate_se3_m5 --checkpoint /tmp/m5_pilot_53/model.pt --num-envs 16 --seed 9501 --output /tmp/m5_eval_pilot_53
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py validate_se3_model_m5 --checkpoint /tmp/m5_pilot_53/model.pt --output /tmp/m5_pilot_53/model_validation.json
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py evaluate --checkpoint /home/ubuntu/dexgym_data/marvin_supermimic_wave_m41/selected_best.pt --reference /home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/cmu_141_16_wave_bimanual.npz --output /tmp/m5_wave_legacy
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source $ISAAC_PYTHON -c "import sys,pytest; sys.path.append('/home/ubuntu/dexgym/.venv/lib/python3.12/site-packages'); raise SystemExit(pytest.main(['-q','tests/supermimic']))"
```

The `train_se3_m5` config fixes WM8, BPTT horizon cap32, TD(lambda), model trust thresholds and strict-pure mode. CLI overrides are explicit. Evaluation uses a separate Newton process and fixed seeded goal set; it records overrides in the report.
The pilot launcher rejects any resume that would exceed 200 total iterations; a 121-iteration request from an 80-iteration checkpoint was tested and failed before new training.

## Remaining gates

- **Passed:** schema/action/FK/reward math tests, 32-step differentiable path unit test, Newton command/Reward/Termination parity, fresh physics resume, independent evaluation protocol, full M4.1 regression.
- **Not passed:** 8/32-step Model Trust, sustained 32-step pilot BPTT, consistent physical improvement across seeds, complete SE(3) arrival, 25-step hold, Medium/Wider trained-policy performance, collision-free exploratory training, dynamic feasibility proof for each sampled target.

The task is a usable, versioned Core integration and a documented failed strict-pure learning pilot. No convergence or generalization claim follows from these results.
