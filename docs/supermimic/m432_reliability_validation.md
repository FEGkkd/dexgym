# M4.3.2 SuperMimic Core reliability validation

Base: `feat/supermimic-core-m431-hardening` at `01326844d6fdfe436354e91bfdd62737d28c4a15`. Development branch: `feat/supermimic-core-m432-reliability`. This patch keeps WM8 masked autoregression, BPTT32, survival return, TD(λ), Task MDP and Robot Adapter math unchanged. No long training or Policy performance tuning was run.

## Reference audit

The SuperMimic [paper](https://arxiv.org/abs/2610.28480) specifies learned differentiable dynamics, short-horizon BPTT, soft survival and TD(λ). The local [TrackerLab source](https://github.com/Renforce-Dynamics/trackerlab-supermimic) was read at `e2db2dea0e81c91ebca815c895b60ee9aa58237a` (MIT): its runner injects observation noise after constructing imagined physical-state observations, and keeps reward/soft termination on physical states; resume checks task, observation and reward contracts. The local [mjlab source](https://github.com/YHFone/supermimic-mjlab) was read at `9aab0a588c6941bf9862a2d444b711fd73e2b832` (Apache-2.0): its World Model optimizer explicitly contains both dynamics and learned termination parameters, BCE uses simulator hard labels, and imagined proprioception is noisy. This patch uses DexGym's existing fixed-base Marvin contracts and copies no third-party source.

## Failure-first evidence and fixes

| Area | Failure observed before fix | Code and passing evidence |
| --- | --- | --- |
| Learned termination | Trainer accepted a World Model optimizer missing all head parameters. `test_m432_optimizer_ownership.py` failed `DID NOT RAISE`. | `validate_optimizer_ownership` rejects missing, duplicate and cross-owned parameters. `build_optimizers` includes the head only in learned mode. Toy positive/negative labels decrease BCE; head weights and Adam state change and resume. |
| Inference compatibility | v2 inference rejected a 16-env checkpoint with a 1-env evaluation configuration; all three new evaluation contract tests failed. | v2's saved manifest SHA is verified first. Inference compares immutable execution fields and explicitly reports evaluation overrides. Training resume retains full config equality. Old v2 Reach and Tracking checkpoints loaded in independent one-env Newton processes. |
| Actor observation noise | The generic module did not exist; the new test failed during import. | Versioned per-feature uniform/Gaussian noise with units, schema, seed and generator state. Collector Replay history is the actual Actor input; World Model labels remain physical. Imagination injects noise after state prediction and reward/termination. Disabled default preserves old output. |
| Shared World Model | Factory accepted matching dimensions without knowing the controller or action scale; six new tests failed. | Factory now requires source and target Robot Adapter signatures to share weights. Signature contains RobotSpec, adapter's executable controller/integration contract, context schema/model range and World Model network/normalizer semantics. Both tasks use the same model and one optimizer; mismatches reject. |

## Automated and physical results

The final local suite: **165 passed** (`tests/supermimic`, Isaac Sim 6 Python, 18.99 s). It includes M4.1 Actor numerical fixture, WM8 masks, BPTT/TD(λ), checkpoint/full and fresh resume, two Goal Schemas, real Marvin adapter signature, noise/normalizer count, and new optimizer tests. An intermediate run found a test-double compatibility bug (`native.cfg` absent); it was fixed and the whole suite rerun green.

Evidence files are in `/home/ubuntu/dexgym_data/marvin_supermimic_m432/`:

| Run | Bounded result | Caveat |
| --- | --- | --- |
| Reach smoke, 16 env × 12 iterations | 192 real transitions, WM 24, Actor 8, Critic 64; 0 nonfinite updates; 125 transitions/s collection; peak Torch allocation 23.7 MB | In-process 12-step evaluation had no completed Episodes and 0 success. |
| Tracking smoke, 16 env × 32 iterations | 512 real transitions, WM 64, 0 nonfinite updates; 145 transitions/s collection; peak Torch allocation 29.8 MB | Trust Gate did not pass (`normalized_delta_rmse` one-step 0.903 vs configured 0.9), so Actor/Critic were correctly withheld. |
| Noisy Reach smoke, 16 env × 12 iterations | 192 real transitions, WM 24, Actor 8, Critic 64; 0 nonfinite; 131 transitions/s collection | Validates real Newton noise, model targets, BPTT and checkpoint path. No Reach convergence claim. |
| Old v2 Reach independent evaluation | 1 env, new seed 987, 100 physical steps, 1 completed truncation, 0 hard failures, 0 success | 16-env training checkpoint loaded with different evaluation protocol. |
| Old v2 Reach repeated evaluation | 1 env, seed 991, 2 separately completed 100-step Episodes, 2 truncations, 0 success | `--episode-count 2` records separate Episodes after physical Reset. |
| New noisy Reach independent evaluation | 1 env, new seed 988, noisy Actor sensing, 100 steps, 1 completed truncation, 0 hard failures, 0 success | Saved model and normalizers frozen. |
| Old v2 Tracking independent evaluation | 1 env, new seed 989, Wave frame 0, hard failure at step 17 | Shows the protocol loads and records a real failure; the short-trained policy did not complete Wave. |
| Old v2 Tracking with perturbed reset | 1 env, seed 990, Wave frame 0, 0.01 rad joint reset perturbation; hard failure at step 17 | Confirms a changed evaluation-only reset parameter is accepted and recorded. |
| M4.1 Legacy Wave | 1356/1356 control steps; 0 contact control/physics steps; 0 illegal joint-state frames; left/right TCP position mean 1.921/4.509 mm; orientation mean 0.273/1.031° | Untouched `selected_best.pt`; original Actor fixture parity is in the test suite. |

The M4.1 checkpoint and Wave reference hashes remained `fd0054a7ef707d46a30b0235713c9e00cc4d7e69d6687fc3015e52aac2cb773f` and `b7af95b289e8aff898df7f5dd06b169f74dba2a506aa8fc5bf589ee3fee8106f`. The M4.3.2 experiments were bounded and do not establish cross-task convergence, learned termination calibration over all Newton failure modes, contact-rich shared dynamics, or real-robot readiness.

## Acceptance gates

| Gate | Status | Evidence and limit |
| --- | --- | --- |
| A Learned termination | Pass for controlled Toy data | Head parameters and Adam state change, BCE falls, checkpoint resume continues. No Newton positive-class calibration claim. |
| B Evaluation compatibility | Pass | v2 16-to-1-env loading, new seeds/perturbation, two independent Episodes, immutable contract rejection, real Newton traces. |
| C Observation noise | Pass for the declared distributions | Disabled parity, seeded uniform/Gaussian, real Replay/clean World Model targets, BPTT gradient, noisy Newton Reach and independent evaluation. |
| D World Model sharing | Pass for matching declared dynamics | Two Toy Goal Schemas and Marvin Tracking/Reach share one model; action/controller/dt/asset/context/normalizer mismatches reject; one optimizer owns shared weights. Actual cross-task long training was not run. |
| E Regression | Pass for software and physical integration | 165 tests, both 16-env smokes, 1356-step M4.1 Wave. Tracking short-run trust still withholds Actor updates, so this gate is not a Tracking convergence claim. |

## Reproduction

Run from the M4.3.2 worktree with the pinned Newton runtime overlay. The commands below were executed; output directories may be changed to new paths.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source /home/ubuntu/isaacsim-6.0.1/bin/python -c "import sys,pytest; sys.path.append('/home/ubuntu/dexgym/.venv/lib/python3.12/site-packages'); raise SystemExit(pytest.main(['-q','tests/supermimic']))"
PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py smoke_core_m43 --config configs/supermimic/m43_reach_smoke.json --num-envs 16 --iterations 12 --eval-steps 12 --output /tmp/m432_reach_smoke
PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py smoke_core_m43 --config configs/supermimic/m43_tracking_smoke.json --num-envs 16 --iterations 32 --eval-steps 12 --output /tmp/m432_tracking_smoke32
PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py evaluate_core_m431 --checkpoint /home/ubuntu/dexgym_data/marvin_supermimic_m431/reach_training/reach_m43_smoke.pt --reference /home/ubuntu/dexgym_data/marvin_supermimic_m42/second_task/procedural_bimanual_reach_v1.npz --num-envs 1 --seed 987 --output /tmp/m432_reach_independent
PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py evaluate --checkpoint /home/ubuntu/dexgym_data/marvin_supermimic_wave_m41/selected_best.pt --reference /home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/cmu_141_16_wave_bimanual.npz --output /tmp/m432_wave_legacy
```

For the noisy Reach run, use the archived `reach_noise_config.json` in the M4.3.2 data directory with the same `smoke_core_m43` command. The Noise Contract and independent evaluation flags are documented in adjacent files.
