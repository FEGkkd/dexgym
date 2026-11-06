# M4.3.1 SuperMimic Core hardening validation

## Provenance and scope

Base: DexGym feat/supermimic-core-m43 at 0ff9d9e6975cde91e958582c4409f9af253f8c1d. Development branch: feat/supermimic-core-m431-hardening, independent worktree /tmp/dexgym-m431. The unrelated untracked Marvin IK files in /home/ubuntu/dexgym were not touched. The protected M4.1/M4.2/M4.3 checkpoint and reference directories were read only. No original MJCF/USD, joint axes, TCP frames, position servo gains, effort limits or interpolation parameters were modified. No long policy training was run.

The [SuperMimic paper](https://arxiv.org/abs/2610.28480) was checked against its differentiable model, short imagination, survival return, TD(lambda), and Algorithm A.1. Actual TrackerLab source was read at e2db2dea0e81c91ebca815c895b60ee9aa58237a in runner.py, networks.py and buffer.py (MIT). Actual [supermimic-mjlab](https://github.com/YHFone/supermimic-mjlab) source was read at 9aab0a588c6941bf9862a2d444b711fd73e2b832 in trainer.py, buffer.py, state.py and reward.py (Apache-2.0). No third-party code was copied. TrackerLab and mjlab are independent engineering references, not the official paper implementation. Marvin remains a fixed-base 42D state, 14D arm action, 28D learned physical delta adaptation.

## Correctness changes

- Core Checkpoint v2 includes a readable canonical manifest and SHA256 of NetworkSpec, actual Goal Tanh switch, normalizer modes/floor/count, World Model integration/output schema, Robot/Action/Controller/effort contract, Task Reward and termination parameters, Episode/success definitions, physics and reset asset. Same-shaped semantic conflicts fail with the conflicting field name. Strict inference-only loading is available for v2. M4.3 v1 has an explicit evaluation-only loader requiring source SHA and acknowledged unrecorded semantics; it cannot silently resume as v2.
- Resume modes are fresh_physics (default), replay, and full. Save rejects full mode before writing if collector state export/import is missing. Replay mode increments Episode IDs on new physics; Full Toy resume retains the existing Episode and reproduced the next update exactly. Model trust is revalidated after every load. Newton uses fresh physics because its simulator state is not bitwise restorable.
- The Newton collector reads real per-environment context before action execution and checks finite shape/dtype/device. Time-varying context requires a declared adapter transition in imagination. A same-state/action, different-context Toy test verifies Replay storage, Model loss, gradients and context advancement.
- The evaluator closes each physical Episode separately across automatic Reset. It records ID/start/length/return, first success/failure, terminal/truncation reason, hold duration and task metrics. Episode success rate is separate from step success fraction. Independent evaluation starts a fresh Newton process, loads a strict Actor, freezes normalizers, disables exploration and saves physical transitions.
- MarvinWaveDirectEnv and MarvinReachDirectEnv now inherit MarvinArmNewtonBaseEnv directly. The base owns the validated 54DoF Newton physics, 14D servo, hand Home commands, contact audit and safe reset writer; it imports no Wave reference or MotionManager. Both environments passed first and partial Reset checks at 16 environments.
- Cross-layer checks compare every collected real transition's observation history, analytic executed command, arrival Reward, hard termination and live reference frame. They uncovered a pre-existing alias: a terminal Reset cleared the same Tensor held by the Collector as the executed action. The base now clones actions before storing last_action. The fixed-seed Tracking sequence that failed at iteration 11 passed at 20 and 60 iterations after this one-line fix.
- The single normalize switch now affects World Model input/delta, Actor observation/goal, and Critic input. Normalizer floor, count cap and version are explicit NetworkSpec fields and checkpoint semantics. Legacy identity normalization remains available; no normalization statistics update in imagination or evaluation.

The eight-step masked autoregression, 32-step BPTT, survival-weighted Actor return, TD(lambda), model parameter freeze and alternating loop were not rewritten.

## Bounded evidence

| Run | Real transitions | WM / Actor / Critic updates | Outcome |
| --- | ---: | ---: | --- |
| Reach Newton, 16 env × 40 cycles | 640 | 80 / 28 / 224 | Zero nonfinite updates; final trusted horizon 32; Trust failures 2 |
| Tracking Newton, 16 env × 60 cycles | 960 | 120 / 20 / 160 | Zero nonfinite updates; final trusted horizon 8; Trust failures 9 |
| Reach fresh-physics resume, 16 env × 4 cycles | 64 new | 8 / 4 / 32 new | Replay absent as declared; fresh transitions collected before trust/BPTT |
| Toy position, 16 env × 30 cycles | 480 | 120 / 29 / 232 | Mean reward 0.278 → 0.440; final position error 0.800 → 2.396, so no convergence claim |
| Toy position+velocity, 16 env × 30 cycles | 480 | 120 / 29 / 232 | Mean reward 0.275 → 0.669; final position error 0.800 → 0.752 |

The separate Reach Newton process loaded a v2 checkpoint and ran 16 complete 100-step Episodes (1,600 real transitions). It recorded zero hard terminations, 16 time-limit truncations, zero Episode successes, and 27.99 mm mean dual-TCP distance. The in-process 12-step Reach smoke had 27.95 mm mean distance and zero completed Episodes; these metrics are deliberately separate. The Tracking 12-step in-process evaluation had mean left/right TCP position error 26.78/12.23 mm; no Episode completed in that short window.

The protected M4.1 selected_best.pt completed all 1,356 Newton control steps with zero hard terminations, zero Newton contact physics steps and zero illegal joint states. Left/right TCP mean position errors were 2.066/4.470 mm; orientation means were 0.291°/1.035°. Maximum command jump was 0.040 rad. The old Actor numerical fixture still passes, and the explicit M4.3 v1 Reach Actor loader matched the original M4.3 source process within 1e-7.

At 16 environments, final Reach/Tracking physical collection throughput was approximately 185/157 transitions per second. Mean World Model update time was about 11.5/11.0 ms; Actor BPTT about 0.276/0.301 s when admitted; Critic gradient step about 2.3/2.4 ms. Peak PyTorch allocated VRAM was about 29/32 MB; Newton/Isaac allocations make this an incomplete device-capacity measure.

## Acceptance gates

| Gate | Status | Evidence and limit |
| --- | --- | --- |
| A Checkpoint semantics | Pass for v2 and explicit legacy evaluation | Same-shape Tanh/Reward/termination/controller/normalizer mismatch tests, strict load and old-source Actor parity |
| B Dynamics context | Pass for zero-dimensional Marvin and nonzero Toy | Real collector validation, context-dependent Model/Replay/Imagination tests; time-varying context was tested only in Toy |
| C Episode evaluation | Pass for mock and Reach Newton | Four-environment fail/success/truncate/reset test and separate-process 16 full Episodes |
| D Physics architecture | Pass for tested Marvin controls | Tracking and Reach are sibling classes; 16-environment partial Reset, hand Home, 500/50 Hz, command/Reward/termination/time parity |
| E Resume | Pass for Toy Full and Newton fresh physics | Exact next Toy update, Replay new Episode ID, missing collector save-time rejection, Newton fresh collection |
| F Regression | Pass for bounded scope | Full CPU suite, two Newton tasks, M4.1 1,356-step playback, original M4.3 Actor parity |

These gates mean framework hardening in the stated scope. They do not establish Reach convergence, stable tracking trust, mixed-motion Newton training, contact-rich shared dynamics, real time-varying Newton context, bitwise Newton Full Resume, or real-robot deployment.

## Reproduction commands

Run from /tmp/dexgym-m431. The pinned Isaac Sim 6 Python and M4.2 Newton runtime overlay are required.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source /home/ubuntu/isaacsim-6.0.1/bin/python -c "import sys,pytest,torch,mujoco,warp; sys.path.append('/home/ubuntu/dexgym/.venv/lib/python3.12/site-packages'); raise SystemExit(pytest.main(['-q','tests/supermimic']))"
    PYTHONPATH=source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/validate_toy_m43.py --output /tmp/m431_toy_30 --iterations 30 --seed 13
    PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py smoke_core_m43 --config configs/supermimic/m43_reach_smoke.json --num-envs 16 --iterations 40 --eval-steps 12 --output /tmp/m431_reach40_final
    PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py smoke_core_m43 --config configs/supermimic/m43_tracking_smoke.json --num-envs 16 --iterations 60 --eval-steps 12 --output /tmp/m431_tracking60_fixed
    PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py evaluate_core_m431 --checkpoint /tmp/m431_reach40_final/reach_m43_smoke.pt --reference /home/ubuntu/dexgym_data/marvin_supermimic_m42/second_task/procedural_bimanual_reach_v1.npz --output /tmp/m431_reach_independent_final
    PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py validate_m431_newton_reset --task reach --output /tmp/m431_reach_partial_reset.json
    PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py validate_m431_newton_reset --task tracking --output /tmp/m431_tracking_partial_reset.json
    PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py evaluate --checkpoint /home/ubuntu/dexgym_data/marvin_supermimic_wave_m41/selected_best.pt --reference /home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/cmu_141_16_wave_bimanual.npz --output /tmp/m431_wave_legacy
