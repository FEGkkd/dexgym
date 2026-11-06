# M4.3 validation and acceptance report

## Scope and provenance

Base: `feat/marvin-supermimic-m42-framework` at `55294a5da4de034fed08018956e7a03a15294229`. Development worktree: `/home/ubuntu/.codex/worktrees/3d7b/dexgym`, branch `feat/supermimic-core-m43`. Final physical artifacts below were generated from tested code at `557ea9f05be40027f9974b1ac5085ad64fe4187f`; Toy artifacts were generated at `ce40d256a806aa106b972a32e2b8fe7609cb6990` (only the Newton memory reporting script changed between these commits). This stage used no long training or hyperparameter search. All new artifacts are under `/home/ubuntu/dexgym_data/marvin_supermimic_m43/`; the M4.1/M4.2 data and assets were not changed.

The 81 M4.2 tests passed before editing. The final M4.3 suite passed **118 tests** before the final documentation commit, including masked short Replay, timing, returns, TD(λ), SO(3), gradients, model freezing, normalization, checkpoint, strict M4.1/M4.2 loading, two network schemas, plugin boundaries and mixed Motion IDs. The static module gate finds no concrete task/motion imports or 14/42/90/146 network constants in `algorithms/supermimic/core`. The exact Python AST of the historical `evaluate_full` function and the task-owned legacy evaluator matches.

## Controlled Toy RL

`scripts/supermimic/validate_toy_m43.py` ran 100 iterations per schema at seed 13, 16 Torch-physics environments, no teacher samples. The same Core Trainer used a 2D state/1D action robot for both a 1D position goal and a 2D position-plus-velocity goal. Each made 400 World Model, 99 Actor and 792 Critic gradient updates, reached a 32-step trusted imagination horizon and had zero nonfinite updates.

| Goal schema | Real reward before → after | Final position error before → after | Success after |
| --- | ---: | ---: | ---: |
| Position 1D | 0.2780 → 0.9209 | 0.8000 → 0.0173 | 100% |
| Position + velocity 2D | 0.2749 → 0.9017 | 0.8000 → 0.0113 | 100% |

Both Toy checkpoints saved and restored network weights, optimizers, normalizers, RNG, Replay and collector state. Reports and exported PyTorch policies: `/home/ubuntu/dexgym_data/marvin_supermimic_m43/toy_rl_ce40d25/`.

## Real Newton integration

Both task configurations use the same `SuperMimicTrainer`, `NetworkFactory`, `SchemaReplay`, `ImaginationEngine`, Critic optimizer, trust evaluator and physical evaluator. Marvin remains a full 54DoF Newton MJWarp articulation at 500 Hz with a 14D arm-only 50 Hz servo contract. The Reach target is a static 1×6 dual-TCP position goal, with a separate 48D observation, Reward, termination and evaluator. Its procedural NPZ supplies only a safe reset pose; it is not a Reach target trajectory.

| Bounded smoke | Real transitions | WM / Actor / Critic updates | Last trusted horizon | Independent Newton metric |
| --- | ---: | ---: | ---: | --- |
| Reach, 16 env × 40 cycles | 640 | 80 / 4 / 32 | 32 | 12-step dual-TCP mean distance 29.72 mm; success 0%; zero hard terminations |
| Tracking, 16 env × 40 cycles | 640 | 80 / 0 / 0 | 0 | Model trust refused Actor; this failed run is preserved |
| Tracking, 16 env × 60 cycles | 960 | 120 / 16 / 128 | 0 at the final check | 12-step left/right TCP mean 26.95/11.12 mm; zero hard terminations |

The Tracking 60-cycle run received Actor/Critic updates during earlier trusted windows, but the final one-step normalized error again exceeded the smoke admission threshold 0.9; **model trust was not sustained**. Reach's 40-cycle run admitted a 32-step horizon with a final normalized open-loop delta RMSE below its zero-delta baseline (see the full report). This is held-out actor-source open-loop evidence, not a closed-loop model-vs-Newton claim. The Reach Policy did not reach its goal and no improvement/convergence claim is made. Both final runs had zero nonfinite World Model, Actor and Critic updates. `reach_smoke_557ea9f/`, `tracking_smoke_ce40d25/` (the failed 40-cycle run), and `tracking_smoke_60_557ea9f/` contain logs, checkpoints and reports.

At 16 environments, Reach/Tracking physical collection ran at **181/167 transitions/s**. Mean WM update time was **11.63/10.98 ms**; mean Actor BPTT update time was **691.6/301.8 ms** where admitted; mean Critic gradient-step time was **2.74/2.36 ms**. Peak PyTorch allocated memory was **28.8/32.3 MB**. Total device memory near the end of each complete loop was **2427/2453 MB** on a 32607 MB GPU (a point measurement, not a device-wide peak). The tiny PyTorch allocation alone must not be used as a capacity estimate for Newton.

The Reach checkpoint was loaded into a new 16-environment Newton process. Because the physical Replay was intentionally absent, the loader returned `requires_fresh_collection=true`, cleared stale model trust, collected 64 new real transitions, made eight new model updates, and only then admitted fresh one-step BPTT. This resume report is under `reach_resume_ce40d25/`. Simultaneous mixed Wave/Procedural **Newton** sampling is not implemented; a synthetic two-length Motion ID test verified the same Trainer's Replay and imagination boundary logic.

## Legacy regression

The protected M4.1 checkpoint SHA256 remained `fd0054a7ef707d46a30b0235713c9e00cc4d7e69d6687fc3015e52aac2cb773f`; its Wave reference remained `b7af95b289e8aff898df7f5dd06b169f74dba2a506aa8fc5bf589ee3fee8106f`. The fixture test verifies old-source Actor actions within `2e-6` absolute tolerance. The full task-owned Newton playback completed **1356/1356** policy steps with zero Newton contact physics steps, zero hard terminations and zero illegal joint states. Left/right TCP position means were **1.87/4.54 mm**, orientation means **0.270°/1.069°**. A second replay through the historical evaluator also completed, with small GPU-physics metric variation. The task-owned function body is AST-identical to the historical one. Full trace and report: `/home/ubuntu/dexgym_data/marvin_supermimic_m43/legacy_wave/`. The M4.2 Procedural Actor checkpoint `pure_v3_seed44/iteration_000200.pt` was strictly loaded; a Wave-reference mismatch was rejected.

## Acceptance gates

| Gate | Result | Evidence and limit |
| --- | --- | --- |
| A: mathematics | Pass | 118 automated tests; masked WM8, 32-step BPTT, survival, TD(λ), SO(3), gradients, value bound |
| B: modularity | Pass | No concrete task import/dimension constants in Core; Toy and two Marvin TaskSpecs use one factory/trainer |
| C: physical integration | Pass for bounded smoke | Real Newton states, action-map and reward timing parity; no contact in tested short episodes |
| D: legacy regression | Pass | Old Actor action fixture and full 1356-step Wave playback |
| E: reusability | Pass for integration; convergence unverified | Reach and Tracking use same Core, different Goal/Observation; Reach success remains 0% |
| F: reproducibility | Pass for tested paths | Configs, Git/asset/reference hashes, checkpoints, RNG, Toy full resume and Newton fresh-data resume |

**Framework Engineering Complete** applies to these two schema plugins and compatibility gates. It does **not** mean universally converged, stable across seeds, contact-rich dynamics validity, or deployment readiness. Learned termination lacks a calibrated positive-failure dataset in this M4.3 smoke. Closed-loop actor-in-model vs actor-in-Newton error, multi-motion Newton mixing, 1024–4096 environment capacity, and long-run cross-task convergence remain unverified.

## Reproduction commands

From the M4.3 worktree, use Isaac Sim 6's Python and the pinned M4.2 Newton runtime overlay:

```bash
cd /home/ubuntu/.codex/worktrees/3d7b/dexgym
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source /home/ubuntu/isaacsim-6.0.1/bin/python - <<'PY'
import torch, mujoco, warp, sys
sys.path.append('/home/ubuntu/dexgym/.venv/lib/python3.12/site-packages')
import pytest
raise SystemExit(pytest.main(['-q', 'tests/supermimic']))
PY
PYTHONPATH=source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/validate_toy_m43.py --output /tmp/m43_toy_recheck --iterations 100 --seed 13
PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py smoke_core_m43 --config configs/supermimic/m43_reach_smoke.json --output /tmp/m43_reach_recheck
PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py smoke_core_m43 --config configs/supermimic/m43_tracking_smoke.json --iterations 60 --output /tmp/m43_tracking_recheck
PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py evaluate --checkpoint /home/ubuntu/dexgym_data/marvin_supermimic_wave_m41/selected_best.pt --reference /home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/cmu_141_16_wave_bimanual.npz --output /tmp/m43_wave_recheck
```
