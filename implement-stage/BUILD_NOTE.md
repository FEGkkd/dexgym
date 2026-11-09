# Hand S0 build and experiment record

| Rung | Scope | Acceptance | Status |
|---|---|---|---|
| F0 | Real manifest → documented config/data audit artifact | `python scripts/supermimic/hand_s0_preflight.py --manifest /home/ubuntu/dexgym/data/processed/hand21_right_geometry_feasible/manifest.json --output /home/ubuntu/dexgym/artifacts/hand_s0/preflight.json` | passed: real manifest audit |
| F1 | Existing Core trust fixes and state-dependent task transition contract | targeted CPU tests plus prior Core suite | passed: 230 tests |
| F2 | Standalone Hand adapter/FK and pre-reset physical collection, exact task parity | FK comparison, reward/event/reset parity in Newton | passed |
| F3 | WM8/BPTT32 training and independent inference/evaluation spine | CPU gradient checks + bounded Newton real training | passed |
| F4 | Current scope: architecture/task/algorithm readiness, source review, report and push | bounded training, held-out evaluation spine and scale validation | passed: bounded validation, review complete; publishing this commit |

## Run record
- 2026-11-09: inspected current remote state; created supermimic worktree at base cf2cdde.
- Original ~/dexgym worktree has uncommitted legacy work; preserved unchanged.
- Existing PPO PID1519240 verified live on RTX5090, using about15GB VRAM.
- Repeated minimal imports of torch2.11.0+cu130 print successfully but exit with
  SIGBUS135 in existing venv, including with libstdc++ preload. Runtime is not
  declared validated; diagnose before trusting tests. Original PPO remains live.

## Deferred
Per explicit user scope update, converged-policy/high-accuracy training is deferred to a later phase. Report/code publication remains required.

## Blockers
Shared venv SIGBUS remains isolated; working persistent Isaac runtime validated by executor and fresh independent doc checks. No active runtime blocker.

- F0: preflight exited successfully; 942 train /446 validation clips, 281793 /122482 frames, zero missing files. Artifact /home/ubuntu/dexgym/artifacts/hand_s0/preflight.json.

- F1: fixed nonfinite trust admission and normalize=False trust; added physical-state-dependent transition hook with score-before-switch test. Targeted25 /full230 tests passed. CUDA seeded witness exit0 using Isaac Python.

## F2 physical task bridge — passed
239 tests passed in 19.92 s. Newton smoke: 16x64=1024 transitions, 30 terminal, 21 goals, FK max6.055631160961639e-7 m; pre-reset capture, exact rewards/commands/termination and finite WM8 update passed. Raw artifacts /home/ubuntu/dexgym/artifacts/hand_s0/newton_smoke.json and f2_suite.log. At that historical checkpoint F3/F4 were pending; current scope/status is below.

## F3 initial actual training — historical record, superseded below
Core suite236 passed in19.87s. Pilot v1 completed150 iterations /38400 real transitions /300 WM updates /50 Actor BPTT32 updates /1000 Critic updates, exit0, zero nonfinite updates. 32-step held-out-lane prediction mean fingertip error1.867 mm; this is model prediction error, not target tracking error. Saved artifacts/hand_s0/pilot_v1/model_150.pt. Pilot v2 continues1000 iterations with fresh physics/replay and preserved curriculum. Final policy accuracy still unassessed.

## Current scope and final validation
User narrowed acceptance to architecture/task/model-based training readiness; stopped our pilot at last logged iteration994 (254464 real transitions,1988 WM,494 Actor,9880 Critic updates; zero nonfinite updates). Original PPO untouched. Saved checkpoint model_500.pt, not994. All446 held-out fixed-clock clips /122036 intended frames evaluated; no claim of policy quality. 1024x16 physical smoke with1000-step replay passed: replay6017040192 bytes, Torch peak6098866688 bytes. Final regression log: /home/ubuntu/dexgym/artifacts/hand_s0/final_regression.log.

Final broad regression first exposed a pre-existing stale expectation: base cf2cdde already sets15mm, while test_right_hand_tracking_config still expected20mm. Corrected test to user15mm contract; production tolerance unchanged. Initial output preserved in final_regression_before_stale_test_fix.log.

Final regression:262 passed,35 torch.jit deprecation warnings,20.50s,exit0. Fresh assumption recheck:clean,semantic_undeclared0, no stale rows; identity unknown, independent acceptance not claimed. A post-review launcher fix resamples fresh goals after restoring curriculum progress; executor-only bounded resume check recorded separately. No live stubs. Publication pending.

Resume order check: model_500 ->502 in2 bounded iterations; restored_curriculum_steps500, new replay/trust validation enforced, finite world updates. Scope/data/test summaries committed in docs/experiments/hand_s0_validation_20261109.json.
