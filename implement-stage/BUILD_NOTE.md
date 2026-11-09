# Hand S0 build and experiment record

| Rung | Scope | Acceptance | Status |
|---|---|---|---|
| F0 | Real manifest → documented config/data audit artifact | `python scripts/supermimic/hand_s0_preflight.py --manifest /home/ubuntu/dexgym/data/processed/hand21_right_geometry_feasible/manifest.json --output /home/ubuntu/dexgym/artifacts/hand_s0/preflight.json` | passed: real manifest audit |
| F1 | Existing Core trust fixes and state-dependent task transition contract | targeted CPU tests plus prior Core suite | passed: 230 tests |
| F2 | Standalone Hand adapter/FK and pre-reset physical collection, exact task parity | FK comparison, reward/event/reset parity in Newton | pending |
| F3 | WM8/BPTT32 training and independent inference/evaluation spine | CPU gradient checks + bounded Newton real training | pending |
| F4 | GRAB training, accuracy verification, source review, report and branch push | independent held-out gates A03/A04 | pending |

## Run record
- 2026-11-09: inspected current remote state; created supermimic worktree at base cf2cdde.
- Original ~/dexgym worktree has uncommitted legacy work; preserved unchanged.
- Existing PPO PID1519240 verified live on RTX5090, using about15GB VRAM.
- Repeated minimal imports of torch2.11.0+cu130 print successfully but exit with
  SIGBUS135 in existing venv, including with libstdc++ preload. Runtime is not
  declared validated; diagnose before trusting tests. Original PPO remains live.

## Deferred
None of the requested trained-policy/report/push deliverables are deferred.

## Blockers
Existing venv process-exit SIGBUS under diagnosis; stdlib preflight is independent.

- F0: preflight exited successfully; 942 train /446 validation clips, 281793 /122482 frames, zero missing files. Artifact /home/ubuntu/dexgym/artifacts/hand_s0/preflight.json.

- F1: fixed nonfinite trust admission and normalize=False trust; added physical-state-dependent transition hook with score-before-switch test. Targeted25 /full230 tests passed. CUDA seeded witness exit0 using Isaac Python.

## F2 physical task bridge — passed
239 tests passed in 19.92 s. Newton smoke: 16x64=1024 transitions, 30 terminal, 21 goals, FK max6.055631160961639e-7 m; pre-reset capture, exact rewards/commands/termination and finite WM8 update passed. Raw artifacts /home/ubuntu/dexgym/artifacts/hand_s0/newton_smoke.json and f2_suite.log. F3 training and F4 precision pending.
