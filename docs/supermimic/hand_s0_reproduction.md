# Hand-only S0 reproduction — implementation record

Status: **task/architecture/model-based-training readiness validation**. The user
explicitly removed converged-policy acceptance from this phase on 2026-11-09.
Actual training and evaluation below are bounded diagnostics. See the final
`hand_s0_readiness_20261109.md` report and updated `implement-stage/SPEC.md`.

## Scope and original code

Base `cf2cddeeddaee7551313ec03a88ee1c45c0a9bc5`, branch `supermimic`, working
directory `/home/ubuntu/dexgym/worktrees/supermimic`. Reuses the original
`right_hand_tracking` environment, official standalone 20-DoF Wuji Hand 2.1
USD, existing GRAB motion bank and SuperMimic Core. No Marvin arm state,
arm world-model weights, objects or tactile observations are used. The running
PPO remains in its existing separate worktree.

## Method correspondence

Primary sources: [TeleDexter](https://arxiv.org/html/2607.11481v1),
[SuperMimic](https://arxiv.org/pdf/2610.28480).

| Component | This implementation | Qualification |
|---|---|---|
| Physical task | Consecutive fingertip subgoals, hold 5–15 steps, random spans, cross-reference switches and cumulative failure budget | Original hand task rules shared between physics and imagination |
| Control | Accumulated joint targets; scale .1 and deadzone .1 | Core raw action passes through tanh exactly once |
| Rate and tolerance | 30 Hz control/reference, 480 Hz physics, 15 mm all-finger success threshold | User's current task; paper uses 30 mm and describes spans at 60 Hz |
| Reward | Sparse hand score, complete hand dense terms, time cost | Object terms removed; Wuji intermediate-frame interpretation is explicit in A14/A16 |
| Actor information | Current fingertip goal and 10-frame hand observation history | No reference joints or future reference frames |
| Learning | Real simulator delta model fitted autoregressively for 8 steps; frozen-model BPTT32, survival and TD(lambda) | No PPO objective, imitation loss or privileged action warm start |
| Discrete events | Exact events in physics and goal progression; learned event/termination models in imagined objective | Real labels supervise the models; approximation error must be reported |
| Scale | Initial 256 environments, reduced hand network widths | Compute adaptation, not paper-scale equivalence |
| Robustness | Nominal fixed-wrist dynamics and original full collisions | Not a claim of full randomized or real-hardware reproduction |

The legacy PPO's simplified dense term remains its default. Its existing
learning curve is not a fair algorithm comparison against the restored dense
term; a comparison needs matched task/reward settings.

## Data and isolation

Input: `/home/ubuntu/dexgym/data/processed/hand21_right_geometry_feasible/manifest.json`.
Eligible >=6-frame clips: 942 train / 446 validation, 281793 / 122482 frames.
Training uses subjects s1–s7 only. Separate simulator lanes provide world-model
checks without optimizer or normalizer access to those lanes. These are not the
final held-out-subject evaluation. Subjects s8–s10 remain evaluation-only.

## Verified checks

- F1: 230 SuperMimic regression tests passed in 18.99 s. NaN trust checks and
  normalization-disabled trust evaluation were repaired; hybrid goal transition
  support preserves existing tasks.
- F2: 239 tests passed in 19.92 s, including old hand terminal tests, new task
  causality checks and FK value/gradient comparisons.
- Newton smoke: 16 environments × 64 control steps, 1024 transitions,
  30 terminal transitions and 21 completed goals. Maximum FK error against the
  actual Newton USD: 6.055631160961639e-7 m. One forced terminal lane confirms
  pre-reset arrival differs from post-reset state (maximum state-coordinate
  difference 1.2632805109024048, mixed state units). Reward, command and hard
  terminal parity passed; a WM8/learned-head update completed with finite loss.

Exact command (from the worktree):

```bash
bash scripts/supermimic/hand_s0_host.sh scripts/supermimic/smoke_hand_s0.py \
  --library /home/ubuntu/dexgym/data/processed/hand21_right_geometry_feasible/manifest.json \
  --output /home/ubuntu/dexgym/artifacts/hand_s0/newton_smoke.json
```

Exit code 0. Raw files: `artifacts/hand_s0/newton_smoke.{json,log}` and
`artifacts/hand_s0/f2_suite.log` under `/home/ubuntu/dexgym`.
These checks establish implementation consistency, not learned tracking precision.

## Open acceptance work

The current phase validates actual optimizer updates, checkpoint/resume,
deterministic evaluation, physical task consistency and scale feasibility.
Converged policy training and three-seed quality acceptance are deferred by
explicit user instruction. A03/A04 record this scope change; measured errors
remain diagnostic. Final source audit, assumption-review status and publishing
are recorded in `hand_s0_readiness_20261109.md`.
