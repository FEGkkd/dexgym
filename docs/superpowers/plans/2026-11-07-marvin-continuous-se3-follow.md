# Marvin continuous dual-TCP SE(3) following implementation plan

**Goal:** Train and independently evaluate an arm-only continuous dual-TCP pose-following policy with the existing SuperMimic Model-Based RL Core.

**Architecture:** A task-owned reference provider converts a validated, 50 Hz procedural Marvin motion into time-indexed dual-TCP poses. A task plugin exposes 10 future SE(3) goals, current-pose observations, differentiable tracking reward, safety termination, and complete-episode metrics. A Newton task environment retains the existing 54DoF robot servo and 14DoF joint-delta action. The shared `SuperMimicTrainer` owns replay, WM8, trust, BPTT32, and TD(lambda).

**Source:** `source/dexgym/assets/supermimic/references/procedural_bimanual_reach_v1.npz` (SHA256 `ca4741f01734389dd62f9e64c66cdc9c237c90da6370cf76b13bcda89e143af2`). This is procedural test data, not human motion capture. Train on forward and time-reversed variants after physical playback validation. The joint witness may be used only for physics reset; Actor, Critic, World Model, reward, and evaluation consume physical state and TCP poses.

**Contracts:** At time `t`, the Actor sees current observation and future goals `t+1..t+10`; action `a_t` yields physical `s_(t+1)`, scored against target `t+1`. End-of-clip holds future goal queries at the last frame; imagination cannot cross frame 200. The 200-step episode truncates at frame 200. Safety failure is distinct from large tracking error.

## Tasks

1. Write failing CPU tests for reference indexing, reverse playback, TaskSpec dimensions, reward time, SO(3) gradients, safety termination, and imagination boundaries. Implement `SE3TCPReferenceProvider` and `MarvinContinuousSE3Follow` until they pass.
2. Write a failing Newton parity test for time advance, pre-reset physical state, reward and action command. Implement `MarvinContinuousSE3FollowEnv` on `MarvinArmNewtonBaseEnv`; register a new Gym task ID.
3. Add a config-driven, task-specific launcher that constructs the existing `NetworkFactory`, `SchemaReplay`, `SuperMimicTrainer`, trust evaluator, and strict checkpoint. Validate WM window 8 and minimum Actor horizon 32 before training. Preserve a disjoint validation environment range.
4. Add independent full-episode Newton evaluation with per-arm TCP position and SO(3) mean/P95/max, completion, contact, joint bounds, and no-noise Actor output. Save traces and a report.
5. Run CPU tests, 16-64 environment Newton smoke, and forward/reverse reference playback. Run bounded 4096-environment strict-pure training in resumable segments, evaluate pre/mid/post checkpoints, and extend only when real physical metrics improve. Keep World Model prediction horizon 8 and Actor imagination horizon 32.
6. Document exact config, measured throughput, model trust, tracking errors, safety, checkpoints, reproducible commands, and limits. Commit and push only after tests and independent evaluation; do not modify the original Wave/M5 assets or checkpoints.
