# M4.3 SuperMimic Core architecture

## Baseline and source audit

This design starts at DexGym `55294a5da4de034fed08018956e7a03a15294229`. The 81 M4.2 SuperMimic tests passed before editing. The unchanged M4.1 Wave checkpoint and M4.2 checkpoints are compatibility fixtures.

The paper [SuperMimic, arXiv:2610.28480v1](https://arxiv.org/abs/2610.28480) was read through its equations, Appendix B multi-interface discussion, Algorithm A.1, and Table A.5. TrackerLab source at `e2db2dea0e81c91ebca815c895b60ee9aa58237a` was read in `runner.py`, `networks.py`, `buffer.py` and the tracking MDP; its repository license is MIT. The independent mjlab implementation at `9aab0a588c6941bf9862a2d444b711fd73e2b832` was read in `algo/trainer.py`, `buffer.py`, `state.py`, and `reward.py`; its license is Apache-2.0. No third-party source is copied. Paper equations guide the Core; fixed-base Marvin state and controller semantics remain DexGym adaptations. TrackerLab's masked replay and configurable network dimensions are implementation references, not a claim of official paper authorship. mjlab's state-spec and termination network separation is another implementation reference.

## Current M4.2 dependencies and defects

```mermaid
flowchart LR
  T[legacy trainer.py] --> R[legacy rewards.py]
  T --> MT[MarvinTrackingTask]
  MT --> I[legacy imagination.py]
  I --> FK[Marvin FK]
  I --> MM[MotionManager]
  I --> N[42D 14D 10x90 networks]
  E[legacy evaluate.py] --> TW[train_wave.evaluate_full]
  RE[legacy replay_buffer.py] --> FIX[42D state, 14D action, 10x56 history]
```

The old trainer also reads `reference.data["q_arm"]` to normalize Critic input, and the old training script imports Wave-specific supervised PD examples. The Motion Library records FPS without enforcing the environment's 50 Hz frame semantics. The legacy `MotionManager` and Newton environment assume one clip per environment batch. These are migration targets; the original classes and checkpoint weight shapes stay unchanged.

## Dependency rule

```mermaid
flowchart TD
  CLI[configured train / evaluate CLI] --> REG[registry and factory]
  REG --> CORE[algorithms.supermimic.core]
  REG --> MARVIN[Marvin robot plugin]
  REG --> TRACK[Marvin tracking task]
  REG --> REACH[Marvin dual TCP reach task]
  MARVIN --> CORE
  TRACK --> CORE
  REACH --> CORE
  CORE --> TORCH[PyTorch]
  LEGACY[M4.1/M4.2 compatibility modules] --> CORE
```

No module in `algorithms.supermimic.core` imports `dexgym.tasks`, `dexgym.motion`, the Marvin FK, or Wave scripts. The plugin registry is outside Core. Legacy files retain their existing behavior and strict checkpoint loaders. New components use explicit `RobotSpec`, `TaskSpec`, `TransitionBatch`, and `TaskState` dataclasses with dimensions and schema versions checked at construction.

## Contracts

- `RobotDynamicsAdapter`: `spec`, `integrate(state, delta, action, dynamics_context)`, `physical_delta(state, next_state)`, `executed_command(state, action)`, `physical_state(env)`, action bounds, optional differentiable kinematics, controller timing and asset hash. The Marvin implementation uses 42D `[q14, qd14, qcmd14]`, predicts 28 physical deltas, and applies the versioned analytic 14D command update. A Toy robot uses different dimensions in tests.
- `TaskMDP`: `spec`, `initial_task_state`, `advance`, `goal`, `observation`, `critic_features`, `reward`, `hard_termination`, `soft_termination`, `valid_imagination_start`, `evaluate_metrics`. It may have `motion_id` and `frame` or an independent static `task_data` target. The task owns success and failure metrics. Arrival reward and termination consume `s_(t+1)` with the advanced task state.
- `ReferenceProvider`: current reference, future goals, clip length, FPS and hash; only tracking task uses it. Non-50 Hz references are rejected on import until SO(3)-correct resampling is implemented.
- `PhysicalCollector`: produces a typed, pre-reset `TransitionBatch` from the actual simulator. The batch contains the executed action and command, previous action, state/next-state, dynamics context, task state before/after, hard terminal/truncation, source, failure reason and environment ID.

`TaskState` contains `task_id`, `motion_id`, integer `frame`, and schema-checked floating `task_data`. Motion tracking uses motion ID and frame with zero task-data width. TCP Reach uses a six-number dual-TCP target with no reference timeline. A single trainer instance has one TaskSpec; mixed task batches require a compatible union schema and are deliberately deferred. Replay still records Task ID for provenance and future weighted sampling.

## Generic algorithms

`NetworkFactory` builds a dimension-driven delta world model, goal encoder/Actor, Critic and optional task termination head. It does not alter old `GoalActor`, `DeltaWorldModel` or `PrivilegedCritic` parameter names or shapes. RunningNorm receives only real training-environment states and legal task goals; held-out environment IDs never update statistics. The world model predicts robot-declared physical delta dimensions; the robot integrates analytic state components. Learned termination is separately task-shaped. Analytic termination is the default for Marvin tracking and Reach.

`SchemaReplay` owns the typed ring with absolute timestamps and Episode IDs, masks terminal transitions correctly, and stores configurable state/action/history/context/task-data widths. It never joins a newly reset physical episode to an old one. The `ImaginationEngine` asks the task for observation/goal/reward/termination/critic features and the robot for integration, so it contains no FK, NPZ or link names. It computes `c[t+1]=c[t](1-d[t+1])`, the survival-weighted H-step Actor objective and TD(λ) targets with the same arrival termination index. World model and Critic parameters are frozen during Actor backward; state/action gradients are retained.

`SuperMimicTrainer` alternates physical collection, masked eight-step model fit, actor-source held-out trust evaluation, legal imagination starts, Actor BPTT and Critic fitting. It owns update counts, finite-gradient protection, schedule, checkpoints and evaluation intervals. It receives adapters, modules and collector by injection. Teacher dynamics collection is tagged; supervised Actor initialization is a separate optional initializer supplied by the task, never imported from Wave in Core.

`PhysicalEvaluator` orchestrates a task-defined metric accumulator over an independent Newton episode. `PolicyBundle` exports only Actor and inference preprocessing schema, normalizers, robot action version, joint order, control frequency and asset hash; it does not imply real-hardware certification. Old M4.1 and M4.2 checkpoints remain loadable only under their exact original schemas.

## Validation sequence and claims

1. Dimension factory and two Goal Schemas fail-before-fix tests, then generic Replay and world-model math tests.
2. Toy Torch physics with two tasks completes collection, WM8, Actor BPTT and TD(λ) with finite gradients and measured reward improvement.
3. Marvin tracking plugin matches M4.2 Reward, Goal, Action and Actor numerical semantics; the M4.1 Wave full Newton inference regression remains unchanged.
4. Reach plugin uses 1×6 dual-TCP goal, independent reward and success metric, and the same Core trainer in a bounded 16–64 env Newton smoke. This is integration evidence, not convergence.
5. Checkpoint save/restart and CPU/GPU inference parity, then documentation and push.

Gate B requires both 10×90 tracking and 1×6 reach to execute the **same Core trainer code**. Gate E requires real Reach Newton transitions and at least one WM, Actor and Critic update, with actual result metrics. Mixed-motion Newton training and contact-rich dynamics sharing remain explicitly unvalidated unless separately exercised.
