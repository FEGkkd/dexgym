# SuperMimic Core M4.3 Implementation Plan

> **For agentic workers:** Execute inline task by task; each code task starts with a failing regression and ends with an independently reviewable commit. No subagent delegation is required.

**Goal:** Make the M4.2 SuperMimic algorithm independent of Wave and Marvin dimensions while running Motion Tracking and a different Dual TCP Reach goal schema through the same trainer.

**Architecture:** Add dimension-driven Core modules and typed Robot/Task/Environment contracts beside the immutable M4.1/M4.2 compatibility classes. Keep Newton physics in task/robot adapters. Build two task plugins, a small Toy benchmark, a task-driven physical evaluator and strict policy export.

**Tech Stack:** Python 3.12, PyTorch, Isaac Lab 3.0/Isaac Sim 6, Newton 1.6.1, MJWarp 3.12, MuJoCo 3.12, pytest.

**Spec:** `docs/supermimic/core_architecture.md` and the user-provided M4.3 requirements.

## Global constraints

- Base exactly `55294a5da4de034fed08018956e7a03a15294229`; work only on `feat/supermimic-core-m43` in the independent worktree.
- Never alter the robot MJCF/USD, controller gains, 14D arm-only action semantics, M4.1/M4.2 checkpoints or reference files.
- Preserve 8-step masked model fitting, 32-step imagination contract, survival return and γ=0.99/λ=0.95 TD returns.
- Do not start long Newton training; Reach verification uses 16–64 environments and bounded steps.
- New Core imports no Marvin, Wave, MotionManager, or task environment modules.
- All checkpoint loading uses strict state dictionaries and explicit schema checks.

---

### Task 1: Baseline and dependency record

**Files:** `docs/supermimic/core_architecture.md`, this plan.

**Interfaces:** Records the dependency direction and the precise baseline used by later tasks.

- [x] Check root/M4.2 worktrees, base SHA and clean status; create isolated M4.3 branch.
- [x] Run all 81 M4.2 SuperMimic tests with the validated Isaac Python runtime.
- [x] Read paper equations/Algorithm A.1 and inspect both repository code trees at the recorded commits and licenses.
- [ ] Commit the architecture audit and plan before changing executable code.

### Task 2: Contracts and dimension factories

**Files:** Create `source/dexgym/algorithms/supermimic/core/{specs,models,normalization}.py`, `source/dexgym/algorithms/supermimic/interfaces/{robot,task,environment,reference}.py`; tests `tests/supermimic/test_m43_specs_factory.py`.

**Interfaces:** `RobotSpec(state_dim,action_dim,delta_dim,context_dim,schema_version)`, `TaskSpec(observation_dim,history_length,goal_dim,goal_horizon,task_data_dim,critic_dim,schema_version)`, `NetworkFactory.build(specs, config)`.

- [ ] Write tests constructing 42/14 + 10×90 tracking and 42/14 + 1×6 Reach Actor/Critic, plus a 3/1 Toy robot; reject mismatched dimensions.
- [ ] Confirm tests fail because the new Core API does not exist.
- [ ] Add immutable specs with positive dimension assertions, Actor/Goal/Critic normalization and dimension-driven models. Keep legacy model classes untouched.
- [ ] Run focused and all existing SuperMimic tests; commit.

### Task 3: Typed generic Replay and world-model fitting

**Files:** Create `core/{transition,replay_buffer,world_model}.py`; tests `test_m43_replay_world.py`.

**Interfaces:** `TaskState(task_id,motion_id,frame,data)`, `TransitionBatch`, `SchemaReplay(schema,capacity,num_envs,device)`, `WorldModel.autoregressive_loss(window,robot)`.

- [ ] Write tests for 1/2/3/5/8 effective transitions, partial resets, ring wraparound, task IDs, variable observation/history and task-data widths, pre-reset next state and FP32 normalizer provenance.
- [ ] Confirm the old fixed Replay fails those tests, then implement typed storage and masked sampling with absolute timestamps and Episode IDs.
- [ ] Fit delta dynamics on eight slots with valid-element normalization; assert nonzero gradients with no full eight-step episode.
- [ ] Run focused/all tests and commit.

### Task 4: Generic imagination, returns, Critic and trainer

**Files:** Create `core/{imagination,returns,optimization,trust_evaluator,trainer,checkpoint}.py`; tests `test_m43_imagination.py`, `test_m43_trainer.py`.

**Interfaces:** `ImaginationEngine.rollout(start,actor,model,critic,robot,task,horizon)`, `SuperMimicTrainer.run_iteration()`. Trainer consumes a `PhysicalCollector` returning `TransitionBatch`.

- [ ] Write two task-dimension gradient tests showing Actor → model → reward and verifying frozen WM/Critic parameters; hand-check survival and TD(λ).
- [ ] Implement task-driven observation/goal/reward/termination and robot-driven integration without task imports in Core.
- [ ] Add actor-source held-out trust, finite-step guards, multi-epoch Critic optimization, mode-tagged collection and per-phase timing. Exclude validation environment IDs from all RunningNorm updates.
- [ ] Test alternating Toy collection/model/Actor/Critic updates and strict Checkpoint restore; run all tests and commit.

### Task 5: Toy model-based RL with distinct goals

**Files:** Create `source/dexgym/tasks/supermimic_toy.py`, `tests/supermimic/test_m43_toy_rl.py`.

**Interfaces:** One known-dynamics Toy robot; two TaskSpec implementations with different goal and reward dimensions, both using the same Core trainer.

- [ ] Test deterministic known transition and finite difference Actor gradient against manual calculation.
- [ ] Run bounded short training for two goals, recording initial/final physical return, WM8 error, Actor/Critic counts and nonfinite counts.
- [ ] Keep the test deterministic and CPU-fast; commit code and measured report.

### Task 6: Marvin robot and Motion Tracking plugin

**Files:** Create `source/dexgym/robots/marvin/supermimic_adapter.py`, `source/dexgym/tasks/marvin_motion_tracking/{task,collector,config}.py`; tests `test_m43_marvin_tracking.py`.

**Interfaces:** New Marvin Robot Adapter delegates exactly to versioned action map and FK; tracking Task owns 10×90 goal, six-link reward, termination and 146D Critic feature contract.

- [ ] Write numerical parity tests against M4.2 reward, action mapping, goal windows and Critic feature order.
- [ ] Implement tracking plugin and generic collector adapter; remove Wave imports from new Core.
- [ ] Verify M4.1 output fixture and full Newton Wave inference regression; commit.

### Task 7: FPS and multiple motion IDs

**Files:** Modify `source/dexgym/motion/{motion_manager,library}.py`; tests `test_m43_motion_fps_ids.py`.

**Interfaces:** `MotionLibrary(...,policy_fps=50)` rejects wrong FPS; `reference_at(motion_id,frame)` and `future_goal_at` preserve per-motion lengths.

- [ ] Write a failing 25 Hz fixture test and two-motion ID/window-boundary test.
- [ ] Enforce actual time spacing and FPS=50 at import; keep Wave hash and numerical frames unchanged.
- [ ] Exercise both IDs through the same tracking Task and generic Replay in a short Torch data-flow test; commit. Mark mixed Newton environments unvalidated unless physically run.

### Task 8: Dual TCP Reach plugin and bounded Newton smoke

**Files:** Create `source/dexgym/tasks/marvin_tcp_reach/{task,env,collector,config}.py`, register in `source/dexgym/algorithms/supermimic/registry.py`; tests `test_m43_reach_task.py`, `test_m43_newton_reach.py`.

**Interfaces:** Reach Task has `goal_dim=6`, `goal_horizon=1`, independent reward/termination/critic features and static target; no 90D padding. Newton collector records actual pre-reset state and target provenance.

- [ ] Write Reach goal/reward/soft-termination/gradient tests and fail before adding plugin.
- [ ] Implement physical Reach environment around unchanged Marvin control profile and 54DoF model; check 14D command, finger home, contact isolation and hard terminal timing.
- [ ] Run a bounded 16–64 env alternating trainer smoke with nonzero WM, Actor and Critic updates; save checkpoint and task metrics. No long training or convergence claim.
- [ ] Commit plugin and physical evidence.

### Task 9: Evaluation, policy bundle and compatibility

**Files:** Create `core/{evaluation,policy_bundle}.py`, `scripts/supermimic/{train_m43,evaluate_m43,export_m43}.py`; update `scripts/supermimic/evaluate.py` to use a task-specific Legacy evaluator adapter; tests `test_m43_policy_bundle.py`, `test_m43_checkpoint_compatibility.py`.

**Interfaces:** `PhysicalEvaluator` obtains metric names and success from Task; `PolicyBundle` stores Actor/normalizers/specs/action schema/joint order/control rate/hashes, and loads on CPU/GPU with strict schema matching.

- [ ] Write mismatched Goal/Action schema rejection and CPU/GPU Actor inference parity tests.
- [ ] Add registry-backed config selection and task-specific physical evaluators without generic code importing `train_wave.evaluate_full`.
- [ ] Load M4.1 and M4.2 checkpoints strictly; replay full Wave Newton reference without fine tuning and verify prior 1356-step safety/accuracy range.
- [ ] Verify save/restart including optimizer, scheduler, normalizer, Replay and Episode boundary; commit.

### Task 10: Documentation, acceptance and push

**Files:** `docs/supermimic/{algorithm_contract,robot_adapter_guide,task_plugin_guide,model_factory,checkpoint_compatibility,m43_validation}.md`.

**Interfaces:** Concrete how-to for new tasks and exact run commands.

- [ ] Re-run all M4.2 and M4.3 tests, static dependency audit and short physical regressions.
- [ ] Report Gate A–F with actual numbers, checkpoint paths, latency/VRAM and explicit missing evidence.
- [ ] Run `git diff --check`, ensure clean branch, commit docs, push `feat/supermimic-core-m43`, and verify remote SHA.
