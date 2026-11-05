# Marvin SuperMimic M4.1 Algorithm Alignment Implementation Plan

> **For agentic workers:** Implement the tasks inline in this isolated worktree. Each behavioral change starts with a failing test and ends with a fresh regression run.

**Goal:** Train a Marvin bimanual tracker with the paper's alternating collect/model/imagination/critic loop, differentiable survival, and environment-local safety termination, then compare independent Newton rollouts against E11.

**Architecture:** Keep the pinned 54DoF Newton environment, 14DoF action map, reference manager, and differentiable FK. Add a stateful collector and alternating training coordinator without changing the legacy E11 entry point. Extend the world model with a learned termination head and use its probability in the Actor return and Critic TD(λ) target. Keep strict paper reward, goal-latent tanh, observation noise, and auxiliary initialization as explicit experiment flags.

**Tech Stack:** Python 3.12, PyTorch, Isaac Lab 3.0, Newton MJWarp, MuJoCo cross-check, pytest, uv, Ruff.

**Spec:** User-provided M4.1 objective in this thread; original paper [SuperMimic §3 and Appendix A/D](https://arxiv.org/pdf/2610.28480).

## Global Constraints

- Pin `cmu_141_16_wave_bimanual.npz` SHA256 `b7af95b289e8aff898df7f5dd06b169f74dba2a506aa8fc5bf589ee3fee8106f`.
- Keep E11 `best.pt`, M4 source branch, source MJCF, USD asset, and old training outputs unchanged.
- Use Newton real physics for collection/evaluation; no PPO or hardware.
- Evaluate left and right TCP position and SO(3) orientation separately across all 1357 frames.
- Preserve the same action map in real Newton and imagined rollouts.

### Task 1: Fix environment-local contact and terminal state capture

**Files:** `source/dexgym/tasks/marvin_supermimic/contact_audit.py`, `env.py`; `tests/supermimic/test_contact_audit.py`.

**Interfaces:** `contact_counts(world_ids, active_count, num_envs) -> Tensor[num_envs]`; environment exposes pre-reset `last_physical_arm_state`, `last_physical_command`, and `last_policy_contact_by_env`.

- [ ] Add failing tests for one contact in environment 1 leaving environments 0/2 un-terminated, invalid world IDs, and the final physics substep being included before `_get_dones`.
- [ ] Run the tests and confirm the intended failure.
- [ ] Map active MJWarp `contact.worldid` into per-environment counts; accumulate before `_get_dones`; retain per-environment counts through reward and reset.
- [ ] Run tests and an actual two-environment Newton step with a forced contact on only one environment.
- [ ] Commit the contact/terminal-state fix.

### Task 2: Add learned soft termination and paper returns

**Files:** `source/dexgym/algorithms/supermimic/networks.py`, `imagination.py`, `replay_buffer.py`; `tests/supermimic/test_soft_termination.py`, `test_world_model.py`.

**Interfaces:** `DeltaWorldModel.termination_probability(state, current_goal)` predicts `d_hat` in `[0,1]`; `lambda_returns(rewards, values, termination_probabilities)` computes Eq. 3; `imagined_rollout` returns survival weights and termination probabilities.

- [ ] Add failing hand-calculated tests for `c_0=1`, `c_{t+1}=c_t(1-d_{t+1})`, reward-before-termination ordering, terminal bootstrap, λ=0/1 targets, and gradients through `d_hat`.
- [ ] Add replay tests that preserve unsafe terminal labels and pre-reset physical next states without joining episodes.
- [ ] Train the termination head with BCE on eight-step real windows, including a terminal at the final transition.
- [ ] Apply paper Eq. 2 to Actor objective and Eq. 3 to Critic targets; confirm BPTT reaches Actor and termination head stays frozen on Actor update.
- [ ] Commit after CPU/GPU numerical and finite-difference tests pass.

### Task 3: Strict paper reward and observation ablations

**Files:** `source/dexgym/algorithms/supermimic/rewards.py`, `networks.py`, `observation.py`; `source/dexgym/tasks/marvin_supermimic/env.py`, `env_cfg.py`; `tests/supermimic/test_rewards.py`, `test_observation_action.py`.

**Interfaces:** `paper_multilink` reward averages squared errors over all six tracked links inside each exponential; optional `goal_latent_tanh`; configurable q/qd observation noise applied to real training and imagination but disabled in deterministic evaluation.

- [ ] Add failing exact-value tests for the four six-link exponential terms and action-smooth/joint-limit terms from Table A.2, plus gradient tests.
- [ ] Add failing tests that tanh bounds the 32D latent only when configured and that noise never modifies command or prior-action fields.
- [ ] Implement flags while retaining legacy E11 profile for comparison.
- [ ] Run real/imagination reward and action-map numerical agreement tests; commit.

### Task 4: Alternate real collection and all three updates

**Files:** `scripts/supermimic/train_aligned.py`, `source/dexgym/algorithms/supermimic/training_loop.py`, `replay_buffer.py`; `tests/supermimic/test_alternating_loop.py`, `test_training_checkpoint.py`.

**Interfaces:** every steady-state iteration collects a bounded number of new Newton transitions, performs eight-step model updates, then 32-step Actor BPTT, then TD(λ) Critic updates. Replay is a ring buffer; checkpoint/resume records update counts and recollects replay if omitted.

- [ ] Add failing event-order tests proving repeated `collect→WM→Actor→Critic` rather than four blocks, and a Pure profile with zero auxiliary supervised updates and zero teacher actions.
- [ ] Implement stateful Newton collection that continues episodes across update boundaries and logs executed noisy actions, terminal labels, and pre-reset next states.
- [ ] Implement model-trust gates, finite-gradient checks, periodic independent full-motion evaluation, bounded stop/resume, and checkpoint selection by physical score.
- [ ] Run small 32–128-environment physical smoke for Pure and optional auxiliary profiles; commit.

### Task 5: Capacity, bounded training, and independent evaluation

**Files:** `scripts/supermimic/train_aligned.py`, `scripts/supermimic/evaluate_wave.py`, `docs/supermimic/marvin_wave_m41.md`; outputs under `/home/ubuntu/dexgym_data/marvin_supermimic_wave_m41/`.

- [ ] Run complete-loop memory/throughput checks at 1024 and 4096 environments, selecting a sustainable size.
- [ ] Train a bounded Pure and optional auxiliary-initialized run with alternating updates, preserving all checkpoints and logs.
- [ ] Reproduce each selected checkpoint in independent full 1357-frame Newton physical rollouts; compare both TCP 6D mean/P95/max, contacts, limits, speed, and torque against E11.
- [ ] Run `uv lock --check`, Ruff check/format, old asset/motion tests, all new SuperMimic tests, and full Newton replay.
- [ ] Document actual results, failures, paper-versus-Marvin differences, artifact hashes, and reproducible commands; commit and push the independent branch.
