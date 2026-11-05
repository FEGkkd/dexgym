# Marvin Wave M4 Implementation Plan

> **For agentic workers:** Execute each step against real Newton state and save measured evidence.

**Goal:** Train and evaluate an arm-only bimanual policy using a learned differentiable world model and 32-step BPTT.

**Architecture:** GPU motion tensors feed Newton DirectRLEnv and a separate replay-driven learner. Differentiable Marvin FK supplies geometric reward for imagined rollouts. The trainer alternates real collection, 8-step model fitting, 32-step imagination, TD(lambda) critic updates, and Newton evaluation.

**Tech Stack:** Python 3.12, PyTorch, Isaac Lab 3.0, Newton MJWarp, MuJoCo, uv.

**Spec:** User's Marvin Bimanual SuperMimic Wave v0 request, 2026-11-05.

## Global constraints

- Pin the dual-arm NPZ by SHA256 and reject asset/report mismatch.
- Preserve 54DoF physics; policy, model and replay use 14 arm DoF.
- Policy dt 0.02 s, physics dt 0.002 s; no reference joint feedforward after reset.
- Validate model predictions before long actor optimization.
- Report Newton tracking separately from imagined reward.
- Preserve previous S0 and M1–M3 results.

## Tasks

- [x] Inspect Git state and create independent branch/worktree at pinned reference commit.
- [ ] Verify NPZ, hashes, schema, FK, replay provenance, and 16-env completion; run `uv lock --check`.
- [ ] Implement GPU Motion Manager, endpoint semantics and tests.
- [ ] Implement shared differentiable action map and 14D Newton DirectRLEnv; smoke actual physics.
- [ ] Implement differentiable 14D FK; verify MuJoCo and finite differences.
- [ ] Implement replay with valid 8-step windows and 10-step histories.
- [ ] Implement 8-step world model, 32-step actor imagination and TD(lambda) critic; test toy gradients and freeze rules.
- [ ] Run zero/random baselines and collect actual Newton transitions.
- [ ] Measure one/eight/thirty-two-step errors against simple baselines.
- [ ] Run bounded SuperMimic training, full-clip Newton evaluation and checkpoint restore.
- [ ] Render policy/video/curve evidence, document results, and push branch.
