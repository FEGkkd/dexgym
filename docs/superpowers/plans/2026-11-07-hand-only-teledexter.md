# Right-hand GRAB tracking implementation plan

> **For agentic workers:** Execute this plan task by task with test-first changes and verification before any completion claim.

**Goal:** Produce a right-hand reachable fingertip sequence library and a small PPO-validated hand-only tracking task.

**Architecture:** Reconstruct GRAB MANO landmarks, feed them to the local Wuji SDK, and compute robot-frame fingertip goals by MuJoCo FK. A new DirectRLEnv samples the resulting sequences and trains a 20D right-hand controller with consecutive subgoals.

**Tech Stack:** Python 3.12, NumPy/SciPy, MuJoCo, local Wuji C SDK 2026.10.26, Isaac Lab 3.0 Newton MJWarp, RSL-RL PPO.

**Spec:** `docs/superpowers/specs/2026-11-07-hand-only-teledexter-design.md`

## Global constraints

- Use only the right hand for observations and actions; hold the other 34 DoFs at reference values.
- Do not read object pose into the task or train on object rewards.
- Use the packaged pinned robot asset and the local pinned Wuji SDK; do not copy vendor files into this repository.
- Keep generated `.npz` files under `data/processed/hand_only_right/` and record their provenance in JSON.
- Stop after a small PPO workflow validation; no large-scale training in this phase.

---

### Task 1: Convert GRAB sequences

**Files:** Create `source/dexgym/data/grab_mano.py`, `source/dexgym/data/wuji_retarget.py`, `scripts/data/build_grab_right_hand_library.py`; test in `tests/data/`.

**Interfaces:** `mano_keypoints(fullpose, template, model) -> ndarray[T,21,3]`; `retarget_sequence(keypoints, sdk, model) -> (q_ref, tip_ref, metrics)`; CLI writes per-motion `.npz` and manifest JSON.

- [x] Write tests for MANO landmark ordering, wrist-relative invariance, SDK right-hand output shape and finite values, and robot FK reconstruction.
- [x] Run the tests and confirm they fail for missing behavior.
- [x] Implement the MANO numeric forward pass and the SDK/MuJoCo adapters using the pinned local sources.
- [x] Generate one pilot motion; inspect clipping, fingertip range, temporal discontinuities and FK reconstruction.
- [x] Process all six available GRAB subjects and validate the saved library and manifest.

### Task 2: Add the right-hand task

**Files:** Create `source/dexgym/tasks/direct/right_hand_tracking/` and its tests; modify `source/dexgym/tasks/__init__.py` and add a PPO config in `source/dexgym/tasks/ppo_cfg.py`.

**Interfaces:** Gym ID `DexGym-Wuji-Right-Hand-Tracking-Direct-v0`; `DirectRLEnv` with 20D action and fixed-width policy observation; sequence loader consumes Task 1 `.npz` files.

- [x] Write tests for sequence sampling, reached-goal dwell, reward terms, termination, and right-hand-only action indices; confirm expected failures.
- [x] Implement task config and environment with Isaac Lab 3.0 documented APIs.
- [x] Run unit tests and a headless environment smoke with synthetic actions.

### Task 3: PPO workflow validation and report

**Files:** Update `README.md`; create `docs/experiments/hand_only_right_pilot.md`.

- [x] Run lint and focused tests, then the existing repository test suite.
- [x] Run a bounded PPO train with a small number of environments and iterations; verify finite metrics and checkpoint creation.
- [x] Evaluate a direct joint replay baseline and a held-out-sequence rollout where practical.
- [x] Record exact commands, library statistics, runtime results, limitations and the configuration proposed for later large-scale training.
