# Allegro Hand Track DirectRLEnv Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first DexTrack migration slice as a registered Isaac Lab `DirectRLEnv` task with a 22-dimensional Allegro/Fly action contract, RSL-RL PPO configuration, and TensorBoard-compatible diagnostics.

**Architecture:** Keep reference loading, source-to-Allegro mapping, frame indexing, and validation in pure Python/Torch code. Keep free-root Allegro asset resolution in `assets/allegro_hand.py`, and keep action accumulation, observations, rewards, termination, and reset in one explicit DirectRLEnv. Register the task through the existing `tasks/direct` entry point and reuse `scripts/rsl_rl/train.py` plus its standard RSL-RL runner logging.

**Tech Stack:** Python 3.12, PyTorch, NumPy, Gymnasium, Isaac Lab/Isaac Sim 6, PhysX, `rsl-rl-lib` 5.0.1, pytest, TensorBoard event files.

**Spec:** `docs/superpowers/specs/2026-09-07-allegro-hand-track-direct-design.md`

## Global Constraints

- Use `DirectRLEnv`; do not implement the first migration as ManagerBasedRLEnv.
- Register `DexterousHands-Allegro-Hand-Track-Direct-v0`.
- First scope is one GRAB Allegro/Fly trajectory with a 22-dimensional action: six free-root pose residuals and sixteen Allegro finger-joint residuals.
- Preserve cumulative residual control: `delta_target += action_scale * dt * action` and `target = reference_frame_target + delta_target`.
- Resolve data and asset roots from `DEXTRACK_DATA_ROOT` and `ALLEGRO_DESCRIPTION_ROOT`; do not hardcode `/home/ubuntu/DexTrack` in task code.
- Do not add TACO, LEAP/Franka, generalist sampling, vision, forecasting, latent/teacher branches, chunking, or curriculum in this plan.
- Preserve source quaternions as validated `xyzw` in the loader and perform one explicit conversion at the simulator/contract boundary required by the installed Isaac Lab API; never silently reorder components.
- Use the existing RSL-RL runner and `logs/rsl_rl/<experiment>/<timestamp>` output; do not add WandB or a second logger.
- Do not launch Isaac Sim or training while implementing pure-Python contract tests.

---

### Task 1: Freeze the GRAB reference data contract

**Files:**
- Create: `source/dexteroushands_lab/tasks/direct/allegro_hand_track/reference_loader.py`
- Create: `source/dexteroushands_lab/tasks/direct/allegro_hand_track/task_contract.py`
- Create: `tests/test_allegro_hand_track_reference_loader.py`
- Create: `data/dextrack/README.md`

**Interfaces:**
- `load_grab_reference(path: str | Path, device: torch.device | str = "cpu") -> GrabReference`
- `GrabReference.hand_target: torch.Tensor` with shape `[T, 22]`
- `GrabReference.object_translation: torch.Tensor` with shape `[T, 3]`
- `GrabReference.object_quaternion_xyzw: torch.Tensor` with shape `[T, 4]`
- `GrabReference.num_frames: int`
- `GrabReference.frame(index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
- `allegro_source_to_sim_joint_order(values: np.ndarray) -> np.ndarray` maps the serialized 16-joint slice without hardcoded task-local indexing.
- `ALLEGRO_TRACK_CONTRACT` is a `TaskDataContract` with task ID, 22 ordered action names (six root channels plus sixteen finger names), control/physics dt, and these exact observation fields: `joint_position[16]`, `joint_velocity[16]`, `hand_pose_velocity[13]`, `object_pose_velocity[13]`, `reference_hand_target[22]`, `object_goal_delta[3]`, and `previous_action[22]`.

- [ ] **Step 1: Write failing loader tests**

  Add tests that locate one `.npy` sample under `DEXTRACK_DATA_ROOT` when the environment variable is set, otherwise skip with a clear message. Assert required keys, `[T,22]`, `[T,3]`, `[T,4]`, finite values, normalized quaternion norms, and final-frame clamping. Add a synthetic invalid dictionary test for missing keys and non-finite values. Add a mapping test that uses named source and destination joint lists and rejects duplicate or unknown names.

- [ ] **Step 2: Run the focused tests and verify they fail for missing modules**

  Run: `pytest -q tests/test_allegro_hand_track_reference_loader.py`

  Expected: collection/import failure because the new package and loader do not yet exist.

- [ ] **Step 3: Implement the loader and contract**

  Use `np.load(path, allow_pickle=True).item()` only after checking that the result is a dictionary. Validate the three required arrays, normalize `object_rot_quat` after rejecting norms below `1e-8`, convert the source hand array into the tested Allegro order, and move tensors to the requested device. Implement `frame()` with `index.clamp(0, num_frames - 1)` so it never wraps. Keep `object_quaternion_xyzw` named with the source convention; add a separate conversion helper for the simulator convention instead of mutating it implicitly.

- [ ] **Step 4: Add data preparation documentation**

  Document extracting `GRAB_Tracking_PK_reduced_300.zip` into a directory containing `data/*.npy`, then exporting `DEXTRACK_DATA_ROOT=/path/to/extracted/GRAB_Tracking_PK_reduced_300`. State that the failed TACO/LEAP downloads are outside this phase and that archives are not committed.

- [ ] **Step 5: Run the focused tests and commit**

  Run: `pytest -q tests/test_allegro_hand_track_reference_loader.py`

  Expected: all loader, frame-clamp, quaternion, and mapping tests pass. Commit with `git add source/dexteroushands_lab/tasks/direct/allegro_hand_track tests/test_allegro_hand_track_reference_loader.py data/dextrack/README.md && git commit -m "feat: add DexTrack GRAB reference contract"`.

### Task 2: Add the external Allegro/Fly asset adapter

**Files:**
- Create: `source/dexteroushands_lab/assets/allegro_hand.py`
- Create: `tests/test_allegro_hand_asset_contract.py`

**Interfaces:**
- `allegro_description_root() -> Path`
- `allegro_hand_usd_path() -> str`
- `ALLEGRO_ROOT_CHANNEL_NAMES: tuple[str, ...]` with `root_x`, `root_y`, `root_z`, `root_rx`, `root_ry`, and `root_rz`; `ALLEGRO_FLY_FINGER_JOINT_NAMES: tuple[str, ...]` with the sixteen actual Allegro finger joint names; and `ALLEGRO_FLY_ACTION_NAMES` as their concatenation
- `ALLEGRO_FLY_TIP_BODY_NAMES: tuple[str, ...]`
- `make_allegro_fly_cfg() -> ArticulationCfg`
- `validate_allegro_fly_dof_contract(robot: Articulation) -> None`

- [ ] **Step 1: Write pure path and name tests**

  Test that an unset `ALLEGRO_DESCRIPTION_ROOT` raises `RuntimeError`, a nonexistent root raises `FileNotFoundError`, and the configured USD path must be a file. Assert sixteen unique finger joint names and five fingertip body names. Do not require Isaac Sim for these path/name tests.

- [ ] **Step 2: Run the focused tests and verify they fail**

  Run: `pytest -q tests/test_allegro_hand_asset_contract.py`

  Expected: import failure because the adapter does not exist.

- [ ] **Step 3: Implement path resolution and articulation configuration**

  Resolve `ALLEGRO_DESCRIPTION_ROOT` with `Path(...).expanduser().resolve()`, append a documented relative USD path, and fail before scene creation if it is absent. Build an `ArticulationCfg` with `UsdFileCfg`, `fix_root_link=False`, enabled self-collisions as required by the converted asset, explicit initial root pose, and an implicit actuator covering only the sixteen finger joints. Keep free-root pose handling in the Direct task rather than pretending root pose is a finger joint.

- [ ] **Step 4: Add a simulator-only asset audit hook**

  Implement `validate_allegro_fly_dof_contract` to compare the loaded articulation joint names against the declared sixteen finger names, verify a free root/expected root DOF representation, verify all limits are finite, and report missing palm/tip bodies. Raise a descriptive `RuntimeError` rather than selecting a fallback asset.

- [ ] **Step 5: Run tests and commit**

  Run: `pytest -q tests/test_allegro_hand_asset_contract.py` and `python -m compileall source/dexteroushands_lab/assets/allegro_hand.py`.

  Expected: path/name tests pass without Isaac Sim; compile succeeds. Commit with `git add source/dexteroushands_lab/assets/allegro_hand.py tests/test_allegro_hand_asset_contract.py && git commit -m "feat: add Allegro Fly asset adapter"`.

### Task 3: Define the Direct task configuration and observation contract

**Files:**
- Create: `source/dexteroushands_lab/tasks/direct/allegro_hand_track/__init__.py`
- Create: `source/dexteroushands_lab/tasks/direct/allegro_hand_track/allegro_hand_track_env_cfg.py`
- Modify: `source/dexteroushands_lab/tasks/direct/allegro_hand_track/task_contract.py`

**Interfaces:**
- `AllegroHandTrackDirectEnvCfg(DirectRLEnvCfg)` with `action_space=22`, fixed `observation_space`, `episode_length_s`, `decimation`, `sim`, `scene`, `robot`, and one cube-like rigid object.
- Config fields `reference_path`, `reference_fps`, `action_scale`, `residual_limit`, `object_goal_pos`, `object_goal_rot_xyzw`, and explicit reward/termination thresholds.
- Registration function `register_task()` in `__init__.py` that only registers the Gymnasium task ID once.

- [ ] **Step 1: Write contract/config shape tests**

  Add tests that instantiate the config without starting Isaac Sim, assert `action_space == 22`, assert `observation_space == ALLEGRO_TRACK_CONTRACT.observation_dim`, and assert the task ID and RSL-RL entry-point strings are stable.

- [ ] **Step 2: Run the tests and verify the configuration is absent**

  Run: `pytest -q tests/test_allegro_hand_track_config.py`.

  Expected: import failure until the task package and config are created.

- [ ] **Step 3: Implement the config and registration shell**

  Follow the existing WUJI DirectTask configuration style. Set simulation dt and decimation explicitly, create a replicated scene, reference the Allegro asset adapter, and spawn one simple rigid cube with deterministic initial pose and friction. Store the external reference path as a config value resolved from `DEXTRACK_DATA_ROOT` at construction. Register `DexterousHands-Allegro-Hand-Track-Direct-v0` with entry points for the environment config and the agent config path defined in Task 5.

- [ ] **Step 4: Run tests and commit**

  Run: `pytest -q tests/test_allegro_hand_track_config.py` and `python -m compileall source/dexteroushands_lab/tasks/direct/allegro_hand_track`.

  Expected: all static config assertions pass. Commit with `git add source/dexteroushands_lab/tasks/direct/allegro_hand_track && git commit -m "feat: scaffold Allegro tracking DirectTask"`.

### Task 4: Implement action processing, reset, observations, reward, and termination

**Files:**
- Create: `source/dexteroushands_lab/tasks/direct/allegro_hand_track/allegro_hand_track_env.py`
- Create: `tests/test_allegro_hand_track_math.py`
- Modify: `source/dexteroushands_lab/tasks/direct/allegro_hand_track/task_contract.py` if observation field constants need exact sizes

**Interfaces:**
- `AllegroHandTrackDirectEnv(DirectRLEnv)` implementing `_setup_scene`, `_pre_physics_step`, `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones`, and `_reset_idx`.
- Pure helpers in the environment module or a small local helper section: `clip_and_accumulate_residual(actions, scale, dt, residual_limit)`, `compute_object_position_error(current, target)`, and `compute_tracking_reward_components(...)`.
- `extras["log"]` keys: `tracking/hand_error`, `tracking/object_pos_error`, `tracking/object_rot_error`, `tracking/palm_distance`, `tracking/fingertip_distance`, `tracking/success`, `control/action_norm`, `control/residual_norm`, `termination/reference_end`, and `termination/object_lost`.

- [ ] **Step 1: Write pure action/reward helper tests**

  Assert action clipping, cumulative residual update, per-dimension limit clipping, zero-action reference behavior, finite reward components, and object-loss/reference-end boolean semantics. Use small Torch tensors and no simulator imports in these tests.

- [ ] **Step 2: Run the tests and verify they fail**

  Run: `pytest -q tests/test_allegro_hand_track_math.py`.

  Expected: missing helper/import failure.

- [ ] **Step 3: Implement DirectRLEnv scene and state buffers**

  In `__init__`, load one `GrabReference`, allocate reference indices, residual targets, previous actions, success/failure flags, episode counters, and joint/body index maps. In `_setup_scene`, add the robot, object, ground plane if required by the selected asset, and clone environments. Fail if the loaded articulation does not satisfy the asset adapter contract.

- [ ] **Step 4: Implement cumulative residual action application**

  In `_pre_physics_step`, clip incoming actions to `[-1, 1]`, update residuals with `scale * control_dt`, clamp residuals to configured limits, and cache the current reference frame. In `_apply_action`, add residuals to the current reference target, clamp finger targets to joint limits, and send finger position targets through Isaac Lab. Apply six root channels using the asset's free-root pose target API; if the installed articulation API requires a pose write instead of a target API, isolate that conversion in one method and keep the 22-D public contract unchanged.

- [ ] **Step 5: Implement observations and diagnostics**

  Concatenate fields in the exact contract order: joint position, joint velocity, global hand pose/velocity, object pose/velocity, current reference residual/target, object-goal delta, and previous action. Assert the resulting last dimension equals `cfg.observation_space`. Populate `extras["log"]` with per-component means and reference index/end flags every step.

- [ ] **Step 6: Implement rewards, dones, and reset**

  Compute hand/reference tracking, object position/orientation, palm and fingertip distances, action norm, residual norm, success, and object-loss terms. Mark reference end on clamped final-frame timeout, object loss on the configured distance/height condition, and reset all related buffers in `_reset_idx`. Keep success separate from timeout so TensorBoard can distinguish them.

- [ ] **Step 7: Run pure tests and compile**

  Run: `pytest -q tests/test_allegro_hand_track_math.py tests/test_allegro_hand_track_reference_loader.py` and `python -m compileall source/dexteroushands_lab/tasks/direct/allegro_hand_track`.

  Expected: helper and loader tests pass; compile succeeds. Commit with `git add source/dexteroushands_lab/tasks/direct/allegro_hand_track tests/test_allegro_hand_track_math.py && git commit -m "feat: implement Allegro tracking DirectRLEnv"`.

### Task 5: Add task registration and RSL-RL PPO configuration

**Files:**
- Create: `source/dexteroushands_lab/tasks/direct/allegro_hand_track/agents/__init__.py`
- Create: `source/dexteroushands_lab/tasks/direct/allegro_hand_track/agents/rsl_rl_ppo_cfg.py`
- Create: `tests/test_allegro_hand_track_registration.py`
- Modify: `source/dexteroushands_lab/tasks/direct/__init__.py`
- Modify: `source/dexteroushands_lab/tasks/direct/allegro_hand_track/__init__.py`

**Interfaces:**
- `AllegroHandTrackPPORunnerCfg(RslRlOnPolicyRunnerCfg)` with actor/critic observation groups, MLP dimensions, rollout length, PPO coefficients, and experiment name.
- `register_tasks()` imports the new package and calls its `register_task()` exactly like existing Direct tasks.

- [ ] **Step 1: Write registration/config tests**

  Add a pure Gymnasium test that calls `dexteroushands_lab.tasks.direct.register_tasks()`, asserts the task ID exists, and reads the `rsl_rl_cfg_entry_point` string. Add a config test that checks actor/critic groups reference `policy` and the configured experiment name begins with `dexteroushands_allegro_hand_track_direct`.

- [ ] **Step 2: Run the tests and verify registration is absent**

  Run: `pytest -q tests/test_allegro_hand_track_registration.py`.

  Expected: task ID assertion fails until the import and registration changes are made.

- [ ] **Step 3: Implement the RSL-RL config and registration**

  Mirror the existing WUJI agent config with one `policy` actor group and one `critic` group (same dimensions for version 1), standard Gaussian MLP actor, value MLP critic, and conservative PPO defaults suitable for the first smoke run. Keep `max_iterations` and `num_steps_per_env` configurable through the existing CLI override path. Add the package to `direct.register_tasks()`.

- [ ] **Step 4: Run registration tests and commit**

  Run: `pytest -q tests/test_allegro_hand_track_registration.py tests/test_allegro_hand_track_config.py`.

  Expected: task ID and agent entry point are present; commit with `git add source/dexteroushands_lab/tasks/direct source/dexteroushands_lab/tasks/direct/allegro_hand_track/agents tests/test_allegro_hand_track_registration.py && git commit -m "feat: register Allegro tracking PPO task"`.

### Task 6: Add Isaac Sim smoke and zero-action validation

**Files:**
- Create: `scripts/environments/smoke_allegro_hand_track.py`
- Create: `tests/test_allegro_hand_track_smoke_contract.py`

**Interfaces:**
- CLI script accepts `--task DexterousHands-Allegro-Hand-Track-Direct-v0`, `--num_envs 1`, `--steps 8`, and `--headless` through the existing AppLauncher conventions.
- Smoke output reports observation/action shapes, finite-value checks, resolved joint/body names, reference indices, and termination flags.

- [ ] **Step 1: Write the static smoke-contract test**

  Test the smoke script's argument parser and a pure `assert_finite_step_tensors` helper with finite and NaN inputs. Do not create an Isaac application in the unit test.

- [ ] **Step 2: Implement the smoke script**

  Launch AppLauncher only when the script is executed, register tasks, make one or two environments, call `reset(seed=0)`, take zero actions for the requested steps, and assert finite observations/rewards. Print the first and last reference index and the action/observation dimensions. Exit nonzero on asset, mapping, or tensor failure.

- [ ] **Step 3: Run static checks**

  Run: `pytest -q tests/test_allegro_hand_track_smoke_contract.py` and `python -m compileall scripts/environments/smoke_allegro_hand_track.py`.

  Expected: static checks pass without starting Isaac Sim.

- [ ] **Step 4: Run the approved simulator smoke**

  With `ALLEGRO_DESCRIPTION_ROOT` and `DEXTRACK_DATA_ROOT` set, run the repository's Isaac Python launcher in headless mode with `--num_envs 1 --steps 8`. Confirm reset/step tensors are finite, the task exposes six free-root channels plus sixteen finger joints, and zero action does not change the residual buffer except through reference-frame progression.

- [ ] **Step 5: Commit**

  Commit with `git add scripts/environments/smoke_allegro_hand_track.py tests/test_allegro_hand_track_smoke_contract.py && git commit -m "test: add Allegro tracking Isaac smoke"`.

### Task 7: Run short PPO and verify TensorBoard output

**Files:**
- Create: `tests/test_allegro_hand_track_tensorboard.py`
- Modify: `docs/migration-plan.md` with the completed task milestone, preserving existing historical notes

**Interfaces:**
- Existing `scripts/rsl_rl/train.py` invocation remains the only training entry point.
- Expected run artifacts: `params/env.yaml`, `params/agent.yaml`, `model_*.pt`, policy metadata, and TensorBoard event files under `logs/rsl_rl/<experiment>/<timestamp>`.

- [ ] **Step 1: Write event-file verification logic**

  Add a test/helper that accepts a run directory, finds an `events.out.tfevents.*` file, reads scalar tags with TensorBoard's event accumulator, and asserts the required tags include `tracking/hand_error`, `tracking/object_pos_error`, `control/action_norm`, `tracking/success`, and at least one termination tag.

- [ ] **Step 2: Run the verifier against a missing directory**

  Run: `pytest -q tests/test_allegro_hand_track_tensorboard.py`.

  Expected: the test reports the required run directory/event file is absent; mark the test as an integration test or skip when `DEXTRACK_RUN_DIR` is unset so normal unit test runs stay simulator-free.

- [ ] **Step 3: Execute a short RSL-RL run**

  Use the existing launcher with `--task DexterousHands-Allegro-Hand-Track-Direct-v0`, `--num_envs 1` or the smallest stable vector count, and `--max_iterations 2` in headless mode. Do not change the logger; let `OnPolicyRunner` create the event file.

- [ ] **Step 4: Verify artifacts and scalars**

  Run the TensorBoard verifier against the printed run directory. Confirm checkpoint save/load, YAML snapshots, policy metadata, episode return/length, all requested tracking/control tags, and reference-end/object-loss diagnostics.

- [ ] **Step 5: Commit and update migration status**

  Commit with `git add tests/test_allegro_hand_track_tensorboard.py docs/migration-plan.md && git commit -m "test: verify Allegro PPO TensorBoard run"`.

## Self-Review Checklist

- [ ] Data schema, quaternion handling, frame clamping, and joint mapping are covered by Task 1.
- [ ] External path resolution, free-root requirements, body names, and joint limits are covered by Task 2.
- [ ] Direct task configuration and task registration are covered by Tasks 3 and 5.
- [ ] Cumulative residual action semantics, observation order, reward components, extras, reset, and termination are covered by Task 4.
- [ ] RSL-RL PPO configuration and existing TensorBoard runner path are covered by Tasks 5 and 7.
- [ ] Static tests run without Isaac Sim; simulator checks are isolated to Task 6 and Task 7.
- [ ] No task depends on TACO, LEAP/Franka, generalist sampling, vision, forecasting, latent/teacher branches, chunking, or curriculum.
- [ ] All paths and interfaces use concrete names; no `TODO`, `TBD`, or unspecified fallback remains.
