# WUJI Reorientation Configurable Training Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Align the WUJI hand reorientation task with the official wuji-mjlab task semantics while adding independently configurable curriculum, domain randomization, disturbances, and extensible PPO/train-play organization.

**Architecture:** Keep the official native USD and current Isaac Sim/PhysX scene unchanged. Add task-local pure helpers for curriculum, disturbance scheduling, and randomization state; let the DirectTask own simulator calls and compose those helpers. Move shared RSL-RL orchestration into a top-level package while preserving the existing shell wrappers.

**Tech Stack:** Python 3.12, Isaac Lab 6.x, Isaac Sim/PhysX, Gymnasium, PyTorch, rsl_rl, pytest.

**Spec:** docs/superpowers/specs/2026-09-09-wuji-reorientation-configurable-training-design.md

## Global Constraints

- Keep the official WUJI native USD selected by WUJI_DESCRIPTION_ROOT; do not modify or regenerate USD assets.
- Preserve the 20-action, 69-frame, 207-actor, and 413-critic policy contract.
- Training defaults enable curriculum, domain randomization, and disturbance; play disables them and observation noise.
- Disabled groups must have no state-changing effect and must emit neutral critic fields.
- Preserve unrelated dirty worktree changes; stage only files belonging to this feature.

### Task 1: Add Structured Config And Pure Training Helpers

Files:
- Modify source/dexteroushands_lab/tasks/direct/wuji_hand_reorientation/wuji_hand_reorientation_env_cfg.py
- Create source/dexteroushands_lab/tasks/direct/wuji_hand_reorientation/curriculum.py
- Create source/dexteroushands_lab/tasks/direct/wuji_hand_reorientation/disturbance.py
- Create source/dexteroushands_lab/tasks/direct/wuji_hand_reorientation/randomization.py
- Test tests/test_wuji_reorientation_config.py
- Test tests/test_wuji_reorientation_helpers.py

Interfaces:
- WujiCurriculumCfg, WujiDomainRandomizationCfg, and WujiDisturbanceCfg are configclass values owned by WujiHandReorientationDirectEnvCfg as curriculum, domain_randomization, and disturbance.
- CurriculumState.reset(env_ids), CurriculumState.update(successes, episode_steps), and CurriculumState.as_progress() operate on tensors without simulator access.
- DisturbanceScheduler.reset(env_ids, now), DisturbanceScheduler.step(now, progress, generator), and DisturbanceScheduler.neutral() return per-environment trigger masks, directions, and velocities.
- DomainRandomizationState.reset(env_ids, defaults, generator) returns sampled per-environment ratios and neutral(num_envs, device) returns unit ratios.

- [ ] Step 1: Write failing config and helper tests. Assert the three nested groups default to enabled with explicit ranges, disabled state is neutral, curriculum updates only enabled schedules, disturbance intervals stay within configured bounds, and randomization reset changes only requested environment IDs.
- [ ] Step 2: Run the focused tests and verify they fail. Run: pytest -q tests/test_wuji_reorientation_config.py tests/test_wuji_reorientation_helpers.py. Expected: FAIL because the nested config groups and helper modules do not exist.
- [ ] Step 3: Implement the nested config groups. Add typed range fields and attach instances to WujiHandReorientationDirectEnvCfg; retain the existing scene, robot, object, contact sensor, and USD-related fields unchanged.
- [ ] Step 4: Implement pure curriculum, disturbance, and randomization state. Keep tensor shapes explicit, use a supplied generator or torch.rand on the configured device, and make disabled behavior return fixed curriculum scale, zero disturbance state, and unit DR ratios.
- [ ] Step 5: Run the focused tests and verify they pass.
- [ ] Step 6: Commit the config/helper deliverable. Stage only the listed files and commit with message feat: add configurable WUJI training helpers.

### Task 2: Integrate Official Task Semantics Into The DirectTask

Files:
- Modify source/dexteroushands_lab/tasks/direct/wuji_hand_reorientation/wuji_hand_reorientation_env.py
- Modify source/dexteroushands_lab/tasks/direct/wuji_hand_reorientation/task_contract.py
- Create source/dexteroushands_lab/tasks/direct/wuji_hand_reorientation/contact.py
- Test tests/test_wuji_reorientation_semantics.py

Interfaces:
- WujiContactGroups.from_body_names(body_names) resolves palm/proximal, distal, fingertip, and finger-pair groups from native USD names without editing the asset.
- WujiHandReorientationDirectEnv owns helper instances and calls reset/step methods from _reset_idx, _pre_physics_step, _get_observations, and _get_rewards.
- The task keeps raw actions in previous_actions and processed EMA targets in joint_targets; first- and second-order action penalties consume raw action history.

- [ ] Step 1: Write failing semantic tests covering raw action history, contact group classification, goal hold/switch counters, cage termination, and neutral critic fields when each feature group is disabled.
- [ ] Step 2: Run the semantic tests and verify they fail. Run: pytest -q tests/test_wuji_reorientation_semantics.py. Expected: FAIL for the current clamped-action history and zero-filled critic fields.
- [ ] Step 3: Add contact.py. Preserve the existing exact filtered contact sensor and map body names into explicit semantic groups; do not change USD paths, materials, or scene spawning.
- [ ] Step 4: Correct action history and reward inputs. Store incoming raw action before clipping, keep EMA/clipped targets in a separate buffer, and calculate action-rate penalties from consecutive raw actions.
- [ ] Step 5: Integrate reset and goal state. Reset helper state and sampled values for only env_ids, preserve the existing tag-frame goal representation, and retain the official 5-step success hold plus 20-step goal switch state machine.
- [ ] Step 6: Integrate curriculum, DR, and disturbance state. Apply supported indexed PhysX properties through existing Isaac Lab asset APIs, record only actually applied values in critic dr_params, inject scheduled object velocity through the rigid object API, and expose actual disturbance tensors in the critic. Unsupported backend fields must be recorded as unsupported rather than fabricated.
- [ ] Step 7: Align reward and termination semantics. Use explicit palm/proximal, distal, fingertip, sliding, detach, hold, self-collision, torque, and cage terms; keep timeout plus cage-drop termination and avoid adding an independent object-height termination.
- [ ] Step 8: Add runtime extras and metadata fields. Log curriculum scale, success count, disturbance state, applied DR ratios, contact counts, cage counter, and termination reason without changing observation dimensions.
- [ ] Step 9: Run semantic tests and the existing contract suite. Run: pytest -q tests/test_wuji_reorientation_semantics.py tests/test_data_contract.py tests/test_policy_metadata.py tests/test_structure.py.
- [ ] Step 10: Commit the task integration with message feat: align WUJI reorientation task semantics.

### Task 3: Make PPO And Train/Play Organization Extensible

Files:
- Modify source/dexteroushands_lab/tasks/direct/wuji_hand_reorientation/agents/rsl_rl_ppo_cfg.py
- Create rsl_rl/__init__.py
- Create rsl_rl/base/__init__.py
- Create rsl_rl/base/train.py
- Create rsl_rl/base/play.py
- Create rsl_rl/cli_args.py
- Modify scripts/rsl_rl/train.py
- Modify scripts/rsl_rl/play.py
- Modify scripts/train.sh
- Modify scripts/play.sh
- Test tests/test_wuji_reorientation_ppo_and_entrypoints.py

Interfaces:
- WujiHandReorientationActorCfg, WujiHandReorientationCriticCfg, WujiHandReorientationAlgorithmCfg, and WujiHandReorientationPPORunnerCfg are independently subclassable config classes.
- rsl_rl.cli_args.add_rsl_rl_args(parser) and parse_rsl_rl_cfg(task_name, args_cli) preserve current CLI behavior.
- rsl_rl.base.train.main() and rsl_rl.base.play.main() are importable entrypoints; wrappers continue to invoke them through run_isaaclab.sh.

- [ ] Step 1: Write failing PPO and entrypoint tests asserting actor [512, 256, 128], critic [512, 512, 256, 128], task-local subclass overrides, importable top-level modules, and wrapper forwarding.
- [ ] Step 2: Run the focused tests and verify they fail. Run: pytest -q tests/test_wuji_reorientation_ppo_and_entrypoints.py.
- [ ] Step 3: Split PPO defaults into actor, critic, algorithm, and runner config classes. Keep the Isaac Lab 6.x RslRlMLPModelCfg fields and the 207/413 observation group mapping.
- [ ] Step 4: Move shared train/play code into top-level rsl_rl. Keep task registration, environment parsing, checkpoint loading, export, metadata, and simulator lifecycle behavior equivalent to the existing scripts.
- [ ] Step 5: Configure play mode to disable curriculum, randomization, disturbance, and observation noise. Training keeps all three groups enabled by default.
- [ ] Step 6: Turn existing scripts into compatibility wrappers and run focused tests.
- [ ] Step 7: Commit the entrypoint deliverable with message feat: organize WUJI PPO train and play entrypoints.

### Task 4: Validate Simulator Integration And Document The Runtime Contract

Files:
- Create docs/wuji_hand_reorientation_training.md
- Create tests/test_wuji_reorientation_runtime_contract.py
- Modify docs/superpowers/specs/2026-09-09-wuji-reorientation-configurable-training-design.md only if validation clarifies a contract detail.

Interfaces:
- Runtime contract tests instantiate the registered task only when Isaac Sim is available and otherwise skip with an explicit dependency reason.
- Documentation lists training defaults, play overrides, supported PhysX randomization fields, and the unchanged USD asset boundary.

- [ ] Step 1: Write runtime contract tests for one/two environments with enabled and disabled groups, finite actor/critic observations and rewards, reset isolation, actual applied DR buffers, disturbance state, metadata, and policy export paths.
- [ ] Step 2: Run simulator-independent tests and inspect skips. Run: pytest -q tests/test_wuji_reorientation_config.py tests/test_wuji_reorientation_helpers.py tests/test_wuji_reorientation_semantics.py tests/test_wuji_reorientation_ppo_and_entrypoints.py tests/test_wuji_reorientation_runtime_contract.py.
- [ ] Step 3: Run the Isaac runtime smoke test in the project environment with two environments and short episode/step counts; verify no USD files or asset source files changed.
- [ ] Step 4: Add runtime documentation and update metadata assertions. State that asset and contact geometry remain from the official USD and unsupported OVPhysX fields are reported explicitly.
- [ ] Step 5: Run the full relevant test set and git diff --check. Run: pytest -q tests/test_wuji_reorientation_*.py tests/test_data_contract.py tests/test_policy_metadata.py tests/test_structure.py && git diff --check.
- [ ] Step 6: Commit documentation and final validation with message docs: document WUJI reorientation training contract.
