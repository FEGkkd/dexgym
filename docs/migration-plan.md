# Migration Plan

## Phase 0: Freeze the contract

Inputs:

- legacy task files and YAML configs;
- current Isaac Lab version;
- the official native WUJI USD description;

Outputs:

- `SPEC.md`;
- architecture brief;
- data contract;
- task migration matrix;
- acceptance tests.

Gate: no implementation starts without a named task ID, asset variant, action
mode, observation schema, reset semantics, and success metric.

## Phase 1: Training architecture

Implement:

- extension package and task registration;
- DirectTask and Manager-Based task packages;
- RSL-RL train/play/export scripts;
- environment listing and asset inspection tools;
- config snapshot and run directory convention.

Gate: both task forms register and construct an Isaac Sim environment.

## Phase 2: DirectTask reference migration

First native task: `ShadowHandReOrientation`.

Implementation order:

1. port asset and joint/body mappings;
2. port action target processing;
3. port reset distribution;
4. port observation field order;
5. port reward terms;
6. port termination and success metrics;
7. compare fixed-seed traces against the old implementation where possible.

Gate: reset, one-step, finite-value, dimension, and reward component tests pass.

## Phase 3: Wuji Hand DirectTask

Implement a native Wuji Hand single-hand reorientation task in Isaac Lab
using the official pre-built WUJI USD and PhysX. The official USD is the only
Isaac asset source for joint order, articulation properties, body mapping, and
contact geometry. The task configuration owns timing, action filtering,
observations, goal generation, reset,
success, cage logic, and reward scaling. The current policy contract is
20 actions and 3 x 69 observation history = 207 inputs.

Current gate: Isaac Sim asset creation, reset, one-step smoke, vectorized
contact isolation, short PPO update, checkpoint loading, policy export,
and offline export validation. These gates pass for the current Wuji DirectTask.
The short policy smoke has not demonstrated reorientation success: success remained zero and the open-loop rollouts showed early cage escapes.
The next task-level milestone is a longer training run after validating initial
contact retention and reward balance. Contact terms must be measured and validated directly in PhysX. The native USD collision-bearing body map
must remain explicit and name-checked.

## Phase 4: Manager-Based arm-hand task

Implement the arm plus Wuji Hand task with local MDP modules. The ownership
scaffold exists under `tasks/manager_based/wuji_arm_hand/`; registration waits
for a validated composite asset and mounting transform.

- `actions.py`: arm joint position plus hand joint position actions;
- `observations.py`: arm state, hand state, end-effector/object relative state;
- `rewards.py`: reach, grasp, pose, action-rate, safety terms;
- `terminations.py`: timeout, object loss, joint-limit or safety failure;
- `events.py`: reset and domain randomization;
- `commands.py`: target object pose or reorientation goal.

Gate: the manager-based task has the same action/observation meaning as its
DirectTask counterpart for the shared fields.

## Phase 5: Sim-to-sim (deferred)

This phase is intentionally outside the current single-hand training scope.
No cross-simulator model conversion, adapter, or trace-comparison code is
included in this repository. It must be specified and validated separately if
the scope is resumed.

## Phase 6: Remaining legacy tasks

Priority order:

1. `ShadowHandReOrientation`;
2. `ShadowHandLiftUnderarm`;
3. `AllegroHandCatchUnderarm`;
4. `ShadowHandCatchUnderarm`;
5. `ShadowHandOver` and two-agent handover;
6. grasp/place and push/block tasks;
7. door, bottle-cap, kettle, scissors, pen, and switch tasks;
8. meta-learning and offline-RL tasks only after the environment contracts are stable.

Old algorithm variants are not part of the first migration gate. They can be
reintroduced after the task behavior is correct.
