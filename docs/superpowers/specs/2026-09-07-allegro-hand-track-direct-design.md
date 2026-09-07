# Allegro Hand Tracking DirectRLEnv Design

## Status

Approved initial scope: single-trajectory GRAB Allegro/Fly task using
`DirectRLEnv`, RSL-RL PPO, and TensorBoard-compatible runner logging.

## Objective

Migrate the first reproducible slice of the legacy DexTrack teleoperation task
into DexGym without preserving the old Isaac Gym runtime. The result is a
registered Isaac Lab task named
`DexterousHands-Allegro-Hand-Track-Direct-v0` that can reset, step, expose a
stable policy contract, run a short RSL-RL PPO update, and emit diagnostic
scalars to the existing TensorBoard log directory.

The first slice deliberately excludes TACO, LEAP/Franka, multi-trajectory
generalist sampling, vision, forecasting, latent/teacher branches, chunking,
and curriculum logic. Those features are follow-up contracts, not hidden
flags in the first task.

## Alternatives and decision

### DirectRLEnv (selected)

The task owns reference-frame progress, cumulative residual targets, reset
semantics, and reward decomposition in one explicit lifecycle. This matches
the legacy behavior closely and makes open-loop trace tests possible before
training.

### ManagerBasedRLEnv (deferred)

Managers are useful for a later arm-hand composition, but splitting reference
progress and accumulated targets across terms would make the first port harder
to audit. Shared terms can be extracted after the Direct task contract is
stable.

### Compatibility wrapper (rejected)

Wrapping the old Isaac Gym task would retain obsolete tensor and simulator
assumptions and make RSL-RL integration fragile. The migration will translate
the behavior into Isaac Lab APIs instead.

## Runtime and external inputs

The task does not contain machine-specific absolute paths. Two external roots
are resolved by configuration/environment injection:

- `DEXTRACK_DATA_ROOT`: extracted GRAB reference data, for example a directory
  containing the `data/*.npy` scalar dictionaries;
- `ALLEGRO_DESCRIPTION_ROOT`: the imported Allegro Fly hand asset and its
  metadata.

The downloaded OneDrive archives remain user data. A preparation step may
extract the selected GRAB archive into a project-local data directory, but
the repository must not vendor the archive or generated object meshes.

The hand asset adapter requires an Isaac-compatible USD generated from the
legacy Fly Allegro description (or an equivalent existing asset). It must be
a free-root articulation exposing six root pose degrees of freedom and the
sixteen Allegro finger joints. A fixed-base asset is not an implicit fallback,
because it would invalidate the legacy 22-dimensional action meaning.

## Package layout

```text
source/dexteroushands_lab/assets/allegro_hand.py
source/dexteroushands_lab/tasks/direct/allegro_hand_track/
├── __init__.py
├── allegro_hand_track_env.py
├── allegro_hand_track_env_cfg.py
├── reference_loader.py
├── task_contract.py
└── agents/
    ├── __init__.py
    └── rsl_rl_ppo_cfg.py
```

`assets/allegro_hand.py` owns asset resolution, joint/body names, initial
state, limits, articulation configuration, and free-root validation. The task
package owns data interpretation and RL behavior. Training scripts remain
thin orchestration layers.

## Data contract

`reference_loader.py` loads one scalar `.npy` dictionary and validates the
following required fields:

```text
robot_delta_states_weights_np: [T, 22]
object_transl:               [T, 3]
object_rot_quat:             [T, 4], xyzw
```

The loader rejects missing keys, non-finite values, invalid frame lengths, and
quaternions with unusable norms. Quaternions are normalized after validation
and remain in `xyzw` order at the task boundary. Frame lookup is integer based
and clamps to the final frame, so a timeout cannot index beyond the sample.

The loader exposes torch tensors on the environment device and keeps the
source-to-Allegro joint mapping in one named, tested function. The mapping is
defined from the actual Allegro joint names; the historical DexTrack slice
permutation is not copied without a test. A unit test uses one real downloaded
sample and asserts the resulting shape, finite values, quaternion norm, and
known joint-name order.

## DirectRLEnv lifecycle

`allegro_hand_track_env.py` implements the repository Direct task contract:

```text
_setup_scene()
_pre_physics_step(actions)
_apply_action()
_get_observations()
_get_rewards()
_get_dones()
_reset_idx(env_ids)
```

Each vectorized environment starts at a reference frame and advances one
frame per configured control step. The reference index, accumulated residual,
previous action, success state, and failure state are reset together. Reset
must be deterministic for a fixed seed and sequence.

## Action semantics

The policy emits a 22-dimensional action in `[-1, 1]`. The task owns clipping,
per-dimension scaling, timestep multiplication, residual accumulation, and
limit enforcement. The first implementation follows the legacy kinematic-bias
mode:

```text
delta_target += action_scale * dt * action
target       = reference_frame_target + delta_target
```

The six root dimensions are applied to the free root pose target and the
sixteen finger dimensions to joint position targets. Finger targets are
clamped to the asset limits. The action application is the only place that
converts policy values into simulator targets; training scripts do not repeat
this logic.

## Observation contract

Version 1 uses one `policy` observation group with a fixed, documented field
order:

1. Allegro joint position;
2. Allegro joint velocity;
3. global hand pose and velocity;
4. object pose and velocity;
5. current reference hand target or residual;
6. object-to-goal position delta;
7. previous action.

The exact flattened dimension and normalization ranges are constants in
`task_contract.py` and are saved with the training configuration. The task may
expose a future privileged `critic` group, but version 1 uses the same stable
policy contract for actor and critic. Adding a field is a contract version
change, not an unnoticed tensor concatenation.

## Reward and termination

The reward is assembled from named components rather than a monolithic copied
expression:

```text
tracking/hand_error
tracking/object_pos_error
tracking/object_rot_error
tracking/palm_distance
tracking/fingertip_distance
tracking/success
control/action_norm
control/residual_norm
termination/reference_end
termination/object_lost
```

The initial total uses hand/reference tracking, object position tracking,
contact-distance shaping, action smoothness, and failure penalties. Lift and
goal bonuses, velocity tracking, orientation weighting, and curriculum remain
explicit follow-up changes. Termination covers reference timeout and clear
object/hand loss conditions; success is recorded separately from timeout.

Every component is reduced over active environments and published through the
standard environment extras. This keeps reward balancing inspectable in
TensorBoard without adding a second logging framework.

## RSL-RL and TensorBoard

`agents/rsl_rl_ppo_cfg.py` follows the existing Direct task agent config and
uses `RslRlOnPolicyRunnerCfg` plus `RslRlPpoAlgorithmCfg`. The existing
`scripts/rsl_rl/train.py` remains the entry point. Runs use the repository
convention:

```text
logs/rsl_rl/<experiment>/<timestamp>
```

The task emits standard `extras["log"]` values; the RSL-RL runner writes the
TensorBoard event files. No WandB dependency and no custom logger are part of
this migration.

## Failure handling and diagnostics

Configuration fails early if either external root is absent, the asset does
not expose the expected six root plus sixteen joint degrees of freedom, or a
reference sample violates the data contract. Runtime checks assert finite
observations, rewards, and target tensors after reset and step. Reference
frame overflow is clamped and recorded as `termination/reference_end` rather
than silently wrapping to frame zero.

The asset preflight reports resolved joint and body names before any training
run. The task does not silently substitute a fixed-base Allegro asset or a
different joint permutation.

## Verification gates

1. Pure-Python reference loader tests using a real GRAB sample;
2. source-to-Allegro joint-name and quaternion contract tests;
3. task registration and observation/action shape checks;
4. Isaac Sim reset/step smoke with one or two environments;
5. zero-action open-loop test for reference-frame and residual behavior;
6. finite-value and termination diagnostics;
7. short RSL-RL PPO update and checkpoint save/load;
8. TensorBoard event/scalar verification for episode and component metrics.

Only after these gates pass will multi-trajectory sampling, other datasets,
additional observation branches, and curriculum be designed as separate
contract changes.

## Acceptance criteria

The first migration is accepted when the named task registers through the
normal DexGym entry point, a real downloaded GRAB sample drives a vectorized
free-root Allegro scene through reset and step, all contract tensors are
finite and have documented dimensions, zero-action behavior matches the
reference bias semantics, a short RSL-RL PPO run saves a checkpoint, and the
run directory contains TensorBoard scalars for return, tracking errors,
control norms, success, and termination reasons.
