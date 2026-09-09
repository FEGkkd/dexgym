# WUJI Reorientation Configurable Training Design

## Goal

Bring the Isaac Lab WUJI reorientation DirectTask into semantic alignment with
the official `wuji-mjlab` reorientation task while keeping PhysX-specific asset
and contact implementation details behind explicit task-owned adapters. Add
structured, independently configurable curriculum, domain randomization, and
disturbance settings. Keep task-local PPO configuration extensible and expose
the train/play orchestration from a dedicated top-level `rsl_rl/` package.

## Scope

The change covers the registered task
`DexterousHands-Wuji-Hand-Reorientation-Direct-v0` and its train/play entry
points. It includes task reset, observation, action history, reward, contact
semantics, termination, curriculum, physical randomization, external object
disturbances, PPO configuration, metadata, tests, and relevant documentation.

It does not convert or redistribute the WUJI robot asset. The official native
USD selected by `WUJI_DESCRIPTION_ROOT` remains the only runtime hand asset.
The project continues to use Isaac Sim/PhysX; the goal is equivalent task data
meaning and training behavior, not MuJoCo/PhysX numerical identity.

## Design

### Structured task configuration

`WujiHandReorientationDirectEnvCfg` owns nested config groups rather than
independent top-level booleans. Each group has an `enabled` field and all
parameters needed to reproduce or disable its behavior.

```python
@configclass
class WujiCurriculumCfg:
    enabled: bool = True
    success_count_threshold: int = 3
    delta_per_loop: float = 0.08
    adaptive_episode_enabled: bool = True
    adaptive_target_steps: int = 800
    adaptive_inc_rate: float = 0.1
    adaptive_dec_rate: float = 0.2
    adaptive_min_scale: float = 0.05
    adaptive_drop_gamma: float = 1.0

@configclass
class WujiDomainRandomizationCfg:
    enabled: bool = True
    object_com_range: tuple[float, float] = (-0.003, 0.003)
    object_mass_scale: tuple[float, float] = (0.4, 1.6)
    object_size_scale: tuple[float, float] = (0.85, 1.15)
    hand_friction_scale: tuple[float, float] = (0.7, 1.3)
    contact_soft_width: tuple[float, float] = (2.0, 5.0)
    contact_finger_width: tuple[float, float] = (1.0, 2.0)
    contact_dmin: tuple[float, float] = (0.5, 1.0)
    pd_stiffness_scale: tuple[float, float] = (0.75, 1.5)
    pd_damping_scale: tuple[float, float] = (0.5, 2.0)
    joint_damping_scale: tuple[float, float] = (0.3, 3.0)
    joint_armature_scale: tuple[float, float] = (0.75, 1.3)
    joint_friction_scale: tuple[float, float] = (0.5, 2.0)
    encoder_bias_range: tuple[float, float] = (-0.01, 0.01)
    link_mass_scale: tuple[float, float] = (0.4, 1.5)
    link_inertia_scale: tuple[float, float] = (0.4, 1.5)

@configclass
class WujiDisturbanceCfg:
    enabled: bool = True
    interval_s: tuple[float, float] = (0.6, 1.8)
    speed_range: tuple[float, float] = (0.05, 0.15)
    warmup_time_s: float = 3.0
    warmup_frac: float = 0.05
    rampup_frac: float = 0.80
```

The exact config class field names are part of the local task API. Defaults
match the official task's documented ranges. `enabled=False` means the group
has no state-changing effect and emits neutral critic fields: zero disturbance
state, unit domain-randomization ratios, and fixed curriculum scale.

The training configuration uses all three groups enabled by default. Play and
evaluation explicitly override them to disabled values, along with actor
observation corruption.

### Task semantic alignment

The DirectTask remains the owner of the Isaac lifecycle, but delegates pure or
backend-facing behavior to focused modules:

- `curriculum.py`: success-count and adaptive-episode curriculum updates;
- `randomization.py`: repeatable per-environment PhysX property sampling and
  application;
- `disturbance.py`: interval scheduling, warmup/ramp scaling, random direction,
  object velocity injection, and reset state;
- `contact.py`: body group resolution and
  PhysX contact aggregation.

The task must preserve the existing public policy contract:

- 20 actions in explicit official USD joint order;
- position-offset action, scale `0.5`, EMA alpha `0.5`, and `0.4 s` warmup;
- one 69-dimensional policy frame and 3-frame 207-dimensional actor input;
- 413-dimensional asymmetric critic input;
- tag-frame cube position and 6D residual orientation;
- raw previous action history, independent from processed EMA targets.

Reset semantics follow the official reorientation task: randomized root pitch,
per-joint reset ranges, randomized cube position/orientation, zero velocities,
goal sampling in the tag frame, and clearing all stateful counters. Native USD
coordinate differences are handled by the task's configured pose values and do
not change frame or field meanings.

Goal progression remains the two-phase state machine: orientation error below
`0.2 rad` for 5 control steps enters a success window; after 20 window steps a
new random goal is sampled. Success does not terminate the episode.

Reward terms retain the official weights and meanings: orientation alignment,
hand pose, first/second-order raw action rate, normalized torque, cage escape,
tip sliding, palm-detach, hold escalation, and finger self-collision. PhysX
contact groups map the official semantic groups explicitly: palm/proximal,
distal link bodies, fingertip bodies, and finger-finger collision pairs. The
implementation may use native USD body names and contact tensors, but it must
expose the same per-environment binary/contact quantities to reward code.

Termination remains timeout plus cage-drop counter. Object height is not a
separate termination unless it is explicitly added to the official task
contract; unused legacy fields must not silently alter behavior.

### PhysX randomization adapter

The randomization module samples from default asset properties on every reset,
then applies indexed values to the selected environment IDs. It uses Isaac Lab
and PhysX APIs where supported:

- object and link mass/inertia/CoM through indexed asset setters;
- actuator stiffness/damping through indexed actuator setters;
- joint damping, armature, and friction through indexed joint setters;
- encoder bias in the observation path;
- object/hand collision material and contact parameters through a PhysX
  material adapter that records the effective sampled value.

The adapter must validate backend capabilities at environment construction. A
capability that cannot be applied on the selected OVPhysX runtime is reported
as a clear configuration/runtime error or as an explicitly recorded unsupported
field; it must never populate critic randomization values as if the change had
been applied. Supported fields are written to a per-environment state buffer
used by the 6-dimensional `dr_params` critic term and logged in `extras`.

Startup-only fields are sampled once after scene creation. Reset-time fields
are resampled for the requested environment IDs only. When the group is
disabled, all properties return to their configured defaults and state buffers
are neutral.

### Disturbance adapter

The disturbance module owns a per-environment next-trigger timer and cached
direction/velocity tensors. After the 3-second warmup, each trigger samples a
uniform speed between the configured minimum and a progress-scaled maximum,
chooses a normalized random world direction, and adds that linear velocity to
the cube. The schedule uses the configured interval range and training
progress/ramp values from the official task. Reset clears all caches and
reschedules the environment.

`perturbation_direction` and `perturbation_velocity` in the critic are the
actual most-recent applied disturbance values. With disturbance disabled they
are zero. Disturbances are not exposed to the actor policy.

### Curriculum adapter

The curriculum module stores per-environment and global training state without
changing observation dimensions. It ports the official success curriculum and
adaptive episode schedule. Curriculum updates are gated by
`cfg.curriculum.enabled`; individual schedules are gated by their nested
fields. The current curriculum scale and effective episode target are logged
in `extras` and represented in the critic progress fields.

No curriculum mutation occurs in play/evaluation mode. Fixed defaults are used
when the group or a schedule is disabled.

### PPO and entrypoint organization

The task's `agents/rsl_rl_ppo_cfg.py` becomes the only owner of task-specific
PPO defaults. It exposes independently subclassable config classes for actor,
critic, algorithm, and runner. The default actor and critic hidden dimensions
are `[512, 256, 128]` and `[512, 512, 256, 128]` respectively. Algorithm
fields remain editable without changing the task environment contract.

The repository gains a top-level `rsl_rl/` package with `base/train.py`,
`base/play.py`, and `cli_args.py`, following the organization used by
`soar-gym`. Existing `scripts/train.sh` and `scripts/play.sh` remain stable
compatibility wrappers and invoke the top-level entrypoints. Train/play code
only handles CLI, environment creation, runner construction, checkpointing,
export, and metadata; task semantics remain in the task package.

### Metadata and observability

Run configuration dumps include all nested curriculum, randomization, and
disturbance fields. Policy metadata continues to identify the 207/413 contract
and adds a runtime section recording enabled groups and effective control
period. Extras expose component metrics without embedding state in reward
values: success count, curriculum scale, disturbance count/speed, sampled DR
ratios, contact group counts, cage counter, and termination reason.

## Testing requirements

Simulator-independent tests must cover:

1. nested config defaults and disabled neutral behavior;
2. action history preserving raw actions while processed targets remain EMA
   filtered and limit-clipped;
3. goal hold/switch state machine and curriculum update functions;
4. cage counter and disturbance scheduling/sampling invariants;
5. randomization sampling from defaults, reset-ID isolation, and neutral
   disabled values;
6. reward contact group semantics and all reward component shapes;
7. actor/critic PPO dimensions and task-local config override behavior;
8. top-level `rsl_rl/` entrypoint structure and compatibility wrappers.

Isaac runtime tests must cover one/two environments with every group enabled
and disabled, finite observations/rewards, reset isolation, actual applied
PhysX parameter buffers, disturbance state, contact mapping, checkpoint
training, and policy export. A backend capability audit is required before
claiming domain-randomization parity.

## Source references

- Official task configuration and event groups:
  https://github.com/wuji-technology/wuji-mjlab/blob/main/src/wuji_mjlab/tasks/reorient/reorient_terms.py
- Official action processing:
  https://github.com/wuji-technology/wuji-mjlab/blob/main/src/wuji_mjlab/tasks/reorient/mdp/actions.py
- Official observations and privileged fields:
  https://github.com/wuji-technology/wuji-mjlab/blob/main/src/wuji_mjlab/tasks/reorient/mdp/observations.py
- Official goal state machine:
  https://github.com/wuji-technology/wuji-mjlab/blob/main/src/wuji_mjlab/tasks/reorient/mdp/commands.py
- Official reward terms:
  https://github.com/wuji-technology/wuji-mjlab/blob/main/src/wuji_mjlab/tasks/reorient/mdp/rewards.py
- Official WUJI reset constants:
  https://github.com/wuji-technology/wuji-mjlab/blob/main/src/wuji_mjlab/tasks/reorient/reorient_constants.py
- Isaac Lab event and asset APIs used by the PhysX adapter:
  https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/events.py

