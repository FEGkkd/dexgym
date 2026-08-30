# Architecture Brief

## System boundary

The project has three boundaries:

1. external legacy implementations used as migration references;
2. this repository's Isaac Lab GPU training and export stack;
3. `sim2sim` (deferred): future adapters and tests that compare Isaac Lab with MuJoCo.

The current acceptance path ends at Isaac Sim/PhysX policy training, checkpointing, and export.

Real-robot deployment is a later consumer of the exported policy and is not
embedded inside the simulation task implementation.

## Main components

### Assets

Asset modules own USD/URDF/MJCF paths, articulation configuration, actuator
settings, joint-name mappings, body-name mappings, default states, and asset
validation helpers. Wuji assets are referenced from the local description
repositories through explicit configuration or environment variables.

### Controllers

Controllers convert normalized policy actions into simulator commands. The
initial mode is joint-position target control. Torque control and task-space
control are separate controllers and must not change the observation contract
implicitly.

### Direct tasks

Each direct task is self-contained. It owns the scene handles and the complete
physics-step lifecycle. This is the authoritative form for porting old
DexterousHands tasks and for behavior comparison against the old code.

### Manager-Based tasks

Manager-Based tasks own configuration composition while MDP modules own terms.
The task configuration selects the robot, object, action terms, observation
groups, reward terms, termination terms, event randomization, and curriculum.

### Training scripts

Training and playback scripts are thin orchestration layers. They import the
extension to register tasks, parse the task registry, construct the environment,
wrap it for the selected RL library, save configs, and export policies. They do
not contain reward or robot-specific logic.

### Validation

Validation is split into static checks, CPU/Python contract tests, Isaac Sim
scene smoke tests, short GPU training tests, checkpoint tests, and sim-to-sim
trace tests.

## Data flow

```text
policy action [-1, 1]
        |
        v
controller/action manager
        |
        v
joint target or torque command
        |
        v
Isaac Lab physics
        |
        +--> observations.policy
        +--> observations.critic (optional)
        +--> rewards
        +--> terminations
        +--> extras/log
```

The same semantic fields are reserved for a future MuJoCo adapter. Flattening
order is defined by the data contract, not by incidental dictionary ordering.

## DirectTask implementation rule

The minimum direct task template is:

```python
class Task(DirectRLEnv):
    def _setup_scene(self): ...
    def _pre_physics_step(self, actions): ...
    def _apply_action(self): ...
    def _get_observations(self): ...
    def _get_rewards(self): ...
    def _get_dones(self): ...
    def _reset_idx(self, env_ids): ...
```

The task must keep action processing, observations, rewards, and reset state
in named methods so each part can be tested independently.

## Manager-Based implementation rule

The manager-based task must have a local task package and local `mdp/` package,
even when the first baseline temporarily reuses upstream Isaac Lab terms. New
Wuji-specific terms are added locally and tested without modifying Isaac Lab.

## Failure containment

- Legacy Isaac Gym code is not imported by Isaac Lab tasks.
- Real-robot code is not imported by simulation tasks.
- Absolute paths are prohibited in task logic.
- Each run and export directory includes the task ID, data-contract version, action schema, observation schema, and normalization metadata.
- Unsupported task features fail at registration or configuration time.
