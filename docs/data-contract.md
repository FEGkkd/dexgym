# Isaac Lab and MuJoCo Data Contract

Contract version: `dexteroushands-v1`

The validated Wuji Hand DirectTask uses the right Wuji Hand 2 Beta 1 asset. Its
policy action order is the MuJoCo actuator order:

```text
r_THJ0 r_THJ1 r_THJ2 r_THJ3
r_FFJ0 r_FFJ1 r_FFJ2 r_FFJ3
r_MFJ0 r_MFJ1 r_MFJ2 r_MFJ3
r_RFJ0 r_RFJ1 r_RFJ2 r_RFJ3
r_LFJ0 r_LFJ1 r_LFJ2 r_LFJ3
```

Isaac Lab maps these entries to semantic native-USD joint names at runtime.
`WUJI_DESCRIPTION_ROOT` selects the external asset checkout; the native USD is loaded at runtime and no machine-local
absolute path is embedded in the extension.

## Identity

Every run and exported policy records:

```text
task_id
contract_version
asset_variant
joint_order
action_mode
control_dt
physics_dt
decimation
observation_schema
quaternion_convention
normalization
```

## Action

The policy emits a float tensor in `[-1, 1]`.

For the first controller:

```text
policy action
  -> scale to joint limits
  -> optional exponential/moving-average filter
  -> joint position target
```

The action vector order is an explicit list of joint names. It must never be
inferred from the order returned by a simulator backend.

## Observation

The Wuji DirectTask uses one 69-dimensional frame and a three-frame
oldest-to-newest history, for a policy input dimension of 207:

```text
20  normalized joint position with uniform noise +/- 0.06
20  normalized qpos error to the processed target
 3  cube position in the tag frame, noise +/- 0.008 with p=0.02 injection
 6  cube orientation residual in tag frame as matrix last-two-rows 6D,
    Gaussian std 0.05 with p=0.02 injection
20  previous raw action
```

The action vector has 20 entries in the explicit Wuji actuator order above.
Actions are clipped to [-1, 1], scaled by 0.5 around the configured default
joint pose, clipped to joint limits, then filtered with EMA alpha 0.5 before
being sent as Isaac joint position targets. A 0.4 s warmup holds the default
target.

The history is term-major and ordered oldest to newest. The public contract
uses `wxyz` for cross-backend quaternion fields; Isaac runtime pose buffers
use the simulator-native `xyzw` layout and convert at the boundary.

## Asymmetric critic

Training returns two observation groups:

```text
policy: [num_envs, 207]  noisy actor input and deployment contract
critic: [num_envs, 413]  privileged input used only by the value network
```

The critic layout is:

```text
207  clean shared policy history (5 terms, 3 frames, term-major)
 60  true relative joint position history
 60  true relative joint velocity history
 45  scaled fingertip position history (offset z=0.5 m, scale 5)
  3  clean cube position in tag frame
  6  clean cube orientation residual
  3  cube linear velocity
  3  cube angular velocity
  3  perturbation direction (currently explicit zero placeholder)
  6  perturbation velocity (currently explicit zero placeholder)
  4  hold/window/episode progress and goal reach count
  1  cage counter progress
  6  domain-randomization ratios (currently explicit one placeholder)
  6  palm rotation 6D
```

The current Isaac migration has the complete tensor and PPO data path, but
perturbation events and physical domain randomization are not migrated yet.
The zero/one fields are explicit finite placeholders and must not be treated
as evidence of semantic parity with the official Wuji randomized training setup.

## Frames

- `world`: simulator world frame;
- `env`: per-environment frame with environment origin removed;
- `base`: robot base frame;
- `palm`: Wuji Hand palm frame;
- `ee`: arm end-effector frame;
- `object`: object root frame;
- `goal`: commanded target frame.

Relative poses must state both source and target frames in their field name or
schema metadata.

## Timing

The contract records:

```text
physics_dt = sim.dt
control_dt = physics_dt * decimation
policy_rate = 1 / control_dt
```

MuJoCo uses the same effective policy period even if its internal integration
step differs.

## Reset and termination

The environment distinguishes:

- `terminated`: physical failure or task success termination;
- `time_out`: episode horizon reached;
- `success`: task success metric, which may be sticky until reset.

These fields are recorded in `extras` and are not hidden inside reward values.

## Export

An exported policy directory contains:

```text
policy.pt or policy.onnx
policy_metadata.yaml
observation_schema.yaml
action_schema.yaml
normalization.yaml
```

The policy is invalid for deployment if the metadata does not match the
requested task ID and contract version.
