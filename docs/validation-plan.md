# Validation Plan

## Static gates

- Python compilation of the extension and scripts;
- package structure check;
- task registry check;
- no Isaac Gym imports in `isaaclab/source`;
- no absolute `/home/...` paths in task code.

## DirectTask gates

For every DirectTask:

1. construct the environment with one or two environments;
2. call reset;
3. verify observation keys and dimensions;
4. apply zero and random actions;
5. verify finite observations, rewards, and done flags;
6. force or wait for timeout and verify reset;
7. verify action targets stay within configured limits;
8. verify reset only changes requested environment IDs.

## Manager-Based gates

- each MDP term returns the documented shape;
- action manager processed actions match the action schema;
- observation groups contain only finite tensors;
- reward and termination terms can be disabled independently;
- event randomization does not change tensor dimensions;
- play configuration disables training-only corruption and randomization.

## Training gates

- short GPU run completes at least one PPO update;
- `env.yaml` and `agent.yaml` are written;
- `model_*.pt` is written;
- play loads the checkpoint;
- TorchScript and ONNX export complete;
- exported metadata matches the task contract;
- JIT and ONNX files pass offline shape/graph validation (207 -> 20 for Wuji).

## Wuji contact-physics gates

These gates establish that the Isaac/PhysX model can support contact-rich
task development. They do not require a scripted pose or an untrained policy
to complete a grasp.

1. the open hand has no probe contact or initial penetration;
2. each of the five distal bodies independently reports object contact;
3. all hand collision prims resolve to the configured WUJI physics material;
4. normal and friction force tensors are finite and mapped to the correct body;
5. an applied subcritical tangential load produces bounded displacement;
6. a load above the configured friction coefficient produces more motion;
7. sliding friction opposes the applied load and remains within `mu + 0.02`;
8. object speed remains below the fixture stability bound;
9. contact-physics pass/fail is separate from grasp/lift outcome pass/fail.

The deterministic fixture and evidence format are documented in
`wuji-contact-friction-validation.md`.

## Sim-to-sim gates (deferred)

These checks are retained for a later milestone and are not required for the
current Isaac-only Wuji training gate.

- action vector names and order match;
- observation vector names and order match;
- normalization matches;
- quaternion convention matches;
- fixed action trace is finite in both simulators;
- policy output shape matches;
- numeric tolerance is recorded separately for kinematics and contact-rich
  dynamics.

## Evidence directory

Each validated run stores:

```text
logs/<rl_library>/<experiment>/<run>/
├── params/env.yaml
├── params/agent.yaml
├── policy_metadata.yaml
├── observation_schema.yaml
├── action_schema.yaml
├── model_*.pt
├── exported/
└── validation/
```
