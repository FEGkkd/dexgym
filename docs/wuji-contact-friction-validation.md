# WUJI Contact and Friction Validation

## Decision

The native WUJI right-hand USD can now be used as the physics asset for
fixed-base contact-rich task development in Isaac Sim and PhysX. The official
USD is loaded directly; it is not regenerated from the URDF.

The task adds a stage-local PhysX material because the official USD collision
prototypes do not author a physics material. The material uses the official
WUJI hand contact friction (`static_friction=0.7`, `dynamic_friction=0.7`,
`restitution=0`, friction combine `max`). The object keeps the official cube
friction `0.3`, so the effective pair coefficient is `0.7`. The source USD
file is never modified.

This is a simulator-readiness result. It establishes the configured Isaac Sim / PhysX
contact behavior only; it does not claim real-hardware equivalence or a small
sim-to-real gap.

## Official USD Runtime Facts

The runtime audit for
`$WUJI_DESCRIPTION_ROOT/hand/body/usd/right/wujihand.usd` reports:

| Property | Official USD runtime value |
| --- | ---: |
| Rigid bodies | 26 |
| Revolute joints | 20 |
| Collision prims | 26 |
| Hand mass sum | 0.518060 kg |
| Native filtered-pair entries | 10 |
| Resolved hand physics material | `/World/Physics/WujiHandPhysicsMaterial` |
| Effective hand/object friction | 0.7 |

All 26 collision prims resolve the stage-local material through the
non-instance rigid-body parents. This is necessary because the collision
subtrees are instance proxies.

## Test Design

1. Environment smoke constructs the DirectTask with one and two environments,
   checks the 207 policy and 413 critic observations, applies actions, and
   verifies the official body/joint/material audit.
2. Five 20 mm fingertip probes independently test thumb, index, middle, ring,
   and pinky contact. The fingertip calibration uses the measured offset from
   each official collision mesh center to its `tip_link` body frame.
3. Each friction probe establishes preload, applies a subcritical tangential
   load (`0.20 * reference normal force`), then applies a supercritical load
   (`0.90 * reference normal force`) on a one-dimensional rail.
4. Fixed-base single, dual, multi, and lift scripts report contact physics
   separately from whether the open-loop pose forms a stable grasp.

The rail is a measurement fixture. It isolates normal/friction response and is
not a grasp controller.

## Measured Evidence

All five official-USD friction probes passed every hard gate: zero initial
contact, preload contact, static friction reporting, friction-cone bound,
bounded static displacement, observed supercritical sliding, opposing sliding
friction, and finite speed.

| Finger | Peak preload normal force | Max friction ratio | Static displacement increment | Sliding displacement increment |
| --- | ---: | ---: | ---: | ---: |
| Thumb | 1.636 N | 0.69940 | 0.016 mm | 8.647 mm |
| Index | 0.841 N | 0.70000 | 0.071 mm | 11.904 mm |
| Middle | 1.042 N | 0.70000 | 0.299 mm | 16.831 mm |
| Ring | 1.553 N | 0.70000 | 0.251 mm | 6.972 mm |
| Pinky | 1.063 N | 0.70000 | 0.177 mm | 8.011 mm |

The maximum accepted ratio is `0.72` (`mu + 0.02`). The final sliding phase
reported opposing friction for all five probes.

The single-finger scripted contact runs also passed for all five fingers. They
reported the expected fingertip body, no undesired palm/intermediate contact,
and a maximum friction ratio between `0.6992` and `0.7000`.

The dual and multi scripts passed their common contact-physics gate and
reported finite forces with a maximum friction ratio near `0.70`, but their
fixed open-loop pose did not produce the required number of fingertip contacts.
The 54 mm cube contacts intermediate official USD link bodies first. Their
`grasp_outcome_pass` and fingertip coverage fields are therefore false and are
not evidence of a learned or complete grasp.

The lift script is also not a successful grasp result. Its current open-loop
pose does not establish a stable fingertip envelope, so lift outcome remains
open. The lift fixture was corrected to hold the object kinematic during
closure and switch to dynamic only when the post-closure gravity-equivalent
load is applied.

Evidence files:

```text
artifacts/wuji_contact_friction_validation/official_thumb_v2/
artifacts/wuji_contact_friction_validation/official_index/
artifacts/wuji_contact_friction_validation/official_middle/
artifacts/wuji_contact_friction_validation/official_ring/
artifacts/wuji_contact_friction_validation/official_pinky/
artifacts/wuji_grasp_validation/official_single_<finger>/
artifacts/wuji_grasp_validation/official_dual/
artifacts/wuji_grasp_validation/official_multi/
artifacts/wuji_grasp_validation/official_lift_v3/
```

## PPO Chain

The official USD task completed a short RSL-RL PPO run with 8 environments and
2 iterations (640 collected transitions). The run had finite observations,
rewards, and updates, and produced:

```text
logs/rsl_rl/dexteroushands_wuji_hand_reorientation_direct_isaac/2026-08-29_15-45-20_official_usd_short/
```

That directory contains `model_0.pt`, `model_1.pt`, `params/env.yaml`,
`params/agent.yaml`, policy/action/observation metadata, and exported
`policy.pt` and `policy.onnx`. The checkpoint was loaded in play mode for 3
steps and both exports were generated successfully. This is a training-chain
smoke test, not task-success evidence.

## What Is Established

- the official native USD loads as a fixed-base 20-DOF articulation;
- official joint names, limits, drives, masses, and body mapping are checked at runtime;
- all official collision bodies are contact-sensor addressable;
- the stage-local WUJI hand material resolves through the instance hierarchy;
- force and friction tensors are finite and mapped to the correct articulation body;
- all five fingertip geometries exhibit the configured Coulomb friction behavior;
- vectorized environment cloning, PPO updates, checkpoint loading, and policy export work.

These results are sufficient to start implementing WUJI fixed-base grasp
rewards, resets, curricula, and longer PPO training in Isaac Lab.

## Remaining Fidelity Work

The native USD is the sole Isaac Sim asset source. The official USD does not
author a hand physics material on its collision prototypes, so the task applies
a stage-local PhysX material with the validated hand coefficient `0.7`; the
source USD remains unchanged.

The remaining work is Isaac-side system identification and task validation:
contact geometry review, material calibration, disturbance and domain
randomization, and task-level grasp/reorientation success. The contact probe
validates the configured force direction and friction-cone behavior; it does not
establish exact real-hardware force magnitudes or sim-to-real performance.

## Reproduction

```bash
export WUJI_DESCRIPTION_ROOT=/path/to/wuji-description
./scripts/smoke.sh \
  --task DexterousHands-Wuji-Hand-Reorientation-Direct-v0 \
  --num_envs 2 --steps 3

./scripts/run_isaaclab.sh \
  scripts/environments/wuji_contact_friction_validation.py \
  --finger index \
  --output artifacts/wuji_contact_friction_validation/official_index

./scripts/train.sh \
  --task DexterousHands-Wuji-Hand-Reorientation-Direct-v0 \
  --num_envs 8 --max_iterations 2 --run_name official_usd_short
```

The friction process exits nonzero if a hard contact gate fails.
`--no_fail_exit` is reserved for exploratory runs.
