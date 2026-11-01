# Phase 1B observation and action schema

The task uses one 52D actor output in this exact order: left wrist translation
(3), left wrist rotation (3), left finger target increments (20), right wrist
translation (3), right wrist rotation (3), right finger target increments (20).
Raw Gaussian actions are logged, nonfinite elements are replaced with zero,
and finite actions are clipped to `[-1, 1]` before target accumulation. Each
wrist translation component is scaled by 0.5 mm per 50 Hz action, each wrist
rotation component by 0.02 rad, and each finger target increment by 0.01 rad.
Accumulated finger targets respect official USD joint limits. The ideal
`pose_command_wrist` controls each free wrist; both sets of finger links retain
their USD masses, inertias, self collisions, drives, and 2 N·m limits.

The 277D actor observation uses joint positions, velocities, target errors,
wrist pose, target error and velocity, fingertip positions relative to the
target object, object pose and velocity relative to the hands, cap pose
relative to the bottle, observed opening progress, remaining angle, observed
bottle height change, and previous clipped action. Bottle and cap true poses
and velocities are **Oracle Object State** in this simulation phase. Axial
progress comes from visible cap and bottle pose geometry. The actor receives
no contact force, normal impulse, friction force, internal thread stick/slip
state, or physical parameter values.

The 421D critic observation concatenates the 277D actor observation with
privileged per-object hand contact forces, finger applied torques, thread
stick/slip and breakaway limits, object masses, weighted fingertip slip, and
full body velocity. `task_contract.py` fixes the field widths; `_get_observations`
checks every field shape at runtime and both 1/2/16/1024-environment smoke
probes check finite complete tensors.

The environment records raw and clipped actions separately, per-hand target
clipping, wrist workspace clipping, actual wrist speed, grouped PPO Gaussian
standard deviations, KL, clip fraction, entropy and value diagnostics. The
Gaussian standard deviation starts at 1.0 and remains learnable.
