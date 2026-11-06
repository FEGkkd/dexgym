# World Model sharing contract

`NetworkFactory.build(..., robot_adapter=robot, shared_world_model=existing)` shares **weights and both World Model normalizers by object identity**. Sharing algorithm code does not imply sharing weights. Building a new model without `shared_world_model` remains independent.

The source model must have been built with an explicit Robot Adapter. The factory computes `world_model_dynamics_v1` from `RobotSpec`, the adapter's readable `checkpoint_contract()` and the World Model's hidden architecture, input/delta normalization, integration and weight semantics. For nonzero context, the adapter must declare `context_schema` and `context_model_range`; equal context dimensions alone are insufficient. Missing signatures fail closed. The signature is a hash plus readable manifest; mismatches identify the differing field.

The adapter contract must describe all transition-relevant controller and physics settings, including action origin/scale, command slew/interpolation, actuator gains/limits, joint bounds, asset and contact configuration. Marvin's `MarvinCoreRobot.checkpoint_contract()` supplies these for the pinned fixed-base arm-only Newton configuration. If a future environment varies payload, friction, contact, damping or controller parameters, expose them in Dynamics Context with a declared supported range or use a separate model. The factory cannot infer undeclared physical changes.

| Change | Share weights? |
| --- | --- |
| Same pinned Marvin robot/controller with Tracking 10×90 and Reach 1×6 goals | Yes, provided the exact adapter signature matches. |
| Different Task Reward, Goal Encoder or Critic | Yes; these are task modules, not dynamics. |
| Action scale, servo gain, physics dt, asset hash or contact behavior differs | No, unless the change is fully represented by a versioned context contract and the trained model covers its declared range. |
| Different context schema or model range | No. |
| Different World Model normalizer floor/mode, architecture or integration version | No. |
| Same dimensions but missing Robot Adapter signature | No. |

Two trainers that use the same World Model object must pass the **same World Model optimizer object**. `SuperMimicTrainer` rejects a second optimizer over shared parameters. Task-specific Actor/Critic optimizers stay separate. Learned task termination heads require optimizer ownership of their own parameters; a mixed learned/analytic multi-task optimizer scheme has not been validated and is outside this patch.

Tests: `test_m432_world_model_sharing.py`, `test_m43_marvin_plugins.py`, and `test_m432_optimizer_ownership.py`.
