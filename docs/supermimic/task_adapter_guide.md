# Task adapter guide

The common `SuperMimicTrainer` receives a collector, replay, robot adapter, task adapter, world model, Actor, Critic, optimizers and trust evaluator. It contains no Wave path or Marvin TCP names.

`RobotDynamicsAdapter` defines state and action dimensions, versioned command mapping, Newton physical-state extraction, interpolated servo commands, and differentiable FK. A different controller, actuator, payload or contact state requires a new action/state schema and appropriately trained world model. The 42D Marvin model does not magically cover object-contact dynamics.

`ReferenceProvider` must expose a stable motion ID, length, `reference_at(frame)` for the state's current time, and `future_goal_at(frame,count)` for the Actor's future frames. `MotionLibrary` loads multiple NPZs, preserves each clip's length and hash, and supports batch motion/frame sampling. A task's arrival reward and termination use `reference_at(frame+1)`. Imagination begins only when the whole planned horizon stays inside a clip.

`TaskMDPAdapter` provides future goals, reward/soft termination through `imagine`, hard termination through Newton, and TD(λ) critic features. The current concrete `MarvinTrackingTask` supports either paper multi-link reward or M4.1 TCP priority reward and analytic or learned termination. A different goal schema needs a corresponding Actor encoder, Critic features, and task termination module. Keep the shared Trainer unchanged.

To add a motion under the existing six-link tracking task, supply an NPZ with Marvin's 14 arm joint order, all 54 robot joints, identical TCP and tracked-link definitions, per-frame time/FPS and the link kinematics arrays loaded by `MotionManager`. Record its SHA256 and provenance. The supplied `procedural_bimanual_reach_v1.npz` is labeled `procedural_test`; it is not human motion capture. Validate the reference first through full Newton Reference PD playback, then train the common entry point.

The initial M4.2 second task changes the reference trajectory and action contract (`robot_fixed_v3`) while reusing `SuperMimicTrainer`. Code reuse and one successful procedural training seed do not prove task generalization. Training a new goal type or contact task remains a separate validation gate.
