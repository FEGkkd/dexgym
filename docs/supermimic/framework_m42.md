# M4.2 SuperMimic framework

## Status

M4.2 is an experimental model based RL framework. The original M4.1 Wave actor and its data are immutable baselines. A single `SuperMimicTrainer` now alternates Newton collection, masked eight-slot world-model fitting, independent held-out model checks, actor BPTT through a frozen world model, and TD(λ) critic fitting. The same entry point accepts the validated Wave reference or a separately identified procedural test reference.

The working physical contract is Marvin's fixed base, 54 simulated joints, 14 controlled arm joints, 40 home-position fingers, 500 Hz Newton MJWarp physics, ten physics substeps per 50 Hz action, unchanged controller gains and effort limits, and collision checks by Newton world ID. The model state is `[q_arm14, qd_arm14, q_command14]`. The model predicts the first 28 state deltas and obtains the next command through the same versioned `ArmActionMap` used by Newton.

## Time and episode contract

One physical transition is `s_t, frame_t, a_t -> s_(t+1), frame_(t+1)`. The arrival frame is `frame_t+1`. Reward and termination compare the pre-reset physical `s_(t+1)` with `reference_at(frame_t+1)`. The Actor sees `future_goal_at(frame_t, 10)`, covering frames `t+1` through `t+10`. The Critic sees `s_t`, the current joint reference, and its future goal. `ReferenceTimeIndex` and `TransitionBatch` assert this relationship.

Replay records the pre-reset state, action, prior action, executed command, pre-reset next state, motion ID, episode ID, frame, terminated, truncated, and source (`actor` or `reference_pd`). It samples eight slots with a validity mask. The terminal transition itself contributes to model and BCE losses; slots after a terminal transition contribute nothing. Invalid sampled slots are zeroed before the model forward pass. Episode IDs, absolute timestamps, and frame checks prevent ring wraparound from creating false trajectories.

Imagination starts only where `start_frame+horizon < clip_length`; no implicit end hold is used. A 1, 8, or 32 step horizon is admitted according to actor-collected held-out model error. WM weights are frozen during BPTT, but state/action gradients remain live. At the 1-step bootstrap stage, the Actor omits the immature terminal Critic value. TD(λ) still trains the Critic.

## Startup modes

| Mode | Real data source | Actor initialization |
| --- | --- | --- |
| `pure_model_based` | Random Actor with configurable small exploration | Random only; reference actions are never supervised targets |
| `teacher_bootstrap_model_based` | Marked Reference PD Newton transitions followed by Actor transitions | Random only |
| `supervised_init_model_based` | Actor Newton transitions | Explicit supervised reference action initialization |

The strict Pure experiment is the clean control. Reference PD is a physical sampling strategy in the second mode, never hidden Actor supervision. The training log reports the two real transition counts separately.

## Versioned action contracts

| Schema | Origin and scale | Per-step maximum command delta | Purpose |
| --- | --- | ---: | --- |
| `legacy_wave_v1` | First Wave frame; Wave excursion plus 0.15 rad | 0.04 rad | Exact M4.1 compatibility |
| `robot_fixed_v2` | Robot home; half joint range | 0.04 rad | Preserved early M4.2 experimental contract |
| `robot_fixed_v3` | Robot home; fixed 0.25 rad arm action envelope, clipped by each joint range | 0.01 rad | Current task-independent small-motion contract |

`robot_fixed_v2` produced unsafe command jumps from small raw actions in Newton. It is kept so its experimental checkpoints cannot be silently reinterpreted. `robot_fixed_v3` changed neither the MJCF nor PD gains. Its envelope is an explicit operating region; tasks outside that region need another versioned action contract and new world-model data. The World Model is shared only when robot asset, action mapping, controller, and physical state meaning match. Contacts, objects, different payloads, and domain randomization need dynamics context or a separate model.

## Model trust and safety

The evaluator separates actor versus Reference PD source and the held-out environment index range. It reports 1, 8, and 32 step action-conditioned open-loop q, qd, TCP position, TCP orientation, constant-velocity baseline, and termination calibration. The independent `validate_dynamics.py` compares Actor-in-model closed loop with Actor-in-Newton from the same physical starting state and reports it separately from action-conditioned open loop.

Current trust limits use an actor-data 1-step q RMSE ≤0.04 rad, mean TCP error ≤12 mm, and improvement over a constant-velocity baseline. Eight and 32 step promotion also require at least eight distinct starts and TCP error ≤50/100 mm respectively. The values are validation gates, not statements that a model is universally accurate. A low aggregate model loss is insufficient evidence of termination reliability; the report records positive/negative labels, recall, false-positive rate, Brier score, and TCP error bins.

All optimizers check nonfinite losses, gradients, norms, and parameters. They skip invalid steps; a post-step nonfinite parameter is restored and its optimizer state cleared. Critic fitting supports 1×1, 2×4, and 5×4 epochs/minibatches with fixed TD(λ) targets.

## Code map

- `algorithms/supermimic/trainer.py`: common alternating schedule.
- `algorithms/supermimic/replay_buffer.py`: provenance-aware masked ring replay.
- `algorithms/supermimic/{imagination,critic_optimizer,termination,trust_evaluator,closed_loop,checkpoint}.py`: mathematical updates and validation.
- `tasks/marvin_supermimic/{robot_adapter,task_adapter,collector,env}.py`: Marvin physics and task-specific behavior.
- `motion/{motion_manager,library,procedural}.py`: current/future reference API and multiple clip catalog.
- `scripts/supermimic/{train,evaluate,validate_dynamics,validate_termination}.py`: common training and physical validation entry points.

The shared 42D dynamics model has no task termination weights in M4.2 v2. `TaskTerminationHead` is a separately checkpointed task module. An optional model-embedded 90D head exists solely to load M4.1 and early M4.2 v1 experiments. The present `MarvinTrackingTask` and imagination path still assume the 10×90 six-link tracking goal, so task plug-in support has been exercised with a second reference under that schema, not a different goal encoder or contact-rich task.

The full physical Replay is saved by default with learned termination checkpoints so boundary-positive labels survive resume. Analytic runs omit Replay by default; they recollect 32 Newton steps and reestablish model trust. Loading an older learned checkpoint without Replay automatically collects 32 fresh Newton boundary-probe steps before resuming Actor updates. Reconstructed environments begin a new Episode ID.
