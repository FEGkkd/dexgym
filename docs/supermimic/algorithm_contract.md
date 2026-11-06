# SuperMimic Core algorithm contract (M4.3)

## Physical time and transition

At a policy boundary, the collector records `state_t`, `task_state_t` and the current Actor history. It executes `action_t` through the robot's versioned action map for ten 2 ms Newton substeps. The captured **pre-reset** result is `state_(t+1)`, and `task_state.frame` advances by one even if the environment subsequently resets. Reward and hard/soft termination compare that arrival state with the arrival task state. A motion task reads `reference_at(frame_(t+1))`; its Actor reads `future_goal_at(frame_t, K)` containing frames `t+1…t+K`. A static reach task advances an episode counter while retaining its six-number target.

`TransitionBatch.validate` checks state/action/context/history widths, finite values, task and motion identity, and exactly one frame advance. `SchemaReplay` stores the executed command and absolute timestamp. The valid mask includes the terminal physical transition. Later slots are invalid if the same environment reset, the ring slot was overwritten, the task or motion changed, or the frame sequence broke. Model and learned-termination losses divide by the number of valid physical elements. A one-step episode therefore still trains the world model.

## Model, imagination and returns

`StateDeltaWorldModel` consumes robot-declared state, action and dynamics context. The robot declares the physical delta width and differentiable integration. For Marvin, the network predicts 28D `Δq, Δqd`; `q_cmd` follows the **same** `ArmActionMap` used by Newton. The M4.1 `legacy_wave_v1` and M4.2 `robot_fixed_v2/v3` interpretations remain versioned in the Marvin adapter. The shared model is safe to reuse only when robot state, action, controller, asset and relevant dynamics context match.

The eight-step model fit is autoregressive. Imagination samples only legal episode starts, freezes world-model and Critic parameters, but retains the action-to-predicted-state gradient. The generic engine asks the Task MDP for observation, future goal, arrival reward, arrival soft termination and Critic features. It knows no robot link or NPZ field.

For horizon `H`, `c[0]=1`, `c[t+1]=c[t](1-d[t+1])`. The Actor maximizes `Σ_t γ^t c[t] r[t] + γ^H c[H] V(s_H)`. TD(λ) uses the same arrival termination probability once at each bootstrap. The optional Critic bootstrap upper bound is enabled only when a Task supplies a proven positive reward upper bound; the bound is `R_max/(1-γ)` and clips bootstrap values above it. The default is off.

## Data isolation and trust

Training environment IDs alone update FP32 World Model input/delta, Actor observation/goal and Critic feature RunningNorm. Held-out Newton environment IDs supply trust data and never update these statistics. The trust evaluator reports separate 1/8/32-step **action-conditioned open-loop** errors and the zero-delta baseline, with distinct start counts. Its normalized option divides physical deltas by training-data standard deviations so joint position and velocity units are not conflated. The 16-environment smoke uses normalized thresholds 0.9/1.0/1.1 **and** requires improvement over the zero-delta baseline. These are smoke admission settings, not a universal deployment guarantee. Policy-closed-loop model-vs-Newton error is not yet measured in M4.3.

`pure_model_based` uses Actor data and no Actor teacher loss. `teacher_bootstrap_model_based` may tag reference-PD physical transitions when the task supplies that policy. `supervised_init_model_based` requires an explicit task-supplied initializer. Wave-specific PD files are absent from the new Core. The historical M4.2 script remains a separate compatibility entry point.

## Paper and implementation relationship

The paper [SuperMimic, arXiv:2610.28480v1](https://arxiv.org/abs/2610.28480) specifies the learned differentiable world model, short-horizon BPTT, survival weighting and TD(λ); its Algorithm A.1 and Appendix B were checked. [TrackerLab](https://github.com/Renforce-Dynamics/trackerlab-supermimic) at `e2db2dea0e81c91ebca815c895b60ee9aa58237a` (MIT) supplies an independent masked-buffer and runner engineering reference. [supermimic-mjlab](https://github.com/YHFone/supermimic-mjlab) at `9aab0a588c6941bf9862a2d444b711fd73e2b832` (Apache-2.0) is another independent state-spec/trainer reference. No third-party source was copied. Marvin's fixed-base 42D state, 14D position servo, 54DoF full physical load, disabled hand contact, six-link tracking reward and TCP Reach reward are DexGym-specific adaptations.

## M4.3.1 executable boundary checks

At the physical boundary, context_t is read from the Robot Adapter before action_t executes. The adapter must return a finite tensor of exact [num_envs, context_dim] shape, dtype and device; an absent nonzero context is an error. RobotSpec.context_evolution is either episode_constant or adapter_transition. In the latter case, imagination calls the adapter after every predicted state to obtain context_(t+1); it cannot silently reuse context_0 for 32 steps. Replay stores the context attached to each real transition.

The Marvin collector verifies actual q_cmd against robot.executed_command(state_t, action_t), compares the Task MDP Reward and hard termination with Newton's pre-reset outputs, and checks the current observation and live reference time. The environment clones the incoming Action before storing it as last_action, because selected last_action rows are reset after terminal physics. Without that clone, Replay would record an overwritten Action Tensor.

The unified NetworkSpec.normalize switch enables or disables every Core RunningNorm: World Model input/output delta, Actor observation/goal and Critic features. Enabled norms use FP32 statistics, an explicit minimum standard deviation and count cap. Disabled World Model delta normalization predicts an unscaled physical delta, and masked loss uses the same unscaled units. Checkpoints bind the switch, floor, cap and algorithm version. Evaluation runs with frozen Actor and normalizer parameters.

Physical Evaluation is an Episode lifecycle, not a fixed-step aggregate. Success is task-defined via success_mask and success_hold_steps. The evaluator distinguishes partial Episodes, hard failures and truncations, and can stop accounting after the first Episode per environment. In-process smoke and independent-process Newton evaluation carry distinct labels. A 12-step Reach smoke with no completed Episodes cannot be reported as a 100-step Episode success rate.
