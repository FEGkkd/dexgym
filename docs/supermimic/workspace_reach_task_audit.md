# Dual TCP workspace reach task audit

This audit concerns the arm-only, fixed-base Marvin task. The physical state is
`q14, qd14, qcmd14`; the Actor controls 14 arm joints at 50 Hz through the
versioned `arm_joint_delta_v1` map. The 40 finger joints remain at Home in the
54-DoF Newton robot and are absent from Actor, Critic and World Model data.
The Goal is one fixed 18D dual-TCP SE(3) pose in the robot base frame. FK is
used to generate targets and to evaluate the pose; the goal joint witness is
discarded and never used as an Actor input, Critic input or reward target.

## Why the earlier pilot was not a workspace-reach result

The original far-goal v3 probes were initially run with the continuous
tracking MDP. That MDP scored full-clip tracking and had Gaussian position and
rotation accuracy terms that were nearly zero at the far targets. The separate
`MarvinWorkspaceSE3ReachHold` task now supplies a broad, differentiable
position/orientation reward, a fixed target, a 600-control-step episode and a
deadline plus sustained hold metric. Its reward, observation and termination
still pass through the shared SuperMimic trainer and Newton collector.

The v3 bank's “Near” level was also too hard for a random-policy bootstrap:
86 training Near targets averaged 0.277/0.253 m of left/right TCP position
displacement and 49.6°/48.6° of rotation. Position P95 was 0.507/0.471 m.
The former 100-iteration training segment collected 400 control steps per
environment, shorter than its 600-step episode. Fresh-physics restart began
another episode, so the first two segments did not observe a complete episode.
The new training segment is 160 iterations × 4 physics steps = 640 steps.

The saved v3 pilot is evidence of failure, not a converged policy. After two
4096-environment, 100-iteration segments, 178 Actor BPTT32 and 3560 Critic
updates were recorded, but a 16-environment independent Newton evaluation had
0/16 successes and six hard failures. On the eleven held-out Near targets,
the zero-action baseline had 0.262/0.289 m final left/right position error and
no safety failure; the Stage 2 policy had 0.689/0.487 m and three failures.
On those 16 real episodes, the Critic predicted initial mean value 1.015,
whereas the actual discounted return was -0.459; mean bias was +1.474 and
predicted/actual return correlation was -0.157. At an exact real starting
state, the 32-step imagined reward contribution was -0.146 while the terminal
Critic contribution was +0.791. This is a plausible source of the bad Actor
gradient; it is not proof that every World Model term is wrong. The saved
Stage 2 model's action-conditioned 32-step joint-position RMSE on its own
Newton trace was 0.000482 rad.

## Corrected task and curriculum

The v4 target bank uses three **measured per-arm TCP pose bins**. Each sampled
start and target is within joint bounds, both endpoints and a 21-point joint
interpolation have no MuJoCo contacts, and each arm must move in position and
orientation. This is a kinematic/static safety screen; it does not prove a
learned policy or a dynamically executed path is safe.

| Level | Allowed position per TCP | Allowed SO(3) angle per TCP | Measured training mean position L/R | Measured training mean angle L/R |
| --- | --- | --- | --- | --- |
| Near | 0.015–0.12 m | 0.035–0.45 rad | 0.055/0.058 m | 10.7°/9.9° |
| Medium | 0.08–0.35 m | 0.15–1.1 rad | 0.170/0.170 m | 29.9°/28.4° |
| Wide | 0.20–1.1 m | 0.4–3.0 rad | 0.428/0.435 m | 83.3°/81.4° |

The versioned bank is
`source/dexgym/assets/supermimic/references/teleop_workspace_pose_curriculum_train_seed20261113.npz`,
SHA256 `9e6373df05ad3b8888ab6a1b5d459f07c0e973e774e49c1e80fd89bcfd85cdb4`.
It contains 256 train, 32 validation and 32 held-out test targets at 50 Hz.
It stores only the reset joint witness and the two target TCP poses; no goal
joint witness is serialized. A zero-action independent Newton baseline on all
eleven held-out Near goals completed 11/11 episodes without hard failure,
with final position error 0.0559/0.0586 m, SO(3) error 0.1818/0.1799 rad,
and 0/11 successes. Physics evidence:
`/tmp/dexgym_workspace_reach_v4_near_zero11/{report.json,physics_trace.pt}`.

The current strict success contract requires both TCPs within 5 mm and 2°,
joint net displacement speed at 50 Hz no greater than 0.15 rad/s, and a
25-step continuous hold that **starts no later than control step 400**. The
episode must finish all 600 control steps with no hard safety failure. The
500 Hz instantaneous velocity peak is reported separately. The old v1 metric
is explicitly retained only for loading the earlier experimental v3
checkpoint; the corrected v2 metric is versioned in new checkpoints.

The new 4096-environment pilot configuration keeps WM8 and BPTT32. Its first
segment covers a full physical episode. Critic bootstrap ramps over 500 Actor
updates, a controlled response to the observed value overestimation. The
curriculum initially samples Near targets; Medium and Wide need separate
Newton evaluation and should not be reported as trained coverage yet.

## 4096-environment Near pilot and controlled ablation

On the v4 bank, the 160-iteration, seed-97 strict-pure pilot collected
2,621,440 Newton transitions, made 5,120 WM updates, 136 Actor BPTT32
updates and 2,720 Critic updates, with zero nonfinite update events. At the
600-step boundary the real collector logged 2,574 normal truncations. The
model's held-out deterministic action-conditioned 32-step TCP position P95
was 0.00108 m and SO(3) P95 was 0.00233 rad. For the exact Actor action
sequence in the first 32 steps of independent Newton evaluation, the model's
reward mean absolute error was 0.0000662 per step and final joint-position
RMSE was 0.000486 rad.

The policy **did not learn stable reach-and-hold**. On the eleven held-out
Near test goals it completed 11/11 episodes safely but had 0/11 successes;
final TCP position errors were 0.274/0.330 m versus zero-action
0.056/0.059 m. It initially moved toward the goals: at step 32 error was
0.049/0.048 m, then drifted to 0.274/0.330 m by step 600. The actor's
mean initial Critic value (4.119) was close to its mean actual discounted
Newton return (4.160), but the across-goal value/return correlation was
-0.283. Mean calibration alone did not give useful per-goal guidance.

A matched seed-97 control changed only the Actor's terminal Critic bootstrap
ramp from 500 to 100000 updates, making its contribution negligible in this
pilot. It also collected 2,621,440 transitions, made 5,120 WM updates and
134 BPTT32 updates with no nonfinite events. Independent Near evaluation was
again 0/11 successes, with final position errors 0.338/0.365 m and zero
hard failures. Therefore weak Critic bootstrap alone did not fix the
long-horizon drift. Further rounds of the same v2 setup are not justified by
these results.

## Finite-horizon task contract v3

A second task-level issue was found by a failing test: the v2 Actor and
Critic did not observe the remaining time of the 600-step finite episode,
and imagined TD(λ)/Actor returns could bootstrap from the terminal timeout
frame. The versioned v3 task appends normalized remaining time to the 69D
Actor observation and 61D Critic features; its analytic soft termination is
exactly one at the terminal time. This keeps the World Model state/action
contract and WM8/BPTT32 formulas unchanged. V1 and v2 checkpoint contracts
remain explicit and their original 68D/60D networks remain loadable. The
v3 change has passed mathematical unit tests; it has not yet demonstrated a
successful physical policy.

A 32-environment, 40-iteration v3 Newton smoke collected 5,120 real
transitions and made 1,280 WM8 updates with zero nonfinite events. The noisy
held-out normalized 32-step error was 2.10 versus a 1.10 trust threshold,
so the gate correctly blocked Actor and Critic updates. This smoke verifies
physical/task integration only. A separate v3 32-step imagination test
confirmed finite nonzero Actor gradients and zero terminal Critic bootstrap
at the episode boundary.

The full v3 seed-97 pilot used 4096 environments and 160 iterations. It made
5,120 WM8, 134 BPTT32 Actor and 2,680 Critic updates. Held-out Near Newton
evaluation had 0/11 successes and five contact failures; final position
errors were 0.231/0.459 m. The v3 Actor moved the arms by an average of
0.628 rad per joint by step 600 and still issued mean bounded action 0.118
at that time. When the dual-TCP goal was set exactly to the current pose, it
issued mean bounded action 0.126, an average commanded increment of
1.26 mrad per step. Thus the learned policy did not know how to hold.

Model-only closed-loop replay of the same Actor predicted the same broad
drift as Newton: by step 600, among the six physically surviving episodes,
the model predicted left/right position error 0.264/0.429 m versus Newton
0.276/0.434 m. One held-out goal's paired 30-environment Newton finite
difference gave learned-model versus physical one-step reward-gradient cosine
0.992 and joint-action-response cosine 0.994. These are local checks, not a
global model guarantee. They support the diagnosis that the current Actor
training/task curriculum, rather than an obviously reversed WM gradient,
caused this failure.

## Hold-first curriculum v5

The v4 Near bank never reset within its 5 mm success radius: it required at
least 15 mm of movement per arm. Consequently the policy had no deliberately
sampled examples of a successful, stationary dual-TCP hold. The v5 bank adds
a separate level 0 Hold class where the TCP goal equals the safe reset pose;
its other levels are the measured Near, Medium and Wide bins above. It has
256/32/32 disjoint train/validation/test goals, evenly distributed among
the four levels. SHA256:
`6a4726337ab5913fb21634382d9dd65177fa86ec9dde2505da728abb338d6d59`.
The target joint witness is discarded; only the reset witness and TCP goal
are stored. The selected curriculum level is included in new training
checkpoint physics contracts, so a strict resume cannot silently change it.

All eight held-out Hold goals were physically maintainable in independent
Newton zero-action evaluation: **8/8 successful full 600-step episodes, zero
contacts, 600-step valid hold**, mean final TCP position error
0.000604/0.000584 m and SO(3) error 0.00131/0.00115 rad. This validates
the Hold task setup and zero-action baseline, not a learned policy.

## Task definition re-audit: arrival, sustained hold, and workspace coverage

The first strict-pure Hold segment used 4096 Newton environments for 160
iterations (2,621,440 real transitions, 5,120 WM8 updates, 140 Actor BPTT32
updates, 2,800 Critic updates, zero nonfinite updates). Its independent
eight-goal Newton test completed every 600-step episode with zero hard safety
failures. Under the versioned **25-consecutive-step arrival** criterion it
scored 8/8 successes. Under the additional **held-to-episode-end** diagnostic
it scored **0/8**: all eight policies left the valid dual-TCP hold region by
the final step. The mean longest hold was 176 steps, and mean final left/right
TCP position errors were 0.0140/0.0147 m, with SO(3) errors 0.0594/0.0406 rad.
The independent trace and report are in
`/home/ubuntu/dexgym_data/marvin_supermimic_workspace_reach_v4/hold_seed97_4096_seg1/eval_hold8_task_audit/`.
The compact checkpoint is
`/home/ubuntu/dexgym_data/marvin_supermimic_workspace_reach_v4/hold_seed97_4096_seg1/fresh_physics.pt`
(SHA256 `0bf5a9e8373a839f3959c35d54b5c52a57d298877822ab5cbdf6d919db1e83c1`).
Its mean absolute bounded arm action was only 0.0027 at step 1 and 0.0038 at
step 600, but the 0.01-rad joint-increment map accumulated a mean 0.015-rad
per-joint physical drift over the episode. The Actor did not learn a stable
stationary command. We therefore report both the original 25-step success
criterion and the new terminal hold metrics; the latter do not alter old
checkpoint/task semantics.

The hold speed criterion is the 50 Hz net joint displacement divided by
0.02 s, as requested. `joint_velocity_abs_max_rad_s` in the current episode
report samples the joint-velocity state once per policy step; it is **not** a
certified 500 Hz substep maximum. A separate 500 Hz audit is required for any
claim about instantaneous peak speed.

The v5 target bank has only 64 training and eight held-out targets per level.
The Hold training TCP locations span roughly 1.0–1.3 m along each Cartesian
axis, but the held-out Hold TCP positions are on average 0.146/0.119 m from
their nearest training position (left/right). Across all four levels the
corresponding means range from 0.113 to 0.175 m. This is evidence of spatial
diversity, **not** dense or arbitrary-pose coverage. A target's FK witness and
collision-free sampled joint interpolation establish only a possible static
path, not a dynamically feasible time-optimal route under the 50 Hz joint
increment controller. Orientation and simultaneous bimanual coverage require
separate measurement. The curriculum's Hold level has zero target displacement;
its 8/8 25-step success is not a reach result. Near, Medium and Wide have not
passed independent learned-policy Newton acceptance.

This static-goal Reach & Hold MDP is a prerequisite diagnostic for a broad
workspace policy. It is **not** the continuous 50 Hz target-stream task used
for teleoperation. The separate `MarvinTeleopWorkspaceFollow` MDP evaluates a
current pose target during each action, but its present generated trajectories
are local strokes around sampled starts. Neither task currently demonstrates
arbitrary reachable dual-TCP SE(3) tracking or deployment readiness. Before
advancing curriculum, the learned policy should pass terminal hold on held-out
Hold goals; then Near movement and both-arm orientation should be checked in
independent Newton episodes. Long-range and changing-goal tests remain separate
acceptance gates.

## Evidence-guided hold stationarity experiment

A versioned near-goal reward variant adds a smooth penalty on bounded joint
increments only while both TCP poses are inside the strict hold neighborhood.
The original reward and checkpoints remain loadable unchanged. With weight 10,
the matched seed-97, 4096-environment, 160-iteration run made 140 Actor BPTT32
updates and had zero nonfinite updates. On the same eight held-out Hold goals,
it scored 8/8 under the brief 25-step criterion but 0/8 held to episode end;
mean longest hold was 198.5 steps and final position error 0.0144/0.0157 m.
One strict Replay/optimizer resume to iteration 320 reached 298 Actor updates,
5,242,880 Newton transitions and zero nonfinite updates. It also scored 0/8
held to episode end, with mean longest hold falling to 61.1 steps and final
position error rising to 0.0425/0.0507 m. More iterations of this exact
configuration are not justified by the physical evidence.

On the 160-iteration model, a 32-step closed-loop imagination starting from
the same eight real held-out Hold states ranked a zero-action policy above the
trained Actor: discounted Actor objectives 3.3502 versus 3.3418 at the
actual 0.28 Critic bootstrap weight. The trained Actor's final decoder had a
positive loss derivative along its output scale (+0.0127), so scaling its
outputs toward zero improved the imagined objective at those starts. In the
saved deterministic training Replay, however, there were 61,440 eligible
early starts (frames 0–40) and 677,376 later starts (frames 128–568). At
iteration 320 the Actor outperformed zero in sampled late, already-drifted
states, but zero outperformed it in sampled early states. At stationarity
weight 10, the Actor output-scale loss derivative was +0.0485 early and
-0.0100 late; a weight of 50 changed the sampled late derivative to +0.0029
without removing the far-goal position gradient. The new v6 reward profile
uses this measured weight of 50 as a controlled test; it is **not yet a
successful policy**. The reward, Task Adapter and checkpoint contracts are
versioned separately from the weight-10 experiment.

## Verified commands

```bash
cd /tmp/dexgym-supermimic-teleop-workspace
PYTHONPATH=source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/generate_teleop_workspace_bank.py --profile reach_curriculum_v4 --duration-s 12 --seed 20261113 --train-clips 256 --validation-clips 32 --test-clips 32 --output source/dexgym/assets/supermimic/references/teleop_workspace_pose_curriculum_train_seed20261113.npz
PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source MUJOCO_GL=egl /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py evaluate_continuous_se3 --task workspace_reach --baseline zero --reference-bank source/dexgym/assets/supermimic/references/teleop_workspace_pose_curriculum_train_seed20261113.npz --num-envs 11 --evaluation-max-difficulty 0 --seed 9301 --output /tmp/dexgym_workspace_reach_v4_near_zero11
PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source MUJOCO_GL=egl /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py evaluate_continuous_se3 --task workspace_reach --checkpoint /home/ubuntu/dexgym_data/marvin_supermimic_workspace_reach_v4/hold_seed97_4096_seg1/fresh_physics.pt --reference-bank source/dexgym/assets/supermimic/references/teleop_workspace_hold_curriculum_seed20261114.npz --num-envs 8 --evaluation-max-difficulty 0 --seed 9301 --output /home/ubuntu/dexgym_data/marvin_supermimic_workspace_reach_v4/hold_seed97_4096_seg1/eval_hold8_task_audit
```

The legacy Wave policy and older workspace checkpoint files were not edited.
No full-workspace arbitrary-pose policy has yet passed physical acceptance.
