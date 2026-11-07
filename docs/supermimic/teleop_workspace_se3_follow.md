# Causal dual TCP workspace tracking pilot

This is a separate Marvin arm-only SuperMimic task intended to test whether a
policy can track **unseen continuous dual TCP SE(3) targets**. The target stream
contract is Marvin fixed-base coordinates at 50 Hz. The policy receives only
the current target and ten frames of its own observation history. It cannot
read future operator commands. This differs from the earlier pinned trajectory
task, which used a ten-frame future target window.

## Contracts

| Item | Definition |
| --- | --- |
| Newton robot | Full 54-DoF Marvin; 40 fingers at Home with the existing disabled hand-contact configuration |
| Controlled joints | Left and right arms, 14 DoF |
| Learned dynamics state | `q14 + qd14 + qcmd14`, 42D |
| Learned delta | `Δq14 + Δqd14`, 28D; command integrated analytically |
| Action | `arm_joint_delta_v1`, 14D, `qcmd_next = clamp(qcmd + 0.01 tanh(action))` |
| Servo | Existing 500 Hz Newton position controller with ten substep command interpolation per 50 Hz action |
| Actor input | 68D proprioception and current TCP error, ten-frame history; current target `[N,1,18]` |
| Critic input | 42D physical state plus current 18D target, 60D |
| Target frame | Marvin fixed base, position xyz plus rotation 6D for both TCPs |
| Target timing | At t the Actor sees target t; action t executes; arrival state t+1 is scored against target t |
| Algorithm | Real Newton transitions, masked 8-step world model, 32-step differentiable imagination, survival weighted Actor BPTT, TD(λ) Critic, model trust gate |

The target at t is held for the action interval. The next target is available
only after that action. This makes the setup causal for a 50 Hz teleoperation
stream. Tracking latency and accuracy must be evaluated together; no
unavailable future target is inserted into the Actor.

## Target distribution and data isolation

The committed `teleop_workspace_bank_v1.npz` is generated with seed
`20261107`: 256 training clips, 32 validation clips, and 32 unseen test clips.
Every clip has 201 frames at 50 Hz. Five families provide independent arm
strokes, opposed strokes, oscillation, move-and-hold, and asymmetric motion;
each has three amplitude levels. The generator samples smooth legal joint
**witnesses** and uses Marvin forward kinematics to produce TCP poses. It
rejects a clip if a sampled joint state violates the safety margin, if a
frame-to-frame command step exceeds 0.009 rad, or if source MuJoCo reports a
contact at any frame. This is a kinematic and collision screen; actual Newton
tracking safety is measured independently.

The witnesses are read only during physical reset. The Actor goal, Critic
features, reward, termination, and World Model training consume TCP targets
and physical arm state; none receives a target joint angle, IK solution, or
joint tracking loss. No teacher action or old Wave Actor is used in this
training run. The generated bank covers a bounded region around Marvin Home;
its left/right TCP xyz extrema and target speeds should accompany any claim
about workspace coverage. It is **not** a survey of the entire mechanical
workspace or of human teleoperation traces.
The bank's observed left TCP xyz envelope is approximately
`[0.117,0.391] × [-0.480,-0.338] × [0.802,1.097] m`; the right envelope is
`[-0.400,-0.153] × [-0.485,-0.334] × [0.796,1.094] m`. The largest adjacent
target position step is 8.43 mm per 20 ms. These measured bounds define the
current tested distribution, not the robot's whole reachable space.

Validation Newton lanes are separate from optimizer lanes. Independent test
episodes start at frame zero, run to frame 200, and use no exploration noise.
An episode succeeds only when both TCPs meet the task's full-clip mean and
95th-percentile position and SO(3) thresholds without a hard safety failure.
The configured mean/P95 limits are 5/10 mm and 2°/4° per arm.
Step-level target proximity is not reported as an episode success.

For this causal task, **inference-only** checkpoint loading can explicitly
accept a new target bank with `--allow-new-reference`. It checks the new
target-stream version and leaves robot, actuator, action, network, reward,
normalization, and controller contracts strict. The report records both bank
hashes. Training Replay resume still requires the original bank hash and
complete training contract.

## Stateful input boundary

`StatefulTeleopPolicy` in `teleop_stream.py` accepts one finite 42D physical
state and one pair of target positions and proper SO(3) matrices at each
timestamp. It rejects frames other than `marvin_fixed_base`, invalid
rotations, and deviations from the 20 ms period beyond 3 ms. It maintains the
same ten-frame history and previous 14D action used in training, then returns
the raw Actor action and the analytically mapped `qcmd_next`. The wrapper is
tested with a mock robot. **It is not a hardware driver or safety controller.**

## Training and evaluation

The current versioned schedule uses 4,096 Newton environments, 384 Replay
steps per environment, a 512×3 World Model, Actor goal latent 64, 512
imagination starts, 8-step model fitting, 32-step Actor BPTT, and five Critic
epochs with four minibatches. Each iteration collects four physics control
steps and performs 32 World Model updates. The model trust gate must admit the
full 32 steps before Actor BPTT. Curriculum difficulty is explicitly passed
as `--curriculum-level 0`, `1`, or `2`; a Replay resume starts new Newton
physics and revalidates model trust. It is not a bitwise simulator resume.

Run from the feature checkout on the pinned Isaac Sim 6/Newton host:

```bash
export PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source
ISAAC_PYTHON=/home/ubuntu/isaacsim-6.0.1/bin/python
BANK=source/dexgym/assets/supermimic/references/teleop_workspace_bank_v1.npz
CONFIG=configs/supermimic/teleop_workspace_4096_full32_v1.json

$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py train_continuous_se3 \
  --config "$CONFIG" --reference-bank "$BANK" --num-envs 4096 \
  --curriculum-level 0 --iterations 100 --output /tmp/teleop_near_segment

$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py train_continuous_se3 \
  --config "$CONFIG" --reference-bank "$BANK" --num-envs 4096 \
  --curriculum-level 1 --iterations 100 \
  --resume /tmp/teleop_near_segment/model.pt --output /tmp/teleop_medium_segment

$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py evaluate_continuous_se3 \
  --task workspace --reference-bank "$BANK" \
  --checkpoint /tmp/teleop_medium_segment/model.pt \
  --num-envs 32 --seed 9101 --evaluation-split test \
  --output /tmp/teleop_medium_eval

$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py evaluate_continuous_se3 \
  --task workspace --allow-new-reference \
  --reference-bank source/dexgym/assets/supermimic/references/teleop_workspace_test_bank_seed20261108.npz \
  --checkpoint /tmp/teleop_medium_segment/model.pt \
  --num-envs 32 --seed 9102 --evaluation-split test \
  --output /tmp/teleop_new_seed_eval
```

The initial generated bank can be recreated with
`scripts/supermimic/generate_teleop_workspace_bank.py`; the committed artifact
has SHA256 `f4137b4a45d6114f330e445ffe84fa689de5c6358a9f1f345ed4b59b0c6639c0`.
Checkpoint loading verifies that hash and the task/action/network contract.
The separately committed 32-clip new-seed test bank has SHA256
`d94b1159b158eec578208978122ad8d2cbf9918186508c4f0ac85e9d70e01b3d`.
A third bank reverses those 32 unseen target streams in time without changing
their reachable poses. Its SHA256 is
`7adf8fb02eefebe5ee0d799e0359531faebfde965feb9ffefbf5e390aea5e874`.
The reverse bank is produced by
`scripts/supermimic/reverse_teleop_workspace_bank.py` and is used for
evaluation only.

## Current independent Newton evidence

All rows below use the **same 32 held-out test clips** at evaluation seed
`9101` and zero reset perturbation. Position errors are left/right means;
orientation errors are left/right means in degrees.

| Policy | Full episodes | Position mm | SO(3) degrees | Hard failures |
| --- | ---: | ---: | ---: | ---: |
| Zero action | 0/32 | 36.5 / 37.0 | 6.23 / 7.00 | 0 |
| Iteration 120, near targets | 0/32 | 13.5 / 12.6 | 3.07 / 3.54 | 0 |
| Iteration 220, near targets | 10/32 | 8.8 / 8.8 | 1.89 / 2.05 | 0 |
| Iteration 320, near+medium targets | 17/32 | 5.0 / 4.9 | 1.23 / 1.26 | 0 |
| Iteration 420, all difficulties | 25/32 | 2.93 / 3.00 | 0.83 / 0.79 | 0 |
| Iteration 520, original test split | 28/32 | 1.81 / 1.73 | 0.46 / 0.46 | 0 |
| Iteration 520, new seed and new bank | 29/32 | 1.77 / 1.66 | 0.56 / 0.47 | 0 |
| Iteration 620, original test split | 30/32 | 1.35 / 1.28 | 0.33 / 0.34 | 0 |
| Iteration 720, original test split | 31/32 | 1.27 / 1.32 | 0.31 / 0.36 | 0 |
| Iteration 720, new seed and new bank | 30/32 | 1.25 / 1.33 | 0.39 / 0.36 | 0 |
| Iteration 720, reversed unseen streams | 30/32 | 1.26 / 1.35 | 0.39 / 0.38 | 0 |
| Iteration 820, original test split | 31/32 | 1.14 / 1.11 | 0.29 / 0.54 | 0 |
| Iteration 820, new seed and new bank | 32/32 | 1.22 / 1.18 | 0.36 / 0.54 | 0 |
| Iteration 820, reversed unseen streams | 30/32 | 1.18 / 1.17 | 0.36 / 0.52 | 0 |
| Iteration 820, new seed with ±0.02 rad reset | 32/32 | 1.37 / 1.32 | 0.39 / 0.59 | 0 |

Iteration 820 is selected by the highest aggregate success across the three
fixed unperturbed test sets: 93/96 episodes, versus 91/96 at iteration 720.
The right TCP's mean orientation error increased somewhat between those two
checkpoints; success and position accuracy improved. At iteration 820, the
original split succeeds on 12/12 near, 10/10 medium and 9/10 wider targets;
the new-seed bank succeeds on all 32, while the reverse bank succeeds on
12/12, 10/10 and 8/10. The remaining failures are mainly wide oscillatory
targets whose SO(3) P95 exceeds the 4° success threshold. The separate
perturbed-start test actually applied a maximum 0.01996 rad arm offset and
completed 32/32 episodes. These are 200-step closed-loop Newton runs, not
World Model rollout scores.

The selected 820-iteration run collected 13,434,880 real Newton transitions,
performed 26,240 World Model updates, 784 Actor BPTT updates and 15,680
Critic updates. There were zero nonfinite optimizer events. The selected
held-out noisy action-conditioned 1/8/32-step normalized World Model errors
were 0.339/0.495/0.579, against zero-delta baselines 1.033/3.215/8.458.
On deterministic Actor data, 32-step action-conditioned open-loop prediction
had a maximum TCP position P95 of 3.01 mm, maximum SO(3) P95 of 0.54°, and
joint-velocity RMSE of 0.0374 rad/s. These model errors are distinct from
the policy's real Newton tracking errors; a separate policy-closed-loop
model-vs-physics metric is not asserted here. The 4,096-environment run used
about 5.42 GiB peak PyTorch allocation and a sampled 23.6 GiB total RTX 5090
device allocation, including Newton.

The selected Replay-bearing checkpoint is archived separately. The committed
[`fresh_physics` inference checkpoint](../../models/supermimic/teleop_workspace_seed97_iter820.pt)
has SHA256 `44740c7ad90627c93e0317ea70ceebaf13e5f849efb35c4714d92a569cc274b6`.
Its Actor and World Model tensors are exactly equal to the selected Replay
checkpoint; on the same perturbed 32-episode Newton protocol the exported
policy also completed 32/32 episodes, with mean tracking metrics differing by
at most `3.5e-5` in reported units across independent simulator runs. The
export omits Replay and cannot restore the old physics environment.
All nine bounded Replay checkpoints, iteration logs, independent Newton
physics traces, and the protected Wave regression are archived under
`/home/ubuntu/dexgym_data/marvin_supermimic_teleop_workspace_seed97_20261107/`.
The selected full checkpoint is
`training/dexgym_teleop_4096_stage9/model.pt`; it records source commit
`a5b25ca4732ed854c71bdb979ee10972caa5fcaf` and is marked `replay`
resume. The committed 10.3 MB export is marked `fresh_physics`.

The combined Newton, asset and SuperMimic regression run passed **228 tests
with 3 skips**. The protected M4.1 Wave checkpoint also completed all
1,356 Newton control steps independently: zero hard terminations, zero Newton
contact control/physics steps and zero illegal joint-state frames. Its left
and right TCP mean position errors were 1.85 and 4.45 mm in this run. This
legacy result is a regression check, not training data for the new Actor.

Absent human target streams, payload variation, contact-rich tasks, network
latency tests and hardware safety validation mean this is **a bounded
workspace learning pilot**, not a general teleoperation deployment policy.
