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

At iteration 520, the original test split succeeds on 12/12 near, 10/10
medium and 6/10 wider targets; the new seed succeeds on 12/12, 10/10 and
7/10. These are 200-step closed-loop Newton runs, not World Model rollout
scores. Remaining wider failures and absent human target streams mean this is
**a bounded workspace learning pilot**, not a general teleoperation
deployment policy.
