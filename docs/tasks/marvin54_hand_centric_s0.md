# Marvin M6 Hand-centric Bimanual Whole-Body S0

The fixed-base task controls 14 Marvin arm joints and 40 Wuji Hand2.2 joints
with one goal-conditioned Actor. It uses synchronized EgoPro hand and upper
body poses and the real-aligned dual-arm MJCF. Isaac Lab runs the existing
Marvin scene with the Newton MJWarp solver at 2 ms per physics step. The
policy supplies one 54-dimensional absolute position target every 20 ms.
Native MuJoCo runs the sim-to-sim evaluation.

The S0 scene uses the position servo gains and force ranges from the
aligned MJCF. A combined affine servo matches MuJoCo's damping behavior
when actuator force saturates. The Newton post-actuator callback restores
the combined gains at the physics boundary after any drive refresh.
The scene uses the aligned MJCF's friction cone and solver settings.

## Data and goal

The input manifest records verified EgoProStandard-body source episodes.
Each source episode belongs to one split before resampling and trimming.
The hand goal describes ten fingertip poses, finger shape in the palm
frame, 25 cross-hand fingertip vectors and relative palm orientation.
The mapped human target and the robot's achievable projection are
stored together. The projected target drives the policy and reward.

The offline solver prioritizes hand geometry, with wrist and elbow pose
as weaker objectives. It enforces joint bounds and checks collision
clearance. A subsequent refinement accepts smaller mapped errors if the
segment still completes a physical replay. Native MuJoCo audits every
integrated 2 ms state and can apply contact-Jacobian corrections to the
reference trajectory. Local corrections isolate separate contact bursts.
They are accepted against full physical replays with bounded fingertip and
hand-shape errors. The native PD controller is a data audit and an
evaluation comparator. The 1,832 accepted clips are kept in episode-disjoint
training, validation and test manifests.

## Training and evaluation

The Actor, World Model, Termination Head and Critic start from random
weights. Physical transitions come from policy exploration. The Actor
updates by backpropagating the imagined tracking return through the
World Model. The task uses hand-centric reward, joint and command
regularization, and physical collision termination. Every physical substep
checks joint excursions, solver health and penetration. Penetration over
3 mm terminates at the first event. The Termination Head predicts this
physical failure signal from state and applied control. The World Model's
contact channels provide the reward signal for penetration over 0.5 mm.

The optional Jacobian observation expresses current fingertip displacement
along the robot's 54 hinge axes using a fixed leverage scale. It enters
the Actor history with the measured joint state and the previous action.
The same 54-output decoder produces bounded commands, which the task
converts to absolute servo targets. The exact training settings are stored
in each checkpoint and its provenance. The evaluator uses a common reward
when comparing runs with different training contact costs.

The collision refinement uses a differentiable clearance estimate between
adjacent finger collision geometries and between the thumb and palm. The
estimate uses capsules for finger segments and an oriented palm box.
Calibration uses poses from the training references. Measured MuJoCo
penetration continues to provide the collision label and the hard
termination signal. The opposing-hand clearance term compares thumb,
finger and palm geometry with a margin calibrated per projected reference
frame. The learned policy keeps the same 54-joint position-control interface.

## Train from the main checkout

The standard run starts from random weights and uses 4096 Newton worlds. It
collects 64 initial control steps, fits the World Model before updating the
Actor, and uses eight-transition dynamics windows and 16-step imagination.
The preview Critic receives the full hand goal window and the remaining
reference steps. With the current reward and discount, bootstrap values and
Critic targets have an upper bound of 954.

The following command uses the 54 DoF aligned MJCF and the localized training
manifest on the Ubuntu host. Choose a fresh output path for each run.

```bash
cd "$HOME/dexgym"
root="$HOME/dexgym_data/marvin54_hand_centric_s0_20261112"
model="$HOME/dexgym_data/marvin54_aris_single_clip_20261112/deploy/dual_arm_marvin.xml"
run="$root/s0_standard_run"
bash scripts/supermimic/host.sh -m dexgym.tasks.direct.marvin54_s0.train \
  --backend newton --model "$model" --library "$root/localized_library/train.json" \
  --output "$run" --seed 42 --num-envs 4096 --iterations 5000 --wm-only-iters 500 \
  --wm-batches 4 --wm-batch 1024 --actor-batch 192 --actor-lr 2e-5 \
  --buffer-steps 96 --exploration 0.10 --direct-tip-feedback --jacobian-feedback \
  --contact-cost 2.4 --finger-clearance-cost 1.8 --cross-clearance-cost 2.2 \
  --wm-contact-weight 0.35 --critic-context preview --reset-sampling uniform \
  --wm-state delta --wm-loss adaptive --wm-contact-intervals observed
```

Each run records the motion and MJCF identities in `provenance.json`, learner
settings in `config.json`, the solver contract in `newton_contract.json`,
metrics in `metrics.jsonl`, and optimizer update counts in `training_proof.json`.
Checkpoints retain network and optimizer states, RNG states and the reset
sampler statistics. A resumed run collects fresh physical transitions for
replay. Use the same observation and learner schema as the checkpoint.

## Compare training variants

| Option | Behavior |
| --- | --- |
| `--critic-context next` | Baseline single-goal value function. |
| `--reset-sampling return` | Estimate discounted measured return in 1 s phase bins, with uniform coverage and a cap on extra probability per source episode. Use the training worlds for feedback. |
| `--wm-state integrated` | Predict 54 velocity deltas. Integrate position over the 20 ms interval using the ten-substep coefficient 0.55. |
| `--wm-loss physical` | Apply fixed arm and hand tolerances and a fingertip FK loss against the measured next state. |
| `--wm-contact-intervals observed --wm-depth-weight 0.10` | Use valid observed intervals for contact supervision, including early physical failures. Add a peak-depth regression in millimeters. State supervision uses full control intervals. |
| `--goal-profile hand-preview` | Add relative fingertip rotation, palm-frame shape and inter-hand feedback, clip-local fingertip velocities, and sparse goals through 0.9 s. |
| `--actor-network layernorm-gelu` | Test LayerNorm and GELU in the 32-dimensional motion encoder and decoder. |
| `--imagination-start-reserve 32` | Match the eligible start phases for 16- and 32-step imagination comparisons. |
| `--optimizer fused-adam` or `--optimizer muon` | Test fused updates, with Muon on the internal linear weights in the latter mode. |
| `--compile-learner --profile-training` | Compile the Actor and World Model forward calls and record synchronized phase timings. |

At 4096 worlds, the standard settings use about 3.2 nominal replay positions
per new physical transition. Test nominal ratios of 8 and 16 using `--utd 1`
with four and eight WM minibatches. Logged state and contact counts report
how many positions actually enter each loss. Fix the real transition budget
when comparing policy quality and measure wall time separately.

## Evaluate Newton and native MuJoCo

Use the episode-disjoint validation manifest when choosing a recipe. For a
final policy, evaluate the test manifest on Newton and on native MuJoCo.

```bash
for backend in newton cpu; do
  bash scripts/supermimic/host.sh -m dexgym.tasks.direct.marvin54_s0.evaluate \
    --backend "$backend" --model "$model" --library "$root/localized_library/test.json" \
    --checkpoint "$run/model_final.pt" --output "$run/test_$backend" \
    --modes nominal perturbed alternate --require-heldout --trace
done

bash scripts/supermimic/host.sh -m dexgym.tasks.direct.marvin54_s0.evaluate_world_model \
  --model "$model" --checkpoint "$run/model_final.pt" \
  --traces "$run/test_newton/nominal" --output "$run/test_newton/world_model.json"
```

The evaluator tracks source-episode overlap and reports clip and episode
statistics, completion, errors on observed prefixes, peak contact depth,
and the number of 2 ms substeps above 0.5 mm. The World Model check replays
recorded actuator targets from measured states and reports horizon errors,
contact calibration and physical failure prediction.
Perturbed starts and wrist-preserving alternate arm configurations test
recovery. The arm-hold comparisons keep one or both arms' initial servo
targets fixed while the Actor continues controlling the other joints.

## Reproduce the selected policy replay

Use the selected 5000-iteration checkpoint for a native replay from the main
checkout. The deployed entry point uses the aligned MJCF beside the checkpoint.

```bash
bash "$root/deploy/replay.sh"
```

The final training retained a single random-initialized lineage. The
contact refinement starts from iteration 2500. The opposing-hand
clearance term starts from iteration 3500. The selected iteration 5000
was evaluated on all 177 held-out Newton clips and one native MuJoCo
clip per held-out source episode. The reports retain every failed
rollout, the error means on complete rollouts, and contact depth sampled
at each 2 ms physics step.

## Early trainer comparison

Each condition ran 180 iterations from seed 42 in 4096 Newton worlds.
The same 30 nominal validation clips were used for each condition.

| Critic and reset setting | Completed clips | Physical substeps above 0.5 mm |
| --- | ---: | ---: |
| Single goal and uniform resets | 7 / 30 | 4.30% |
| Preview Critic and uniform resets | 10 / 30 | 3.54% |
| Single goal and return-weighted resets | 9 / 30 | 0.93% |
| Preview Critic and return-weighted resets | 9 / 30 | 0.98% |

The preview Critic with uniform resets completed the most nominal clips
and had the highest mean reward per required step in this pilot. The
return-weighted sampler reduced shallow penetration and increased early
fingertip tracking error. The standard command uses the preview Critic
with uniform resets. The 5000-iteration replay metrics above come from the
previous trained checkpoint.

The logs, traces and stage commands on the training host are under
`/home/ubuntu/dexgym_data/marvin54_hand_centric_s0_20261112/supermimic_standard_20261112`.
