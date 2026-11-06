# M5 full-horizon 4096-environment training pilot

This is a strict-pure, IK-free Marvin bimanual SE(3) Reach & Hold experiment. The
policy starts from random weights. Newton provides physical transitions; FK is
used only for the task's TCP goals and reward. The task still requires both TCPs
within 5 mm and 2 degrees, with a 25-step valid hold before step 125, and a
complete safe 200-step episode. No controller, robot asset, or reference data
changed.

## Versioned training plan

`configs/supermimic/m5_se3_full32_wm256_4096_bootstrap_v2.json` uses:

| Setting | Value |
| --- | ---: |
| Newton environments | 4096 |
| Physics collection per iteration | 4 policy steps, 16,384 real transitions |
| Replay length | 256 transitions per environment |
| World Model | 3 hidden layers of 256; 8-step masked autoregression |
| World updates | 32 per iteration, batch 2048 |
| Actor latent / imagination batch | 32 / 256 |
| Actor rollout | exactly 32 steps when trusted; no 1-step Actor update |
| Critic | TD(lambda), 5 epochs × 4 minibatches |
| Critic terminal Bootstrap | ramp from 0 to 1 over 50 successful Actor updates |
| Total cap | 400 iterations, saved in at most 40-iteration segments |

The 32-step Actor gate requires held-out noisy action-response and deterministic
current-Actor geometry to pass. It does not change the World Model's 8-step
prediction loss. The Replay and collector do not use a teacher, IK, joint-goal
supervision, or a prior Wave Actor. The checkpoint stores the training config;
resume rejects a different bootstrap schedule. Old checkpoints retain the
original full-bootstrap behavior through the default zero-warmup setting.

The 40-iteration `full32_wm256_v1` diagnostic admitted six 32-step Actor
updates, but independent Newton tracking regressed: right mean TCP position
error grew from 20.08 to 29.36 mm. On the same checkpoint and Replay starts,
the terminal Critic gradient norm was 2.03 times the direct reward gradient
norm, with cosine −0.63. This is evidence of an unreliable early Critic
direction; it is not proof that the World Model alone caused the regression.
The v2 schedule gradually introduces the terminal Critic while keeping
32-step BPTT and TD(lambda) intact.

## Fixed-goal independent Newton results

All evaluations below use a fresh Newton process, 64 Near goals from seed
9501, 200 policy steps per environment, zero exploration noise, and the same
success and safety definitions. Position values are left/right mean TCP error
over the full episodes; SO(3) values are geodesic errors.

| Seed 60 checkpoint | Strict success | Position (mm) | SO(3) (deg) | Mean longest valid hold | Hard failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Actor, WM warmup only (v1 iteration 4) | 0/64 | 19.91 / 20.08 | 3.81 / 3.91 | 0 | 0 |
| Full-bootstrap v1 iteration 40 | 0/64 | 20.19 / 29.36 | 4.04 / 5.21 | 0 | 0 |
| Bootstrap-ramp v2 iteration 40 | 0/64 | 19.12 / 19.26 | 3.73 / 3.82 | 0 | 0 |
| Bootstrap-ramp v2 iteration 120 | 3/64 | 9.46 / 10.29 | 1.95 / 1.98 | 4.08 | 0 |
| Bootstrap-ramp v2 iteration 240 | 60/64 | 4.05 / 3.53 | 1.12 / 0.93 | 139.12 | 0 |
| Bootstrap-ramp v2 iteration 320 | 64/64 | 2.10 / 1.99 | 0.50 / 0.47 | 181.59 | 0 |
| Bootstrap-ramp v2 iteration 400 | 64/64 | 1.81 / 1.70 | 0.43 / 0.42 | 183.31 | 0 |

The 240-iteration evaluation completed all 64 episodes. Its contact-event and
joint-violation counts were zero, and the maximum 500 Hz joint speed was
0.702 rad/s, below the existing 5 rad/s hard safety threshold. The hold
success criterion uses 50 Hz net joint displacement speed, as explicitly
selected for this task; the 500 Hz peak remains a separate safety report.

The 400-iteration policy also completed a new Near goal set (seed 9502) at
**64/64** strict success, 1.75/1.66 mm mean position and 0.42/0.42 degree
mean SO(3) error, with no contact or hard failure. On an untrained Medium set
(seed 9503), it achieved 59/64 strict success. On a Wider Feasible set (seed
9504), it achieved 16/64. These are evaluations of one training seed, not
multi-seed Medium/Wider evidence. Wider Feasible remains a clear curriculum
limit.

## Independent training-seed variation

Three random Actor/World Model/Critic seeds used the same 4096-environment
config, with no teacher data. At the equal 240-iteration budget, each seed
completed 64 fresh Newton episodes on Near evaluation seed 9501:

| Training seed | Strict success | Left/right mean position (mm) | Left/right mean SO(3) (deg) | Mean longest valid hold |
| --- | ---: | ---: | ---: | ---: |
| 60 | 60/64 | 4.05 / 3.53 | 1.12 / 0.93 | 139.12 steps |
| 61 | 42/64 | 5.13 / 4.22 | 1.08 / 1.09 | 84.22 steps |
| 62 | 53/64 | 4.65 / 3.52 | 1.15 / 0.97 | 114.33 steps |

All three admitted only horizon-32 Actor rollouts by this point. Their
Actor/Critic update counts were 202/4040, 198/3960, and 202/4040. All had
zero nonfinite updates and zero Newton hard failures or contacts in the 64
evaluation episodes. The 42–60/64 range shows genuine learning but material
seed variability at 240 iterations. Equal-budget 400-iteration continuations
were run for all three seeds:

| Training seed | Near seed 9501 | Near seed 9502 | Mean position on 9501 (L/R mm) | Mean SO(3) on 9501 (L/R deg) | Mean longest hold on 9501 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 60 | 64/64 | 64/64 | 1.81 / 1.70 | 0.43 / 0.42 | 183.31 steps |
| 61 | 64/64 | 64/64 | 1.84 / 1.93 | 0.43 / 0.42 | 182.61 steps |
| 62 | 64/64 | 64/64 | 2.10 / 1.85 | 0.45 / 0.47 | 181.59 steps |

This is **384/384 strict Near successes** across three independent training
seeds and two held-out goal seeds. All 384 episodes completed with zero hard
termination, contact events, or joint violations. Each seed used 6,553,600
real transitions and 12,800 WM8 updates. Actor update counts were 362/358/360
and Critic counts were 7240/7160/7200; all Actor updates used horizon 32, and
all three runs had zero nonfinite updates. The result supports repeatable Near
task learning at this bounded budget. It does not establish the same success
rate over arbitrary goal distributions or other physical domains.

On that same 240-iteration real Newton trace, TCP position P95 World Model
prediction errors were 0.78/1.35, 1.26/1.63, and 2.32/3.07 mm for
action-conditioned open-loop horizons 1/8/32. Independent Actor-in-model
closed-loop errors were 1.13/1.31, 1.24/1.59, and 1.87/1.72 mm. These are
different evaluations and must not be combined into a single trust number.
On the final 400-iteration seed60 model against the unseen Newton goal set
9502, action-conditioned open-loop TCP position P95 errors were 0.18/0.24,
0.85/0.71, and 1.33/2.08 mm for 1/8/32 steps. Actor-in-model closed-loop
position P95 errors were 1.06/0.98, 0.79/0.92, and 0.65/0.68 mm. The
corresponding 32-step SO(3) P95 errors were 0.34°/0.40° open loop and
0.23°/0.20° closed loop. These are on-policy physics comparisons, separate
from the held-out Replay trust gate.

The 400-iteration run collected 6,553,600 real transitions, made 12,800 WM8
updates, 362 Actor updates (all horizon 32), and 7,240 Critic updates with
zero nonfinite updates. Across its ten separately resumable 40-iteration
segments, the recorded wall time summed to 1,006.4 seconds. Mean per-iteration
collection, World Model, and Actor/Critic times were 0.572, 0.264, and 1.292
seconds. Mean BPTT32 and Critic optimization times per admitted Actor update
were 1.363 and 0.039 seconds, respectively.
The same 400-iteration schedule took 984.8 and 985.6 seconds for seeds 61
and 62. Effective full-loop real-transition throughput for the three seeds
was approximately 6,512, 6,655, and 6,650 transitions/s, including model,
Actor, Critic, checkpoints and startup overhead.

The complete 4096-environment loop allocated about 3.9 GB at the PyTorch
peak. A `nvidia-smi` sample during training showed 20.2/32.6 GB total device
memory in use, including Newton/Isaac allocations. It was a snapshot, not a
measured total-device peak. There was no observed OOM, so the environment
count was not reduced.

The full Newton/asset/SuperMimic test suite passed **215 tests** with 3 skips
after the checkpoint export path was added. The task has demonstrated
repeatable Near-goal learning for three tested training seeds and two tested
evaluation seeds. This is not evidence of full Medium/Wider curriculum
convergence, contact task transfer, or real-robot readiness.

The full 400-iteration Replay checkpoint is approximately 3.4 GB. The
versioned export tool made `models/supermimic/m5_se3_reach_hold_seed60_iter400.pt`
(3,297,889 bytes; SHA256
`fab9ccff3382b71679e1eb360d421fb6dcfbb7e5ac859ec0e5f904305ae6ff91`).
It retains Actor/WM/Critic tensors, normalizers, optimizer state, RNG and the
readable semantic manifest, while explicitly declaring `fresh_physics` resume
with no Replay. Every module Tensor was compared exactly with the full
checkpoint. An independent Newton process loaded this exported file and
completed the Wider Feasible evaluation. The full Replay checkpoint remains
separate from the small Git LFS model asset. Its archived copy is
`/home/ubuntu/dexgym_data/marvin_supermimic_m5_se3_reach_hold/m5_full32_4096_seed60_iter400/model_full_replay.pt`
with SHA256
`39e632f87f52a0a0fa8c77213d385c3df147b37f973ef4296270808ec2f51af2`.
The exact v2 config SHA256 is
`009d568182b4b4fde3fa4dbc693c9640bc136094792bb3a77a14428c37e57b58`.
The separate `m5_full32_4096_multiseed_240` and
`m5_full32_4096_multiseed_400` directories hold the other full checkpoints,
update logs, and independent Newton traces. All were created without
overwriting earlier M4/M5 experiment data.

## Protected Wave regression

The original M4.1 selected Actor was loaded from the repository's byte-identical
Git LFS asset and ran the complete historical 1,356-step Wave in a fresh
Newton process. It completed without hard termination, contact physics steps,
or illegal joint states. Left/right TCP mean position errors were 1.82/4.36
mm, and SO(3) errors were 0.265°/1.020°. The checkpoint SHA256 remains
`fd0054a7ef707d46a30b0235713c9e00cc4d7e69d6687fc3015e52aac2cb773f`.
The full report and physical policy states are archived as
`m5_full32_wave_legacy_regression/` under the selected seed60 experiment
directory. No old checkpoint, reference, asset, controller, or action schema
was changed to achieve this replay.

## Reproduction

Use a new output directory for each run. `--resume` restores Replay and
optimizer state but starts fresh Newton physics with new Episode IDs, then
revalidates model trust.

```bash
export PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source
export ISAAC_PYTHON=/home/ubuntu/isaacsim-6.0.1/bin/python

$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py train_se3_m5 \
  --config configs/supermimic/m5_se3_full32_wm256_4096_bootstrap_v2.json \
  --iterations 40 --seed 60 --output /tmp/m5_full32_bootstrap_seed60_iter40
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py train_se3_m5 \
  --config configs/supermimic/m5_se3_full32_wm256_4096_bootstrap_v2.json \
  --iterations 40 --seed 60 \
  --resume /tmp/m5_full32_bootstrap_seed60_iter40/model.pt \
  --output /tmp/m5_full32_bootstrap_seed60_iter80
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py evaluate_se3_m5 \
  --checkpoint /tmp/m5_full32_bootstrap_seed60_iter240/model.pt \
  --num-envs 64 --seed 9501 --level near \
  --output /tmp/m5_full32_bootstrap_eval_seed60_iter240
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py export_core_checkpoint \
  --checkpoint /tmp/m5_full32_bootstrap_seed60_iter400/model.pt \
  --output models/supermimic/new_m5_export.pt
```

The first command reproduces the initial 40-iteration segment; the same resume
pattern continues at 120, 160, 200, 240, 280, 320, 360 and 400 iterations.
Use `--seed 61` or `--seed 62` for the other independent training runs.
Selected full Replay checkpoints and physical traces are preserved under
`/home/ubuntu/dexgym_data/marvin_supermimic_m5_se3_reach_hold/`.
