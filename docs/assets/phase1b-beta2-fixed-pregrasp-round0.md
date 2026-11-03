# Phase 1B Beta 2 fixed pregrasp round 0

Date: 2026-11-03. This is one Reset distribution stage of the unified
`DexterousHands-WujiHand2-Bimanual-PregraspCapTurn30-Direct-v0` task, not a
separate GraspHold or CapTurn policy. The same 52D actor controls both hands
throughout each Episode. The policy is trained from scratch with a learnable
Gaussian std initialized at 1.0.

The fixed training Reset is local hand-shape seed **66003** in the committed
Beta 2 PGS Mimic pregrasp bank (SHA-256
`6811aa8bd8f58454ddd4f1f1ae780c10a4cdf1532b03fdb353a830041a2a9d63`).
The frozen manifest assigns that seed to training. Its zero-action native
contact probe did not form valid bilateral contact; an active feasibility probe
did at the first 50 Hz action. The one-row training mode checks the manifest
and rejects reserved evaluation seeds. The three held-out local seeds remain
9103120, 9103176, and 9103231. All are one relative template family.

## Pre-training controls

| Control, 200 episodes from seed 66003 | Success | Left/right contact-step rate | Bottle-stable step rate | Max observed opening |
| --- | ---: | ---: | ---: | ---: |
| Zero action | 0/200 | 100% / 0.14% | 100% | 0.035 rad |
| Random Gaussian action, std 1.0 | 0/200 | 76.2% / 6.74% | 17.3% | 1.784 rad |

The random control shows that a large transient relative angle does not pass
the contact, bottle stability, pitch, and continuous Success gate. Its raw
action elements exceed `[-1,1]` 31.74% of the time. The zero control does not
obtain sustained right-cap contact. Neither control supports a learned-skill
claim. Raw JSON episode rows and logs are under the ignored
`logs/phase1b_beta2_fixed66003/` directory.

Native smoke passed 15 zero-action steps at 1 and 1024 environments with
finite 277D actor and 421D critic observations. At 1024 environments, measured
step throughput after scene initialization was 4955 transitions/s, maximum
constraint error 3.63 μm, and right contact rate zero at the final step.

## Controlled PPO baseline

Only the Reset sampler changes from the multi-seed Beta 2 Level A baseline.
The task physics, Success, reward, 52D action scales, and Gaussian actor
distribution remain unchanged. The initial run is 1024 environments,
seed 1101, 1000 requested PPO updates with 32 rollout steps per environment.
The requested count is 32,768,000 PPO transitions. The experiment is subject
to the remaining 12 GPU-hour budget; observed learning and action saturation
will determine whether to stop before 1000 updates.

```bash
export ISAACLAB_PYTHON=/home/ubuntu/miniconda3/envs/allo_track/bin/python
export WUJI_DESCRIPTION_ROOT=/tmp/wuji-description-v2026.9.11
export PYTHONPATH=/home/ubuntu/dexgym
scripts/run_isaaclab.sh rsl_rl/base/train.py \
  --task DexterousHands-WujiHand2-Bimanual-PregraspCapTurn30-Direct-v0 \
  --num_envs 1024 --seed 1101 --max_iterations 1000 \
  --run_name beta2_fixed66003_gaussian_std1_seed1101_1024_1000iter \
  --hand_variant beta2 --fixed_pregrasp_seed 66003 --solver_type pgs \
  --init_action_std 1.0 --action_distribution gaussian \
  --unified_reward_profile baseline \
  --source_bank_snapshot docs/assets/beta2_phase1b_pregrasp_bank.json --viz none
```

Training checkpoint selection requires deterministic 200-episode evaluation
on the fixed training state and separately on the frozen held-out local seeds.
The fixed-state result can demonstrate acquisition from this initial pose;
only the latter tests within-template generalization. Beta 2 Level B and
original-distance natural pregrasps remain unqualified and cannot inherit a
near-contact score.

## Seed 1101 results and stop decision

The 1024-environment scratch run reached iteration 209 (6,881,280
transitions) before an intentional SIGINT stop. The requested 1000 updates
were not completed. The latest saved checkpoint is `model_200.pt`; `model_100.pt`
has the same independent Success rate with less action saturation. Raw log,
checkpoints, TensorBoard events, and curve PNGs are under the ignored paths
`logs/phase1b_beta2_fixed66003/` and
`logs/rsl_rl/dexteroushands_pregrasp_capturn30_phase1b/2026-11-03_19-23-11_beta2_fixed66003_gaussian_std1_seed1101_1024_1000iter/`.

| PPO update | Fixed train seed, 200 episodes | Three frozen local eval seeds, 200 episodes | Eval bilateral support | Deterministic raw action outside `[-1,1]` |
| ---: | ---: | ---: | ---: | ---: |
| 25 | 0/200 | Not tested | 0% fixed | Not recorded here |
| 100 | 200/200 | 200/200 | 89.7% | 11.1% |
| 150 | Not tested | 200/200 | 90.7% | 13.2% |
| 200 | 200/200 | 200/200 | 96.0% | 18.5% |

All 200-update evaluation episodes on the three frozen local seeds had no
failure or constraint-failure termination. Their mean final pitch error was
1.96 µm. The three rows are variations of **one** relative grasp template;
this is within-template evidence only. The deterministic action overflow at
update 200 includes left wrist rotation 72.5%, right wrist rotation 55.8%,
left fingers 10.0%, and right fingers 18.1% of their respective elements.
The stochastic sampled-action overflow in training rose from about 31.7% at
initialization to 41.3% at update 209. The wrist-rotation Gaussian stds
fell during this period (left roughly 1.00 to 0.81; right roughly 1.00 to
0.92), so the rising overflow is not explained by increasing std alone.
Fixed-state independent Success was
already saturated at update 100, so extending this distribution solely for
reward would increase action saturation without measurable deterministic
Success gain. The next controlled experiment uses an independent PPO seed and
2048 environments.

`model_100.pt` counterfactuals, each 200 episodes on the fixed state, were
0/200 with no right-hand drive/contact, 0/200 with no left-hand drive/contact,
and 0/200 with reversed wrist turning. A separate no-hands/no-drive control
was also 0/200, with zero positive opening. These controls use the same
Success predicate as the learned policy.

The one-environment 500 Hz successful trace for `model_100.pt` opened 0.78435
rad by actual relative rigid-body pose, with a matching Mimic joint angle and
0.374509 mm axial travel (pitch prediction 0.374500 mm). About 93.6% of its
positive angular progress occurred while right-cap contact force was present.
The maximum constraint error was 5.98 µm. This does **not** clear the full
physics gate: PhysX screw velocity differed from adjacent-pose angular speed
by 1.624 rad/s MAE with -0.0436 rad integrated bias. The deepest reported
PhysX signed separation was -2.395 mm; a cap collision-proxy point SDF reached
-1.741 mm. These are distinct geometric quantities and neither alone is an
exact penetration depth. Raising PGS position iterations from 8 to 16 on a
controlled one-environment replay did not remove the discrepancy (velocity
MAE 1.685 rad/s; signed separation minimum -2.584 mm), so solver iteration
count alone is not a demonstrated remedy.

The 2048-environment seed 2202 replication uses the same fixed Reset,
physics, reward, and Gaussian std 1.0. Its raw log is
`logs/phase1b_beta2_fixed66003/train_seed2202_2048_400.log` and its requested
run name is `beta2_fixed66003_gaussian_std1_seed2202_2048_400iter`.

Checkpoint SHA-256: `model_100.pt`
`693217aa606753ee2057711fbac35fd41b0119e7ecc0a0ae6c7af5b845e2d33a`;
`model_200.pt`
`9103aa4ca81b9153078426d390ccc4c52fb883f9ccbea2f9823cec7efc4a0ba3`.
After setting the three environment variables in the training command above,
the frozen within-template evaluation can be reproduced with:

```bash
CKPT=logs/rsl_rl/dexteroushands_pregrasp_capturn30_phase1b/2026-11-03_19-23-11_beta2_fixed66003_gaussian_std1_seed1101_1024_1000iter/model_100.pt
scripts/run_isaaclab.sh scripts/environments/evaluate_phase1b.py \
  --checkpoint "$CKPT" --hand-variant beta2 --policy learned \
  --level A --action-distribution gaussian --episodes 200 \
  --num-envs 256 --seed 9117 --headless \
  --output logs/phase1b_beta2_fixed66003/model100_heldout_A_200.json
```

Adding `--fixed-reset-seed 66003` instead tests the training row and sets
`held_out=false` in the result. For the no-right and no-left counterfactuals,
replace `--policy learned` with `--policy no_right` or `--policy no_left`
and include the fixed seed; `reverse`, `zero`, `random`, and `no_hands` use
the same evaluator and Success definition.

## Seed 2202, 2048-environment baseline failure

The second scratch Gaussian seed used 2048 environments and exactly the same
fixed Reset, physics, and baseline reward as seed 1101. Scene initialization
took about 15 minutes, then PPO produced complete updates at roughly 6600
transitions/s with about 23.8 GB GPU memory. It reached iteration 113
(7,471,104 actual transitions) before an intentional stop. The raw run is
`logs/rsl_rl/dexteroushands_pregrasp_capturn30_phase1b/2026-11-03_19-56-15_beta2_fixed66003_gaussian_std1_seed2202_2048_400iter/`;
`model_100.pt` SHA-256 is
`b5b8db19156910e0b42281ba5ec7760afc989d18d4c07807bca43616489c7599`.
It scored **0/200** on the fixed training Reset and **0/200** on the three
frozen local evaluation seeds. In both, right-cap contact and bottle-stable
step rates were zero. The fixed replay's left-bottle contact rate was 99.7%,
but contact alone did not hold the bottle; finger target clipping was 9.9%.

The seed-2202 right-cap contact rate fell from about 3% at update 2 to zero
by update 29 and stayed there through update 113. A 500 Hz replay of its
`model_50.pt` showed no effective right-cap force: bottle height fell from
0.500 m to 0.480 m within 0.3 s, while the successful seed-1101 replay kept
the bottle near 0.498 m and developed right-cap force by 0.1 s. The failed
policy then let the cap move away from the right wrist as the bottle dropped.
Training reward improved from roughly -13 to -2 despite zero success because
the policy reduced negative progress and constraint terms. This is an
observed contact-avoidance local optimum, so extending the unchanged run to
400 updates was not justified. The two-seed comparison curves are under
`logs/phase1b_beta2_fixed66003/baseline_two_seed_*.png`.
For a diagnostic only, zeroing the left actor's 26 action values in a
200-episode replay raised the bottle-stable step rate from 0% to 69.5%,
while right-cap contact and Success remained zero. This isolates harmful
left-hand motion as one failure cause; the task training never freezes the
left hand.

## Controlled reward follow-up

A new scratch run uses the **same seed 2202, 2048 environments, fixed Reset,
Gaussian std 1.0, solver, and 52D action**. Its only change is
`--unified_reward_profile contact_coverage`, adding bounded left bottle
stability, right side proximity, and right contact aids. The numerical
reward tests passed 7/7 with unrelated ROS pytest plugin autoload disabled;
the tested maximum non-turn dense return remains below the 25-point Success
bonus. The run log is
`logs/phase1b_beta2_fixed66003/train_seed2202_contact_coverage_2048_400.log`.
Independent 200-episode evaluation remains required before claiming a fix.
