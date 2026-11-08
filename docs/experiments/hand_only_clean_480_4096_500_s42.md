# Clean right-hand tracking: 480 Hz controller and PPO

**Historical task semantics:** This report used the dual-arm `hand_x2` asset
and strict 51/89 motion bank. The Clean task ID was later switched to the
standalone Hand 2.1 model. Use commit `3c57d1e` to reproduce this run.

**Date:** 2026-11-08. This run uses the independent
`DexGym-Wuji-Right-Hand-Clean-Tracking-Direct-v0` task. The original task ID,
motion bank, and drive remain available.

## Task contract

The new task loads `data/processed/hand_only_right_collision_clean/manifest.json`
and `hand_only_drive_corrected_hand_x2.usda` with full hand collisions. Newton
runs at 480 Hz; 16 physics steps per action give exactly 30 Hz policy control.
The reward, 105D observation, 20D incremental right-hand action, stride-four
subgoal, 10 mm all-five-finger threshold, two-step dwell, and 45-step (1.5 s)
per-subgoal timeout are inherited from the original task. The independent
task has a 30-second global episode cap so the longest clean clips can finish;
the original task's cap remains eight seconds.

The local clean manifest has 51 training motions and 89 held-out validation
motions. Its SHA-256 is
`0d4ff25d9880e1819b5e88178c1003fd2e523bf8e5c60f3a7764ac32dcb202aa`.
The tuned USD SHA-256 is
`8d3d246283856441c3c9d463be38773a4c3a0be93268299d0287b0c4fae8ecc0`.
The data files and run artifacts remain local and are not committed.

## Pretraining gates

The fixed-target hold test used the same 178 validation pose pairs as the
500 Hz controller experiment, now with 480 Hz, full contacts, and the `hand_x2`
USD. After 1.5 seconds, **178/178** samples had every fingertip within 10 mm
and tail joint speed below 0.05 rad/s; the largest final finger error was
5.632 mm. All 178 also met the two-step 10 mm condition at the end. The local
record is `artifacts/hand_only_clean_480/hold/summary.json`.

The Joint Replay baseline used actual `env.step()` calls at 30 Hz. One
environment ran each complete clean motion from frame zero. Its action moved
the incremental joint target toward the current reference subgoal. The audit
independently checked the observed five-finger error and two-step dwell, the
target frame change, and same-step terminal completion, including the last
transition before automatic reset.

| Split | Motions completed | Subgoals reached | Mean fingertip error during active steps |
| --- | ---: | ---: | ---: |
| Train | 51/51 | 2,952 | 4.256 mm |
| Validation | 89/89 | 6,506 | 4.092 mm |

Neither split had a target timeout, global episode truncation, or unfinished
motion. Transient fingertip errors did exceed 10 mm while the controller
approached some new subgoals; success was counted only after the threshold
and dwell conditions were met. Local records are
`artifacts/hand_only_clean_480/joint_replay_train.json` and
`joint_replay_validation.json`.

## PPO protocol

RSL-RL PPO uses seed 42, 4096 training environments, and 32 rollout steps per
environment. The policy and value MLPs retain the original two 256-unit
layers and original PPO hyperparameters. The run was launched with the
following command when the requested endpoint was 1000 iterations:

```bash
PYTHONPATH=source isaaclab train --rl_library rsl_rl \
  --task DexGym-Wuji-Right-Hand-Clean-Tracking-Direct-v0 \
  --num_envs 4096 --max_iterations 1000 --seed 42 \
  --run_name clean_480_4096_1000_s42 physics=newton_mjwarp
```

The local training log is
`artifacts/hand_only_clean_480/ppo_train_4096_1000.log`; checkpoints are
under `logs/rsl_rl/dexgym_wuji_right_hand_clean_tracking/`. The user then
requested a stop at iteration 500. The saved `model_500.pt` checkpoint has
internal iteration index 500, after 65,667,072 logged environment steps, and
is the endpoint used in this report. The
interrupt was handled after the log reached iteration 508; those later updates
are not evaluated here. The separate 1024×200 startup trial was interrupted
when the user increased the training scale; its partial checkpoints are not
used for conclusions.

## Held-out comparison protocol

Each controller is evaluated on the clean validation split with seed 42,
256 environments, and 1000 policy steps per environment. All 89 validation
motions are seen. The same script counts a subgoal only when the task pays
its +10 success bonus, and counts a motion complete only when its final
subgoal and same-step termination occur. The Joint Replay controller receives
the hidden reference joint angles and therefore serves as a privileged
physical upper baseline, not as a policy input.

| Controller | Mean tip error | Strict subgoals | Completed episodes | Completion rate |
| --- | ---: | ---: | ---: | ---: |
| Zero action | 9.555 mm | 15,390 | 434 | 8.19% |
| PPO iteration 250 | 5.429 mm | 96,730 | 2,590 | 88.43% |
| PPO iteration 500 | 5.393 mm | 84,803 | 2,238 | 95.68% |
| Privileged Joint Replay | 4.402 mm | 109,707 | 2,958 | 100% |

The baseline JSON files are
`artifacts/hand_only_clean_480/eval_zero_validation_s42.json` and
`eval_joint_replay_validation_s42.json`.

At iteration 500 the policy completes a larger fraction of started episodes
than at iteration 250, while reaching fewer subgoals per fixed wall-clock
policy step. Its mean ended-episode length is 101.4 steps versus 82.2 at
iteration 250. The completion fraction and throughput describe different
properties; neither alone is a full measure of tracking quality.

An independent evaluation seed (43) shows the same pattern, with all 89
validation motions seen in both policy runs:

| Policy | Mean tip error | Strict subgoals | Completed episodes | Completion rate |
| --- | ---: | ---: | ---: | ---: |
| Iteration 250, seed 43 | 5.424 mm | 96,739 | 2,491 | 87.77% |
| Iteration 500, seed 43 | 5.386 mm | 85,479 | 2,191 | 95.64% |

Thus the original 105D/20D TELEDEXTER-style task learns a strong tracking
policy on the clean library under the corrected Newton drive. The iteration-500
checkpoint improves complete-sequence reliability over the iteration-250
checkpoint on both evaluated seeds, but advances fewer goals per fixed number
of policy steps. Its completion rate is still below privileged Joint Replay.
The validation bank contains only 89 screened GRAB motions, so these results
do not establish performance on the excluded motions or on physical hardware.

The evaluated `model_250.pt` and `model_500.pt`, exact run parameters, raw
training log, and all evaluation JSON files are preserved locally under
`/home/ubuntu/dexgym/artifacts/hand_only_clean_480/`. They are Git-ignored.
The SHA-256 of `model_500.pt` is
`baf59af87a18711621a5f70081b65e3e433ba8b4800120bc42d482160b34b704`.

## Subgoal-duration audit

The evaluator was rerun on both checkpoints and both validation seeds, still
with 256 environments and 1000 policy steps per environment. For every paid
subgoal bonus it records the task's pre-step `goal_age + 1`: the actual number
of 30 Hz control steps spent on that subgoal, including the two required
within-threshold steps. The additional measurement does not change the task,
action, reward, or checkpoint. Small changes in aggregate counts relative to
the earlier evaluation runs reflect run-to-run simulation variation.

| Seed | Policy | Mean steps per reached goal | p90 | p95 | Noncompletion terminations at goal age 45 |
| --- | --- | ---: | ---: | ---: | ---: |
| 42 | Iteration 250 | 2.483 | 3 | 5 | 329 |
| 42 | Iteration 500 | 2.962 | 6 | 8 | 102 |
| 43 | Iteration 250 | 2.483 | 3 | 5 | 341 |
| 43 | Iteration 500 | 2.939 | 6 | 8 | 99 |

On a per-motion comparison, iteration 500 takes longer on 80/89 validation
motions for seed 42 and 82/89 for seed 43. Its longer goal time comes with
fewer noncompletion terminations at the 45-step limit and higher
completed-episode rates. The termination count can include an unsafe-state
termination on the same step; it is not an exclusive cause label. This confirms
that the later policy advances more slowly while completing sequences more
reliably. The measurements do not identify the reward weights as the cause.

The current policy observation omits the accumulated position command
`q_cmd` although actions update it incrementally. Adding `q_cmd - q` would
make a separate 125D observation variant for an ablation with the same reward.
The current dense term is the mean of `exp(-50 e_f)` over five fingers, so
`exp(-50 × mean_error)` is only a lower bound on its average, not its exact
value. Its scale relative to the constant +10 reach bonus and 0.01 time cost
makes a reward-speed tradeoff plausible. A reward ablation should follow the
observation ablation and compare completion, goals per 1000 steps, per-goal
duration, and 45-step terminations on the same validation bank.

Timing JSON files are local under
`artifacts/hand_only_clean_480/eval_policy{250,500}_timing_s{42,43}.json`.
