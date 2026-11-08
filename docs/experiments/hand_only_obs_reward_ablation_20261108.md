# Right-hand clean tracking: observation and reward ablations

**Date:** 2026-11-08. The existing 105D clean task and its 500-iteration PPO
checkpoint are the reference. Both ablations keep the same screened GRAB
MotionBank (51 train, 89 validation), full-contact `hand_x2` USD, 480 Hz Newton
physics, 30 Hz policy, 20D incremental action, stride-four subgoals, 10 mm
five-finger success threshold, two-step dwell, and 45-step goal timeout.

## Controlled changes

| Task | Observation | Reach bonus | Dense coefficient | Time cost | Action-change coefficient |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean baseline | 105D | 10 | 1.0 | 0.01 | 0.02 |
| `Clean-Obs125` | 125D | 10 | 1.0 | 0.01 | 0.02 |
| `Clean-Hybrid125` | 125D | 10 | 0.1 | 0.05 | 0.02 |

The extra 20 observation channels are the accumulated PD position command
minus measured joint angle, `q_cmd - q`. They are appended after the original
105 fields. The hybrid reward is a scoped ablation of the user's identified
dense-versus-speed tension; it is not the paper's full reward equation. Since
subgoal stride is fixed at four throughout this comparison, multiplying sparse
reward by the stride would only change its constant scale.

The new variants have separate Gym IDs and PPO experiment directories. Both
start from random policy initialization because the 125D actor input layer
cannot load a 105D checkpoint. They use 4096 training environments, seed 42,
the same PPO hyperparameters, and a checkpoint numbered 500. Validation uses
the same 89 held-out clean motions and compares deterministic policy rollout
at 256 environments × 1000 steps, including mean fingertip error, strict
subgoals per 1000 steps, completed-episode rate, subgoal duration, 45-step
noncompletion terminations, and Action Std. The intermediate checkpoint uses
evaluation seed 42; the endpoint uses seeds 42 and 43. Training still uses
one seed, so differences smaller than run-to-run training variation require
further replication.

## Observation-only result

The observation-only policy was trained from scratch with 4096 environments,
seed 42, through checkpoint 500. Its Action Std rose from 0.30 at iteration 0
to 0.70 at 250 and 1.46 at 500. For the 105D baseline, the corresponding
values were 0.30, 0.65, and 1.19. Adding command error alone therefore did
not reverse the upward Std trend.

| Policy and evaluation seed | Mean tip error | Strict subgoals | Completed episodes | Completion rate | Mean steps per reached goal | Goal-age-45 noncompletion terminations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 105D baseline, iteration 250, seed 42 | 5.420 mm | 96,840 | 2,556 | 88.60% | 2.483 | 329 |
| 105D baseline, iteration 500, seed 42 | 5.417 mm | 84,683 | 2,256 | 95.67% | 2.962 | 102 |
| 125D observation, iteration 250, seed 42 | 5.224 mm | 105,045 | 2,838 | 93.63% | 2.350 | 193 |
| 125D observation, iteration 500, seed 42 | 4.695 mm | 105,082 | 2,809 | 93.54% | 2.349 | 194 |
| 105D baseline, iteration 500, seed 43 | 5.379 mm | 85,403 | 2,198 | 95.69% | 2.939 | 99 |
| 125D observation, iteration 500, seed 43 | 4.695 mm | 105,107 | 2,690 | 93.79% | 2.354 | 178 |

All evaluation rows cover 256 environments for 1000 policy steps and see all
89 held-out motions. Completion rate is the share of episodes ending with
the final subgoal reached; it is not the share of 89 distinct motions that
could be completed. The 125D policy improved mean fingertip precision and
subgoal throughput at iteration 500, but had about two percentage points lower
episode completion and more age-45 noncompletion terminations than the 105D
baseline. Its action Std was higher. This is a performance tradeoff rather
than evidence that the partial-observation issue alone caused rising Std.

## Hybrid reward at iteration 250

At the same iteration 250 and evaluation seed 42, the 125D hybrid policy has
mean fingertip error 5.058 mm, reaches 100,740 strict subgoals, and completes
2,721 episodes (92.68%). Its reached subgoals take 2.439 steps on average,
with 215 goal-age-45 noncompletion terminations. Action Std is 0.64, versus
0.70 for the observation-only policy. The hybrid reward improves fingertip
precision slightly at this checkpoint but reduces subgoal throughput and
completion rate relative to the observation-only variant. The endpoint
evaluation is needed to judge whether this trend persists.

## Endpoint comparison

The hybrid run completed checkpoint 500 after 65,667,072 environment steps
and 1712.93 seconds of training. The two endpoint evaluations agree on the
main tradeoff:

| Policy and evaluation seed | Mean tip error | Strict subgoals | Completed episodes | Completion rate | Mean steps per reached goal | Goal-age-45 noncompletion terminations | Deterministic action RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 105D baseline, seed 42 | 5.417 mm | 84,683 | 2,256 | 95.67% | 2.962 | 102 | — |
| 125D observation, seed 42 | 4.695 mm | 105,082 | 2,809 | 93.54% | 2.349 | 194 | 0.584 |
| 125D hybrid, seed 42 | 4.718 mm | 109,171 | 2,897 | 94.70% | 2.275 | 162 | 0.535 |
| 105D baseline, seed 43 | 5.379 mm | 85,403 | 2,198 | 95.69% | 2.939 | 99 | — |
| 125D observation, seed 43 | 4.695 mm | 105,107 | 2,690 | 93.79% | 2.354 | 178 | 0.584 |
| 125D hybrid, seed 43 | 4.694 mm | 109,370 | 2,800 | 94.66% | 2.273 | 158 | 0.539 |

The hybrid policy reaches about 4% more strict subgoals than the
observation-only policy at the same 256,000 evaluation steps and improves
completed-episode rate by about one percentage point. Its mean fingertip
error is effectively the same at seed 43 and 0.023 mm worse at seed 42.
Compared with the original 105D policy, the hybrid policy is more accurate
and advances subgoals faster, but its completed-episode rate remains about
one percentage point lower. The privileged Joint Replay baseline reached
109,707 subgoals with 100% episode completion at seed 42 in the prior clean
experiment; it receives reference joint angles unavailable to these policies.

| Training iteration | 105D baseline Action Std | 125D observation Action Std | 125D hybrid Action Std |
| ---: | ---: | ---: | ---: |
| 0 | 0.30 | 0.30 | 0.30 |
| 100 | 0.38 | 0.38 | 0.37 |
| 200 | 0.54 | 0.58 | 0.53 |
| 250 | 0.65 | 0.70 | 0.64 |
| 300 | 0.74 | 0.84 | 0.74 |
| 400 | 0.96 | 1.13 | 0.93 |
| 500 | 1.19 | 1.46 | 1.23 |

The reward change reduces Action Std relative to the observation-only task,
but all three runs show a sustained increase. Thus these two reward weights
do not fix the user's concern about increasing policy distribution width.
Action Std describes the stochastic training policy; the evaluation metrics
above use its deterministic inference action. A rising Std alone does not
establish whether a deterministic policy is usable, but it does show this
reward ablation failed its intended Std trend check. The 125D hybrid variant
is the stronger candidate on fingertip precision plus subgoal throughput;
the original 105D policy still has higher complete-sequence reliability.
Further training to iteration 1000 was not needed to decide whether the
current reward change reverses the Std trend. Any further reward or PPO
entropy changes should be evaluated as a separate ablation.

## Reproduction and artifact scope

The new task IDs are `DexGym-Wuji-Right-Hand-Clean-Obs125-Direct-v0` and
`DexGym-Wuji-Right-Hand-Clean-Hybrid125-Direct-v0`. Both use the existing
`data/processed/hand_only_right_collision_clean` bank and independent PPO
experiment directories. Checkpoints, raw training logs, and validation JSON
files are saved locally under
`/home/ubuntu/dexgym/artifacts/hand_only_ablation_125/`; this ignored data
is not part of the Git commit. The iteration-500 checkpoint SHA-256 values
are recorded below:

- Observation-only: `6ab771ba46c49fb5afd512e699ebee2b7ed2bcf1d0d0712943b6aec0274a8399`
- Hybrid reward: `437aaa68a5b54da02fcf563c17c03a046d4876780f8133bea09ff1c0c7df36d4`

Both policies were trained with one PPO seed. Seeds 42 and 43 above change
validation episode sampling, not policy initialization. The 89 held-out
motions are from the screened clean library, so these results do not
establish performance on excluded GRAB motions or hardware.
