# Hand 2.1 15 mm PPO training diagnosis (2026-11-09)

## Evaluation contract

The 4,096-environment PPO run stopped at iteration 3812. This diagnosis uses
the exact active 15 mm task with the completed 20→40 frame curriculum,
5–15-step fingertip hold, random forward spans, and cross-trajectory goals.
Each evaluation has 446 validation environments and 900 policy steps, for
401,400 environment transitions. The initial reset covers every held-out
validation segment once; subsequent resets use the task's random RSI. Seed
20261109 is used in the main comparison; checkpoint 3600 and 3800 are also
compared with seed 20261110. This is a sampled evaluation, not exhaustive
replay of every goal of every segment. Later goal streams diverge with policy
behavior, so counts compare identical protocols rather than step-for-step
paired target outcomes.

`mean` executes the deterministic tanh policy mean, `sample` uses the
policy's learned Tanh Gaussian distribution, and `privileged` reads the
current `q_ref` for a controller ceiling. A reached goal, a failure-budget
timeout, and an episode time-limit truncation are counted separately.

| Controller, seed 20261109 | Reached | Goals / 1000 steps | Budget timeout | Episode truncation | Mean tip error | Reward / step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Privileged `q_ref` | 30,021 | 74.79 | 29 | 109 | 6.90 mm | 9.17 |
| Deterministic PPO 1900 | 28,175 | 70.19 | 14 | 197 | 12.39 mm | 6.56 |
| Deterministic PPO 3600 | 28,555 | 71.14 | 57 | 146 | 11.68 mm | 7.02 |
| Deterministic PPO 3800 | 27,679 | 68.96 | 79 | 182 | 11.95 mm | 6.91 |
| Sampled PPO 3800 | 25,796 | 64.27 | 132 | 226 | 13.77 mm | 5.47 |

| Second seed, deterministic PPO | Reached | Goals / 1000 steps | Budget timeout | Mean tip error | Reward / step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Checkpoint 3600 | 28,579 | 71.20 | 56 | 11.62 mm | 6.95 |
| Checkpoint 3800 | 27,830 | 69.33 | 72 | 11.88 mm | 6.93 |

## What the evidence supports

1. **The task is feasible at the controller level.** Privileged replay
   reaches 30,021 targets and times out 29 times. PPO is useful but still
   below the reference controller's target throughput. The data bank or
   Newton drive is therefore not a broad feasibility blocker for this sample.
2. **The final checkpoint is not the best of the measured checkpoints.**
   Checkpoint 3600 reaches 876 more goals than checkpoint 3800 with the first
   seed, and 749 more with the second. The training curve's apparent plateau
   hides a real checkpoint-selection problem. Checkpoint 3600 is the current
   measured candidate for goal throughput, but only two seeds and three
   checkpoints have been evaluated.
3. **Learned action noise has a measurable cost.** With checkpoint 3800,
   switching from deterministic to sampled actions reduces goal throughput
   from 68.96 to 64.27 goals per 1000 steps, increases timeouts from 79 to
   132, and raises mean fingertip error from 11.95 to 13.77 mm. The mean
   absolute action rises from 0.266 to 0.488, and the fraction of steps with
   at least one `|action| ≥ 0.95` rises from 6.7% to 39.9%. The end-of-run
   latent Std was about 0.61. These comparisons show the cost of sampling;
   they do not isolate the entropy coefficient as its cause. The Tanh
   Gaussian remains bounded, so this is not the old hard-clipping bug.
4. **Reward and goal throughput are not perfectly aligned in this sample.** Checkpoint 3800
   has better mean tip error and higher reward per step than checkpoint 1900
   on the first seed, yet reaches 496 fewer targets and has 65 more budget
   timeouts. Because the nonsparse per-step term is bounded between -0.1 and
   0, the average sparse bonus per reached goal is bounded by 93.49–94.92
   for checkpoint 1900 and 100.22–101.67 for checkpoint 3800. The task's
   sparse bonus multiplies an accuracy score by `(span + 5)` or `100` for a
   cross-trajectory target, while the time cost is only 0.1 per policy step.
   This can reward more precise successes even when goal throughput falls.
   The checkpoint comparison shows the mismatch in these rollouts; the exact
   causal contribution of each reward term still needs an ablation.
5. **Short-span failure budgets cause occasional timeouts even with q_ref control.** All 29
   privileged timeouts occurred at spans 2–19. Deterministic checkpoint 3800
   had 74 of its 79 timeouts in that range; sampled checkpoint 3800 had 114
   of 132. `1.5 × span` allows only a few outside-threshold frames for small
   spans. The absolute timeout rate remains low, so this is an edge case,
   not a catastrophic task-definition failure. These are not unavoidable
   failures on a fixed list of goals because later goal streams differ by
   controller. Cross-trajectory targets had
   no budget timeouts in this evaluation, consistent with their 300-frame
   budget.

The training log does not record validation goal throughput, goal timeout by
span, or goal duration. Its reward and mean episode length cannot substitute
for those task-level metrics. An adaptive PPO learning rate around
`1.5 × 10^-3` late in training and the large cross-target reward may also
contribute to fluctuations, but this evaluation does not establish either
as a root cause. The next controlled tests should separately measure reward
components at successful goals and vary action sampling spread, while
selecting checkpoints by held-out goals per 1000 steps and timeout rate.

Reproduce the target-level evaluation with
`scripts/evaluate_hand21_teledexter_policy.py`. Local result JSON files are
under `artifacts/hand21/ppo_15mm_4096_50k_s42/validation_*.json` and remain
excluded from Git with the GRAB data and checkpoints.

On this machine, the Isaac Lab Python runtime sometimes exits with `SIGBUS`
while cleaning up after `EVAL_DONE`; all seven 900-step JSON result files
were written and validated before that shutdown error. This runtime issue is
separate from the policy outcomes reported here.
