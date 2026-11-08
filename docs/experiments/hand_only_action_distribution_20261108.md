# Hybrid125 action-distribution diagnosis

**Historical task semantics:** These task IDs now use the standalone Hand 2.1
model and new GRAB bank. Use commit `3c57d1e` to reproduce these old results.

**Date:** 2026-11-08. This follows the 4096-environment Hybrid125 PPO run
documented in [the observation and reward ablation](hand_only_obs_reward_ablation_20261108.md).
The task, screened 51/89 train/validation motion bank, 480 Hz full-contact
Newton controller, 30 Hz policy, 125D observation, 20D incremental action, and
hybrid reward are unchanged. The experiment separately tests entropy and the
action distribution.

## Order and decision criteria

1. Audit the raw action before Isaac Lab's `clip_actions=1.0` wrapper and the
   task's own `clamp(-1, 1)`. Compare stochastic and deterministic inference
   from the same Hybrid125 checkpoint.
2. Resume that checkpoint for about 100 iterations with identical optimizer
   state and training seed in two branches: entropy coefficient 0.005 versus
   0. Compare the Action Std slope from checkpoint 500.
3. If clipping remains material, run short independent bounded-action pilots
   with a tanh-squashed Gaussian and Beta distribution. Use the same task and
   evaluation protocol, and inspect both learning and action support before
   any long training.

Validation uses all 89 held-out clean motions, 256 environments, and 1000
policy steps per environment. Seed 42 is used throughout; seed 43 replicates
the original clipping audit and the low-completion anomaly noted below. A
subgoal counts only
when the environment pays its strict five-finger success bonus; completed
episodes require the final subgoal. The action audit reports the fraction of
all 20D raw action components outside [-1, 1] and the fraction of environment
steps with at least one such component. The mean-action clipping fraction is
measured on the states encountered during stochastic execution. These are
pre-wrapper measurements; the executed action is clamped to [-1, 1].

## Stage 1: original Gaussian checkpoint 500

The original RSL-RL Gaussian is state-independent in standard deviation and
has unbounded samples. Its checkpoint-500 Action Std was 1.23. RSL-RL samples
the raw action; Isaac Lab clips it before `env.step()`, and the task also
clamps the received action.

| Evaluation seed | Execution | Raw component clipping | Steps with any clipping | Mean action outside bounds on sampled states | Mean tip error | Strict subgoals | Completed-episode rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | Deterministic mean | — | — | — | 4.718 mm | 109,171 | 94.70% |
| 42 | Gaussian sample, then clamp | 51.46% | 100% | 27.51% | 7.045 mm | 48,440 | 81.61% |
| 43 | Deterministic mean | — | — | — | 4.694 mm | 109,370 | 94.66% |
| 43 | Gaussian sample, then clamp | 51.49% | 100% | 27.67% | 7.023 mm | 49,838 | 80.57% |

The seed-42 stochastic run has raw action RMS 1.571 and executed action RMS
0.817. Its per-joint raw clipping rates range from 40.29% to 61.34%.
Because more than half of the scalar raw actions are changed by the wrapper,
many different Gaussian samples produce the same executed action. PPO still
computes likelihoods for the sampled raw actions, so this mapping does not
by itself prove an invalid policy gradient; it does create broad saturated
regions in which different policy outputs have the same environment effect.
The stochastic policy's lower tracking performance is consistent across both
evaluation seeds. It does not by itself determine how much of the rising Std
is caused by the entropy bonus; the paired continuation below tests that.

## Stage 2: paired entropy continuation

Both branches loaded the same `hybrid125_model_500.pt`, including optimizer
state, then trained 101 logged iterations (500 through 600) with 4096
environments and seed 42. The clean task and Gaussian distribution were
identical. The entropy-free branch has a separate task ID and PPO log
directory; its runtime `agent.yaml` confirms `entropy_coef: 0.0`. The control
retains `entropy_coef: 0.005`.

| PPO iteration | Entropy 0 Action Std | Entropy 0.005 Action Std |
| ---: | ---: | ---: |
| 500 | 1.23 | 1.23 |
| 525 | 1.17 | 1.30 |
| 550 | 1.13 | 1.36 |
| 575 | 1.09 | 1.44 |
| 600 | 1.06 | 1.49 |

Each continuation took about 346 seconds. Removing entropy reversed the Std
slope over this 100-iteration window, while the matched positive-entropy
control kept increasing. This supports the entropy bonus as a major driver of
the rising Std in this setup. It does not imply that all action clipping has
disappeared: the Gaussian is still unbounded and its mean can itself exceed
the execution bounds. Endpoint clipping and performance are audited below.

The same seed-42 stochastic validation protocol at the two endpoint
checkpoints shows the remaining saturation:

| Policy | Action Std | Raw component clipping | Policy mean outside bounds on sampled states | Mean tip error | Strict subgoals | Completed-episode rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid125 checkpoint 500, entropy 0.005 | 1.23 | 51.46% | 27.51% | 7.045 mm | 48,440 | 81.61% |
| Checkpoint 600, entropy 0 | 1.06 | 49.12% | 31.33% | 6.781 mm | 59,975 | 91.24% |
| Checkpoint 600, entropy 0.005 | 1.49 | 60.11% | 40.61% | 6.792 mm | 57,482 | 90.71% |

The entropy-free branch clips about 11 percentage points fewer raw components
than its paired control, but nearly half are still clipped. The policy mean
itself lies outside bounds for about 31% of components on the states visited
by this branch. Removing entropy improves the Std trajectory and stochastic
tracking, while leaving a substantial unbounded-action saturation problem.

Deterministic seed-42 validation also reveals a performance cost:

| Policy | Mean-action clipping | Mean tip error | Strict subgoals | Completed-episode rate |
| --- | ---: | ---: | ---: | ---: |
| Original checkpoint 500, entropy 0.005 | — | 4.718 mm | 109,171 | 94.70% |
| Checkpoint 600, entropy 0 | 24.70% | 5.391 mm | 98,308 | 92.21% |
| Checkpoint 600, entropy 0.005 | 33.15% | 4.581 mm | 109,990 | 95.93% |

Thus removing the entropy bonus reverses Std growth and reduces stochastic
clipping, but it also lowers deterministic tracking quality over this
continuation. Lower Action Std alone is not sufficient as the model-selection
criterion.

## Stage 3: bounded-distribution pilots

Three policies started from random initialization with seed 42, 4096
environments, entropy coefficient 0, and were continued from their existing
checkpoints to iteration 500. They differ in their action distributions: unbounded
Gaussian with hard clipping, tanh-squashed Gaussian with change-of-variables
log probability and latent-Gaussian KL, or Beta rescaled to [-1, 1]. All
three start with approximately 0.30 per-action standard deviation. The Beta
head uses a symmetric concentration initializer to match the Gaussian
initial spread; the native RSL-RL Beta initializer would start near 0.48.
Because the Beta head has twice as many actor outputs and a different
parameterization, these are short viability pilots rather than a strict
single-parameter ablation.

At checkpoint 100, stochastic validation with seed 42 gives:

| Distribution | Raw component clipping | Mean tip error | Strict subgoals | Completed-episode rate | Mean steps per reached goal | Goal-age-45 noncompletion terminations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian plus hard clamp | 0.40% | 6.088 mm | 72,112 | 78.56% | 3.225 | 494 |
| Tanh Gaussian | 0% | 6.177 mm | 68,787 | 67.01% | 3.130 | 863 |
| Beta | 0% | 6.516 mm | 69,290 | 84.82% | 3.489 | 291 |

Both bounded distributions remove clipping by construction. At this early
checkpoint, Beta improves complete-sequence reliability but advances fewer
goals and has higher fingertip error than the unbounded Gaussian control.
Tanh Gaussian is worse on completion at the same training budget. These
comparisons do not establish late-training rankings; in the unbounded
Gaussian, clipping was only 0.40% at iteration 100 but exceeded 50% by the
prior iteration-500 run with entropy 0.005. Deterministic validation and any
longer training are considered separately below.

Deterministic validation at the same checkpoint and seed gives:

| Distribution | Mean-action clipping | Mean tip error | Strict subgoals | Completed-episode rate | Goal-age-45 noncompletion terminations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian plus hard clamp | 0.10% | 5.399 mm | 80,101 | 83.35% | 404 |
| Tanh Gaussian | 0% | 5.543 mm | 74,610 | 71.22% | 784 |
| Beta | 0% | 5.720 mm | 90,722 | 91.79% | 208 |

Beta's higher complete-sequence rate and subgoal count at iteration 100 make
it the strongest bounded pilot, despite its somewhat higher average fingertip
error. Tanh Gaussian is behind both alternatives on completion and subgoals
at this short budget. To test whether the Beta advantage persists after more
learning, extend Beta and the entropy-free unbounded Gaussian control from
their own checkpoint 100 to checkpoint 250, keeping the tanh pilot at 100.

At checkpoint 250, the stochastic and deterministic seed-42 validation
results were:

| Distribution | Execution | Raw component clipping | Mean tip error | Strict subgoals | Completed-episode rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Gaussian, entropy 0 | Stochastic | 0.68% | 6.822 mm | 68,819 | 90.42% |
| Beta, entropy 0 | Stochastic | 0% | 6.634 mm | 62,611 | 82.22% |
| Gaussian, entropy 0 | Deterministic | 0.41% | 6.494 mm | 82,019 | 90.43% |
| Beta, entropy 0 | Deterministic | 0% | 6.023 mm | 80,758 | 83.03% |

Beta's checkpoint-100 advantage did not persist to checkpoint 250 in this
single training seed. The Beta run still avoids clipping. At the user's
request, all three from-scratch distribution arms stopped at checkpoint 500;
no earlier iterations were retrained. The separate 500-to-600 continuation
above is an entropy diagnostic and is not one of these capped distribution
runs.

At checkpoint 500, seed-42 validation gives the final matched comparison:

| Distribution | Execution | Logged Action Std | Raw component clipping | Mean tip error | Strict subgoals | Completed-episode rate | Mean steps per reached goal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian, entropy 0 | Stochastic | 0.19 | 0.95% | 6.113 mm | 62,640 | 72.95% | 3.707 |
| Tanh Gaussian, entropy 0 | Stochastic | 0.21* | 0% | 6.640 mm | 71,026 | 83.39% | 3.369 |
| Beta, entropy 0 | Stochastic | 0.21 | 0% | 6.920 mm | 57,520 | 87.16% | 4.262 |
| Gaussian, entropy 0 | Deterministic | 0.19 | 0.67% | 5.828 mm | 63,348 | 12.42% | 2.658 |
| Tanh Gaussian, entropy 0 | Deterministic | 0.21* | 0% | 6.276 mm | 83,476 | 82.28% | 2.814 |
| Beta, entropy 0 | Deterministic | 0.21 | 0% | 6.526 mm | 66,630 | 83.33% | 3.587 |

*For tanh Gaussian, RSL-RL logs the latent Gaussian Std before `tanh`;
the other two columns describe their respective action distributions and
should not be read as identical physical action spread.*

At the capped iteration 500, both bounded distributions have zero clipping.
Tanh Gaussian advances more subgoals and has lower mean fingertip error than
Beta, with 3.8 percentage points lower stochastic complete-sequence rate.
Beta takes longer per reached subgoal. Against the entropy-free unbounded
Gaussian, tanh Gaussian improves stochastic subgoal throughput and completion,
but its mean fingertip error is higher. These rankings come from one training
seed and one common evaluation seed for the distribution comparison.

The Gaussian's very low deterministic completion repeats with seed 43:
63,505 strict subgoals, 12.57% completed episodes, and 1,878 goal-age-45
noncompletion terminations, versus 12.42% completion and 1,904 such
terminations at seed 42. Both evaluations saw all 89 held-out motions. Its
stochastic policy completed 72.95% under seed 42. The different action modes
visit different state sequences, so this does not identify a specific failed
finger or motion; it establishes that low Action Std and low raw clipping do
not ensure a reliable deterministic policy.

The original positive-entropy Hybrid125 checkpoint 500 remains stronger in
deterministic tracking (109,171 subgoals, 94.70% completion, 4.718 mm error
at seed 42), while its sampled raw actions suffer 51.46% clipping and only
48,440 stochastic subgoals. The bounded arms remove that severe clipping and
improve stochastic subgoal count, but no bounded checkpoint here matches the
original policy's deterministic precision and throughput. The results favor
keeping action-distribution changes separate from later reward and GRAB
sampling work.

## Local artifacts

Raw logs and JSON results are in
`/home/ubuntu/dexgym/artifacts/action_distribution_diagnostic/` and remain
Git-ignored. Checkpoint-500 SHA-256 values are:

| Distribution | Checkpoint SHA-256 |
| --- | --- |
| Gaussian, entropy 0 | `7d20c4e1271f1960f6b72a9e1e411d0ae3641c7653773383161dc3e0fb2c736d` |
| Tanh Gaussian, entropy 0 | `df8340bb5020729ab928c3f3c6a3d16b02cb4a1a9dff1c24b76277fd6768fbae` |
| Beta, entropy 0 | `4c03e1aa9c65f9d807c3b0d45344512f712e63ede32452196d8eda675598148c` |

The source GRAB data and clean motion bank are unchanged. The five-finger
threshold remains 10 mm with two consecutive reached steps; no reward,
observation, or reference-trajectory processing change was made here. The
Python regression suite passes with 260 tests, 3 skipped; both bounded
distributions also passed a two-iteration real-Newton PPO smoke test before
the scaled runs.
