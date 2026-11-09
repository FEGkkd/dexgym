# Hand 2.1 TELEDEXTER action audit, 4096 environments × 300 PPO iterations

**Date:** 2026-11-09. **Task:** `DexGym-Wuji-Hand21-TeleDexter-Direct-v0`.
The run starts from random initialization with seed 42, 32 policy steps per
iteration, the official Hand 2.1 right-hand model, the GRAB geometry-feasible
motion bank, Newton MJWarp, and the existing 0.1 deadzone residual controller.
The saved agent configuration confirms `TanhGaussianDistribution`, latent
`init_std: 1.0`, and `entropy_coef: 0.005`. The task keeps the 20 mm fingertip
threshold and the TELEDEXTER-style subgoal curriculum.

**Historical reward caveat:** This run predates the cross-trajectory reward
attribution fix. It assigned the weight-100 sparse bonus when a successful
step selected a cross-trajectory target for the *next* step, rather than when
that cross-trajectory target was actually reached. Its action statistics and
checkpoint describe that earlier reward definition. They do not measure PPO
learning under the corrected completed-goal reward.

## Measurement

The audit observes each sampled PPO rollout action immediately before the
environment step. It does not alter the sampled action, policy, update, task,
or reward. Counts are stored for every iteration and every exact active goal
span from 0 through 40 frames. Span 0 denotes a **cross-trajectory target**:
the policy has switched to another motion at the same reference frame and is
now chasing that target. It measures time spent pursuing cross-trajectory
targets, not just the instant at which a switch is triggered.

- **Latent policy mean over limit:** fraction of 20 action coordinates with
  `|μ| > 1`, where `μ` is the Gaussian mean before tanh. The executed
  deterministic mean `tanh(μ)` remains within `[-1, 1]` by construction.
- **Latent Gaussian sample over limit:** fraction with `|z| > 1`, measured
  equivalently as `|tanh(z)| > tanh(1)` on the action actually passed to the
  wrapper. This is an action that would have exceeded the old hard action
  bounds without tanh; it is **not** a clipping event in this run.
- **Executed action saturation:** fraction with `|tanh(z)| ≥ 0.95`. The CSV
  also records the fraction of environment steps with at least one saturated
  coordinate. All table percentages below are coordinate-wise unless marked
  otherwise.

## Result

The run completed **300 iterations and 39,321,600 environment steps** in
842.63 seconds. The exact-span CSV has 12,300 rows (41 spans × 300 iterations),
and every iteration sums to 4096 × 32 environment steps. The final checkpoint
is `model_299.pt` and records iteration 299.

| PPO iterations | Latent Std at window end | Mean `|μ|>1` | Sample `|z|>1` | Executed `|a|≥0.95` |
| --- | ---: | ---: | ---: | ---: |
| 0–49 | 0.86 | 1.34% | 31.39% | 6.58% |
| 50–99 | 0.78 | 4.34% | 28.51% | 5.40% |
| 100–149 | 0.73 | 4.75% | 26.02% | 4.39% |
| 150–199 | 0.71 | 4.63% | 24.34% | 3.65% |
| 200–249 | 0.70 | 4.94% | 24.01% | 3.52% |
| 250–299 | 0.69 | 4.54% | 23.50% | 3.24% |
| All 300 | — | 4.09% | 26.29% | 4.46% |

The learned latent Std fell from 1.00 to 0.69 while the entropy coefficient
stayed at 0.005. The latent mean violation rose during early learning, but
Gaussian sample violations and executed near-bound actions both declined.
In the final 50 iterations, 46.06% of environment steps had at least one of
20 action coordinates with `|a|≥0.95` even though only 3.24% of individual
coordinates were saturated. Tanh prevents hard clipping; it does not remove
near-bound actions.

| Active target | Environment steps | Executed saturation, all 300 | Executed saturation, last 50 |
| --- | ---: | ---: | ---: |
| Cross-trajectory target, span 0 | 9,945,785 | 5.56% | 4.13% |
| Span 1–5 | 5,361,389 | 3.39% | 2.25% |
| Span 6–10 | 5,810,378 | 3.88% | 2.57% |
| Span 11–20 | 13,116,381 | 4.51% | 3.03% |
| Span 21–29 | 5,087,667 | 4.01% | 3.43% |
| Span 30–40 | 0 | — | — |

Cross-trajectory targets have higher observed saturation than ordinary short
subgoals. This is descriptive: target type, trajectory difficulty, and
curriculum stage are not controlled separately. The largest observed span
was 29 frames. The curriculum needs 25,600 steps per environment to reach
its maximum, while this run supplied 9,600, so spans 30–40 were not sampled.

At iteration 299, PPO logged 14.6 mm mean episode fingertip error, 0.0699
goal-reach fraction per policy step, and about 30.1 reached goals per episode.
These training metrics do not establish held-out tracking performance.

## Artifacts and runtime note

- `artifacts/hand21/action_audit_4096_300_s42_exact_span.csv`: raw counts by
  exact span and PPO iteration.
- `artifacts/hand21/action_audit_4096_300_s42_summary.json`: calculated rates,
  span bins, PPO log milestones, and run metadata.
- `artifacts/hand21/hand21_action_audit_4096_300_s42_model_299.pt`: final
  checkpoint.
- `artifacts/hand21/ppo_action_audit_4096_300_s42.log`: training log.

The host's shared virtualenv had multiple native files returning kernel I/O
errors. This run used temporary local package overlays without changing the
project model or the shared virtualenv. MuJoCo, Warp, Newton, NumPy, SciPy,
Isaac Lab, and Omni Client matched their installed versions. The intact USD
Exchange binding was version 3.0.0 rather than the damaged 3.0.1 copy.
Environment creation, all 300 PPO iterations, CSV writing, and the final
checkpoint completed. A Bus error occurred during process cleanup afterward,
so the shell exit status was 135. Repeating the experiment on a repaired
runtime would remove that provenance caveat.

The normal entrypoint for a healthy installation is:

```bash
PYTHONPATH=source python scripts/train_hand21_action_audit.py \
  --task DexGym-Wuji-Hand21-TeleDexter-Direct-v0 \
  --num_envs 4096 --max_iterations 300 --seed 42 \
  --run_name hand21_action_audit_4096_300_s42 physics=newton_mjwarp
```
