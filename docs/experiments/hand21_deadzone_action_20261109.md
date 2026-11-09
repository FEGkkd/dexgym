# TELEDEXTER deadzone residual action on Hand 2.1 (2026-11-09)

**Subsequent task update:** The current trainable task ID is
`DexGym-Wuji-Hand21-TeleDexter-Direct-v0`; the Clean family described below
remains available for diagnostics. See
[the current task report](hand21_teledexter_task_20261109.md).

The official Hand 2.1 Clean Tracking task and its Obs125, Hybrid125, and
action-distribution variants now follow TELEDEXTER Appendix C.1's residual
joint target equation. For the bounded policy output `a_t` and previous
position-controller target `q_cmd,t-1`, each of the 20 joints receives

```text
deadzone_0.1(a) = sign(a) * max(abs(a) - 0.1, 0)
q_cmd,t = clip_to_joint_limits(q_cmd,t-1 + 0.1 rad * deadzone_0.1(clip(a_t, -1, 1)))
```

Actions in `[-0.1, 0.1]` hold the previous command exactly. The largest
single-step target change is 0.09 rad per joint. The previous bounded action
remains in the policy observation, and the 125D variants still expose
`q_cmd - q`. The action-change reward penalty and all tracking rewards and
terminal rules are unchanged. The historical 54-DoF task keeps its original
0.08-rad residual mapping with no deadzone.

The privileged Joint Replay and smoke controllers now invert the soft
deadzone when selecting an action for a desired joint target. The full-bank
replay script also excludes the 16 segments that cannot form a stride-four
subgoal, matching the task's sampler. PPO experiment directories have a new
`deadzone` name, so earlier checkpoints are not silently treated as if they
used this action mapping.

## Validation

- Unit tests cover the deadzone boundary, positive and negative residuals,
  action and joint-limit clipping, exact holding, inverse mapping for small
  joint steps, and the legacy task configuration.
- Hybrid125 Newton Joint Replay smoke: four environments, 120 policy steps,
  201 subgoals reached, 2.80 mm mean fingertip error; same-step terminal
  completion and goal timeout checks passed. The replay controller reads
  reference joint targets and is not a learned policy.
- Hybrid125 with 4,096 environments initialized and executed one 30 Hz step.
- Entropy0 Hybrid125 PPO completed one 16-environment iteration and 512
  environment steps; mean action standard deviation was 0.30. This checks
  the training pipeline, not tracking performance after learning.
- The validation Joint Replay entry point loaded all 446 eligible segments
  and ran one action step without a FK or motion-count mismatch.

Local logs are under `artifacts/hand21/` and are not committed. No 500- or
1,000-iteration PPO run was started as part of this action change.
