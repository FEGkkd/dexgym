# Phase 1 environment-transition budget audit — 2026-11-01

Historical snapshot: the user subsequently raised the cumulative limit to
100,000,000 transitions. The current accounting and 2048-environment results
are in [the PGS Mimic large-batch audit](phase1-pgs-mimic-large-batch-audit.md).
The 25-million figures below describe the earlier decision boundary only.

The Phase 1 research limit is **25,000,000 environment transitions** or
approximately **12 GPU hours**, whichever is reached first. This audit
reconstructs logged PPO interactions from all 24 local TensorBoard run
directories. It corrects the previous cumulative estimate in the strict-hold
report; it does **not** establish that the total research budget remains
available.

## Reproducible accounting

For each run, the script reads `num_envs` and `num_steps_per_env` from its
saved configuration and counts distinct TensorBoard update-step records.
It multiplies the three values, then sums the runs. Distinct steps avoid
counting duplicate event records within a resumed run. Configured maximum
iterations and checkpoint iteration labels are not used as actual update
counts. The run-level JSON output remains in ignored
`artifacts/phase1/phase1_budget_audit.json`.

```bash
/home/ubuntu/miniconda3/envs/allo_track/bin/python \
  scripts/environments/audit_phase1_budget.py \
  --budget 25000000 \
  --output artifacts/phase1/phase1_budget_audit.json
```

Observed output:

```text
PHASE1_BUDGET_AUDIT {"logged_ppo_transitions": 24124160, "upper_bound_remaining_total_transitions": 875840, "all_transitions_exhausted_proven": false}
```

The previous reported cumulative PPO count, 23,990,272, was **133,888 too
low**. The sidewall-manifold report's pre-final-round cumulative count is
correspondingly corrected from 23,171,072 to **23,304,960**. The final
25-update, 1024-environment strict-hold run added 819,200 transitions.

The **875,840** remainder is an **upper bound**: deterministic evaluations,
smoke tests, physical probes, and interrupted rollouts without logger records
are omitted. The actual all-purpose remaining transition budget is lower,
but cannot be reconstructed exactly from the available logs. GPU process
time was not reliably recorded either, so this audit makes no claim about
the separate 12-GPU-hour limit. A 1024-environment, 50-update run at a
32-step rollout would require 1,638,400 transitions and already exceeds
even the optimistic remainder. No further PPO run was started for this
accounting change.

## Research status at the budget boundary

The latest Mimic strict-hold checkpoint still achieved **0/200** strict
GraspHold successes on the held-out pregrasp set and **0/200** on the
training-family evaluation. A later scripted exact-target candidate achieved
**50/50** right-sidewall tripod steps in fresh-process Reset replay, but
failed strict hold because its reported screw speed remained above
3 rad/s while actual angle changed very little. It is not a validated bank
sample. CapTurn30 was not trained from a Mimic-compatible stable grasp
start. These observations are detailed in the
[strict-hold audit](phase1-strict-hold-audit.md) and
[Mimic grasp replay audit](phase1-mimic-grasp-replay-audit.md).

Accordingly, Phase 1 remains **partial**: the DirectRLEnv, batched physics,
PPO run path, checkpoints, and independent evaluation exist, while learned
grasp and cap-turn success are unproven. A new training extension first
needs a physically consistent screw speed under bilateral contact, then a
strictly validated Reset state and a deliberately bounded experiment. Raw
reward or checkpoint creation alone does not satisfy the learning gates.
