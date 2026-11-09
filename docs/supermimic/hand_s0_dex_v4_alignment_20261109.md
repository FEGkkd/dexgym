# Hand S0 v4: executed controls, exploration, normalization and loss

The four requested alignments are implemented in the independent
`source/dexgym/tasks/direct/hand21_supermimic/` task. The default is now v4.
**Implementation checks pass, but large-scale training readiness does not:
the preserved action deadzone causes zero Actor gradients in the short runs.**
No large-scale training was started. No surrogate gradients or additional
Actor gating were introduced to hide this result.

Reference: [Dex-Supermimic, commit 2becb738](https://github.com/CumulusAlpha/Dex-Supermimic/tree/2becb7380fed86171056f9bf4338b2a6a0c14b01),
particularly `common.py`, `networks.py`, and the fixed-variant collection,
world fitting and imagination code in `runner.py` under
`src/dex_supermimic/rl/dex_supermimic/`.

## Changes

| Part | v4 behavior |
| --- | --- |
| World-model action input | Actual executed 20D joint target `u`, after deadzone, accumulated increment and joint-limit clipping. Supervised WM8 uses captured `executed_command`; BPTT derives it differentiably from predicted state and Actor action. |
| Actor / collection | Actor returns `tanh(MLP(...))`; collector applies `clamp(action + sigma * noise, -1, 1)`. Previous-action observations store this bounded action. Physics, imagination and evaluation apply no second tanh. |
| Actor normalization | Empirical normalization of the complete 1265D input: ten 125D observations plus current 15D goal, with separate statistics for each history coordinate. Update real inputs before acting; use `std + 0.01`. No replay/imagination/evaluation updates. |
| WM normalization | Running population mean/variance on real `(state, context, executed_target)` transitions. Input std `sqrt(var + 0.01^2)`; delta std `sqrt(var + 0.0001^2)` for denormalization. Cap 2e8 samples; restore host count on resume. |
| Termination normalization | Same running input normalization on real arrival state and arrival task features. Existing architecture, BCE supervision and optimizer ownership retained. |
| Critic normalization | Running normalization from all H+1 detached imagined feature vectors after Actor optimization. Recompute values under new statistics before constructing fixed TD(lambda) targets. No real-collection Critic norm updates. |
| Physical WM loss | Coordinate scale `max(delta_std, floor)`, position floor 0.003 rad and velocity floor 0.05 rad/s. Mean over 40 dimensions, sum masked time steps, divide by batch size. |

Actor discounted-return and Critic mean-square objectives already use the same
reductions; they are retained. The physical WM loss now follows Dex's reduction.
The **total** world/head objective is not claimed to be identical to Dex:
existing termination BCE still averages valid steps and shares the WM optimizer;
`reach_head` still adds its unchanged mean BCE on true reached-event labels.
We did not import Dex's independent failure optimizer, mixed failure sampling,
contact objective, failure-probability cap or soft task counters.

`env.py`, `env_cfg.py`, `logic.py` and `subgoals.py` are unchanged. So are the
reach network, label definition, reward computation, reach loss weighting,
15 mm threshold, holds, curriculum and target switching. Observation construction
only dispatches previous action through the version-matched adapter so v4 avoids
a second tanh; v2/v3 retain their original behavior.

The 500 WM-only rounds, UTD 0.4, prefill 64, network sizes, batches and independent
LR schedule from v3 are unchanged. This remains the standalone single hand,
without objects/tactile sensing or a future reference encoder.

## Validation and the remaining deadzone issue

- Regression suite: 311 passed in 23.47 s after integrating concurrent PPO changes from main, with 35 existing torch.jit warnings.
- Numerical oracles cover normalization formulas/resume, logged commands,
  coordinate floors and masked time reduction, reach BCE invariance,
  pre-action norm timing, post-tanh exploration, frozen evaluation norms,
  imagined terminal inclusion, refreshed Critic targets and checkpoint compatibility.
- Full-size default run: 256 environments, 64 prefill steps, two iterations,
  17664 transitions, 20 WM optimizer steps, no Actor/Critic updates (500-round
  default WM-only schedule retained).
- Diagnostic changes only `wm_only_iters` to 1. Two rounds execute 20 WM,
  one Actor optimizer call and 20 Critic steps. Resume adds 64 fresh collection
  steps and one round: cumulative 30 WM, two Actor calls, 40 Critic steps,
  34560 real transitions. All nonfinite counters are zero.
- **Both Actor calls have gradient norm 0; decoder parameters remain exactly
  equal to initial parameters.** An optimizer call is not proof of useful learning.
- Cause: the retained TELEDEXTER controller uses
  `u = clip(q_cmd + 0.1 * sign(a) * max(abs(a)-0.1, 0), limits)`.
  Inside `abs(a)<0.1`, exact `du/da` is zero. The small-output Actor initialization
  enters this region. Collection noise can create WM data outside the deadzone,
  but is absent from deterministic imagination and does not itself provide an
  Actor gradient through that deadzone.
- Separate diagnostic: load the two-round checkpoint, prefill real replay,
  then add `atanh(0.4)` to the output bias **only in memory for this probe**.
  One full BPTT32/batch1024 update has gradient norm 5.088889 and maximum Actor
  parameter change 0.000146993. The executed-command path carries gradients
  outside the deadzone; this does not establish learning from original initialization.
  No perturbed weights or initialization change are shipped as the default.
- Strict held-out evaluation: 16 environments x 32 steps completes using the
  resumed v4 checkpoint, without normalization updates. This checks execution,
  not convergence or tracking accuracy.
- Checkpoint audits confirm real normalizer counts 17664 then 34560; Critic
  counts are 0 during WM-only, then 33792 and 67584 (33 x 1024 per Actor rollout).

**Before large training:** resolve the exact controller deadzone versus
small-output deterministic Actor initialization. This change deliberately does
not remove the deadzone, change initialization, add imagination noise or use a
straight-through derivative. Increasing training duration is not established as
a remedy. The shortened diagnostic does not prove what happens after all 500
default WM-only rounds.

## Compatibility and evidence

V4 cannot resume v3 weights under the new action/normalization contract. Explicit
legacy configurations remain in `agents/hand_s0_v2.json` and `agents/hand_s0_v3.json`.
The generic checkpoint module only gains optional normalizer/output-contract
hooks; old manifests remain unchanged. PPO files are untouched.

Raw logs, checkpoints, diagnostic scripts and reports are at
`/home/ubuntu/dexgym/artifacts/hand_s0/dex_align_v4/` on the Ubuntu host.
The machine-readable summary is
[hand_s0_dex_v4_alignment_20261109.json](../experiments/hand_s0_dex_v4_alignment_20261109.json).
The first evaluation invocation omitted required `--mode` and exited at CLI
parsing; the corrected `--mode subgoals` run completed normally.
Checkpoints contain the pre-change parent SHA because verification preceded the
commit; source file hashes in the summary identify the implementation tested.
