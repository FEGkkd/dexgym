# Hand 2.1 PPO frame tracking (2026-11-09)

The active `DexGym-Wuji-Hand21-TeleDexter-Direct-v0` task uses the official
standalone right-hand model and the existing 30 Hz GRAB motion bank. This
revision replaces the former goal-triggered random-span schedule with a
time-driven one-frame schedule. The separate Hand S0 SuperMimic task is not
changed.

At reset, a motion and a valid reference start frame `s` are sampled. The
first policy target is frame `s+1`. At every 30 Hz policy step, Newton runs
16 physics steps, then the five fingertip errors are evaluated against the
current target. The next target is exactly the next GRAB frame, regardless
of success. The last frame is scored before the episode ends. A fresh motion
is sampled on the next reset. Random target spans, the 20→40-frame
curriculum, 5–15-step hold, and random cross-trajectory switches are disabled.
Segments need at least two frames; the current bank has 950 eligible train
and 450 eligible validation segments under that rule.

All five fingertips being within **15 mm** marks that frame as hit. Each hit
receives the existing TELEDEXTER-style sparse term with fixed span one,
`6 × 0.75 × Σ_f exp(-β_f e_f)`, with `β_thumb=100` and `β_other=90`. Every
step also receives `0.1 × mean_f exp(-50 e_f) - 0.1`. Reward is computed for
the frame just evaluated, before the target advances. The 0.1-rad residual
action with 0.1 deadzone, 125D observation, Tanh Gaussian latent Std 1.0,
and PPO entropy coefficient 0.005 remain unchanged.

Consecutive frames outside the 15 mm threshold increment the miss counter;
a hit resets it. The episode terminates on the 46th consecutive miss, after
scoring that frame. Joint-limit/finite-state safety checks and the 30-second
cap remain. The fixed reference stride and curriculum bound are both one.
The new PPO experiment name contains `frame_tracking` so fresh runs cannot
silently resume the earlier random-span experiment.

The Newton smoke check ran 16 training environments for 120 policy steps.
It verified 1,907 nonterminal frame advances, each exactly one frame, and
reported a 0.0006 mm initial FK discrepancy. A 16-environment, one-iteration
PPO workflow check completed 512 environment steps and logged a fixed span and
curriculum bound of one, latent action Std 1.00, and finite rewards. Future
TensorBoard runs log `frame_hit_fraction`, `mean_frames_hit`,
`reference_stride_frames`, and `mean_consecutive_misses`. These are
functional checks; no PPO learning claim is made for the new definition.

An additional privileged joint-controller check covered all 450 validation
segments at the first reset and ran 20 policy steps per environment. It hit
8,935 of 9,000 scored frames (99.28%), with 1.41 mm mean fingertip error,
2.28 mm mean worst-finger error, 19 ordinary segment-end resets, and no
failure-budget timeout. This is a short controller baseline, not policy
performance or a full-sequence replay. The
host's Isaac runtime emitted a bus error during Python shutdown after both
smoke checks and the privileged check had written their results. Earlier
`model_3600.pt` policy-play and checkpoint metrics belong to the old task.
