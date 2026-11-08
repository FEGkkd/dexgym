# Right-hand RL task switched to official Hand 2.1 (2026-11-09)

## Scope and compatibility

The existing Clean task ID and its Obs125, Hybrid125, Entropy0, TanhGaussian,
and Beta variants now load the **official standalone 20-DoF Wuji Hand 2.1 USD**
and `data/processed/hand21_right_geometry_feasible/manifest.json`. The
original `DexGym-Wuji-Right-Hand-Tracking-Direct-v0` remains the historical
54-DoF Marvin/Beta 2 task. No dual-arm or left-hand joints are present in the
switched Clean task.

The Clean task retains its 20D incremental joint action, 105D observation,
30 Hz policy, four-frame subgoal stride, all-five-finger 10 mm success rule,
two-step dwell, 45-step goal timeout, and reward weights. The 125D variants
retain command error in the observation and their prior reward/distribution
choices. Newton uses 480 Hz, 16 simulation steps per action, full hand
collisions, official grouped SI PD gains, and the validated `implicitfast`
integrator. Fingertip positions come from the official USD tip frames in the
right-wrist frame, matching the new Hand 2.1 FK used for `tip_ref`.

The new bank contains 1,404 segments. The sampler keeps all data on disk but
only samples segments with at least six frames for stride 4: 942 training
segments and 446 validation segments. It retains the s1–s7 train / s8–s10
held-out subject split. PPO experiment directories use a new `hand21` prefix
so earlier checkpoints and logs cannot be silently mixed with this model.
Earlier PPO checkpoints have not been evaluated on the new model or library.

## Verification

- Focused registration, task logic, terminal timing, and MotionBank tests:
  18 passed.
- Four-environment Clean and Hybrid125 120-step Newton action loops: passed;
  both reached 197 subgoals with mean fingertip error 3.02 mm under the
  privileged joint-target controller used by the smoke script. This is a
  task wiring check, not PPO performance.
- Hybrid125 same-step completion and goal-timeout terminal checks: passed.
- Hybrid125 4,096-environment initialization and one policy step: passed;
  initial FK mismatch was at most 0.009 mm.
- Entropy0 Hybrid125 PPO, 16 environments, one iteration: completed 512
  environment steps, 125D actor/critic inputs, and a valid PPO update.
  One iteration does not measure learned tracking quality.

The supporting smoke and PPO logs are local under `artifacts/hand21/` and
are not committed. The model/data validation and direct-control GRAB video
are described in [the companion report](wuji_hand21_model_data_20261109.md).

## Training entry point

After the GRAB library is present under `data/processed/`, the entropy-free
Hybrid125 variant can be launched with:

```bash
PYTHONPATH=source isaaclab train --rl_library rsl_rl \
  --task DexGym-Wuji-Right-Hand-Clean-Hybrid125-Entropy0-Direct-v0 \
  --num_envs 4096 --max_iterations 500 --seed 42 \
  --run_name hand21_entropy0_4096_s42 physics=newton_mjwarp
```

This command names a candidate training run; only the one-iteration PPO
workflow check above was executed during the model switch.
