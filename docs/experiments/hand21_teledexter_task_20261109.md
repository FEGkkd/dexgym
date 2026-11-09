# Hand-only TELEDEXTER task on Wuji Hand 2.1 (2026-11-09)

## Active task and data gate

`DexGym-Wuji-Hand21-TeleDexter-Direct-v0` is the current right-hand PPO task.
The old 54-DoF and Clean/ablation Gym IDs retain their historical environment
configurations for diagnostic scripts, but have no active PPO registration.
Historical code, reports, checkpoints, and local GRAB files were retained.

The task uses the official 20-DoF Hand 2.1 USD in Newton, full hand collision,
480 Hz physics, and 30 Hz policy actions. Actions use the 0.1-rad residual
target with a 0.1 soft deadzone; the observation is the 125D vector including
the accumulated joint-command gap. The train/validation split remains
s1–s7 / s8–s10.

Before changing the subgoal schedule, a **full Joint Replay** drove the
reference joint targets through the actual 30 Hz environment action loop with
20 mm all-five-finger success, two-step dwell, and fixed stride four. All
1,388 eligible segments completed with no goal timeout or episode truncation:

| Split | Completed segments | Reached subgoals | Mean active fingertip error |
| --- | ---: | ---: | ---: |
| Train | 942/942 | 70,567 | 3.57 mm |
| Validation | 446/446 | 30,679 | 2.98 mm |

The largest transient fingertip error was 89.1 mm on train and 102.6 mm on
validation while approaching new goals. Completion means each goal was
eventually reached within the task rules, not that every intermediate frame
was inside 20 mm. The privileged controller receives reference joint angles;
these numbers measure data and controller feasibility, not PPO performance.
The local JSON records are `artifacts/hand21/joint_replay_20mm_{train,validation}_full.json`.

## Paper-style hand-only task rules

The following rules adapt TELEDEXTER Appendix C.2–C.4 to a task with no
object and a 30 Hz GRAB bank:

- A goal is reached when all five fingertip errors are below **20 mm** for
  `Nstay` consecutive policy frames. `Nstay` is sampled uniformly from 5–15
  at reset and after every success.
- Each successful in-trajectory transition samples its next forward span
  uniformly from 1 through the current curriculum bound. TELEDEXTER's
  40→80 frames at 60 Hz correspond to **20→40 GRAB frames at 30 Hz**.
  The upper bound follows the paper's cubic schedule with `sigma_min=0.7`
  over 25,600 per-environment policy frames. RSI samples initial states from
  the first 90% of each reference segment.
- With probability 0.1 after a successful goal, the next target switches
  to another segment of the same train/validation split at the **same frame
  index**. Destination sampling is uniform among other segments containing
  that index. At a segment's last frame the task attempts this switch; if no
  valid destination exists, the episode completes.
- For a reached in-trajectory goal, the sparse bonus is `(span + 5) × score`,
  where `score = 1.5 × 0.5 × Σ_f exp(-β_f e_f)`, with β = 100 for the thumb and
  90 for each other finger. A step that performs a cross-trajectory switch
  uses weight 100. The hand-only dense term stays the mean of
  `exp(-50 e_f)`, scaled by 0.1; the time cost is 0.1 per step. No object
  reward or action-change penalty is added.
- An out-of-tolerance policy frame increments `n_fail`; success resets it.
  The current goal terminates on the first step with
  `n_fail > 1.5 × actual_span`. After a cross-trajectory switch, the next
  goal receives a flat 300-frame failure budget. Joint-limit and finite-state
  safety checks and the 30-second episode cap remain in place.

The fixed-stride `max_goal_steps=45` rule is only used by the diagnostic
Clean task; the new task uses the cumulative failure budget above.

## PPO and validation status

The active RSL-RL PPO config uses a **Tanh Gaussian** actor with latent
`init_std=1.0`, `entropy_coef=0.005`, 32 rollout steps, and the existing
0.1-rad/0.1-threshold residual deadzone. Tanh bounds sampled actions within
`[-1, 1]` before the environment's safety clamp. The experiment directory is
`dexgym_wuji_hand21_teledexter_20mm_tanh_std1_entropy005`. The old PPO
configs are no longer attached to Gym task IDs.

The new task passed unit tests for the schedule, sampled span, variable
dwell, hand-only sparse reward, same-step failure-budget termination, and
valid cross-trajectory indexing. Newton smoke runs reached 80 goals with
the normal 0.1 switch probability in eight environments over 120 steps;
with switch probability forced to one, 56/56 reached goals switched to a
different segment at the same frame. A 4,096-environment one-step check
passed. A 16-environment, one-iteration PPO check with the requested Tanh
Gaussian configuration completed 512 environment steps; the reported latent
action standard deviation was 1.00, and the entropy estimate was finite.
This is a
functional check; no large-scale PPO learning result is claimed.

Local smoke and PPO logs are under `artifacts/hand21/` and are not committed.

The active task can be launched for a future 4,096-environment PPO run with:

```bash
PYTHONPATH=source isaaclab train --rl_library rsl_rl \
  --task DexGym-Wuji-Hand21-TeleDexter-Direct-v0 \
  --num_envs 4096 --max_iterations 500 --seed 42 \
  --run_name hand21_tanh_std1_entropy005_s42 physics=newton_mjwarp
```

This command is documented for the next training stage; only the one-iteration
workflow check above was run during task construction.
