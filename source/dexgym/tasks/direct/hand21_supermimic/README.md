# Standalone Hand21 TeleDexter / SuperMimic task

Gym task: `DexGym-Wuji-Hand21-TeleDexter-SuperMimic-Direct-v0`.
The environment class is `Hand21SuperMimicEnv` in `env.py`; task parameters are
`WujiHand21SuperMimicCfg` in `env_cfg.py`. Edit this directory to change SuperMimic
rules. Its environment, configuration, `logic.py` and `subgoals.py` are independent
copies with no imports from the PPO task. The physical collector and imagined
rollouts both use this directory's rules.

The PPO task has its own [implementation](../hand21_ppo/README.md).

This task uses the official standalone right Wuji Hand 2.1: 20 controlled joints,
fixed wrist, full hand collisions, Newton MJWarp at 480 Hz and commands at 30 Hz.
There are no manipulated objects or tactile observations. GRAB supplies wrist-frame
robot fingertip goals through the geometry-screened motion library.

All five fingertips must enter 15 mm and remain for 5–15 steps. The task retains
the existing random-span curriculum, cross-trajectory switches, reset sampling,
and failure budgets. This trains causal fingertip subgoal following; it does not
establish exact human joint-angle reproduction or continuous teleoperation quality.

## Code ownership

| Component | Task implementation |
| --- | --- |
| Physical environment and configuration | `env.py`, `env_cfg.py` |
| Shared physical/imagination goal rules | `subgoals.py`, `logic.py` |
| Standalone robot state, controls and differentiable FK | `supermimic_robot.py`, `hand_kinematics.py` |
| Actor/Critic inputs, reward and termination | `supermimic_task.py` |
| World model and supervised reach event head | `supermimic_model.py` |
| Pre-reset transition collection | `supermimic_collector.py` |
| Registered experiment configuration | `supermimic_cfg.py`, `agents/hand_s0.json` |
| Component construction and checkpoint physics contract | `supermimic_runner.py` |
| Training, evaluation, smoke and data preflight | `supermimic_train.py`, `supermimic_evaluate.py`, `supermimic_smoke.py`, `supermimic_preflight.py` |

The generic replay, WM8 optimizer, BPTT32 Actor, TD(lambda) Critic,
and checkpoint machinery remain in `dexgym.algorithms.supermimic.core`.
No task module imports implementation from `scripts` or a repository-root config
directory. Scripts with the old Hand S0 names are thin compatibility launchers.

Actor input is a ten-step history of 125-dimensional proprioception/current-goal
observations plus the current five fingertip positions (15 values). It sees no
reference joint angles or future reference frames. The Critic may use privileged
reference landmarks. The world model learns from real pre-reset transitions;
Actor and Critic update on fixed 32-step model rollouts each iteration after
the world-model update.
All environment lanes contribute to training. Validation subjects remain reserved
for policy evaluation.
The v2 hand experiment removes prediction-error gating, its thresholds and
world-model warmup.

## Entry points

Use an installed DexGym / Isaac Lab 3.0 environment. Set `LIBRARY` to the processed
manifest. CLI `--config` can override the packaged JSON; omit it for the registered
default. The default experiment has 256 environments and a 250,000-iteration total
schedule, so use `--iterations` to bound a workflow check.

```bash
LIBRARY=/absolute/path/to/hand21_right_geometry_feasible/manifest.json

python -m dexgym.tasks.direct.hand21_supermimic.supermimic_preflight \
  --manifest "$LIBRARY" --output /tmp/hand_s0/preflight.json

python -m dexgym.tasks.direct.hand21_supermimic.supermimic_smoke \
  --library "$LIBRARY" --num-envs 16 --steps 64 --output /tmp/hand_s0/smoke.json

python -m dexgym.tasks.direct.hand21_supermimic.supermimic_train \
  --library "$LIBRARY" --iterations 2 --output /tmp/hand_s0/train

python -m dexgym.tasks.direct.hand21_supermimic.supermimic_train \
  --library "$LIBRARY" --resume /tmp/hand_s0/train/model_2.pt \
  --iterations 2 --output /tmp/hand_s0/resume

python -m dexgym.tasks.direct.hand21_supermimic.supermimic_evaluate \
  --library "$LIBRARY" --checkpoint /tmp/hand_s0/resume/model_4.pt \
  --mode subgoals --num-envs 16 --steps 64 --output /tmp/hand_s0/eval
```

Training and subgoal evaluation construct the environment with `gym.make` and
load both configurations from the registry. `fixed_clock` evaluation deliberately
uses a task-local diagnostic subclass, advances one reference frame per control
step, and reports uncompensated errors over all eligible validation clips. It
does not replace the training MDP.

Isaac Lab's generic PPO CLI does not implement this custom learner. Use the task
module entry points above. On the recorded Ubuntu host, the existing
`scripts/supermimic/hand_s0_host.sh` runtime wrapper is required for its repaired
Python dependencies; it accepts `-m` with the same task module names, and its script commands remain compatible. See
[readiness evidence](../../../../../docs/supermimic/hand_s0_readiness_20261109.md)
and [integration checks](../../../../../docs/supermimic/hand_s0_task_integration_20261109.md).

Checkpoint module names and executable contracts are preserved. Resume restores
model/optimizer state and curriculum progress, then starts fresh physics/replay
and immediately resumes the fixed-horizon update schedule. It is not a bitwise
continuation of physical trajectories.
This integration establishes software readiness, not a converged policy or real-robot performance.

Legacy v1 experiment JSON is migrated to v2 with a warning. Resuming a gated checkpoint
records the schedule changes in `trainer_config_migration`; other configuration and model
contracts remain strict. New logs use `imagination_horizon=32` and contain no trust-gate metrics.

Current v2 checks are recorded in the [gate-removal report](../../../../../docs/supermimic/hand_s0_no_actor_gate_20261109.md).
