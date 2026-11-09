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

The v3 direct Actor consumes ten 125-dimensional observation frames and the
current 15-dimensional fingertip goal, concatenated into a 1265-dimensional input.
It has no separate goal encoder and sees no future reference frames. Its hidden
layers are ELU MLP (512, 256, 128), with output weights orthogonally initialized
at gain 0.01 and zero bias. The adapter applies tanh exactly once in both physics
and imagination. The Critic uses ELU MLP (512, 512, 256, 128) with its existing
162-dimensional privileged features. The world model uses (2048, 2048, 2048)
with LayerNorm/SiLU. Existing hand normalization and reward/termination heads remain.

The hand-local `supermimic_schedule.py` owns v3 training scheduling. The packaged
`agents/hand_s0.json` sets UTD=0.4 (2,3 physical steps alternating), 64 physical
steps of initial replay collection, and 500 complete WM-only iterations. Each
iteration performs 10 WM updates of 2048 windows. Iteration 501 starts Actor
updates with 1024 imagined rollouts of 32 steps, followed by 5 Critic epochs of
4 minibatches. The 500-iteration delay is fixed, never based on model error.
All environment lanes contribute to training; GRAB validation subjects remain
reserved for policy evaluation. The default number of environments remains 256.

Learning-rate decay is independent of the segment and total training lengths:
`trainer.lr_decay_iters=250000`, then a constant `trainer.lr_final_ratio=0.03`
floor, starting from `trainer.lr=2e-4`. `trainer.wm_only_iters` counts optimizer
iterations, whereas `trainer.warmup_steps` counts vectorized physical collection
steps. Precollection does not advance the optimizer iteration counter.

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

The changed architecture makes v3 checkpoints incompatible with v1/v2 weights;
start v3 training from scratch. An explicit `--config` pointing to the packaged
`agents/hand_s0_v2.json` retains the historical v2 architecture and no-warmup
schedule for old checkpoint evaluation/resume. Legacy v1 JSON still migrates to
v2 with a warning, not to v3. New v3 checkpoints bind the UTD, WM-only duration,
precollection and learning-rate schedule in the strict trainer configuration.
On fresh-physics resume, 64 new steps refill replay; the restored iteration
preserves the 2/3-step collection phase and does not restart the 500-iteration wait.

The bounded v3 checks and exact scope are recorded in the
[Dex alignment report](../../../../../docs/supermimic/hand_s0_dex_alignment_20261109.md).
