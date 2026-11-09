# Hand S0: removal of Actor Trust Gating (2026-11-09)

The standalone Hand21 SuperMimic task now uses the `hand21_s0_v2` experiment.
Its Actor and Critic update every iteration on fixed 32-step model rollouts,
after the world-model updates. Model prediction error does not permit, delay,
or shorten these updates. The physical task remains in `hand21_supermimic`;
PPO remains independently defined in `hand21_ppo`.

## Changes

- Removed the task-specific `HandModelTrust` implementation and its old import shim.
- Removed trust thresholds, model-validation lane allocation, and the 200-update
  world-model warmup. All physical environment lanes now contribute training data;
  GRAB validation subjects are still reserved for policy evaluation.
- Preserved WM8 training, BPTT32, TD(lambda), exploration, gradient clipping,
  finite-number checks, the standalone hand model, and the physical task rules.
- New Hand logs report `imagination_horizon` and omit trust-gate metrics.
- Legacy v1 experiment JSON migrates with a warning. Old gated checkpoints migrate
  only the scheduling fields and record their changes in `trainer_config_migration`.
  Other trainer, task, robot, network, optimizer, reference, and physics contracts
  are still checked. Resume restores models/optimizers/curriculum and collects fresh
  physics transitions; it does not promise identical trajectory continuation.
- Generic Core retains its old schedule for unrelated existing tasks; the Hand
  entry point supplies no trust evaluator and explicitly selects `every_iteration`.

## Verification

All commands exited successfully on the recorded Ubuntu Isaac Lab / Newton host.
These are bounded software and physics-simulation checks, not real-robot trials
or evidence of policy convergence.

| Check | Result |
| --- | --- |
| Regression suite | 285 passed, 35 existing torch.jit deprecation warnings, 19.89 s |
| Fresh default run | 256 environments, 8 iterations, 2,048 real transitions, 16 WM updates, 8 Actor updates, 160 Critic updates |
| Old checkpoint resume | `pilot_v2/model_500.pt` to iteration 502; 512 additional real transitions, 4 WM updates, 2 Actor updates, 40 Critic updates |
| Update schedule | Every fresh and resumed iteration performed the Actor update at horizon 32, including the first iteration |
| Numerical checks | World/Actor/Critic nonfinite counters remained zero |
| Configuration/checkpoint tests | Immediate updates without validation, fixed-horizon resume, explicit legacy migration, rejection of unrelated config changes |

Machine-readable segment reports are preserved in
[`hand_s0_no_actor_gate_20261109.json`](../experiments/hand_s0_no_actor_gate_20261109.json).
Raw logs and checkpoints remain under
`/home/ubuntu/dexgym/artifacts/hand_s0/no_gate/`; the full regression log is
`/home/ubuntu/dexgym/artifacts/hand_s0/no_gate_regression.log`.
No large-scale training was launched. `policy_accuracy_passed` remains `null`.

## Reproduce the short checks

```bash
LIBRARY=/home/ubuntu/dexgym/data/processed/hand21_right_geometry_feasible/manifest.json
bash scripts/supermimic/hand_s0_host.sh \
  -m dexgym.tasks.direct.hand21_supermimic.supermimic_train \
  --library "$LIBRARY" --iterations 8 --output /tmp/hand_s0_no_gate/fresh
bash scripts/supermimic/hand_s0_host.sh \
  -m dexgym.tasks.direct.hand21_supermimic.supermimic_train \
  --library "$LIBRARY" \
  --resume /home/ubuntu/dexgym/artifacts/hand_s0/pilot_v2/model_500.pt \
  --iterations 2 --output /tmp/hand_s0_no_gate/resume
```
