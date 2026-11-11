# DexGym

Isaac Lab 3.0 / Newton dexterous-hand tasks. The supported SuperMimic task is
**Wuji Hand 2 · Cube Reorientation**, trained through the reference-aligned
`dex_reference` engine. Independent PPO tasks remain available.

## Cube SuperMimic

- Task ID: `DexGym-Wuji-Hand2-Cube-Reorientation-SuperMimic-Direct-v0`.
- [Task and complete usage guide](docs/tasks/hand2_cube_supermimic.md).
- [Task implementation](source/dexgym/tasks/direct/hand2_cube_supermimic/): environment,
  reward/goal rules, training, evaluation, rendering and TensorBoard.
- [Training engine](source/dexgym/algorithms/supermimic/dex_reference/): real collection,
  replay, world model, differentiable Actor updates, TD(lambda) Critic and checkpoints.
- [Cube and hand assets](source/dexgym/assets/wuji_hand2_cube/).

The 4096-environment, 8000-iteration run completed 2387 goals in 512 independent
native evaluation episodes (4.662 goals/episode). See the
[final evaluation](docs/experiments/cube_reorient_final_evaluation.md) for the
remaining gap to the public reference policy.

Training uses 64 initial collection steps, 500 WM-only iterations, WM horizon 8
and fixed Actor horizon 32. Ordinary terminal Critic bootstrap is active from the
first Actor update. There is no Model Trust Gate, progressive 1/8/32 admission,
or Critic bootstrap weight ramp.

The early Marvin experiments and object-free Hand21 SuperMimic implementation
have been removed. Their source and results remain in Git history before this
cleanup; they are not runnable tasks on current main. Shared robot assets used
by the retained PPO environments remain packaged.

## Install and verify

Use Python 3.12 and the Isaac Lab 3.0 / Isaac Sim 6 / Newton GPU environment.
`uv sync --locked --group dev` installs project dependencies. The verified Ubuntu
host uses a process-local wrapper, without changing the shared PPO environment:

```bash
bash scripts/supermimic/host.sh --pytest tests -q
bash scripts/supermimic/host.sh -m dexgym.tasks.direct.hand2_cube_supermimic.validate_native --output /tmp/cube_native.json
bash scripts/supermimic/host.sh -m dexgym.tasks.direct.hand2_cube_supermimic.smoke --output /tmp/cube_smoke.json
bash scripts/supermimic/host.sh -m dexgym.tasks.direct.hand2_cube_supermimic.train --num-envs 4096 --output /path/to/new_run
```

The host wrapper uses cached binary overlays whose historical directory names
include `hand_s0` and `marvin_supermimic_m42`. These are runtime dependencies,
not task definitions or training methods. They must remain available on this
host. `DEXGYM_RUNTIME_OVERLAY` can select another compatible overlay root.
Other machines must configure their own compatible runtime.

## Right-hand fingertip tracking

The original hand-only task uses GRAB right-hand MANO poses retargeted by the
local WujiHand2 SDK. It controls 20 right-hand joints with PPO, holds both arms
and the left hand at Home, and follows five robot-FK fingertip targets in the
right-wrist frame with full hand collisions and a 10 mm all-finger goal threshold.
It uses no object state. Local recordings and generated
motions are excluded from Git; [data/README.md](data/README.md) describes the
required layout and conversion command. The
[pilot report](docs/experiments/hand_only_right_pilot.md) records task settings,
data counts, and the small PPO check. The
[4096-environment PPO report](docs/experiments/hand_only_4096_1000_s42.md)
records the 1000-iteration training and held-out tracking results. The
[direct position-target hold probe](docs/experiments/hand_only_position_hold_20261108.md)
measures joint and fingertip drift when a GRAB reference target is held for
1.5 seconds. The [three backend controls](docs/experiments/hand_only_backend_controls_20261108.md)
compare zero gravity, one-joint response, and source MuJoCo versus Newton.
The [layered Hand-only drive validation](docs/experiments/hand_only_drive_layered_validation_20261108.md)
records the isolated USD gain correction, timestep and collision sweep, and
324-pose fixed-target acceptance test. The
[self-collision post-optimization report](docs/experiments/hand_only_collision_postopt_20261108.md)
describes the separate GRAB refinement library and full-contact Newton check.

The current Clean task family uses the official standalone
[Wuji Hand 2.1 model](source/dexgym/assets/wuji_hand21/README.md) and the
full s1–s10 geometry-screened GRAB library. The 54-DoF original task ID
remains available for historical comparisons. See the
[model-switch report](docs/experiments/hand21_rl_model_switch_20261109.md)
for the exact task IDs, compatibility note, and smoke results.
The [deadzone action report](docs/experiments/hand21_deadzone_action_20261109.md)
records the current action equation and its validation.
The [current frame-tracking task report](docs/experiments/hand21_frame_tracking_task_20261109.md)
defines the active one-frame target schedule and Newton smoke check. The
[historical TELEDEXTER task report](docs/experiments/hand21_teledexter_task_20261109.md)
records the full-library Joint Replay and earlier random-span training rules. The
[current-rule privileged replay](docs/experiments/hand21_teledexter_privileged_replay_current_rules_20261109.md)
reports target timing and failure by exact goal span under the historical 20 mm setting. The
[15 mm training record](docs/experiments/hand21_teledexter_15mm_4096_50k_run_20261109.md)
records the 4,096-environment run stopped at iteration 3812. The
[checkpoint diagnosis](docs/experiments/hand21_15mm_training_diagnosis_20261109.md)
compares held-out target throughput, timeouts, and policy sampling. The
[Hand 2.1 motion distribution report](docs/experiments/hand21_motion_distribution_20261109.md)
summarizes joint coverage, fingertip speeds, and goal jumps by reference span. The
[4096-environment action audit](docs/experiments/hand21_action_audit_4096_300_s42.md)
records the 300-iteration PPO run, latent Gaussian limit rates, and executed
action saturation by target span and cross-trajectory target. That run used
the earlier switch-step reward timing and predates the completed-goal reward
fix. Earlier
right-hand Gym IDs remain available for diagnostics but no longer register PPO
training configurations.

```bash
PYTHONPATH=source python scripts/smoke_hand21_teledexter.py \
  --num-envs 8 --steps 120 --split validation
PYTHONPATH=source python scripts/train_hand21_action_audit.py \
  --task DexGym-Wuji-Hand21-TeleDexter-Direct-v0 \
  --num_envs 4096 --max_iterations 300 physics=newton_mjwarp
```

当前双臂双手模型已升级为与 Cube 任务同源的 Hand2.2，使用显式 SI 驱动并默认开启手部接触。详见[资产升级与 Newton 验证](docs/robots/marvin_hand22_upgrade_20261111.md)。
