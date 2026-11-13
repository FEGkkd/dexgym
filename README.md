# DexGym

DexGym shares one SuperMimic training engine across **Wuji Hand 2 Cube
Reorientation** and **Marvin M6 + dual Wuji Hand2.2 S0**. Both tasks collect
physical transitions in Isaac Lab 3.0 with Newton. The cube environment is
registered with Gymnasium. The Marvin S0 task uses a dedicated Motion Library
training entry point.

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

The earlier PPO task implementations are available in
[the pre-removal revision](https://github.com/FEGkkd/dexgym/tree/f2f416e/source/dexgym/tasks).
The [dual-arm Hand2.2 assets](docs/robots/marvin_hand22_upgrade_20261111.md)
and [motion data tools](data/README.md) remain available independently.

## Marvin Hand-centric Bimanual S0

- [S0 training, validation and native MuJoCo replay](docs/tasks/marvin54_hand_centric_s0.md).
- [S0 task implementation](source/dexgym/tasks/direct/marvin54_s0/) for one
  goal-conditioned 54 DoF Actor, learned dynamics and a physical failure head.

## Wholebody S0 Teleoperation

- [Wholebody S0 guide](source/dexgym/tasks/direct/wholebody_s0/README.md) covers MANUS and wrist Tracker MCAP processing, dual-arm retargeting, reference motion screening, Newton training, and native MuJoCo replay.
- Dedicated [PPO and SuperMimic tasks](source/dexgym/tasks/direct/wholebody_s0/) learn bounded residuals on the same 54-DoF retargeted references. The retargeted pose reward tracks both wrists, ten wrist-local fingertip poses, and the relative pose of the two wrists.

## Install and verify

Use Python 3.12 and the Isaac Lab 3.0 / Isaac Sim 6 / Newton GPU environment.
`uv sync --locked --group dev` installs project dependencies. The verified Ubuntu
host uses a process-local runtime wrapper:

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
