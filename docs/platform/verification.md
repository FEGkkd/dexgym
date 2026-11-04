# DexGym 2.0 host verification — 2026-11-04

These are observed results from the isolated refactor worktree, not claims of
policy quality. Commands use `uv` for the temporary uv 0.13.7 executable used
on this host. All training and checkpoint paths below are relative to the
DexGym repository; the large checkpoint and export files remain ignored by Git.

## Source and host

| Item | Observed value |
| --- | --- |
| Isaac Lab | official `v3.0.0`, commit `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`; installed wheel 3.0.0 |
| Newton / Warp | 1.6.1 / 1.17.0 |
| PyTorch / CUDA build | 2.11.0 / 13.0; `torch.cuda.is_available()` true |
| RSL-RL | `rsl-rl-lib` 5.5.1 |
| Python / uv | 3.12.11 / 0.13.7 |
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB, compute capability 12.0 |
| Driver / host CUDA | 595.84 / `nvidia-smi` 13.2; `nvcc` 13.2.86 |
| Disk before install | 955 GB available |
| Backend | `physics=newton_mjwarp`, `--viz none`, CUDA device 0 |
| Source protection | original dirty branch `feat/teledexter-paper-reproduction` left untouched; worktree branch `refactor/isaaclab3-newton` from `eeba754` |

This host's Conda Python loads an older C++ runtime when `omni.client` imports.
The initial official baseline failed with missing `GLIBCXX_3.4.30`. A minimal
`import omni.client` reproduced the error; using the process-only
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` made the import and the
subsequent Newton runs succeed. No system or Isaac Lab library was changed.
The ambient ROS Jazzy pytest plugin lacks its own `lark` dependency; tests
passed with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

## Gate A — static and package

- `uv lock --check`: passed; lock resolved 193 packages.
- `uv sync --group dev`: passed; installed DexGym 2.0.0 and Isaac Lab 3.0.0.
- `uv run ruff check source/dexgym tests/newton scripts/smoke_newton.py`: passed.
- `uv run ruff format --check source/dexgym tests/newton scripts/smoke_newton.py`: passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/newton`: 3 passed.
- `uv run isaaclab list_envs --keyword DexGym`: exactly the Direct and Manager IDs.
- `uv run isaaclab train --rl_library rsl_rl --help` and corresponding `play --help`: inspected before bounded runs.
- `git diff --check`: passed. Local asset manifest hashes validated by pytest.

## Gate B — official Cartpole Newton baseline

```bash
env PYTHONPATH= LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  timeout 600s uv run --locked isaaclab train --rl_library rsl_rl \
  --task Isaac-Cartpole-Direct --num_envs 16 --max_iterations 5 \
  --viz none physics=newton_mjwarp
```

Passed with exit code 0. RSL-RL logged 5 learning iterations, 1,280 total
steps, and finite loss values. The official Cartpole USD and ground layer were
loaded; Newton initialized on CUDA. Checkpoint `model_4.pt` was written in the
local `logs/rsl_rl/cartpole_direct/` run directory.

## Gate C — DexGym runtime and PPO

The official `isaaclab zero_agent --task <ID> --num_envs 16 --max_steps 16
--viz none` command also exited 0 for both IDs without a `physics=` override.
The logs loaded `isaaclab_newton` and reported `(16, 4)` observations and
`(16, 1)` actions, confirming the configured default is Newton.

Both commands below passed at 16 environments and 400 steps, including full
joint position/velocity, observation and reward finite-value assertions,
random goal sampling, goal change after reset, and timeout detection:

```bash
env PYTHONPATH= LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  uv run --locked python scripts/smoke_newton.py \
  --task DexGym-Joint-Tracking-Direct-v0 --steps 400 --num-envs 16
env PYTHONPATH= LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  uv run --locked python scripts/smoke_newton.py \
  --task DexGym-Joint-Tracking-Manager-v0 --steps 400 --num-envs 16
```

| Task | Observation | Batched action space | Timeout count |
| --- | --- | --- | --- |
| Direct | `(16, 4)` | `(16, 1)` | 16 |
| Manager | `(16, 4)` | `(16, 1)` | 16 |

Both final training commands passed with exit code 0:

```bash
env PYTHONPATH= LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  timeout 600s uv run --locked isaaclab train --rl_library rsl_rl \
  --task DexGym-Joint-Tracking-Direct-v0 --num_envs 16 \
  --max_iterations 10 --viz none physics=newton_mjwarp
env PYTHONPATH= LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  timeout 600s uv run --locked isaaclab train --rl_library rsl_rl \
  --task DexGym-Joint-Tracking-Manager-v0 --num_envs 16 \
  --max_iterations 10 --viz none physics=newton_mjwarp
```

Each logged 10 learning iterations and 2,560 total steps. Final checkpoints:

- Direct: `logs/rsl_rl/dexgym_joint_tracking/2026-11-04_17-34-09/model_9.pt`.
- Manager: `logs/rsl_rl/dexgym_joint_tracking/2026-11-04_17-34-20/model_9.pt`.

Both `model_9.pt` files have finite actor/critic tensors and iteration 9.
Compared with `model_0.pt`, the actor plus critic parameter L1 changes were
59.72523 (Direct) and 48.43628 (Manager). This confirms updates occurred; it
does not establish tracking accuracy.

The official `isaaclab play --rl_library rsl_rl --task <ID> --checkpoint
<model_9.pt> --num_envs 1 --viz none physics=newton_mjwarp` command loaded each
final checkpoint and entered policy playback. The two bounded runs returned 0
after a 35-second SIGINT (`timeout --preserve-status --signal=INT 35s`). Each
exported `policy.pt`, `policy.onnx`, and `policy.onnx.data` under its local run's
`exported/` directory. The CLI's headless play loop has no finite step limit
without video, hence the deliberate interrupt.

## Scope and remaining limits

Only the official Cartpole asset and `newton_mjwarp` backend were validated in
this run. Archived Shadow, Allegro, Wuji, WujiHand2, Wrist 6DoF, and bottle
assets are preserved with provenance but are outside the active Newton tasks.
PhysX, Isaac Sim/Kit, contacts, grasping, ROS 2, and Arm S0 were not tested.
The PPO run was a chain test, not a trained policy quality evaluation. Local
checkpoints and exports are ignored by Git and must be regenerated with the
commands above on another host.
