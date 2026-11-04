# DexGym 2.0

DexGym is a small downstream [Isaac Lab 3.0](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)
reinforcement learning package. It registers two joint target tracking tasks
against Newton `newton_mjwarp`. DirectRLEnv owns the step logic; the
ManagerBasedRLEnv variant composes actions, observations, rewards, commands,
reset events, and terminations through Isaac Lab managers. Both use the
official Cartpole asset and a shared RSL-RL PPO configuration.

| Gym ID | Architecture | Observation | Action |
| --- | --- | --- | --- |
| `DexGym-Joint-Tracking-Direct-v0` | DirectRLEnv | cart position, velocity, goal, error (4) | cart position target (1) |
| `DexGym-Joint-Tracking-Manager-v0` | ManagerBasedRLEnv | same (4) | cart position target (1) |

## Install

Use Python 3.12, an NVIDIA GPU/driver supported by the
[Isaac Lab 3.0 installation guide](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/installation),
and [uv](https://docs.astral.sh/uv/). The lockfile resolves
`isaaclab[rsl-rl]==3.0.0` from NVIDIA's index, PyTorch 2.11.0,
and CUDA toolkit 13.0 dependencies selected by Isaac Lab v3.0.0. No Isaac Sim or Kit installation is required for
Newton headless runs. On the verified Conda-based host, prepend
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` to runtime commands
to avoid Conda's older C++ runtime. The full evidence is in
[host verification](docs/platform/verification.md).

```bash
uv sync --locked --group dev
uv run isaaclab list_envs --keyword DexGym
```

## Train and play

Check `uv run isaaclab train --help` and `uv run isaaclab play --help` on your
installation before changing the commands. These bounded runs are workflow
checks; ten iterations do not imply a trained controller.

```bash
uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct \
  --num_envs 16 --max_iterations 5 physics=newton_mjwarp
uv run isaaclab train --rl_library rsl_rl --task DexGym-Joint-Tracking-Direct-v0 \
  --num_envs 16 --max_iterations 10 physics=newton_mjwarp
uv run isaaclab train --rl_library rsl_rl --task DexGym-Joint-Tracking-Manager-v0 \
  --num_envs 16 --max_iterations 10 physics=newton_mjwarp
uv run isaaclab play --rl_library rsl_rl --task DexGym-Joint-Tracking-Direct-v0 \
  --num_envs 1 --checkpoint /path/to/model.pt physics=newton_mjwarp
```

## Test

```bash
uv run ruff check source/dexgym tests scripts/assets scripts/smoke_newton.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
uv run python scripts/smoke_newton.py --task DexGym-Joint-Tracking-Direct-v0 --steps 400
uv run python scripts/smoke_newton.py --task DexGym-Joint-Tracking-Manager-v0 --steps 400
uv run python scripts/smoke_newton.py --task DexGym-Joint-Tracking-Direct-v0 \
  --steps 400 --policy logs/rsl_rl/dexgym_joint_tracking/<run>/exported/policy.pt
```

The smoke script asserts 16-environment observation/action shapes, finite
rewards, observations and joint states, randomized goals, and episode timeouts.
Official `isaaclab play` also exports JIT and ONNX policies into the selected
run's `exported/` directory. The active code reads Isaac Lab 3.0
`ProxyArray` values through `.torch` and uses indexed simulation writes.
MuJoCo WXYZ quaternions must pass through `dexgym.utils.quaternion`; Isaac Lab
3.0 data is XYZW.

## Assets and limits

See [asset inventory](docs/platform/assets.md) and the packaged
`source/dexgym/assets/manifest.json`. Original Shadow, Allegro, Wuji,
WujiHand2 Beta, Wrist 6DoF, bottle, source hashes, and PhysX validation
materials are retained. Only Cartpole is registered as a Newton RL task. A
separate **54 DoF Marvin M6 + WujiHand2 Beta 2 asset** now loads in Newton MJWarp
for offline structural, kinematic, control and dynamics validation. The pinned
converted USD is packaged; its source MJCF and meshes remain in the external
repository. `make_robot_cfg()` now defaults to the **no-hand-contact** USD:
all 54 hand collision meshes are disabled while hand mass, inertia, mounting,
and 40 joint controls remain. Pass `hand_contacts=True` to load the preserved
full-contact model. The [collision validation](docs/robots/marvin_wuji_no_hand_contacts.md)
includes actual Newton contact and inertial checks. To reproduce the original
source conversion, obtain an authorized source checkout, set
`DEXGYM_MARVIN_MJCF` to its `dual_arm_marvin.xml`, then run:

```bash
uv run --no-sync python scripts/assets/import_marvin_wuji.py
USD=artifacts/marvin_wuji/generated/6e0d569e-self-collision/dual_arm_marvin/dual_arm_marvin.usda
uv run --no-sync python scripts/assets/validate_model_structure.py --usd "$USD"
uv run --no-sync python scripts/assets/validate_fk_parity.py --usd "$USD"
uv run --no-sync python scripts/assets/validate_dynamics.py --usd "$USD" --mode smoke --num-envs 16
```

The [Marvin/Wuji validation report](docs/robots/marvin_wuji_validation.md)
contains the complete commands and measured limits. FK parity and 54-channel
control passed; physical fixed-TCP wrist tracking and backend dynamics still
need calibration. An [opt-in Newton controller profile](docs/robots/marvin_wuji_controller_tuning.md)
reduced the tested Home-centered fixed-TCP wrist error to about 0.56 mm per
arm; it is not a real motor calibration. The old
PhysX-specific Python asset adapters are preserved but not imported by this
package. Historic tests and validation scripts are in `archive/isaaclab2/` and
excluded from the current runtime and test suite. No real robot connection is part of this platform.

The older reports under `docs/assets/` and `docs/superpowers/` are historical
Isaac Lab 2.x research records; their commands do not describe the current CLI.

A future `tasks/direct/arm_s0_ee_tracking/` may use this asset for
six-dimensional TCP goals, joint-increment actions, wrist rotation,
full workspace sampling, PPO/imitation evaluation, policy export, and ROS 2
integration. No S0 task exists and this work did not run PPO or connect a robot.

Version, command, checkpoint and gate outcomes from the host run belong in
`docs/platform/verification.md`; check that report before claiming a specific
installation has passed the training loop.
