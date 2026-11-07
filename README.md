# DexGym

DexGym contains Newton MJWarp robot environments for Isaac Lab 3.0 and a
**complete, usable SuperMimic model-based RL framework**. The SuperMimic Core
collects real physical transitions, fits an eight-step masked autoregressive
world model, checks model trust, optimizes an Actor through differentiable
imagination and BPTT, and fits a TD(λ) Critic with survival-weighted returns.
Robot dynamics and task MDPs are separate plugins; the same Core trainer runs
different observation, goal and reward schemas. This is a software and
algorithm capability, **not a claim that every task or policy has converged**.

## Tasks

| Gym ID | Goal | Status |
| --- | --- | --- |
| `DexGym-Marvin-SuperMimic-Wave-Direct-v0` | 10×90 bimanual motion tracking | M4.1 validated Actor completed all 1,356 Newton control steps; Core training and evaluation are available. |
| `DexGym-Marvin-SuperMimic-Reach-Direct-v0` | Static 1×6 dual TCP position | Separate MDP, real Newton collection and short Core training smoke passed; no convergence claim. |
| `DexGym-Marvin-SuperMimic-SE3-Reach-Hold-Direct-v0` | Static 1×18 dual TCP SE(3) and hold | IK-free, 4,096-environment WM8/BPTT32 training: three random 400-iteration seeds achieved 384/384 strict successes across two independent Near goal sets, without safety failures. One seed reached 59/64 Medium and 16/64 Wider Feasible; harder curricula remain open. |
| `DexGym-Marvin-SuperMimic-Continuous-SE3-Follow-Direct-v0` | Moving 10×18 dual TCP SE(3) targets | Strict-pure 4,096-environment WM8/BPTT32 training completed 1,000 iterations. The selected 800-iteration checkpoint completed 64/64 Newton episodes on the pinned forward/reverse procedural trajectories under 0.01 and 0.02 rad reset perturbations. This does not establish unseen-trajectory generalization. |
| `DexGym-Marvin-SuperMimic-Teleop-Workspace-SE3-Direct-v0` | Causal current 1×18 dual TCP SE(3) target at 50 Hz | Strict-pure 4,096-environment WM8/BPTT32 training ran 820 iterations. The selected policy completed 93/96 full Newton episodes over three disjoint unseen or reversed target banks, with no safety terminations. Tested within a bounded procedural workspace; human teleoperation and real hardware remain unvalidated. |
| `DexGym-Joint-Tracking-Direct-v0` | Cartpole joint target | Isaac Lab DirectRLEnv, RSL-RL PPO example. |
| `DexGym-Joint-Tracking-Manager-v0` | Cartpole joint target | Isaac Lab ManagerBasedRLEnv, RSL-RL PPO example. |
| `DexGym-Wuji-Right-Hand-Tracking-Direct-v0` | Right WujiHand2 five-fingertip sequence tracking | TeleDexter-inspired hand-only DirectRLEnv; GRAB retargeting and a bounded PPO workflow check passed. No convergence claim. |

The Cartpole examples use PPO. **SuperMimic does not use PPO.** The SuperMimic Marvin
environment simulates all 54 DoF at 500 Hz, controls only 14 arm joints at
50 Hz, and commands the 40 hand joints to Home. Hand states and actions are
absent from the SuperMimic policy and world model. The validated no-hand-contact
configuration and controller parameters are unchanged.

## Right-hand fingertip tracking

The separate hand-only task uses GRAB right-hand MANO poses retargeted by the
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
1.5 seconds.

```bash
PYTHONPATH=source uv run python scripts/data/build_grab_right_hand_library.py \
  --data-root data --output data/processed/hand_only_right
PYTHONPATH=source uv run python scripts/smoke_right_hand_tracking.py \
  --num-envs 8 --steps 120 --split validation
PYTHONPATH=source uv run isaaclab train --rl_library rsl_rl \
  --task DexGym-Wuji-Right-Hand-Tracking-Direct-v0 \
  --num_envs 16 --max_iterations 10 physics=newton_mjwarp
```

## SuperMimic code

- [Core](source/dexgym/algorithms/supermimic/core/): typed transitions, masked Replay, state-delta world model, imagination, Actor/Critic factories, normalization, trust checks, physical evaluation and versioned checkpoints.
- [Interfaces](source/dexgym/algorithms/supermimic/interfaces/): robot, task, reference and physical collector contracts.
- [Marvin task plugins](source/dexgym/tasks/marvin_supermimic/): shared Newton arm physics and independent Wave, position Reach and SE(3) Reach & Hold tasks.
- [Configs](configs/supermimic/) and [scripts](scripts/supermimic/): bounded examples, training, evaluation and diagnostics.
- [How to add a task](docs/supermimic/task_plugin_guide.md), [architecture](docs/supermimic/core_architecture.md), [checkpoint rules](docs/supermimic/checkpoint_compatibility.md), [main integration validation](docs/supermimic/main_release_validation.md), [M5 full-horizon physical results](docs/supermimic/m5_full32_4096_training.md), [pinned continuous SE(3) following](docs/supermimic/continuous_se3_follow_training.md), and [causal workspace tracking](docs/supermimic/teleop_workspace_se3_follow.md).

WM8, BPTT32, soft termination, survival weighting and TD(λ) are covered by
the [SuperMimic tests](tests/supermimic/). The trained M4.1 Wave Actor remains
a legacy checkpoint; newer action and goal schemas are checked rather than
silently interpreted as that policy. Shared World Model **weights** require
matching dynamics and controller signatures; sharing Core code does not imply
that any robot or contact task can reuse the same learned model.

## Assets included

The repository contains the pinned Marvin/Wuji MJCF and its 70 STL dependencies
under [source_model](source/dexgym/assets/marvin_wuji/source_model/), both
validated USD variants under [usd](source/dexgym/assets/marvin_wuji/usd/),
the [Wave and procedural references](source/dexgym/assets/supermimic/references/),
the Wave physical reset states and validation report, and the validated
[M4.1 Wave checkpoint](models/supermimic/m41_wave_selected_best.pt) and the
[M5 SE(3) Reach & Hold checkpoint](models/supermimic/m5_se3_reach_hold_seed60_iter400.pt)
and the [continuous SE(3) following checkpoint](models/supermimic/continuous_se3_follow_seed72_iter800.pt)
and the [causal workspace tracking checkpoint](models/supermimic/teleop_workspace_seed97_iter820.pt)
in Git LFS. Run `git lfs pull` after cloning before loading these checkpoints.

Pinned SHA256 values: Marvin MJCF
`6e0d569ec28c8ac26a84bfc9a48652b7a11a2f113193c2d3ea4191621280b3ce`,
Wave reference
`b7af95b289e8aff898df7f5dd06b169f74dba2a506aa8fc5bf589ee3fee8106f`,
M4.1 checkpoint
`fd0054a7ef707d46a30b0235713c9e00cc4d7e69d6687fc3015e52aac2cb773f`,
M5 exported checkpoint
`fab9ccff3382b71679e1eb360d421fb6dcfbb7e5ac859ec0e5f904305ae6ff91`.
Selected continuous SE(3) policy
`cd9eb16b1b1ce36ad32be2728865a2eb478ec0c507b195bb62ad10ad7849c689`.
Selected causal workspace policy
`44740c7ad90627c93e0317ea70ceebaf13e5f849efb35c4714d92a569cc274b6`.
See the [asset notice](source/dexgym/assets/marvin_wuji/source_model/NOTICE.md)
for source and license attribution. The committed M5 file is a 3.2 MB
`fresh_physics` checkpoint: model, normalizers, optimizer state and semantic
contract are intact, but its 3.4 GB training Replay is deliberately omitted.
Full Replay checkpoints and physics traces are separate experiment artifacts.
They are archived under
`/home/ubuntu/dexgym_data/marvin_supermimic_m5_se3_reach_hold/`.

## Install and verify

Use Python 3.12, an NVIDIA GPU and the Isaac Lab 3.0/Newton environment.
`uv sync --locked --group dev` installs the project dependencies. The physical
commands below were verified on the project's Isaac Sim 6 host, which uses a
pinned MuJoCo/Warp overlay at
`/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides`.
That overlay is a runtime dependency on this host, not a robot model asset.
Consult [platform verification](docs/platform/verification.md) for host setup;
another host must supply compatible Isaac Lab, Isaac Sim, Newton and GPU
libraries. On the verified Conda host, set
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` if its older C++
runtime takes precedence.

```bash
uv sync --locked --group dev
git lfs pull

# Algorithm, schema, asset and legacy parity tests on the verified host.
PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source \
  /home/ubuntu/isaacsim-6.0.1/bin/python -c \
  "import sys,numpy; sys.path.append('/home/ubuntu/dexgym/.venv/lib/python3.12/site-packages'); import mujoco,pytest; sys.exit(pytest.main(['-q','tests/newton','tests/assets','tests/supermimic']))"

# Bounded Newton integration of two task MDPs; outputs go outside Git.
PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source \
  /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py \
  smoke_core_m43 --config configs/supermimic/m43_reach_smoke.json \
  --num-envs 16 --iterations 12 --eval-steps 12 --output /tmp/dexgym_reach_smoke

# Full, independent Newton replay of the pinned Wave Actor (1,356 control steps).
PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source \
  /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py \
  evaluate --checkpoint models/supermimic/m41_wave_selected_best.pt \
  --reference source/dexgym/assets/supermimic/references/cmu_141_16_wave_bimanual.npz \
  --output /tmp/dexgym_wave_legacy
```

The test command runs the SuperMimic mathematical/integration suite and the
Cartpole and asset tests. The latest verified run passed 221 tests with 3 skips.
The short Reach run verifies the training path, **not** policy success. The
Wave replay validates a trained legacy policy without modifying its weights.
See [M4.3.2 reliability results](docs/supermimic/m432_reliability_validation.md)
and [M5 full-horizon pilot results](docs/supermimic/m5_full32_4096_training.md) for
measured limits and reproducible training/evaluation commands.

The Cartpole examples remain available through `uv run isaaclab train` and
`uv run isaaclab play`; see [platform verification](docs/platform/verification.md)
for their exact commands. No real robot connection or deployment is included.
