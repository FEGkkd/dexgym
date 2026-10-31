# DexGym

WujiHand2 Beta 双手自由腕与螺纹瓶的独立 Isaac Lab 基础场景见
[资产、控制和物理验证说明](docs/assets/wujihand2-beta-bimanual-bottle.md)。此场景不注册 RL task；
[Phase 0 双手抓握初态、接触监测与 Reset 验证](docs/assets/phase0-bimanual-grasp-reset.md)
记录了当前可复现的验证结果。悬空稳定抓握尚未通过，因此未提供已验证初态库。
原有单手训练与导出接口继续使用第一代 Wuji Hand 资产。

DexGym is a focused Isaac Lab reinforcement-learning workspace for
dexterous-hand research. It contains the migrated task extension, DirectTask
and Manager-Based task boundaries, RSL-RL train/play entry points, contract
tests, and WUJI official-USD contact validation tools.

> The repository does not redistribute Isaac Sim, Isaac Lab, or the official
> WUJI description assets. Those dependencies are installed or checked out
> separately and are selected through environment variables.

## 项目定位 / Project scope

这是一个面向灵巧手强化学习的 Isaac Lab 工程。当前优先保证单手固定基座
任务的定义、训练、checkpoint、playback、策略导出和接触摩擦验证链路完整。
DirectTask 是迁移旧 DexterousHands 任务的核心形式，Manager-Based 保留为
可组合 MDP 和后续机械臂加灵巧手任务的组织形式。

This is an Isaac Lab project for dexterous-hand reinforcement learning. The
current release prioritizes a complete fixed-base single-hand path: task
registration, reset/step, PPO training, checkpoint loading, policy export, and
contact/friction validation. DirectTask is the authoritative form for migrating
legacy DexterousHands tasks. Manager-Based provides the composable MDP boundary
for future arm-hand work.

## 当前可运行任务 / Runnable tasks

| Task ID | Workflow | Status |
| --- | --- | --- |
| `DexterousHands-Shadow-Hand-Pose-Direct-v0` | DirectTask | architecture baseline |
| `DexterousHands-Shadow-Hand-Reorientation-Direct-v0` | DirectTask | migration baseline |
| `DexterousHands-Wuji-Hand-Reorientation-Direct-v0` | DirectTask | official native WUJI USD, 20-DOF, 207/413 PPO chain and contact/friction gates validated |
| `DexterousHands-Allegro-Repose-ManagerBased-v0` | Manager-Based | Isaac Lab manager baseline |

WUJI uses the official native USD at
`hand/body/usd/right/wujihand.usd`, resolved from
`WUJI_DESCRIPTION_ROOT`.

WUJI 运行时只支持这个官方原生 USD。仓库不包含模型转换逻辑、转换后资产或旧资产回退路径。

The WUJI runtime supports only this official native USD; no converted asset or
legacy asset fallback is included.

The WUJI arm-hand directory is an unregistered scaffold; arm plus hand training
is not claimed by this release. Sim-to-sim is also deferred.

## 环境要求 / Requirements

- Linux x86_64 with an NVIDIA RTX-class GPU. Isaac Sim's current practical
  floor is RTX 4080 with 16 GB VRAM; this workspace was verified on an RTX 4090.
- Isaac Sim `6.0.1` or a compatible `6.0.x` runtime.
- Isaac Lab on the Isaac Sim 6.0.x compatibility line. The local verification
  used Isaac Lab commit `ffff603eafc6b74264a5261cc0183d6a65390d78`.
- Python 3.12 in the Isaac Sim runtime. The verified runtime used Python 3.12.3,
  Torch `2.10.0+cu128`, Gymnasium `1.2.1`, and RSL-RL `5.0.1`.
- Conda or Miniconda for the project wrapper environment. Conda does not
  replace the Isaac Sim Python runtime.
- The external `wuji-description` checkout for the WUJI task, containing
  `hand/body/usd/right/wujihand.usd`.

Isaac Sim and Isaac Lab are external dependencies because their binaries and
licenses are not part of this repository. Install them using the upstream
instructions before starting a GPU task.

## 安装 / Installation

### 1. Clone and create the project environment

```bash
git clone https://github.com/FEGkkd/dexgym.git
cd dexgym
./scripts/setup_conda.sh
```

### 2. Select Isaac Lab, Isaac Sim, and WUJI assets

```bash
export ISAACLAB_ROOT=/path/to/IsaacLab
export ISAACLAB_PYTHON=/path/to/isaacsim-or-isaaclab/python
export WUJI_DESCRIPTION_ROOT=/path/to/wuji-description
source scripts/activate_dexgym.sh
test -f "$WUJI_DESCRIPTION_ROOT/hand/body/usd/right/wujihand.usd"
./scripts/install.sh
```

`ISAACLAB_PYTHON` must be the Python executable from the Isaac Sim runtime,
not the ordinary system Python and not only the Conda wrapper. The launcher
also auto-detects a sibling `../IsaacLab`, `../.venv-isaacsim`, or
`$HOME/IsaacLab` layout when those variables are omitted.

### 3. Check registration and run a smoke test

```bash
./scripts/list_envs.sh
./scripts/smoke.sh \
  --task DexterousHands-Wuji-Hand-Reorientation-Direct-v0 \
  --num_envs 2 --steps 3
```

### 4. Start training

Headless training:

```bash
./scripts/train.sh \
  --task DexterousHands-Wuji-Hand-Reorientation-Direct-v0 \
  --viz none --num_envs 256 --max_iterations 5000
```

GUI training, with an Isaac Sim window:

```bash
./scripts/train.sh \
  --task DexterousHands-Wuji-Hand-Reorientation-Direct-v0 \
  --viz kit --num_envs 1 --max_iterations 100 --run_name gui
```

Runs are written to `logs/rsl_rl/<experiment>/<timestamp>`. The training
script saves environment and agent configuration, data-contract metadata, and
RSL-RL checkpoints.

### 5. Play and export a checkpoint

```bash
./scripts/play.sh \
  --task DexterousHands-Wuji-Hand-Reorientation-Direct-v0 \
  --num_envs 1 \
  --load_run 2026-08-29_15-45-20_official_usd_short \
  --checkpoint model_1.pt
```

Playback exports `policy.pt`, `policy.onnx`, `policy_metadata.yaml`,
`observation_schema.yaml`, `action_schema.yaml`, and `normalization.yaml` in
the run's `exported` directory. The exact `--load_run` value is the directory
name under the experiment log directory.

### 6. Run tests and contact validation

```bash
./scripts/test.sh
./scripts/check_structure.sh
./scripts/run_isaaclab.sh scripts/environments/wuji_contact_friction_validation.py \
  --finger index \
  --output artifacts/wuji_contact_friction_validation/index
```

The contact validation measures contact force direction, the configured
Coulomb coefficient, bounded subcritical displacement, and supercritical
sliding. It is separate from grasp success and does not claim that the current
open-loop pose or short PPO run has learned a stable grasp.

## 仓库架构 / Repository architecture

```text
source/dexteroushands_lab/
  assets/                 USD, joint/body maps, asset adapters
  tasks/direct/            explicit DirectRLEnv task definitions
  tasks/manager_based/     ManagerBasedRLEnv task boundaries and local mdp/
  utils/                  data contracts and policy metadata
scripts/
  rsl_rl/                 train, play, CLI parsing
  environments/            registration, smoke, USD and contact probes
  tools/                   structure and export checks
tests/                    simulator-independent contract tests
docs/                     installation, architecture, migration, validation
logs/                     ignored runtime outputs
artifacts/                small tracked validation summaries and local results
```

The runtime data path is:

```text
policy action [-1, 1]
  -> DirectTask action processing or Manager-Based action manager
  -> Isaac Lab articulation targets
  -> Isaac Sim / PhysX
  -> policy observations, privileged critic, reward, termination, extras
  -> RSL-RL PPO update and metadata/checkpoint export
```

Read [docs/INSTALL.md](docs/INSTALL.md) for the complete bilingual setup
reference, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for ownership and data
flow, and [docs/wuji-contact-friction-validation.md](docs/wuji-contact-friction-validation.md)
for the official-USD physics evidence.

## 已知边界 / Known boundaries

- The current WUJI task is a fixed-base single-hand DirectTask. Arm plus hand
  is not registered.
- The short PPO run proves numerical training-chain connectivity, not
  reorientation success or a production checkpoint.
- The WUJI runtime accepts only the official native USD selected by
  `WUJI_DESCRIPTION_ROOT`; converted robot assets are not supported.
- Cross-simulator comparison and real-robot deployment are deferred.

## License and attribution

The migrated extension is released under Apache-2.0. It builds on the
DexterousHands project by PKU-MARL and the Isaac Lab / Isaac Sim ecosystem.
WUJI assets remain under the license and terms of their separate official
repository; do not redistribute them through this repository.
