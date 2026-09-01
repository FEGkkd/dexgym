# Installation and Configuration / 安装与配置

## 中文

### 依赖关系

DexGym 是 Isaac Lab 扩展，不包含 Isaac Sim 和 Isaac Lab 本体。运行时分为
两部分：Conda 环境只负责项目工具（Python 3.12、pytest）；Isaac Sim 的
Python 解释器负责启动 Kit、加载 PhysX 和运行 Isaac Lab。

最低建议配置：

- Ubuntu/Linux x86_64；NVIDIA RTX GPU，建议至少 RTX 4080 16 GB；
- Isaac Sim 6.0.1 或兼容的 6.0.x；
- Isaac Lab 的 6.0.x 兼容分支；本工程已在 commit
  `ffff603eafc6b74264a5261cc0183d6a65390d78` 上验证；
- 外部 `wuji-description` 仓库，且存在
  `hand/body/usd/right/wujihand.usd`；
- Conda/Miniconda。

### 配置步骤

```bash
git clone https://github.com/FEGkkd/dexgym.git
cd dexgym
./scripts/setup_conda.sh

export ISAACLAB_ROOT=/path/to/IsaacLab
export ISAACLAB_PYTHON=/path/to/isaacsim-or-isaaclab/python
export WUJI_DESCRIPTION_ROOT=/path/to/wuji-description
source scripts/activate_dexgym.sh
./scripts/install.sh
```

检查变量：

```bash
printf '%s\n' "$CONDA_DEFAULT_ENV"
printf '%s\n' "$ISAACLAB_ROOT"
printf '%s\n' "$ISAACLAB_PYTHON"
printf '%s\n' "$WUJI_DESCRIPTION_ROOT"
test -f "$WUJI_DESCRIPTION_ROOT/hand/body/usd/right/wujihand.usd"
```

当前任务只加载上述官方 USD，不执行模型转换，也不回退到旧的本地转换资产。

### 训练和 GUI

```bash
./scripts/list_envs.sh
./scripts/smoke.sh --task DexterousHands-Wuji-Hand-Reorientation-Direct-v0 --num_envs 2 --steps 3
./scripts/train.sh --task DexterousHands-Wuji-Hand-Reorientation-Direct-v0 --viz none --num_envs 256 --max_iterations 5000
```

使用 `--viz kit` 请求 Isaac Sim GUI；使用 `--viz none` 运行无窗口模式。
远程服务器没有显示器时使用 headless；桌面 GUI 还需要正确的 NVIDIA 显示和
Kit 环境变量。

### 常见问题

- `Isaac Sim Python was not found`: 设置 `ISAACLAB_PYTHON`，它必须是
  Isaac Sim 运行时中的可执行 Python。
- `WUJI_DESCRIPTION_ROOT is required`: 设置外部 WUJI description 路径。
- `Official WUJI Hand USD was not found`: 检查 USD 的相对路径和大小写。
- 任务列表为空：确认 `PYTHONPATH` 包含仓库的 `source/`，并通过仓库脚本
  启动，不要直接使用系统 Python。
- 训练输出中的 `model_*.pt`、视频和 ONNX 文件默认被 Git 忽略；它们是
  运行产物，不是安装依赖。

## English

### Dependency model

DexGym is an Isaac Lab extension. It does not include the Isaac Sim or Isaac
Lab distributions. Runtime responsibilities are split deliberately: Conda
holds project tools such as Python 3.12 and pytest, while the Isaac Sim Python
executable starts Kit, loads PhysX, and runs Isaac Lab.

Recommended baseline:

- Ubuntu/Linux x86_64 with an NVIDIA RTX GPU; RTX 4080 with 16 GB VRAM or
  better is the practical floor;
- Isaac Sim 6.0.1 or a compatible 6.0.x runtime;
- Isaac Lab on the 6.0.x compatibility line; this workspace was verified at
  commit `ffff603eafc6b74264a5261cc0183d6a65390d78`;
- an external `wuji-description` checkout containing
  `hand/body/usd/right/wujihand.usd`;
- Conda or Miniconda.

### Setup

```bash
git clone https://github.com/FEGkkd/dexgym.git
cd dexgym
./scripts/setup_conda.sh

export ISAACLAB_ROOT=/path/to/IsaacLab
export ISAACLAB_PYTHON=/path/to/isaacsim-or-isaaclab/python
export WUJI_DESCRIPTION_ROOT=/path/to/wuji-description
source scripts/activate_dexgym.sh
./scripts/install.sh
```

Verify the paths and launch a smoke test:

```bash
./scripts/list_envs.sh
./scripts/smoke.sh --task DexterousHands-Wuji-Hand-Reorientation-Direct-v0 --num_envs 2 --steps 3
```

The task loads only `/hand/body/usd/right/wujihand.usd`.
It does not convert robot models or fall back to previously generated assets.

Train headless or with the desktop window:

```bash
./scripts/train.sh --task DexterousHands-Wuji-Hand-Reorientation-Direct-v0 --viz none --num_envs 256 --max_iterations 5000
./scripts/train.sh --task DexterousHands-Wuji-Hand-Reorientation-Direct-v0 --viz kit --num_envs 1 --max_iterations 100 --run_name gui
```

Run `./scripts/test.sh` for contract tests. Use
`./scripts/run_isaaclab.sh scripts/environments/wuji_contact_friction_validation.py`
for the official-USD contact/friction probe.

### Runtime behavior

`ISAACLAB_PYTHON` must be the executable from the Isaac Sim runtime. The
launcher auto-detects common sibling layouts, but explicit variables are
recommended for reproducibility. Isaac Sim binaries, Isaac Lab, WUJI USD files,
checkpoints, videos, and large time series are intentionally external or
ignored generated output.
