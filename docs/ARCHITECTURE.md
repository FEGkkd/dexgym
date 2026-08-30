# Architecture / 架构

## 中文说明

### 1. 仓库边界

`DexGym` 只拥有 Isaac Lab 任务扩展和实验工具。Isaac Sim、Isaac Lab、官方
WUJI description 和运行生成的 checkpoint 不在 Git 仓库中。资产路径通过
`ISAACLAB_ROOT`、`ISAACLAB_PYTHON`、`WUJI_DESCRIPTION_ROOT` 注入，任务代码
不依赖机器本地绝对路径。

### 2. 目录职责

```text
source/dexteroushands_lab/
├── assets/                 资产适配、USD 路径、关节/刚体命名映射
├── tasks/
│   ├── direct/             DirectRLEnv 任务，显式掌握完整生命周期
│   └── manager_based/      ManagerBasedRLEnv 任务和本地 MDP 边界
└── utils/                  action/observation contract 和 policy metadata
scripts/
├── rsl_rl/                 训练、playback、checkpoint 和策略导出
├── environments/            注册、smoke、USD、接触和摩擦验证
└── tools/                  结构检查与导出 shape 检查
tests/                      不启动仿真的契约测试
docs/                       安装、迁移矩阵、验证和物理证据
```

### 3. DirectTask 核心

每个 DirectTask 自己实现：

```python
_setup_scene()
_pre_physics_step(actions)
_apply_action()
_get_observations()
_get_rewards()
_get_dones()
_reset_idx(env_ids)
```

WUJI Reorientation 是当前核心任务。官方 native USD 在运行时加载，使用
20 个关节，policy observation 为 3 x 69 = 207，asymmetric critic 为 413。
动作先在 `[-1, 1]` 内裁剪，映射到关节目标，执行 EMA 过滤，再送到 Isaac
关节 position target。官方 USD 没有修改；WUJI 手摩擦材料在 task stage
中以 `mu=0.7` 的局部 PhysX material 绑定。

### 4. Manager-Based 边界

Manager-Based 任务由环境配置组合 scene、action、observation、reward、
termination、event 和 curriculum。每个项目任务保留本地 `mdp/`，即使初始
baseline 复用 Isaac Lab terms。当前 `wuji_arm_hand/` 只提供 ownership
scaffold 和 action/observation/reward/termination/events/commands 文件，
没有注册为可训练任务，等待真实机械臂加手复合资产与安装变换验收。

### 5. 训练数据链路

```text
RSL-RL policy action
        ↓
scripts/rsl_rl/train.py or play.py
        ↓
Gymnasium task registry
        ↓
DirectRLEnv / ManagerBasedRLEnv
        ↓
Isaac Lab articulation and PhysX
        ├── observations.policy
        ├── observations.critic (training only)
        ├── reward / terminated / timeout
        └── extras and policy metadata
```

训练脚本是薄编排层：加载任务、解析配置、创建环境、包装 RSL-RL、保存
`env.yaml`/`agent.yaml`、checkpoint 和 contract metadata。关节映射、奖励和
接触逻辑属于任务/资产模块，不放进启动脚本。

### 6. 验证层级

1. Python compile 和结构检查；
2. contract/unit 测试；
3. Isaac Sim 1/2 环境 reset-step smoke；
4. 官方 USD 刚体、关节、collision、material audit；
5. 五指接触、静态摩擦和超临界滑移探针；
6. 短 PPO、checkpoint、JIT/ONNX export；
7. 长训练、任务成功、sim-to-sim 和 sim-to-real，均是后续阶段。

## English description

### System boundary

`DexGym` owns the Isaac Lab task extension and experiment tools only. Isaac
Sim, Isaac Lab, the official WUJI description checkout, and generated
checkpoints remain external. `ISAACLAB_ROOT`, `ISAACLAB_PYTHON`, and
`WUJI_DESCRIPTION_ROOT` select these dependencies; task code contains no
machine-specific absolute asset path.

### Ownership model

- `source/dexteroushands_lab/assets`: USD adapters, joint/body maps, defaults,
  actuator settings, and asset checks.
- `tasks/direct`: explicit `DirectRLEnv` lifecycle and task semantics. This is
  the primary migration form for legacy DexterousHands tasks.
- `tasks/manager_based`: composed environments and local MDP ownership for
  reusable arm-hand work.
- `utils`: backend-independent action/observation contracts and policy metadata.
- `scripts/rsl_rl`: thin train/play/export orchestration.
- `scripts/environments`: task registration, smoke tests, USD inspection, and
  contact/friction probes.
- `tests`: simulator-independent contract checks; GPU scene checks are run by
  the scripts in `scripts/environments`.

### DirectTask contract

A DirectTask keeps `_setup_scene`, `_pre_physics_step`, `_apply_action`,
`_get_observations`, `_get_rewards`, `_get_dones`, and `_reset_idx` as named
methods. This makes action processing, observations, reward, termination, and
reset semantics independently inspectable.

The validated WUJI task loads the official native USD, exposes 20 action joints,
uses a 207-dimensional policy history and a 413-dimensional asymmetric critic,
and applies EMA-filtered joint-position targets. The source USD is not edited;
the task adds the official `mu=0.7` hand material as a stage-local PhysX
material because the native collision prototypes do not author one.

### Training and evidence path

```text
policy action [-1, 1]
  -> task/action manager
  -> Isaac Lab articulation targets
  -> Isaac Sim / PhysX
  -> policy and critic observations, reward, termination, extras
  -> RSL-RL PPO update, checkpoint, export, and metadata
```

Validation is intentionally layered: static structure, contract tests, Isaac Sim
scene smoke, native-USD audit, contact/friction probes, short PPO and export,
then longer task-success, sim-to-sim, and sim-to-real milestones. Passing the
contact probe does not claim that the current open-loop pose forms a stable
multi-finger grasp or that a short PPO run has learned reorientation.
