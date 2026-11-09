# Hand S0 task-package integration — 2026-11-09

> Historical report. Current default is v4; see [v4 alignment and the unresolved Actor deadzone](hand_s0_dex_v4_alignment_20261109.md). Do not infer current training readiness from this older report.

> Historical v1 evidence below includes trust-gated updates. The current standalone
> Hand S0 v3 has no error gate and uses a fixed WM-only phase; see the [current checks](hand_s0_dex_alignment_20261109.md).

本次将已验证的独立单手 SuperMimic 链路正式整合进 DexGym 任务体系。
验收范围是代码组织、任务注册、实际执行与 checkpoint 兼容性；未启动大规模训练，
也不宣称策略已达到遥操作精度要求。

## 代码归属与入口

正式 Gym ID：`DexGym-Wuji-Hand21-TeleDexter-SuperMimic-Direct-v0`。

任务实现统一位于
[`source/dexgym/tasks/direct/right_hand_tracking`](../../source/dexgym/tasks/direct/right_hand_tracking/README.md)：

- 环境、单手模型接口、目标切换、奖励、终止、采集器、世界模型及信任检查均属于任务包。
- 训练、评估、物理短测、数据预检的业务代码已迁入 `supermimic_*.py`。
- 默认配置从 `configs/supermimic/hand_s0_train.json` 移至任务包的 `agents/hand_s0.json`，通过 `supermimic_cfg_entry_point` 注册，并包含在 wheel 中。
- 原有 `scripts/supermimic/*hand_s0*.py` 训练/评估/短测/预检入口只保留兼容调用；主机运行时脚本负责依赖环境和启动，不承载任务逻辑。
- 通用 SuperMimic 算法仍位于 `dexgym.algorithms.supermimic.core`。

训练和实际子目标评估通过 Gym 注册加载配置并调用 `gym.make`。
`fixed_clock` 评估保留任务包内的诊断子类，其逐帧推进规则与训练 MDP 有明确区分。
既有 PPO ID 和其 15 mm 规则保持原样。单手模型仍为官方 standalone Wuji Hand 2.1，
20 个控制关节、固定腕部、480 Hz 物理与 30 Hz 控制，不使用物体或触觉。

本次没有改动机器人/任务/网络的 checkpoint 类路径或可执行契约。
另修正了两项入口问题：`--iterations 0` 明确报错；源码提交号按实际模块所在源码仓库记录，
避免从其他工作目录启动或把安装包放在其他 Git 仓库下时误记提交号。
无法确定源码提交的安装包明确记录 `unversioned-package`。

## 验证结果

所有最终检查退出码为 0。机器可读结果见
[`hand_s0_task_integration_20261109.json`](../experiments/hand_s0_task_integration_20261109.json)。

| 检查 | 结果与边界 |
| --- | --- |
| 回归 | 273 项通过，18.66 秒；35 条既有 `torch.jit` 弃用警告 |
| Ruff | 本次迁移的模块、CLI 与测试全部通过 |
| 正式注册入口 Newton 短测 | 16 环境 × 64 步，1,024 条 transition；30 次终止、21 次目标完成；终止前状态保留验证通过 |
| 可微 FK 对物理 FK | 最大误差 `6.055631160961639e-7 m`，小于 `1e-4 m` 验收阈值 |
| WM8 短测 | 真实采集数据上一次更新成功，loss 与梯度有限 |
| 旧 checkpoint 恢复 | `pilot_v2/model_500.pt` → 新任务入口 → 第 502 轮；256 环境新增 512 条真实 transition、4 次 WM 更新，非有限计数增量为 0 |
| 恢复协议 | 模型/优化器与课程步数恢复；新物理状态、新 replay 重新收集，信任门控重新验证；该短段 `allowed_horizon=0`，没有新增 Actor/Critic 更新 |
| 评估入口 | 新保存的 `model_502.pt` 在验证集完成 16 环境 × 64 步，严格契约加载通过；无探索噪声、归一化不更新 |
| wheel | 成功构建并解包；从 `sys.path` 移除源码路径并切换工作目录后，配置、环境注册、训练/评估模块及单手 USD 资产均可加载 |

短评估只检验流程，不用于声称泛化或收敛。它不是对所有验证轨迹的穷尽评估。
本次也不以两轮恢复训练代替之前的 BPTT32 架构验收；原有证据见
[完整链路验收报告](hand_s0_readiness_20261109.md)。

## 使用方式

安装 DexGym 后，可以直接以任务模块启动；默认配置不依赖当前工作目录。
详见 [任务说明与命令](../../source/dexgym/tasks/direct/right_hand_tracking/README.md)。
旧命令若显式指定旧 JSON 路径，应移除 `--config` 以采用注册默认值，或指定自己的覆盖配置。

在本次 Ubuntu 主机的已验证运行时中，短测命令为：

```bash
bash scripts/supermimic/hand_s0_host.sh \
  -m dexgym.tasks.direct.right_hand_tracking.supermimic_smoke \
  --library /home/ubuntu/dexgym/data/processed/hand21_right_geometry_feasible/manifest.json \
  --num-envs 16 --steps 64 --output /tmp/hand_s0_smoke.json
```

原始日志、wheel、checkpoint 和逐目标数据保留于主机
`/home/ubuntu/dexgym/artifacts/hand_s0/task_integration/`；回归日志为同级
`task_integration_regression.log`。这些大文件未纳入 Git。
`policy_accuracy_passed` 保持为 `null`，未启动后续长期训练。


## 后续任务拆分

PPO 与 SuperMimic 现已分别定义在 `source/dexgym/tasks/direct/hand21_ppo/`
和 `source/dexgym/tasks/direct/hand21_supermimic/`。
两个任务分别拥有独立的 `env.py`、`env_cfg.py`、`logic.py`、`subgoals.py`。
配置直接继承 Isaac Lab 的 `DirectRLEnvCfg`，两个任务之间没有继承或实现导入。
PPO 的训练配置也移至自己的 `ppo_cfg.py`。

旧路径保留导入转发，任务注册直接指向新目录。SuperMimic 世界模型声明稳定的
checkpoint 类型标识，其他模型、参数与任务契约仍执行原有严格检查。

拆分后验证通过 281 项测试（35 条既有弃用警告）。两套环境与 PPO 训练配置的全部有效字段与拆分前 aa88be2 一致。
PPO 与 SuperMimic 各完成 16 环境 × 64 步物理短测；PPO 强制跨轨迹测试完成 52 次切换，SuperMimic 的终止前采集与 WM8 更新通过。
旧 model_500.pt 经新入口恢复到 502，新增 512 条真实 transition、4 次世界模型更新，非有限计数为 0。新 checkpoint 的验证集评估与 wheel 构建/解包加载通过。
原始结果保留于 `/home/ubuntu/dexgym/artifacts/hand_s0/task_split/`，回归日志为同级 `task_split_regression.log`。
