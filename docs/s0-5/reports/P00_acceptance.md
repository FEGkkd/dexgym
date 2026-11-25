# P00 验收报告

| 项 | 值 |
|---|---|
| Phase / Spec | P00 · v0.1.0 |
| Run ID | S05-P00-20261125-ubuntu-audit |
| Commit / Working tree | Ubuntu main `21a813a1950891c712e421b5ced883763bc904cb`，tracked diff `ca0ca092d3120dc5249eccadf0c4c26eb56e22d22a4935ccd72481c4b555c2bc`。发布、运行与 Teleop 版本见 [审计](../REPOSITORY_AUDIT.md) |
| Execution status | READY_FOR_REVIEW，待人工审核 |

## 1–4 计划、实施、覆盖与文件

核验现场仓库、S0 系列路径、原始输入/Dataset、运行依赖及现有资产。交付主设计与三层参考、动作/观测/奖励/时序契约、公平评估、30 个 ID、逐阶段依赖和验证，P00 直接覆盖 S05-REQ-029。其余 29 个需求有明确的计划模块、测试与验收方法，见 [需求矩阵](../TRACEABILITY_MATRIX.md)。

变更限 Git 工作树的 `docs/s0-5/`。顶层 12 份文件见 [索引](../README.md)，11 份 `specs/P00–P10` 见 [路线图](../ROADMAP.md)，本报告在 `reports/`。记录命令、源码摘要、数据元信息和测试的证据与脚本在 [evidence/P00](../evidence/P00/README.md)。

## 5–9 实际命令、验证与定量证据

| 级别 | 执行 | 实际结果与范围 |
|---|---|---|
| 环境 | SSH，`git log/status/diff/ls-files`、lscpu/free/df/nvidia-smi、三解释器导入 | 四个 checkout 的 SHA 和 tracked/untracked 身份，实际项目 overlay import 成功。裸环境与 wrapper 不同 |
| 仓库代码 | 只读源检索、类/函数/行号和文件 SHA，追踪 Teleop、历史目标/IK/训练/执行/评估 | S0-4 发布与现场训练代码有差异，main 缺三目录，均定位替代位置 |
| 数据集成 | `ros2 bag info -s mcap`，Jazzy + Teleop/采集 overlay 的 ROS CDR 反序列化，NumPy `allow_pickle=False` | 4 个 bag 信息命令退出 0，四输入流共 8 种指定 topic 样本实际反序列化成功。30 组派生文件可枚举，1 组成套读取 dtype、shape、SHA |
| 物理见证 | 加载现有主模型，原生 MuJoCo 10 个 2 ms 子步 | 54/54/54/54，20 ms 有限状态，0 warning；只覆盖此模型和起始条件 |
| 资产回归 | 三个现有 tests/assets 文件，Wuji Python 首次，repo wrapper 复试 | 首次 2 collection errors，Isaac Lab 不在路径。wrapper 6 passed、3 skipped，单独 binding 1 passed、2 skipped；跳过详情见证据 |
| 设计一致性 | `python docs/s0-5/evidence/P00/check_documents.py` | 检查 30 IDs、11×25 结构、需求归属、文档引用、17 行验收与 8 个 JSON 见证。stdout 原文在证据目录 |
| 历史实验读取 | S0-4 state/reviews、checkpoint SHA、servo/bank/provenance | 最新完整 19k，原脚本选 15k，仿真难段 P95 38.07/36.86 mm（15k/19k）。未当作 P00 重跑 |

S0-5 单元/集成生产功能、GPU 物理训练完整路径、性能、PPO/SuperMimic、真实随机化和真机动作均为 NOT_RUN，本阶段只对计划与已有可安全验证项出证据。详细命令及退出见 [证据清单](../evidence/P00/README.md)。

## 10 验收矩阵

| Criterion | Expected | Actual | Status | Evidence |
|---|---|---|---|---|
| G01 实际仓库 | checkout、dirty、环境、资源、数据和资产 | 现场 SSH 四个 checkout、三个解释器、模型及 checkpoint | PASS | [审计](../REPOSITORY_AUDIT.md) |
| G02 历史问题 | 代码与结果分级 | 六类问题有行号、版本和未验证原因 | PASS | [审计](../REPOSITORY_AUDIT.md) |
| G03 主设计 | 架构、模块和边界齐全 | M1–M9 完整 | PASS | [主设计](../MASTER_DESIGN.md) |
| G04 需求 | 稳定 ID 覆盖关键目标 | 30 个连续唯一的稳定 ID | PASS | [需求](../REQUIREMENTS.md) |
| G05 接口 | 数据、目标、R/O/A、时序、评分 | C1–C6 与 E1–E5 | PASS | [契约](../CONTRACTS.md) · [评估](../EVALUATION_PROTOCOL.md) |
| G06 Spec | P00–P10 全部独立 | 11 份，各 25 个编号段 | PASS | [路线](../ROADMAP.md) |
| G07 各阶段可执行条件 | 输入、输出、依赖、验收 | 每份 `5,9,10,19,24,25` | PASS | [路线](../ROADMAP.md) |
| G08 需求验证 | 模块、单元、集成、量化、证据 | 30 行验证责任与证据路径 | PASS | [追踪](../TRACEABILITY_MATRIX.md) |
| G09 未解决问题 | 风险与后验需证 | I01–I14 与 R01–R12 | PASS | [问题](../OPEN_ISSUES.md) · [风险](../RISK_REGISTER.md) |
| G10 强基线 | B0/B1/B1+ 合理且公平 | E1/E3 与 P03，Oracle 单列，B2 共环境适配 | PASS | [评估](../EVALUATION_PROTOCOL.md) |
| G11 因果 | 禁未来及不可部署字段 | 捕获与接收、前缀不变、特权隔离 | PASS | [契约](../CONTRACTS.md) |
| G12 Teleop-only | 溯源、split、筛除 | 原包验证与独立 bank 设计、EgoPro 排除 | PASS | [契约](../CONTRACTS.md) |
| G13 可复现与失败 | 命令、哈希、计分覆盖和版本 | C6/E3、证据和未评项 | PASS | [证据](../evidence/P00/README.md) |
| G14 真机安全 | 独立审核、许可和计量 | M8/P10，分 Shadow、HIL、真机 | PASS | [主设计](../MASTER_DESIGN.md) |
| G15 项目状态 | 持久授权和版本状态 | 状态、版本基线、下一个授权项 | PASS | [项目状态](../PROJECT_STATE.md) |
| G16 报告 | 可复审，记录偏差与失败 | 真实验收报告含执行及复现记录 | PASS | [证据](../evidence/P00/README.md) |
| G17 范围 | 仅 P00，不开始后续 | 写入限文档和证据；未改 S0-4 源文件，未启动训练或发布真机动作 | PASS | [项目状态](../PROJECT_STATE.md) |

## 11–14 失败、限制、问题和规格差异

Wuji 环境缺 Isaac Lab，已用现有 runtime wrapper 验证可收集的测试，原错误保留。Wuji Python 缺独立 mcap 包，改用现场 ROS2 MCAP 插件和两套既有工作区解析成功。跳过由现有测试可选前置资产条件触发，报告不将其算作通过。内存/磁盘/GPU 是当时采样。原始采集对齐与 54 DoF 模型只有样本见证；全量谱系、端到端时间、难姿态归因和 S0-5 性能在后续阶段验收。未改变核心规范。可重跑脚本的历史路径和 checkout SHA 对应到明确的本机现场。

## 15–16 证据与复现

按 [证据说明](../evidence/P00/README.md) 运行 Ubuntu 只读探查与本地文档检查。`host.json` 存包路径、资源和 git diff 指纹，`code.json/source.json` 逐模块快照行号，`followup.json` 保存解析、重试和异常，`s04.json` 是历史结果。所有设计门的通过仅针对 P00 产物。

## 17–18 阶段裁定及下一项

P00 **READY_FOR_REVIEW**。用户人工确认后方可改 **ACCEPTED**。P01 需显式授权、确认采集原包与标定入口、可用解析 overlay、独立数据输出容量，冻结 P01 source lineage、split 和同步验收规则。然后才可开始 P01 的测试先行和独立 Bank 重建。
