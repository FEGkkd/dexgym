# DCR-005 | Policy-First SuperMimic Redesign v2.0.0

| 字段 | 决策记录 |
|---|---|
| Original Design | 早期主路线依次要求硬件资格、在线新建 54 DoF 联合 IK、PPO 主任务，后续才对照 SuperMimic。小阶段通过后逐项人工放行。 |
| Discovered Problem | 现有遥操已具备在线名义链路。依赖新名义控制器和完整硬件计量会推迟检验残差策略。S0-4 的固定手和 SDK FK 评分无法单独确定人体几何保真。 |
| Supporting Evidence | P00 仓库审计、P01 HMB 历史仿真资格、P02 成对自由度和严格必要界、S0-4 MuJoCo Warp 及 SuperMimic 源码有可复用证据。原 14/54 DoF 运动学结果未提供 Policy 比较。 |
| User Decision | 项目用户要求 Existing Teleop + 54 DoF Residual RL，SuperMimic 主算法，simulation-first，训练和集成分离。仅修订设计与发布候选。 |
| Proposed Change | V1-A–D DAG，从 HMB 构建离线 RMB，以现有 Marvin IK + Wuji SDK 名义动作训练残差。先资格化 WM，后做静态、原速、混合、实际物理评估和独立仿真包。整体授权只在独立人工指令后生效。 |
| Alternatives | 历史联合 IK/QP、PPO-first、先做实机计量、重新采集数据、把 SDK FK 当主目标。前三项留在独立研究或硬件阶段，冻结现有 HMB 用作首版开发数据，独立 mapped intent 作主评分。 |
| Expected Benefit | 可更早比较同条件 B0、B54 的机器人精度，分别定位映射、几何可改善域、闭环执行和模型误差。 |
| Risks | 缓存/实时分布差、M1 连续资格、54 轴激励不足、WM 利用、功能几何退化、候选实测三方审阅、预算或 SDK 依赖失败。每项见风险登记。 |
| Affected Requirements | S05-REQ-001 至 030 稳定迁移，新增 031 至 035。 |
| Affected Phases | P03–P10 迁移至 V1 仿真与独立后续阶段。P00/P01 历史签核及 P02 履历保留。 |
| Affected Contracts | C1–C6，分别绑定 RMB、时间、动作安全、观测、学习、取证和发布。 |
| Affected Tests | 文档/状态正负控、谱系、前缀、物理轴顺序、滤波 parity、WM 梯度/漏洞利用、冻结评分、导出独立闭环。 |
| Compatibility Impact | 冻结原 v0.1.x Git 对象及原凭据。V2 原始策略与 WM 优先重置权重，另设运行时 ABI、模型资产和资格。既有实时入口、START 处理和权限维持其原有用途。 |
| Remaining Hardware Obligations | P01-T、C2-RT、物理 robot-base、硬件 Servo 与外部可溯源计量，独立 Final Test、正式物理评分、Shadow、HIL、现场安全与逐级人工批准。 |
| Approval Status | DRAFT_READY_FOR_REVIEW，PENDING_HUMAN_REVIEW。后续整体执行授权尚缺。 |

P02 原有限签核继续有效。M0 用于不可变审计，M1 尺度 1 用于原历史静态范围，M2 用于压缩诊断，Minimax 用于位置诊断，平滑捏合仍是未冻结候选。申请将固定 M1 扩至 V1-A **连续历史仿真参考生成**，以已签署的 Train 常量为输入，且先通过输入切换、连续性与独立几何门。正式物理评分、动态资格和 M1 新用途各自需要对应审批或试验证据。 [原 P02 交接](../reports/P02_final_handoff.md) · [原有限签核](https://github.com/FEGkkd/dexgym/blob/c5af9af94d7b9d3f437c14df128441d1183bec56/docs/s0-5/reports/approvals/P02_LIMITED_ACCEPTANCE_SIM0.md)

[总体设计](../MASTER_DESIGN.md) · [追踪矩阵](../TRACEABILITY_MATRIX.md) · [风险登记](../RISK_REGISTER.md)
