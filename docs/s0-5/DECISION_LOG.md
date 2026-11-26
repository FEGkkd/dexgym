# S0-5 决策记录 v2.0.1

历史 D-001–D-024 和 DCR-001–003 保存在已审核的 v0.1.x 基准。DCR-004 与有限签核来自独立的 SIM0 Git 对象。既有人工事件沿用其签核范围。 [原决策](https://github.com/FEGkkd/dexgym/blob/b8d8b401bdda59ad00a37f0a3bf6878b462ed756/docs/s0-5/DECISION_LOG.md) · [DCR-004](https://github.com/FEGkkd/dexgym/blob/c5af9af94d7b9d3f437c14df128441d1183bec56/docs/s0-5/dcr/DCR-004_bimanual_simulation_research_branch.md)

| 决策 | V2 候选 |
|---|---|
| D-025 | 现有 Teleop 名义链 + 54 DoF 残差。P03 新联合 IK 迁至可选对照。 |
| D-026 | 复用 SuperMimic，资格化 54 轴 WM 后启用 Actor 更新，PPO 属于可选同条件研究。 |
| D-027 | 仿真阶段自有 Servo 和安全门。物理资格独立于 V1 release。 |
| D-028 | P02 原签核范围保持历史静态。M1 连续参考用途等待人工批准及 V1-A 技术核验。 |
| D-029 | V1 工程单种子、最终科研多种子及独立 Test 各自留痕。按实际证据授予候选等级。 |
| D-030 | 设计人工批准与独立 Campaign 整体授权分离。授权后按 DAG 和预算自主推进技术门，人工事件来源必须可核查。 |

v2.0.0 变更由 [DCR-005](dcr/DCR-005_policy_first_supermimic.md) 提交。现版本候选状态 PENDING_HUMAN_REVIEW。

| 决策 | v2.0.1 候选 |
|---|---|
| D-031 | 在预 Actor 冻结支持及见证域上测离线绝对精度，完整域和原相对门继续生效。 |
| D-032 | 现有实际 Teleop 的两份 MCAP 因果闭环为首版发布链路，T1/T2 绑定同一目标与执行。 |
| D-033 | 实际诊断归因先于有界补训，EgoPro 只在 Teleop-only 已通过且缺口被证实后开启。 |
| D-034 | M1 连续性仅可在正式训练前比较有限平滑候选，保留原目标和源索引。 |
| D-035 | 中间失败模型的独立诊断导出不改变技术状态。 |

[变更依据](dcr/DCR-006_teleop_validation_egopro.md)
