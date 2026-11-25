# DexGym Wholebody S0-5

S0-5 是面向 Marvin M6 双臂和 Wuji Hand 2.2 双手的 54 自由度实时残差控制器。当前阶段和授权范围见项目状态。

| 阅读顺序 | 文档 |
|---|---|
| 审核入口 | [项目状态](PROJECT_STATE.md) · [P01-A 验收报告](reports/P01_A_acceptance.md) · [P00 Review Fix](reports/P00_review_fix.md) · [原始 P00 验收报告](reports/P00_acceptance.md) |
| 事实依据 | [执行基线](EXECUTION_BASELINE.md) · [仓库与 Teleop 审计](REPOSITORY_AUDIT.md) · [历史数据使用](HISTORICAL_DATA_USAGE.md) · [审计证据](evidence/P00/README.md) |
| 设计与验收 | [主设计](MASTER_DESIGN.md) · [接口契约](CONTRACTS.md) · [评估协议](EVALUATION_PROTOCOL.md) |
| 需求与计划 | [需求](REQUIREMENTS.md) · [追踪矩阵](TRACEABILITY_MATRIX.md) · [路线图与阶段规格](ROADMAP.md) |
| 变更管理 | [决策记录](DECISION_LOG.md) · [风险](RISK_REGISTER.md) · [未决问题](OPEN_ISSUES.md) |

主设计 `M1–M9` 约束各阶段。接口、评估及阶段规格各有版本。P00 中的候选参数须按各阶段证据确认。项目状态文件记录授权和验收状态。

## 阶段报告发布

每个 S0-5 阶段或子阶段完成后，将报告提交到 `docs/s0-5/reports/` 下的对应位置，推送到 `FEGkkd/dexgym` 的 `s0-5` 分支。推送后核对远端报告，并在交付时附上报告链接和提交 SHA。报告发布与阶段签核分开记录。
