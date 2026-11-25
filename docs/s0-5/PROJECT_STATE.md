# S0-5 项目状态

[phase_state.json](phase_state.json) 记录当前阶段与授权事件。[P00 Review Fix 报告](reports/P00_review_fix.md) 保存本轮审查结论。

| 阶段 | 状态 | 说明 |
|---|---|---|
| P00 | READY_FOR_REVIEW | 六项修订完成后提交审核，等待人工 ACCEPTED |
| P01–P10 | NOT_STARTED | P01 正式门 P01-A 至 P01-F 均未启动 |

**版本。** 原始 P00 由 GitHub `s0-5` 的 `cc05f09847254ddae3b76581fb9e1feed4f31065` 发布。Review Fix 在 Ubuntu 的独立 `/home/ubuntu/dexgym_worktrees/s0-5` 以此为审查基线。GitHub 分支与实际运行路径核对见 [执行基线](EXECUTION_BASELINE.md)。原始 P00 主仓库、S0-4 发布/训练、Teleop 的当时身份见 [旧审计](REPOSITORY_AUDIT.md)。

**复验与交付。** 设计／核心接口／评估规范更新至 v0.1.1，P01 Spec 为 v0.2.0 待人工冻结。旧 P00 指标和原始验收报告留存，历史 38 文件可按 Git SHA 复验。新检查器校验当前证据、引用和合法转换。除文档及审计代码外，后续模块仍属于设计阶段。[本轮证据](evidence/P00_review_fix/README.md)

**独立终评。** 当前资格未证实，`Final Test = BLOCKED`。P01 可在授权后建设开发 Train/Validation 视图，P09 启动必须持有独立封存清单、母源证据和访问日志。[历史用途](HISTORICAL_DATA_USAGE.md)

**下一项授权。** 用户确认 P00 ACCEPTED 并授权 P01-A 前，项目停在 P00。随后签核 v0.2.0、核验执行基线和原包访问，审批约束和受控样本。P01-C/D 用开发数据实测并冻结时钟、QC、资源和划分后，才可申请全量 P01-E。[P01 Spec](specs/P01_teleop_data.md)
