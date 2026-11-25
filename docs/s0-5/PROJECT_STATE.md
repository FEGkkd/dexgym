# S0-5 项目状态

[phase_state.json](phase_state.json) 保存阶段和子阶段状态。

| 阶段 | 当前状态 | 授权范围 |
|---|---|---|
| P00 | ACCEPTED | 用户签核 `09c1c9e3c63797083a82493e2a40dd8bb4b1db6d` |
| P01 | IN_PROGRESS | 本次授权限 P01-B |
| P01-A | ACCEPTED | 人工签核仅覆盖 A1 元数据门 |
| P01-B | BLOCKED | B1 旧 CollectionStatus 解码失败，B2 实机来源及配对缺证 |
| P01-C–F、P02–P10 | NOT_STARTED | 待各自授权 |

Master Design、Contracts、Evaluation Protocol 均批准 v0.1.1。P01 Spec v0.2.0 的格式、语义和验收方法已批准。QC 数值阈值、时钟参数和全量预算将在 P01-C/D 另行实测和冻结。[人工批准记录](reports/approvals/P00_ACCEPTED_P01_A.md)

实际开发在 Ubuntu `/home/ubuntu/dexgym_worktrees/s0-5`。原始 [P00 验收](reports/P00_acceptance.md) 和 [P00 Review Fix](reports/P00_review_fix.md) 留作历史快照。Final Test 仍为 BLOCKED。[历史用途](HISTORICAL_DATA_USAGE.md)

P01-A 在 `P01-A-20261126-09c1c9e-r03` 完成实测。32 个独立真人捕获母源和其启动标定来源通过元数据核查，89 个来源未确认的录制隔离。15 项单元测试和 4 项真实集成测试通过，独立 Final Test 缺封存证明。完整读包、时钟和遥测资格由 P01-B/C 验证。[验收报告](reports/P01_A_acceptance.md) · [运行凭据](evidence/P01_A/review_receipt.json)

P01-B 完成有界 schema 与解码审计。两种格式和旧 HandState 的原始字段已运行验证，仍有 1 个 CollectionStatus 版本失败。实机可追溯的动态系统辨识候选为 0。当前停在 P01-B 审核。[人工授权](reports/approvals/P01_A_ACCEPTED_P01_B.md) · [审核报告](reports/P01_B_acceptance.md) · [运行凭据](evidence/P01_B/review_receipt.json)
