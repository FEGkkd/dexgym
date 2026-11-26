# S0-5 项目状态

[phase_state.json](phase_state.json) 保存阶段和子阶段状态。

| 阶段 | 当前状态 | 授权范围 |
|---|---|---|
| P00 | ACCEPTED | 用户签核 `09c1c9e3c63797083a82493e2a40dd8bb4b1db6d` |
| P01 | IN_PROGRESS | 本次授权限 P01-C |
| P01-A | ACCEPTED | 人工签核仅覆盖 A1 元数据门 |
| P01-B（历史） | BLOCKED | v0.2.0 原 B1 FAIL / B2 BLOCKED |
| P01-B-HMB | ACCEPTED | 用户签核 `8b0f399` 的有界解码门 |
| P01-C | BLOCKED | C1 与模拟到达下的 C2 算法通过。真实捕获至控制器的时间上界无证据 |
| P01-T | BLOCKED | 动态资格为 0，P03 正式启动的强制前置 |
| P01-D–F、P02–P10 | NOT_STARTED | 待各自授权 |

原 Master Design、Contracts、Evaluation Protocol v0.1.1 和 P01 v0.2.0 留作历史版本。项目用户批准 DCR-001 的依赖范围，活动规格为 P01 v0.2.1 和三份 v0.1.2 设计。QC 数值阈值、时钟参数和全量预算将在 P01-C/D 另行实测和冻结。[DCR-001](dcr/DCR-001_p01_hmb_telemetry_dependencies.md) · [原批准记录](reports/approvals/P00_ACCEPTED_P01_A.md)

实际开发在 Ubuntu `/home/ubuntu/dexgym_worktrees/s0-5`。原始 [P00 验收](reports/P00_acceptance.md) 和 [P00 Review Fix](reports/P00_review_fix.md) 留作历史快照。Final Test 仍为 BLOCKED。[历史用途](HISTORICAL_DATA_USAGE.md)

P01-A 在 `P01-A-20261126-09c1c9e-r03` 完成实测。32 个独立真人捕获母源和其启动标定来源通过元数据核查，89 个来源未确认的录制隔离。15 项单元测试和 4 项真实集成测试通过，独立 Final Test 缺封存证明。完整读包、时钟和遥测资格由 P01-B/C 验证。[验收报告](reports/P01_A_acceptance.md) · [运行凭据](evidence/P01_A/review_receipt.json)

用户已接受 B-HMB 并只批准 P01-C。119 条旧辅助记录的独立恢复保留，历史 B1/B2 结果不变。实机可追溯的动态系统辨识候选为 0。[人工授权](reports/approvals/P01_B_HMB_ACCEPTED_P01_C.md) · [冻结实施规格](specs/P01_C_calibration_and_causal_sync.md)

P01-C Run `P01-C-20261126-8b0f399` 在七个开发母源的 23 个真包窗口完成有界几何和因果核心测试。2,248/2,300 拍在模拟到达下联合有效，真实 VERIFIED 联合时序资格为 0。55 项单测和 3 项真包集成测试通过。P01-C 停在 BLOCKED 审核节点，无后续阶段授权。P01-D 的 QC 范围、阈值与切分仍待实测证据及人工冻结。[验收报告](reports/P01_C_acceptance.md) · [运行凭据](evidence/P01_C/review_receipt.json)
