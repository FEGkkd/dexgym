# S0-5 项目状态

[phase_state.json](phase_state.json) 保存阶段和子阶段状态。

| 阶段 | 当前状态 | 授权范围 |
|---|---|---|
| P00 | ACCEPTED | 用户签核 `09c1c9e3c63797083a82493e2a40dd8bb4b1db6d` |
| P01 | IN_PROGRESS | 本次授权限 P01-C 审查修复与依赖调整 |
| P01-A | ACCEPTED | 人工签核仅覆盖 A1 元数据门 |
| P01-B（历史） | BLOCKED | v0.2.0 原 B1 FAIL / B2 BLOCKED |
| P01-B-HMB | ACCEPTED | 用户签核 `8b0f399` 的有界解码门 |
| P01-C（历史结论） | BLOCKED | 原运行与原 C2 真实时钟缺证状态保留 |
| P01-C（历史 HMB 仿真资格） | READY_FOR_REVIEW | C1-GEO 与 C2-SIM 通过有界真包复验，待人工签核 |
| C2-RT | BLOCKED | 设备物理捕获、控制器接收和跨时钟误差界未证 |
| 物理 robot-base 注册 | BLOCKED | 缺独立外部测量 |
| P01-T | BLOCKED | 动态资格为 0，P03 正式启动的强制前置 |
| P01-D–F、P02–P10 | NOT_STARTED | 待各自授权 |

原 Master Design、Contracts、Evaluation Protocol v0.1.1 和 P01 v0.2.0 留作历史版本。项目用户批准 DCR-001 的依赖范围，活动规格为 P01 v0.2.1 和三份 v0.1.2 设计。QC 数值阈值、时钟参数和全量预算将在 P01-C/D 另行实测和冻结。[DCR-001](dcr/DCR-001_p01_hmb_telemetry_dependencies.md) · [原批准记录](reports/approvals/P00_ACCEPTED_P01_A.md)

实际开发在 Ubuntu `/home/ubuntu/dexgym_worktrees/s0-5`。原始 [P00 验收](reports/P00_acceptance.md) 和 [P00 Review Fix](reports/P00_review_fix.md) 留作历史快照。Final Test 仍为 BLOCKED。[历史用途](HISTORICAL_DATA_USAGE.md)

P01-A 在 `P01-A-20261126-09c1c9e-r03` 完成实测。32 个独立真人捕获母源和其启动标定来源通过元数据核查，89 个来源未确认的录制隔离。15 项单元测试和 4 项真实集成测试通过，独立 Final Test 缺封存证明。完整读包、时钟和遥测资格由 P01-B/C 验证。[验收报告](reports/P01_A_acceptance.md) · [运行凭据](evidence/P01_A/review_receipt.json)

用户已接受 B-HMB 并只批准 P01-C。119 条旧辅助记录的独立恢复保留，历史 B1/B2 结果不变。实机可追溯的动态系统辨识候选为 0。[人工授权](reports/approvals/P01_B_HMB_ACCEPTED_P01_C.md) · [冻结实施规格](specs/P01_C_calibration_and_causal_sync.md)

原 P01-C Run `P01-C-20261126-8b0f399` 在七个开发母源的 23 个真包窗口完成有界测试。2,248/2,300 拍在模拟到达下联合有效，真实 VERIFIED 为 0。原验收结论为 BLOCKED。[原验收报告](reports/P01_C_acceptance.md) · [原运行凭据](evidence/P01_C/review_receipt.json)

审查修订 Run `P01-C-REVIEW-20261126-0d9d351` 通过 70 项单测和 3 项真包集成验证。原/新因果姿态联合掩码一致，修订去掉跨安全恢复的速度。依 DCR-002，C1-GEO/C2-SIM 进入 READY_FOR_REVIEW，未授予后续阶段授权。P01-D 需人工签核并单独批准启动，阈值和切分仍待冻结。P03 保留 P02、P01-T、C2-RT 与独立物理基座四项前置。[修订报告](reports/P01_C_review_fix.md) · [DCR-002](dcr/DCR-002_simulation_and_real_clock_qualification.md) · [运行凭据](evidence/P01_C_review/review_receipt.json)
