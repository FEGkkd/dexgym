# S0-5 项目状态

[phase_state.json](phase_state.json) 保存阶段和子阶段状态。

| 阶段 | 当前状态 | 授权范围 |
|---|---|---|
| P00 | ACCEPTED | 用户签核 `09c1c9e3c63797083a82493e2a40dd8bb4b1db6d` |
| P01 | ACCEPTED | 人工签核历史 HMB 数据主线和三项历史用途 |
| P01-A | ACCEPTED | 人工签核仅覆盖 A1 元数据门 |
| P01-B（历史） | BLOCKED | v0.2.0 原 B1 FAIL / B2 BLOCKED |
| P01-B-HMB | ACCEPTED | 用户签核 `8b0f399` 的有界解码门 |
| P01-C（历史结论） | BLOCKED | 原运行与原 C2 真实时钟缺证状态保留 |
| P01-C（历史 HMB 仿真资格） | ACCEPTED | 项目用户签核 `eb00f25`、最终凭据与 SHA |
| P01-D | ACCEPTED | 用户签核 `34e44b6`，绑定原 D1 配置和运行凭据 |
| P01-E | ACCEPTED | 用户签核绑定正式 Run、manifest、计划及凭据 SHA |
| P01-F | ACCEPTED | 签核绑定 F1 receipt、P02 handoff 和 E1 manifest SHA |
| C2-RT | BLOCKED | 设备物理捕获、控制器接收和跨时钟误差界未证 |
| 物理 robot-base 注册 | BLOCKED | 缺独立外部测量 |
| P01-T | BLOCKED | 动态资格为 0，P03 正式启动的强制前置 |
| Final Test | BLOCKED | 缺乏独立封存证据 |
| P02 | READY_FOR_REVIEW | 历史静态研究有限签核申请，正式评分域独立待审 |
| P03–P10 | NOT_STARTED | 依各自独立前置 |

原 Master Design、Contracts、Evaluation Protocol v0.1.1 和 P01 v0.2.0 留作历史版本。项目用户批准 DCR-001 的依赖范围，活动规格为 P01 v0.2.1 和三份 v0.1.2 设计。QC 数值阈值、时钟参数和全量预算将在 P01-C/D 另行实测和冻结。[DCR-001](dcr/DCR-001_p01_hmb_telemetry_dependencies.md) · [原批准记录](reports/approvals/P00_ACCEPTED_P01_A.md)

实际开发在 Ubuntu `/home/ubuntu/dexgym_worktrees/s0-5`。原始 [P00 验收](reports/P00_acceptance.md) 和 [P00 Review Fix](reports/P00_review_fix.md) 留作历史快照。Final Test 仍为 BLOCKED。[历史用途](HISTORICAL_DATA_USAGE.md)

P01-A 在 `P01-A-20261126-09c1c9e-r03` 完成实测。32 个独立真人捕获母源和其启动标定来源通过元数据核查，89 个来源未确认的录制隔离。15 项单元测试和 4 项真实集成测试通过，独立 Final Test 缺封存证明。完整读包、时钟和遥测资格由 P01-B/C 验证。[验收报告](reports/P01_A_acceptance.md) · [运行凭据](evidence/P01_A/review_receipt.json)

用户已接受 B-HMB 并只批准 P01-C。119 条旧辅助记录的独立恢复保留，历史 B1/B2 结果不变。实机可追溯的动态系统辨识候选为 0。[人工授权](reports/approvals/P01_B_HMB_ACCEPTED_P01_C.md) · [冻结实施规格](specs/P01_C_calibration_and_causal_sync.md)

原 P01-C Run `P01-C-20261126-8b0f399` 在七个开发母源的 23 个真包窗口完成有界测试。2,248/2,300 拍在模拟到达下联合有效，真实 VERIFIED 为 0。原验收结论为 BLOCKED。[原验收报告](reports/P01_C_acceptance.md) · [原运行凭据](evidence/P01_C/review_receipt.json)

审查修订 Run `P01-C-REVIEW-20261126-0d9d351` 通过 70 项单测和 3 项真包集成，随后最终边界补丁通过 78 项单测和 3 项集成。原 READY_FOR_REVIEW 历史事件保留，现有用户签核绑定最终提交与凭据。P03 保留 P02、P01-T、C2-RT 与独立物理基座四项前置。[修订报告](reports/P01_C_review_fix.md) · [最终补丁](reports/P01_C_final_patch.md) · [人工签核](reports/approvals/P01_C_ACCEPTED_P01_D.md)

P01-D Run `P01-D-20261126-eb00f25` 产出已获人工批准的冻结方案与有界成果。32 个母源的 24/8 开发 split、QC、schema 及仿真用途保持已批准的值。[D1 验收](reports/P01_D_acceptance.md) · [原运行凭据](evidence/P01_D/review_receipt.json) · [人工批准](reports/approvals/P01_D_ACCEPTED_P01_E_E0.md)

E0 的原技术审核事件和证据继续保留。项目用户独立批准配置与实施锁的精确 SHA 及正式 Run ID `P01-E-E1-20261126-3f90652`。[E0 预检报告](reports/P01_E_preflight.md) · [E0 签核与 E1 授权](reports/approvals/P01_E_E0_ACCEPTED_P01_E_E1.md)

正式 E1 历史仿真 HMB 已由用户签核。P01-F 独立审计 Run 为 `P01-F-20261126-b22f8be`。[E1 验收报告](reports/P01_E_acceptance.md) · [人工签核](reports/approvals/P01_E_ACCEPTED_P01_F.md) · [P01-F 规格](specs/P01_F_independent_hmb_acceptance.md)

F1 重算全部拍及有效段，重新校验 1,540 个正式分区文件并从 32 个母源的原始数据复验 4,790 对消息。用户签核 P01-F、P01 数据主线与三类历史用途，独立授权 P02；当前 Run 为 `P02-20261126-54c2e06`。[P01-F 验收](reports/P01_F_acceptance.md) · [P01 历史技术汇总](reports/P01_acceptance.md) · [人工签核和 P02 授权](reports/approvals/P01_ACCEPTED_P02.md)

P02 的受约束静态试验有 36 个已复验 54 DoF 姿态，另有 2 个 UNKNOWN；主队列十指同拍 ≤3 mm 为 0/32。10 个已评分姿态的至少一个原映射指尖超过当前链长的必要可达界。验收只覆盖静态研究，DCR-003 提请裁定工作空间注册与正式评分范围。下一阶段须等待人工审核，且 P03 前继续要求 P01-T、C2-RT 与物理 robot-base 独立 ACCEPTED。[P02 验收](reports/P02_acceptance.md) · [DCR-003](dcr/DCR-003_p02_workspace_registration.md)

用户批准对 DCR-003 做专项研究。独立 P02 复查 Run P02-REFINE-20261126-8753921 完成 114,532 拍检查、冻结 Train 注册、同目标预算和 minimax 诊断。候选 M1 只具历史仿真静态适用域；M0 保留为原目标审计列。原 P02、HMB 文件和资格未变，P02 仍等待人工签核。正式评分域仍待独立批准，P03 前置继续有效。[复查报告](reports/P02_review_refinement.md) · [DCR-003 补证](dcr/DCR-003_workspace_refinement_evidence.md)

P02-FINAL-20261126-ecd12ec 已完成冻结 M1 的 14/54 DoF 自由度配对及独立审计。C2-RT、物理 robot-base 和 P01-T 的联合测量计划已准备，三项继续独立 BLOCKED。P02 有限签核、物理评分域和 A1/A2 无运动现场资格须分别审批。P03 尚未启动。后续并行工作按协议登记。[有限交接](reports/P02_final_handoff.md) · [并行协议](MULTI_AGENT_PROTOCOL.md) · [测量计划](specs/HARDWARE_SHARED_MEASUREMENT_PLAN.md)
