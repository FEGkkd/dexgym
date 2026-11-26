# S0-5 v2.0.0 | Policy-First 设计审阅入口

**DRAFT_READY_FOR_REVIEW · PENDING_HUMAN_REVIEW · execution_authorized=false**

项目名称为 **Generalist Real-Time Residual Motion Controller for High-Precision Bimanual Dexterous Teleoperation**。V2 面向现有 Marvin IK + Wuji SDK 遥操作系统，以共享 54 DoF SuperMimic 残差策略为核心。当前交付是设计修订候选，整条 V1 主线尚处于 NOT_STARTED。人工批准设计之后，项目用户另行授权整体 V1 Campaign。

| 阅读顺序 | 文件 |
|---|---|
| 研究设计与路线 | [总体设计](MASTER_DESIGN.md) · [路线图](ROADMAP.md) · [DCR-005](dcr/DCR-005_policy_first_supermimic.md) |
| 要求和接口 | [需求](REQUIREMENTS.md) · [追踪矩阵](TRACEABILITY_MATRIX.md) · [技术契约](CONTRACTS.md) |
| 证据和发布 | [评估协议](EVALUATION_PROTOCOL.md) · [阶段状态](PROJECT_STATE.md) · [自主研发协议](MULTI_AGENT_PROTOCOL.md) |
| 版本身份和实查 | [设计审核报告](reports/P00_V2_design_review.md) · [冻结候选](design_freeze_candidate.json) |

阶段实施入口按 [V1-A](specs/V1-A_offline_rmb.md)、[V1-B](specs/V1-B_residual_environment.md)、[V1-C0](specs/V1-C0_world_model.md)、[V1-C1](specs/V1-C1_static.md)、[V1-C2](specs/V1-C2_dynamic.md)、[V1-C3](specs/V1-C3_mixed.md)、[V1-D](specs/V1-D_evaluation_deployment.md) 阅读。旧 P00–P02 保持已记录资格，旧活动规格和批准在固定 Git 对象可重放。 [原设计入口](https://github.com/FEGkkd/dexgym/blob/b8d8b401bdda59ad00a37f0a3bf6878b462ed756/docs/s0-5/README.md)

只读核验本次候选可在仓库根目录运行：

```bash
python -B docs/s0-5/evidence/P00_V2/check_design.py
python -B docs/s0-5/evidence/P00_V2/test_check_design.py
python -B docs/s0-5/evidence/P00_V2/verify_legacy.py
```

检查器只读仓库、Git 对象与候选状态。Campaign 需在另行批准后创建独立持久台账。

历史 P00 检查器在原版本运行。当前 checkout 可用原脚本的 `--history-only --require-history` 复验 P00 固定对象。完整历史结构和原合成测试由 `verify_legacy.py` 从各自 Git 版本隔离重放。
