# S0-5 v2.0.1 | Policy-First 设计审阅入口

**DESIGN_READY_FOR_REVIEW · PENDING_HUMAN_REVIEW · execution_authorized=false**

S0-5 基于现有 Marvin IK + Wuji SDK 与 54 DoF SuperMimic 残差控制。v2.0.1 将离线精度、实际现有 MuJoCo 遥操的两份录包闭环、诊断迭代及最终双回归纳入首版发布。EgoPro 是在 Teleop-only 已有效且确认覆盖缺口时开启的补充支线。

[总体设计](MASTER_DESIGN.md) · [路线](ROADMAP.md) · [要求](REQUIREMENTS.md) · [追踪](TRACEABILITY_MATRIX.md) · [契约](CONTRACTS.md) · [评估](EVALUATION_PROTOCOL.md) · [自主研发](MULTI_AGENT_PROTOCOL.md)

[阶段状态](PROJECT_STATE.md) · [DCR](dcr/DCR-006_teleop_validation_egopro.md) · [审核报告](reports/P00_V2_1_design_review.md) · [冻结候选](design_freeze_candidate.json) · [v2.0.0 原冻结候选](https://github.com/FEGkkd/dexgym/blob/f78ee27f4762fedca9529bf697560bb267f8c704/docs/s0-5/design_freeze_candidate.json)

```bash
python -B docs/s0-5/evidence/P00_V2_1/check_design.py
python -B docs/s0-5/evidence/P00_V2_1/test_check_design.py
```

检查器只读设计和历史 Git 对象，合成控制在独立 fixture 上运行。V1 全部阶段等待用户另行授予 Campaign 权限。
