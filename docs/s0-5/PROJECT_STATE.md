# S0-5 项目状态 v2.0.0

设计包提交状态为 **DESIGN_READY_FOR_REVIEW**，设计内容为 **DRAFT_READY_FOR_REVIEW**，批准状态为 **PENDING_HUMAN_REVIEW**。本次授权的工作范围是 Design Revision 与正常 Git 发布。经独立 Reviewer 复核的技术设计候选等待项目用户和外部 Reviewer 审阅。

| 工作 | 当前登记 | 依据 |
|---|---|---|
| P00 / P01 | ACCEPTED，历史签核范围沿用 | [P00 签核](reports/approvals/P00_ACCEPTED_P01_A.md) · [P01 签核](reports/approvals/P01_ACCEPTED_P02.md) |
| P02 | s0-5 历史账本保留 READY_FOR_REVIEW 事件。另有 P02 历史静态有限 ACCEPTED 的真实用户签核，范围来自 SIM0 的独立已提交对象 | [P02 原交接](reports/P02_final_handoff.md) · [原有限签核](https://github.com/FEGkkd/dexgym/blob/c5af9af94d7b9d3f437c14df128441d1183bec56/docs/s0-5/reports/approvals/P02_LIMITED_ACCEPTANCE_SIM0.md) |
| M1 连续仿真用途 | PENDING_HUMAN_REVIEW，V1-A 连续性验证另设技术门 | [DCR-005](dcr/DCR-005_policy_first_supermimic.md) |
| P03-SIM0 | 独立历史支线，实际研究状态由原工作树记录 | [DCR-004 历史](https://github.com/FEGkkd/dexgym/blob/c5af9af94d7b9d3f437c14df128441d1183bec56/docs/s0-5/dcr/DCR-004_bimanual_simulation_research_branch.md) |
| V1-A 至 V1-D | 全部 NOT_STARTED。Campaign 授权尚未授予 | [状态机](phase_state.json) · [自主研发协议](MULTI_AGENT_PROTOCOL.md) |
| Final Test、正式物理评分、P01-T、C2-RT、物理 robot-base、真实部署 | BLOCKED 或待独立人工批准 | [评估协议](EVALUATION_PROTOCOL.md) · [硬件共同计划](specs/HARDWARE_SHARED_MEASUREMENT_PLAN.md) |

`phase_state.json` 保留原历史事件与字段，V2 状态在独立键 `policy_first_v2`。设计批准身份与执行授权身份分别核对；技术状态字段只表达审核后可复验的技术结论。 [V1 历史账本](https://github.com/FEGkkd/dexgym/blob/b8d8b401bdda59ad00a37f0a3bf6878b462ed756/docs/s0-5/phase_state.json)
