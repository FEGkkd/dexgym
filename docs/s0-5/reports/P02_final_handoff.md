# P02 最终有限交接与硬件前置准备

**P02 保持 READY_FOR_REVIEW。** 本轮 Run P02-FINAL-20261126-ecd12ec 以 ecd12ecd6657539e44830dde07c6055c6c507192 为审核基线。用户认可五项研究用途，正式物理评分域待审批。原 P02、专项研究、P01 HMB、模型、split、QC、失败和报告均保留。[冻结计划](../evidence/P02/final_handoff/work_plan.json) · [运行凭据](../evidence/P02/final_handoff/execution_receipt.json)

## 签核身份和工作包

| 复验对象 | SHA-256 |
|---|---|
| Refinement receipt | 0adfc8976ebfa7ee0845d42e5d291caa70f94cf172e72d4a7d2fcb3c16a15aa0 |
| M1/M2 Train fitted candidates | 577195a8eb66699e11cf8bcf910383086672c384723d61fa90c13ed90652baa7 |
| Static scoring candidate | 96aac52666d7b03d6a3c087cb845f824a2ed4403f1cf35ae87af0264dd5f4f94 |

主代理冻结输入，执行 A/B 在 Ubuntu 的官方实验并封存证据。三名子代理各有独占文件范围，B 未参与 A 的实现。[长期协议](../MULTI_AGENT_PROTOCOL.md) · [工作包、命令、产物与 SHA](../evidence/P02/final_handoff/agent_work_manifest.json)

| 工作包 | 结论 |
|---|---|
| A M1 配对 | 31/32 主样本具两组合法见证，其中 30 改善、1 退化。配对十指最大误差中位差 −22.09 mm。54 DoF 的 3/5/10/15 mm 同拍命中为 5/8/14/15。原两个 UNKNOWN 及一个 14 DoF 超时保留。 |
| B 独立审计 | PASS_STATIC_RESEARCH_ONLY。核验 71 个模型资产、38 个样本身份和 35 个严格配对。提出硬约束、功能非退化、最差指误差和连续性四层名义算法规格。 |
| C 硬件前置 | 核验 19 个 Ubuntu 实际工作区源码文件，5 个不同于 Git HEAD。共同计量方案和现场清单已建立。物理测量与运行二进制尚待取证，三个资格各为 BLOCKED。 |

[M1 配对报告](P02_M1_dof_pairing.md) · [独立数学与评分审计](P02_final_independent_audit.md) · [硬件前置证据](P02_hardware_prerequisite_readiness.md)

Ubuntu **20 项 P02 单测通过**，A 与 B 进程均为退出 0。未运行 M2 完整配对，也未增加 Oracle 预算。历史 S0-4 对照原样保留。实际 Servo、物理工作空间和真实时钟仍依赖下游独立测量。[测试输出](../evidence/P02/final_handoff/p02_unit_tests.txt) · [新 Run 文件 SHA](../evidence/P02/final_handoff/archive_sha256.json)

## 提交的两项独立申请

1. **P02 历史静态研究有限签核。** 申请批准有界交接。M0 是不可变历史意图审计，M1 是尺度 1 的历史仿真静态开发候选，M2 是空间压缩诊断，Minimax 是指尖位置可达诊断，平滑捏合仍为未冻结候选。保留原意图、投影、域外、未知和失败分母。正式物理评分域继续作为独立未决义务，终局精度挑战维持。P02 原共同评分要求与阶段状态保留。[DCR-003 限定条款](../dcr/DCR-003_limited_handoff_candidate.md) · [机器可读候选](../evidence/P02/final_handoff/limited_scope_candidate.json)
2. **硬件专项测量授权。** 申请由项目用户与现场负责人确认工位、设备身份、外部计量与证书、时钟参考、采集权限、数据保存位置及安全负责人。按联合计划先核对现场与具体允许的测量项，再安排实际执行。涉及目标写入、通电状态改变或运动的工作须有明确范围、工位安全审核和独立动作批准。[联合测量计划与现场清单](../specs/HARDWARE_SHARED_MEASUREMENT_PLAN.md)

P01 保持 ACCEPTED。C2-RT、P01-T、物理 robot-base、独立 Final Test 保持 BLOCKED。P03 仍须 P02、P01-T、C2-RT、物理 robot-base 各自签核，并取得正式评分范围及启动授权。P03、RL 和真机动作未执行。
