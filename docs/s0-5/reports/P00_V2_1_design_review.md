# S0-5 v2.0.1 | Teleop 与 EgoPro 补充设计审核报告

**DESIGN_READY_FOR_REVIEW · PENDING_HUMAN_REVIEW · execution_authorized=false**

基线为 `f78ee27f4762fedca9529bf697560bb267f8c704`，原冻结候选 SHA-256 为 `dda0c8254d0c8e637bbe1ee14d1bdfbf8b6b7309a8209263fc350a0c3e4561ea`。项目用户确认 Policy-first、54 DoF、SuperMimic 和自主研发方向。v2.0.0 原设计、凭据、P00–P02 历史和 HMB 身份保存在固定 Git 对象。全部新技术阶段均为 NOT_STARTED。 [原冻结候选](https://github.com/FEGkkd/dexgym/blob/f78ee27f4762fedca9529bf697560bb267f8c704/docs/s0-5/design_freeze_candidate.json) · [变更请求](../dcr/DCR-006_teleop_validation_egopro.md) · [当前候选](../design_freeze_candidate.json)

## 1. 离线低误差

D0 使用真实 MuJoCo 物理步进，按同一 Teleop Validation 靶点比较 B0/B54。预 Actor 冻结静态合法见证域 S5 和具有实际连续 Servo 见证的动态 D10；可改善域仍另报。冻结目标独立于 Policy。绝对候选为 5/10 mm，保持原相对、完整域、功能、安全和 World Model 门。S5 至少 40 个不同目标、6 个来源及 20% 静态支持。D10 至少 1,000 个不重复动态拍、6 个来源及 10% 原定连续有效域。全计划失败复合分位数保留，覆盖不足记 INCONCLUSIVE，实际超出记 FAIL。科研最终 3/5 mm 等挑战沿用。 [V1-A](../specs/V1-A_offline_rmb.md) · [D0 规格](../specs/V1-D_evaluation_deployment.md) · [评估协议](../EVALUATION_PROTOCOL.md)

## 2. 实际现有 MuJoCo Teleop

D1 以用户指定 Teleop 分支为基础，验证 Git、Ubuntu 独立 worktree、二进制、SDK、模型和实际 ROS 图。GitHub 与 Ubuntu 对照工作树为 `6d0b0b5b2601618f1b0052d93ef2403eea9b064a`。用户日常 4sensor 工作树含修改。本次对两树均仅只读核对。正式运行需隔离 ROS Domain 与运行目录，并验证节点清单、仿真驱动、唯一命令写入和回退机制。 [实际链路](../specs/V1-D_teleop_replay_and_refinement.md) · [运行时契约](../specs/V2_runtime_contract.md)

## 3. 指定 MCAP 闭环

只读核对两个实际文件和相邻元数据。角色为 TELEOP_DEPLOYMENT_REGRESSION。

| 录包 | 字节 | 元数据时长 | SHA-256 |
|---|---:|---:|---|
| s0_sim_20261119_172809 | 132,641,216 | 69.260778050 s | `ffd5353918e38e6527b534aa3263baa20d7cb3f8a868d91f33b8644cbd437817` |
| s0_sim_20261117_181022 | 229,433,886 | 119.700535066 s | `cbac62bb6d58ce2fa159d547408cf3534a29c32ca8b0bba1cddc8c4bc99ba1f5` |

元数据列出双侧 MANUS/Tracker、流程状态和命令反馈。原 CDR、可用 START、初态和历史来源重叠需在正式预检逐项解析。回放只输入人端合法样本和必要事件，在线 Marvin IK + Wuji SDK 产生名义命令，读取本次 MuJoCo 实测反馈。原指令、反馈和 RMB 只在允许的离线角色使用。T0/T1 用于归因映射或接口变化，T1/T2 在相同实际配置和独立评分器下比较残差。每包每方法至少两次冷启动，独立 Final Test 继续阻塞。 [只读身份记录](../evidence/P00_V2_1/historical_inputs.json) · [回放与对照](../specs/V1-D_teleop_replay_and_refinement.md)

## 4. 部署反馈与修订

先核 START、在线名义、SDK 状态、坐标、观测归一化、时钟和历史、Servo、投影、失效与回退。D2 保留原 checkpoint、问题证据、变化、成本、新 checkpoint 和双回归，最多两个有据的迭代。D3 对最终 SHA 同时重新做离线和两包实际回归。录包根及其新增衍生训练样本禁止准入，历史重叠须披露。 [D2、D3](../specs/V1-D_teleop_replay_and_refinement.md)

## 5. 何时引入 EgoPro

Teleop-only 已通过 D0 和 D1，保留 Policy-T，D2 经独立复核确认接口问题不能解释缺口，且 E0 有可重复的几何或动态分布证据、候选样本和剩余预算时，才考虑独立 EgoPro Bank。历史 Marvin54、S0-1、S0-4 资产只作来源线索。E1 单独验证骨架、源级划分、时间及缺字段 mask、坐标、名义与意图和无对象支持。执行授权须明确包含 E 阶段。 [数据与准入](../specs/V1-E_conditional_egopro_augmentation.md)

## 6. 如何归因 EgoPro 增益

E2 以已保留的 Policy-T 为起点，80/20 为采样候选，允许 EgoPro 10%–30%。混合域的真实 54 轴状态与动作、1/4/8 步模型、梯度、Servo、边界与模型利用差必须先获独立资格。基础分布资格继续可查。E3 分别比较 Policy-T 与 Policy-E 的缺口、Teleop Validation 和指定 MCAP。预算允许时预登记等额 Teleop-only 继续训练对照；未运行等额对照时，观察只能归于整套补充方案。 [补充训练契约](../specs/V2_1_egopro_training_and_regression.md)

## 7. 如何保留原 Teleop

固定旧 Teleop 静态、动态、全域和功能条件，既验原绝对/相对及安全门，也对 Policy-T 做额外非退化。两包分开审查失败与覆盖。混合平均无权抵消退化。任何保留或新域收益门失败均走 Policy-T 路径，整体交付仍需现有图实际双回归。 [保留与回退](../EVALUATION_PROTOCOL.md)

## 8. 整体资源和权限

训练上限 20M，C0/C1 含 B smoke 共 2M，C2 8M，C3 8M，D2 和 E2 各有独占 1M。2M 评估里保留 650k 给 D0–D3 及 E，分为 200k、150k、100k、200k。预 Actor 动态正控最多 100k，计入原 C0 的 700k 评估池。保留 1 GPU、48 GPU 小时、72 小时总时长及 250 GiB。新增或耗尽资源时由授权与实际计量约束，支持不足可提交较低等级及证据。 [资源与自主推进](../MULTI_AGENT_PROTOCOL.md)

## 9. M1 连续性与诊断导出

硬阈值不达连续性门时，只在正式 Actor 前比较预登记的平滑区间，以 Train 定标、独立验证连续性及功能非退化。保留 M0、硬 M1、旧阈值掩码和原分母，重新验证域。独立诊断只允许有效已有 checkpoint 和数据、功能、安全、因果、导出证据，最高 EXPORTABLE。正式独立仿真及实际 Teleop 等级分别绑定同一候选的实际证据。 [平滑和能力域](../specs/V1-A_offline_rmb.md) · [分级](../specs/V1-D_evaluation_deployment.md)

## 10. 设计检查与待审风险

独立 Reviewer 的发现和复验证据保留，包括冻结输入、近零比率、两包配对、E 分支授权、新域 WM、剩余预算、最终策略身份和等级分离。新版检查器校验 41 项需求和 14 个阶段，合成控制覆盖拒绝路径及 Policy-T/Policy-E 正常路径，原候选在原 Git 对象独立重放。未发生数据生产、训练、回放部署或阶段启动。 [独立 Reviewer](../evidence/P00_V2_1/independent_findings.md) · [主代理实际检查](../evidence/P00_V2_1/verification_results.json) · [工作包](../evidence/P00_V2_1/agent_work_manifest.json)

请在设计审核中确认 5/10 mm 和覆盖候选、M1 平滑范围、回放样本支持与分包非退化、EgoPro 缺口及保留门、D/E 子预算与诊断出口。主要待实测风险是同录包的可重建起始条件、历史训练重叠、动态低误差域可用覆盖和新域动力学误差。项目审批与下一条独立 Campaign 指令继续分开。
