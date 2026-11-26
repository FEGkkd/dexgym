# S0-5 Master Design v2.0.1

**Generalist Real-Time Residual Motion Controller for High-Precision Bimanual Dexterous Teleoperation**

`DRAFT_READY_FOR_REVIEW`。V2.0.1 是待审设计补充。当前授权仅覆盖设计修订、源码与历史证据审计、检查器及合成检查。本候选的人工批准和 V1 Campaign 启动是两个独立事件。原活动设计在候选获批前保有其原有范围及历史基准身份。[设计变更](dcr/DCR-006_teleop_validation_egopro.md) · [项目状态](PROJECT_STATE.md)

## M1 研究问题

在同一双臂双手 54 DoF 机器人、实时可用人端信号、名义参考、意图、仿真 Servo、安全约束和评分时间轴下，SuperMimic 学到的闭环残差能否相较现有 Marvin IK + Wuji SDK 改善十指空间精度、臂手协调、动态跟踪、稳定保持及恢复？以执行后几何、功能表现、约束介入、失败和延迟回答。使用 **Existing Teleop + 54 DoF Residual RL**，基础命令链沿用现有名义算法。V1 先验收离线绝对精度，再复用现有 MuJoCo Teleop 对指定录包做闭环验证、诊断及有界修订。首版发布以离线和 Teleop 回归均通过为目标。[现有 Teleop](https://github.com/FEGkkd/whole-body-teleop/tree/codex/whole-body-sim-phase1) · [评估协议](EVALUATION_PROTOCOL.md)

核心训练和评价以合格 Teleop MANUS 双手骨架、双腕 Tracker 及机器人反馈为主。Teleop-only Policy 达到核心有效性后，E0–E3 可有界使用独立 EgoPro Bank 补充有证据的参考缺口。EgoPro 仅标准化人体几何与轨迹，分别记录来源和 mask。仿真采用无物体机器人与场地碰撞几何。对象状态、图像、视觉策略、触觉、VLA、自主任务执行和对象接触动力学在范围之外。[HMB Reader 契约](specs/HMB_reader_contract.md)

## M2 结构与信号所有权

| 向下游依次传递 | 所有者与控制契约 |
| --- | --- |
| 可用人端输入 | HMB Reader 提供历史模拟因果数据，仿真部署用等型的实时可用样本、时间、时龄、权限与世代状态。独立记录源时标及到达时标。 |
| 关节名义参考 | Marvin IK 为臂，Wuji SDK 为手。V1 训练输入为 `CACHED_CAUSAL_NOMINAL_REFERENCE`。V1-D1 在现有软件链路内由 Marvin IK + Wuji SDK 实时生成 `q_nominal_online`。两种路径的启动状态、时钟、采样和分布差异须实测。 |
| 独立人意图 | 冻结注册、手部地标、腕与功能几何从人端数据生成。独立生成器与评分过程不得读取被评控制器的输出或机器人实测姿态。可选独立合法见证另存其几何与状态。 |
| 残差 Actor | 当前帧及过去四帧的因果观测和机器人反馈输入共享 MLP 512/512/256，近零初始化，输出 54 维有界归一化动作。V1-B 冻结特征顺序和模型接口。 |
| 仿真命令与反馈 | `q_requested = q_nominal + scale ⊙ a`，`q_issued = SafetyProjection(q_requested)`；Servo 演化 `q_applied`，MuJoCo 提供 `q_measured, qd_measured`。硬投影、路径护栏、执行器和回退有独立裁决权，Actor 和 World Model 只能提出动作或作可微预测。 |
| 衡量与发布 | 独立评估进程只依据 MuJoCo 执行结果选 checkpoint，验证 B0/B54 等条件、连续回放、失败和性能门。部署包保存 Actor、接口及可验证运行依赖。 |

原 Teleop START 与映射保留作 T0。D1 固定实际现有工程、软件、模型、域和事件，T1/T2 共享在线名义计算和安全，仅切换 S0-5 残差。独立评分由人端输入及冻结的事件/目标生成，录包中的机器人命令和反馈只作历史诊断。真实时钟、多频率和实物命令权另外验证。[技术契约](CONTRACTS.md) · [仿真部署规格](specs/V1-D_evaluation_deployment.md)

## M3 参考、功能几何与可改善域

HMB 是冻结的真人数据及 `SIMULATED` 时钟产品。RMB 由 V1-A 新建，将 `q_nominal, qd_nominal`、原 SDK 输出、名义 FK 与状态存为一层，将独立人意图存为另一层，并可增加独立可行几何。世界系目标包含腕与十指，局部手型、同手捏合、跨手指对及双腕相对 SE(3) 分项。保留 M0 原意图参照、映射差、投影差和物理执行差。在同一点同时刻另有合法可行几何时，执行相对 mapped intent 的误差向量等于相对可行几何的误差向量加投影向量。各项百分位独立计算。

P02 既有有限交接覆盖历史仿真静态研究。V2.0 提请将冻结的尺度 1 M1 授予 V1 连续仿真参考用途，依赖 Train-only 身份、连续性和阈值切换技术检查。M0 为不可变审计列，M2、Minimax 分别用于压缩和位置可达诊断，硬阈值连续性未通过时可在允许的候选区间验证平滑过渡，训练前冻结分支、连续性、功能非退化以及域成员。原 M1 和 M0 逐拍保留。保留严格必要条件排除、UNKNOWN、独立合法残差改善见证、困难连续运动和全计划域。共同可改善域和绝对低误差能力域的规则和样本身份须在正式 Actor 学习之前冻结。静态合法见证、动态连续 Servo 见证和可改善空间分别登记，训练后的输出和结果无权改变其分母。[V1-A 规格](specs/V1-A_offline_rmb.md) · [RMB 契约](specs/V2_rmb_contract.md) · [P02 最终交接](reports/P02_final_handoff.md) · [P02 历史有限签核](https://github.com/FEGkkd/dexgym/blob/c5af9af94d7b9d3f437c14df128441d1183bec56/docs/s0-5/reports/approvals/P02_LIMITED_ACCEPTANCE_SIM0.md)

## M4 环境与算法

初始候选为正常重力、50 Hz 控制、500 Hz 物理和位置 Servo，2 ms 子步每拍 10 次。优先移植已审 MuJoCo Warp、CPU 几何、Torch FK 与命令投影，按实际 SHA、54 轴命名、路径约束、动作响应与 CPU/GPU Servo 验证决定资格。残差初始尺度左右臂 ±8°、双手 ±5°。四组动作依次为左臂 7、右臂 7、左手 20、右手 20。执行等价性用 B0 核对。几何 Reward 使用独立 mapped intent，多尺度世界指尖、局部手型、对捏、双手、腕和速度，并按同一 Reward 单位设定 Critic value bound。[环境规格](specs/V1-B_residual_environment.md)

主算法复用现有 SuperMimic。以 MuJoCo 执行 transition 填充 Replay，学习 54 轴动力学及必要的失败预测。通过独立模型资格后，从合法 Replay 状态做短时 imagination，可微几何回报训练 Actor，Critic 匹配 λ-return。不断采样 MuJoCo 纠正分布。模型状态须包含已证必要的 Servo 记忆，任务适配须绑定请求、投影、下发、滤波后与有效区间量。模型资格覆盖 1/4/8 步误差、静止、动作梯度、各轴实际激励、约束边界与源级留出，Actor 更新服从资格以及运行期模型利用检查。[World Model 规格](specs/V1-C0_world_model.md)

采样器可访问合格时序、根和分段索引。Actor 白名单排除未来时刻、离线插值、录制和轨迹索引、Oracle q 及特权标签。训练环境能在想象之后的步骤推进外生参考，单次 Actor 决策仍只消费当前及历史合法信息。实际闭环评分及发布部署继续使用该信息边界。[RMB 契约](specs/V2_rmb_contract.md) · [环境规格](specs/V1-B_residual_environment.md)

## M5 阶段与资源

在人工设计批准和独立有界 Campaign 授权后，V1-A 建参考，V1-B 建任务与通过接口/物理门，V1-C0 用有界真实仿真采样资格化 World Model，V1-C1/C2/C3 依次做静态、原速动态和混合，V1-D0 至 D3 依次开展离线精度、现有 Teleop 回放、诊断迭代和离线/部署双回归。V1 候选主训练实物理上限为 2000 万 transitions，包含探索、WM-only warmup 及失败重试。C3 的 10M 划为 8M Teleop-only、1M 部署修订与 1M EgoPro/匹配对照条件池。2M 独立仿真评估按冻结子桶另记。显存、时钟时间、GPU-hours、评估和重试均使用有限台账，以授权预算和运行预检为准。新联合 IK/QP、PPO、新人体录制、最终 START、真实计量和辨识不构成该 V1 仿真主线依赖。历史相关证据和可选对照均可追溯。[阶段路线](ROADMAP.md) · [自主研发规范](MULTI_AGENT_PROTOCOL.md)

## M6 验收与研究外推

V1 比较相同缓存参考下的 B0 零残差和 B54 全身残差。B14、S0-4、PPO 或新联合 IK 可在匹配条件下另作消融或后续对照。D0 先验证冻结低误差域的绝对精度候选（静态 5 mm、动态 10 mm），同时保留相对、完整域、安全及非退化门。D1 使用 T0、T1、T2 在现有工程做配对录包闭环。D2 先修接口，再按证据及预算补训；D3 重跑同一候选的离线与两包回归。诊断导出是独立较低等级出口。V1 工程结果沿用单种子及历史开发集范围。独立 Final Test 在有独立新数据、审计与批准前保持阻塞。后续正式科研对照仍须预注册、多种子、独立数据、完整物理评分和原有挑战标准。真机另需 C2-RT、P01-T、物理 robot-base、Servo、外部计量、实物安全，以及各级 Shadow/HIL/现场人工批准。[评估协议](EVALUATION_PROTOCOL.md) · [部署规格](specs/V1-D_evaluation_deployment.md) · [共享硬件计量](specs/HARDWARE_SHARED_MEASUREMENT_PLAN.md)

## M7 设计、执行与证据身份

Design Freeze 候选锁定版本、基线 Git 对象、文档 SHA、需求映射、DAG、量化门、待审参数、预算和自主更改边界。后续生效要分别记录项目用户真实设计批准和 V1 Campaign 执行授权。Main Agent 按实际技术证据、独立审查和预算推进，通过的技术记录使用 `TECHNICAL_PASS`，最终交付为 `READY_FOR_REVIEW`。保留每次失败、原始报告及不可变历史证据，任何阶段、目标域或指标变化须按契约重新获得资格。[追踪矩阵](TRACEABILITY_MATRIX.md) · [自主研发规范](MULTI_AGENT_PROTOCOL.md) · [设计冻结候选](design_freeze_candidate.json)

[实际 Teleop 规格](specs/V1-D_teleop_replay_and_refinement.md) · [EgoPro 条件分支](specs/V1-E_conditional_egopro_augmentation.md) · [v2.0.0 总体设计](https://github.com/FEGkkd/dexgym/blob/f78ee27f4762fedca9529bf697560bb267f8c704/docs/s0-5/MASTER_DESIGN.md)
