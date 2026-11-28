# DCR-009 | Home-Anchored Per-Recording Human Intent Mapping

**v2.1 设计提案。DCR 状态 `PENDING_HUMAN_REVIEW`，阶段 `DCR_PENDING_HUMAN_REVIEW`，执行授权 `false`。**

| 身份 | 登记 |
| --- | --- |
| Repository / branch | `FEGkkd/dexgym` / `s0-5` |
| 本次已核对远端审核基线 | `cf07a88bc93d1aa9737b41fb83b9392921449142`，含用户给出的 `324d771e29d436b206211f444a6d800338f871ce` 及其后高误差可视化提交。 |
| 已有设计冻结身份 | v2.0.1 `490770e1df45832d0693380563152bd19eefa3f4`，冻结候选 SHA-256 `cd5bff29a5b5f683fae328898aa9060b40fc4be00e193a22d9bf18cb2d10b60b`。 |
| 本次决定 | 项目用户授权审计、设计与提案提交、独立审查、静态和合成设计检查。DCR 批准、物理标定资格、实施、Pilot、全量重建和训练各自另需有效资格。 |

## 原设计和发现的问题

v2.0.1 使用 P02 Train-only 统一注册的单位尺度 M1 构建独立仿真意图。M1 以固定世界旋转和平移映射原始 Tracker，随后经两侧腕姿安装、TCP／掌偏置、机器人局部形态构造及 20 mm 近捏合修正生成几何。P02 的有限历史用途是静态仿真研究。M1 的拟合矩阵没有独立物理注册资格。现有 V1-A 在普通分区边界以固定 Home 重建名义状态，DCR-009 提议未来以录制母源或真实新世代为状态边界。[原 V1-A](../specs/V1-A_offline_rmb.md) · [P02 评分候选](../evidence/P02/refinement/scoring_freeze_candidate.json) · [既有名义候选](DCR-008_causal_teleop_nominal_reference_candidate.md)

历史有界诊断中的 32 个母源都从实测掌心朝下 Home 到达了相应目标并完成仿真回放。该诊断采用首个合格四路拍，新的因果稳定窗口尚无真实合格证据。其 99,562 个共同拍上，M1 到共享原生目标的腕位置 P95 为 693.28 mm，共享目标到已下发腕 FK 为 79.11 mm，下发到实测为 18.66 mm。末端位置误差尾部 4,979 拍中，有 4,972 拍的最大链路差位于 M1 与共享目标之间。M1 拟合的全局轴转换与历史垂直标定候选约差 95.73°。候选把已记录的向上轴送到 MuJoCo 向上轴，M1 在此对照中送到了水平面。该证据界定帧语义分歧及三例的相对几何差异，选择正式目标和物理标定仍需独立取证。[全量分层诊断](../reports/TELEOP_full_start_handoff_layer_audit.md) · [中点与相对几何](../campaign/evidence/teleop_full_start_handoff_mapping_geometry.json) · [同拍见证](../reports/TELEOP_high_error_mapping_witnesses.md)

DCR-008 的原联合有效 Train 分母为 85,382。原固定 M1 精确双腕 IK 覆盖上界仅 25.08%，历史诊断也记录了连续性和受限命令问题。该必要界的适用条件是指定几何与签名模型。新目标须重新检验自己的可达性、路径、连续性和残差改善域。既有 DCR-008 的候选结果、未通过项和审批状态按其自身回执管理。[DCR-008 独立评审](../campaign/DCR-008_independent_review.md)

## 拟议决定和数学定义

保留 Existing Teleop + 54 DoF Residual RL 的结构。Teleop 给出策略无关名义动作，独立的人体意图定义目标，MuJoCo 实测提供实际执行几何。Policy 以后用残差减小独立目标与实际执行之间的误差。Marvin IK、Wuji SDK、因果输入、Servo 和安全所有权沿现行链路审计及复用。

统一采用列向量和 `T_AB` 约定，将 B 帧坐标表达为 A 帧。沿几何提案记 `H` 为人体 Tracker 世界（用户记号 `W`），`W` 为机器人共同 Base／仿真世界（用户记号 `B`），`B_s` 为左右各臂求解基座。`K_s` 是有侧别的 Tracker 刚体，`A_s` 是匹配的解剖腕根，`N_s` 是 Marvin TCP，`U_s` 是评估腕／掌站点。固定安装定义 `T_HA_s(t) = T_HK_s(t) T_K_sA_s`。Home 和人体均取相同语义的解剖腕参考。

对每个 root 或有证据的新世代 `k`，取 `m_H,k = (p_H,A_L,start + p_H,A_R,start)/2`，`m_W,home = (p_W,A_L,home + p_W,A_R,home)/2`。固定 `R_cal ∈ SO(3)` 由独立标定提供。锁存 `t_k = m_W,home − R_cal m_H,k`，`T_WH,k = (R_cal,t_k)`，并计算 `T_WA_s^*(t) = T_WH,k T_HA_s(t)`。以 `T_B_sN_s^* = (T_WB_s)^−1 T_WA_s^* T_A_sN_s` 送入机器人求解适配，以 `T_WU_s^* = T_WA_s^* T_A_sU_s` 保存对应站点目标。固定 TCP／掌参考的右乘含旋转和对应原点偏置。双腕间距和相对 SE(3) 的严格保真断言在匹配的 `A_s` 参考成立。各层差值单列，缩放、左右独立对齐、夹紧或重定时均无权改写独立目标。[拟议坐标与几何规定](../specs/V2_1_home_anchored_geometry_proposal.md)

Robot Home 是经过签名模型和当次新鲜稳定反馈确认的起点，默认掌心朝下。Human START 是本段在冻结因果截止处有资格的人体姿态。人体初始掌心朝上、相对或左右不同都应保留。Home 只提供中点和安全过渡起点。每母源默认初始化一次，有独立证据的重启、时钟／标定／不连续拓扑变化、严重失效或安全复位才开新世代。普通存储分区延续同一会话。对可观察且有资格的真实 START，按相同因果稳定窗验收；明确存在但不合格的事件记 `START_UNQUALIFIED`。只有可靠来源中缺失或不可观测的真实 START 才可重建，采用第一个完整合格尾随稳定窗的终止控制拍并标记 `RECONSTRUCTED_START`。所有原前缀与不合格样本保留原分母和时间轴，Train 与 Validation 的原始身份保持不变。[拟议运行时与候选参数](../specs/V2_1_home_anchored_runtime_proposal.md)

## 标定资格、过渡与失效

现有 `R_cal`、左右 tracker-to-palm 与 Wuji/TCP 桥仅有具备身份的历史校准及仿真诊断证据。需要独立确认真实轴定义、物理标定和重复佩戴／安装的可用范围。老 P02 M1、以双腕连线或掌面拟合每个 `R_k`、以 Home 法向重定向人体起始姿态都无物理资格。后续有界校准须绑定测量夹具、独立重测、方向和位置误差、两侧互换负控、标定世代、版本和审查结果。允许先行完成无执行的设计检查。未取得相应授权和标定即阻塞所依赖的 Pilot 或全量构建。[标定审计与拟议资格](../specs/V2_1_home_anchored_geometry_proposal.md)

先到达质量合格的 Home，再通过现有 Marvin IK／Wuji SDK／Servo 及独立硬安全层达到锁存的 START。先有同世代实测反馈、姿态和位置容差、稳定速度、规定 dwell 与路径证据，才切入正式历史时间轴。记录独立的 Home、预定位、START、回放和安全停止状态。若目标解析证明不可达则标 `UNREACHABLE` 并保存界和作用域。无法验证、超时、未解或有限搜索用 `UNKNOWN`／未到达，保留安全拒绝、HOLD、FALLBACK、失效回退与可观察状态。不能到达的根／世代保留原目标，不进入正式回放。不得压缩原人意图来通过安全门。[拟议运行时规定](../specs/V2_1_home_anchored_runtime_proposal.md)

逐层保存原输入、独立 START、原生 raw/filtered/workspace/step/IK 目标、Marvin 和 Wuji 原始结果、接纳、名义、下发、可证生效、反馈与独立 FK。79.11、76.74、18.66 mm 对应不同链路和样本域。Workspace 投影、限速、IK、Servo 及延迟的独立影响须在后续有界验证区分，避免将夹具或状态失真交给 RL。早期原生回放和离线缓存只有在明确输入、同状态、同资产、同会话时束下才可比较，正式在线等价另需指定录包证据。

## 数据、评分和旧门槛

提议新增 `START_HOME_ANCHORED_V1`。`M0_TARGET`、`M1_TARGET`、`START_TARGET`、`Q_NOMINAL`、`Q_ISSUED`、`Q_MEASURED` 及可证明的 `Q_APPLIED` 保留独立身份。机器人局部形态适配单独版本化，与人体刚体位置变换及可行见证严格分开。Reward 提请使用已批准后冻结的 `START_TARGET`，延续世界十指、腕位姿、局部手型、同手捏合、跨手、相对 SE(3)、动态速度及连续性。姿态误差定义在同帧 SO(3)，名义、发布和反馈对独立目标没有生成权。新的几何、Reward 目标及同条件 B0/B54、低误差域和可改善域使用前须重新取证。历史评分和最终科研门保留。[RMB 和 Gate 提案](../specs/V2_1_home_anchored_data_gate_proposal.md)

32 个母源，Train / Validation 为 24 / 8，分区 228 / 80，计划拍 86,338 / 30,275，共 116,613。HMB 联合有效 85,382 / 29,150，共 114,532。原五帧候选 82,559 / 28,237。每个 split 原 85% 和 70% 门槛与口径仍冻结。提请批准的主动跟踪只能计入因果、合法且确有当前跟踪证据的拍。仅发布、START 成功、HOLD、FALLBACK、FAILED、UNKNOWN、不可达或缺测均无主动资格。五帧仍使用原签名候选窗，禁止纳入跨分区额外窗来改变既有分子。统计逐类报告原计划分母、源和遮罩。未来跨分区会话连续性另列诊断。

## 备选和风险

1. 完全沿用旧 M1 作为主目标。能保留旧实验同口径，仍存在轴约定分歧、动态连续性与名义资格缺口。
2. 按每段双腕方向或掌法向拟合新的全局旋转。该自由度须有独立物理基准才能纳入范围，当前提案采用固定 `R_cal`，每段仅锁存共同平移。
3. 以每侧单独锁存和 Home 姿态归零，可能保住执行起姿。两手间距、掌面方向与跨手几何因此受改变。当前提案保留这些原始量。
4. 复用共享诊断程序及已有证据做后续受控适配。32/32 历史成功可用于风险评估，先满足新的稳定窗、模型、标定与独立路径资格。

风险包括历史安装漂移、真值独立性不足、两个模型的手部几何差异、初态不可达、长时 HOLD、时间世代与历史泄露、标定或执行模型误差被错判为映射误差。Train 候选检测覆盖稳定、时延、切换和连续性，并保留负控与失败。只在 Train 固结算法与参数，Validation 可用于冻结后的审查。资源估算以已有诊断实际为参考，正式资源限额及触发条件须在各后续阶段锁定，失败与重试计入限额。[阶段一审计与风险](../reports/V2_1_home_anchored_design_audit.md)

## 影响范围和阶段门

| 提请修订对象 | 拟议变化及当前约束 |
| --- | --- |
| `MASTER_DESIGN.md`、`PROJECT_STATE.md`、`ROADMAP.md` | 增平行 v2.1 设计分支。批准之后修订目标定义、依赖及状态记录，并保留原冻结索引。 |
| `CONTRACTS.md`、`V1-A_offline_rmb.md`、`V2_rmb_contract.md` | 加入独立 START 和名义／发布／应用／反馈 schema，改为真实世代重置。历史候选窗与两个覆盖门原样保留。 |
| `V2_residual_task_contract.md`、`EVALUATION_PROTOCOL.md` | 经批准后替换未来候选的主几何来源，复验因果遮罩、Reward、可达域和公平对照。 |
| P02、DCR-008、已冻结设计和历史回执 | 引用原路径与字节身份，继续支持旧版复算及受限资格。 |

`DESIGN_PROPOSAL → DCR_PENDING_HUMAN_REVIEW → DESIGN_APPROVED → IMPLEMENTATION → PILOT_VALIDATION → PILOT_APPROVED → FULL_RMB_BUILD_AUTHORIZED → FULL_RMB_BUILD → INDEPENDENT_AUDIT → READY_FOR_REVIEW`。

阶段一提交提案、静态检查、合成正负控、SHA 收据和独立评审，此处暂停。阶段二需项目用户明确批准 DCR，按风险受控地实施并完成六个不同 Train 母源 Pilot，然后等待项目用户再批准。阶段三还需要合格坐标标定、模型一致或已批准桥接、Home→START 安全、独立评分与 Reward 以及 85%／70% 口径冻结、独立 Reviewer 同意，并取得新的明确全量 Campaign 授权。RMB 的独立审计和人工验收先于任何未来 SuperMimic 正式训练。阶段通过和人工授权是两类不同证据。不存在自动跨越暂停点的通道。[独立审核](../reports/V2_1_home_anchored_independent_review.md) · [提案机读状态](../evidence/V2_1_START_PROPOSAL/design_candidate.json)
