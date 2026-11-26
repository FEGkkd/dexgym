# S0-5 v2.0.1 EgoPro 补充训练与双域回归

状态 DRAFT_READY_FOR_REVIEW。V1-E2/V1-E3 属于可选技术支线。所有数字为 PRE_REGISTERED_CANDIDATE，待设计批准。Main Agent 核对授权与预算，World Model & Training Agent 负责训练，Data Agent 和 Deployment Agent 分别锁定来源和回归。Independent Reviewer 复验触发、实际物理、安全、比较及分母。 [E0/E1 数据资格](V1-E_conditional_egopro_augmentation.md) · [部署闭环](V1-D_teleop_replay_and_refinement.md)

## 1. 前置、输入与候选身份

进入 V1-E2 的前置为有效的 G-D0、G-D1、D2 接口优先诊断、G-E0 和 G-E1，以及明确包含 E 支线的有效 Campaign 授权。输入为冻结的 Teleop-only 最佳合格全 checkpoint（Policy-T）、独立封存的 EgoPro Reference Motion Bank、两类数据的角色登记、已验证接口与安全 SHA、留出计划、原始部署基线及计量预算。Policy-T 选自冻结 Teleop Validation 的实际 MuJoCo rollout，另存 D1 录包回归的实际清单和失败。

在 Policy-T 的完整 checkpoint 上派生独立 Run。Actor 初始权重、共享 Builder、joint manifest、命令顺序、残差尺度、冻结归一化器、实际控制输入、独立人体意图和 (q_{nominal}) 契约保持同一选择。对不同来源，使用其经资格化的几何/时序掩码，保留来源在 Replay 元数据中。Actor 输入白名单继续生效，来源标签、split、episode 和轨迹号均留在数据/审计层。CPU、Warp、Servo、接触与碰撞模型须按本版本资产核对。冻结归一化器支持在两域实测；坏值或过度超界阻塞该候选。归一化变更需受影响阶段检查、公平性和重新导出，不作为默认 E2 调参手段。

## 2. 采样与课程

在每个合法新窗口的采样事件先以确定性独立 RNG 选择来源，再在该来源内按母源、动作分层和片段分配。候选 (p(T)=0.80,p(E)=0.20)，允许分别为 0.70–0.90、0.10–0.30，严格互补。资格阶段独立的动作实验与 WM-only 探索分开记账；学习阶段至少按每个 50k 实际有效训练 transition 的完整区块报告接受窗口和实际有效 transition 的比例、计划/有效分母、各来源及母源计数、拒绝原因与重采样。最后不足 50k 的区块合并至上一完整区块核验。配比以**实际生效的参考采样**为验收口径，不能把同一帧重采或无效动作计作覆盖。用分层采样纠正已登记的短轨迹偏差；超出批准配比或缺少必要标签时停下并保留原运行账。库与合批不跨来源、时间重启、无效边界或参考世代连窗。

Teleop 部分保留静态、原速动态和困难组，静态总比率不低于 15% 且 D0 域索引不变。EgoPro 部分只抽取 G-E0 已证缺口，保留其掩码、各功能点的覆盖和无法验证的场景。训练侧可使用以参考为驱动的 MuJoCo 探索，指定 MCAP 的原始轨迹及派生目标均不进入训练、Replay、强化采样、统计拟合或 WM 资格拟合。允许 D2 汇总描述缺口并在**已授权 Train 来源**中寻找代表动作，登记因故查阅的回归次数。

## 3. E2-WM 混合域资格

### 收集、拟合及资格分离

E2 开始先锁 Actor 和 Critic 的更新，执行 E1 已准入片段上的安全物理收集和 WM 拟合。以 EgoPro Train 根预登记拟合、调试和独立资格三个互斥集合。保留原 Teleop WM 留出，与新 EgoPro 隔离留出分别复核。EgoPro Validation 和 Teleop Validation 供 Policy 评估，仍和 WM 拟合/调试分开。环境编号不推导母源隔离。只有重跑时完全相同资产、Servo、初态、动作及状态时钟的既有**实际** MuJoCo traces 可作固定物理真值，记实际生成 Run ID 和内容 SHA；动作依赖新 Policy 的闭环对照另执行。

### 量化门

按 C0 原阈值，在 Teleop、EgoPro 两域独立评估实际 54 轴正负激励、每轴相位和响应证据、拟合一/四/八步及实际 horizon、固定目标保持、逐轴中心差分、同指和臂手耦合、饱和和接触边界、时钟和 Servo 慢状态、实物理与 imagination 的差异及反利用探针。逐域保持 C0 量级的至少 1,024 合法隔离锚点、4,096 预登记配对动作案例、每轴 64 合法有限差分初态且至少 32 证实有效方向。多动作试次按同起点簇及来源处理。每域单列缺测、合法重复、实际控制失败、碰撞和各误差指标，合并平均不能替代任何单域门。 [详细阈值与负控](V1-C0_world_model.md)

可在样本计划中预先标明开环/敏感性等试验能共享的同初态、同时间和同执行控制序列，完整记录每个统计分母。这样可复算多种指标；物理计费仍以实际运行次数为准。只有预算能保留足量独立样本、E3 完整回归以及 D3 保留金时才启动。预先估算不足时关闭 E 支线，使用 Policy-T。不能删减某一域或用 imagination 代替 MuJoCo。

### 检查点与门的绑定

独立资格凭据绑定训练和资格 registry/manifest、来源隔离与实际比例、缺口 UID、样本计划和原始实际物理 traces、初始与合格 WM SHA、checkpoint lineage、horizon、codec、Servo、安全、控制时钟、Builder、输入指纹及 Reviewer 证据。只有有效的 `E2_WM_QUALIFIED` 可以发第一条混合域 Actor/Critic 更新。资格后每消耗 250k 实际训练 transition、每次数据/模型/档位更新或异常后执行域分层 watchdog。失配立即锁 Actor/Critic；可在同预算重获资格，或退回 Policy-T。 [SuperMimic 契约](V2_supermimic_contract.md)

## 4. E2 补充学习与对照

Policy-E 使用相同 Actor/R/O/A、安全、Servo 和真实 transition 计数，在当前混合域 WM 合格之后继续训练。Replay 保存输入来源、真实目标和执行、终止和数据缺失；引擎延续真实采样、safety/Servo、WM 更新、短程 imagination、真实评估的顺序。封存每次成果和失败，checkpoint 选择依据固定的实际 MuJoCo 分层比较。Reward 和 imagined 结果作为诊断。

候选默认 Policy-T 和 Policy-E。若实际预算允许，在首个 E2 运行前预登记等额 `Policy-T+`，它从同 Policy-T checkpoint、同配置和同超参数只用 Teleop Train 继续。比较须记录实际训练、WM-only 与 Actor-exposed transitions、相同资格证据、更新、scoring 和随机种子；只在两臂的实际补充 physical transitions 相等且 Actor 更新预算匹配时称为等额。若某臂提前失败或曝光量不同，保留其结果并标明预算不匹配。缺等额对照时只报告“在计划内实际观察到扩充候选较 Policy-T 改善”，无法由该比较单独估计数据源贡献。禁止凭单种子宣称最终科研显著性。

## 5. E3 物理回归协议

E3 前锁定 Policy-T、Policy-E（如有 Policy-T+）、共享 Teleop Validation、EgoPro Validation、暂不移入训练的匹配缺口组、D0 两个绝对域、D1 两个指定 MCAP、共同目标/时间轴/安全、失败复合 P95、配对分母和缺失/近零规则。EgoPro Validation 独立报告，复用 E0 已登记的工作空间、SO(3)、手型、Pinch、跨手和停反向分箱，按根等权并分列帧汇总。训练前登记 E0 缺口组的预期改善指标和配对方式，能力域和困难段归属只来自计划/独立几何。

候选增益门在已证缺口的独立验证中满足预先择一的主指标，十指复合误差 P95 或恢复 P95 相对 Policy-T 改善至少 10%，并报告另一指标的非退化及完整缺口覆盖。恢复 P95 按扰动结束后到首次持续五拍达标的时间计，未恢复值进入失败复合分位数；20 ms 以下看绝对候选差并记收益不可判，位置误差近零和非有限口径沿 D0/D1。覆盖增加由 E0/E1 的冻结分箱及独立验证重算，需结合合格实际轨迹。E3 新候选对 Policy-T 的 Teleop 保留须满足下表全部要求。统一使用预先冻结的共同支持与失败分母，样本不足记 INCONCLUSIVE。

| 对照域 | 必须通过 |
|---|---|
| Teleop 静态与原速动态 | 保持 D0 绝对、B0 相对门及功能/安全条件。Policy-E 相对 Policy-T 在两类固定 max-ten P95 均≤ `T + max(5%×T, 0.25 mm)`，完整域相对 T 同样适用。任何复合门非有限仍不能通过。 |
| 功能与失败 | 静止 RMS、Pinch、局部手型、跨手、相对腕、失败、饱和、安全介入和控制周期沿现行非退化门，额外在 Policy-E/Policy-T 间做同条件检查。硬安全违规为零。 |
| 两包实际 MuJoCo Teleop 回归 | Policy-E 在每包和配对汇总均保持原 T2/T1 部署门。对同图的 Policy-T 也使用 `T + max(5%×T, 0.25 mm)` 位置误差非退化及同协议功能、失败、硬安全和时序门。冻结合法零残差旁路和因果比较。 |
| EgoPro 扩充域 | 实际覆盖及上节预登记主指标通过，记录命令失效、无对象可行性与 UNKNOWN。单独发布完整 E Validation。 |

E3 对照采用 V1-D 的独立真实 MuJoCo 软件反馈、指定原始输入和因果时间轴。仅在未改变时钟、事件、模型资产、图、参数、初态、输入与 bootstrap 身份，且完全对应冻结分母的情况下复用原实际对照 traces，Reviewer 重算。E 候选应实际重新跑。每次回归有明确用途和预算，优先保留完整 Teleop 门。E3_FAIL/INCONCLUSIVE 时撤销 Policy-E 资格，锁定 Policy-T；如果 Policy-T 未取得所需部署资格，先按 D2/D3 正常门与预算处理，再按实际等级交付。 E3 通过也须由 D3 完成最终发布回归。

## 6. 预算、失败与交付

训练上限从 C3 原账户留出 1M，含全部 E2 暖场、WM 定标、训练、恢复和失败。如果同时跑等额 T+，E 和 T+ 各最多 500k，合计不超过 1M。默认无 T+ 时 E 账户最高 1M。最多两次混合域 WM 资格尝试和两项学习变体，两个变体上限包含可选 T+。D2/E 合用 100k 评估池。所有额外 MuJoCo 分步计入实际执行数，训练与评估两账互斥。同一持久迹线只能在相同条件下复算；不得以宣称缓存模拟置换独立必要的实际试验。最终 D3 200k 单列。预算不足或达到安全与审核停止条件时封存，保留原样本、checkpoint、终止与超额预估。 [预算治理](../MULTI_AGENT_PROTOCOL.md)

在 `v1/<campaign>/V1-E2/<run>/` 保存 `gate_receipt.json`、`wm_qualification.json`、`sampling_audit.json`、`budget_delta.json`、Policy-T/Policy-E/T+身份、环境指纹、原始及想象迹线、负控、源及资产 SHA、全部日志和独立审阅； E3 同级路径保存 `gate_receipt.json`、`paired_scores.json`、`teleop_retention.json`、`recorded_replay_receipts.json`、`source_coverage.json`、`selection.json` 和 `rollback_receipt.json`（需要时）。政策选择凭据必须标明是否为 Policy-T、E3 证据及归因范围。

## 7. 测试

- 单元正控检查互补采样、有效帧记账、已有实际轨迹哈希一致、按初态聚类、双域 WM 标签、近零值处理、候选切换和 rollback。合成负控拒绝 MCAP 谱系混入、源间交叉窗、弱化掩码、伪造传感器字段、混合域未获资格直接更新、复用失配 Servo、越过预留、跨域平均掩盖退化、缺失部署样本和候选等级冒用。
- 获授权后的实际验证包括合法与拒绝两种执行路径、独立留出 WM 全套诊断、两域实际物理对照、两包完整部署回归、人工可复验停止证据和 Policy-T 回退。原始硬安全事件与失败全记录。
