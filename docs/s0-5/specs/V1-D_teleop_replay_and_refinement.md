# V1-D 现有 MuJoCo Teleop 闭环回放与修订实施规格 v2.0.1

状态 `DRAFT_READY_FOR_REVIEW`。本文针对 V1-D1/D2/D3 的真实软件链路仿真。执行必须等待设计审核和单独 Campaign 授权。本文所称实际闭环来自现有 MuJoCo 软件执行及当次反馈。真实设备仍受独立门禁。[V1-D 总规格](V1-D_evaluation_deployment.md) · [运行时契约](V2_runtime_contract.md)

## 1. 目标、前置与输出

D1 需要 G-D0、当前功能与安全证据、离线 Policy SHA、预留预算。D1 出具实际 Teleop 软件集成及 T0/T1/T2 配对证据，性能不达门可在功能与安全合格时进入 D2 定位。D2 输出分类收据、退回或修订决定、完整旧新 checkpoint 历史和复跑证据。D3 用最终固定 Actor、映射、Servo、数据和评分 SHA 重跑离线与指定录包，按真实结果裁定发布等级。为诊断可另外保存无正式 D 阶段推进权的导出包。现有用户工作区与未提交修改原样保留。

## 2. 实际工程与源码锚点

指定仓库分支为 [现有 Teleop](https://github.com/FEGkkd/whole-body-teleop/tree/codex/whole-body-sim-phase1)，本设计审阅时分支 Git 对象为 `6d0b0b5b2601618f1b0052d93ef2403eea9b064a`。执行时重核并锁定实际来源、基线和集成 diff。使用隔离 Ubuntu worktree、独立 build/install/log 和 Run 根、独立 ROS Domain/namespace、明确防串网及无硬件路径。签名 manifest 记录实际 worktree/dirty 状态、编译器和可执行产物、SDK 与 ABI、launcher 的 base/overlay 参数、asset 及模型 include、坐标、关节清单、Servo 配置、仿真/profile 时钟、进程 PID 和命令图。模块节点以实际组件与运行轨迹取证，预期入口含 `whole_body_teleop` 和 `whole_body_mujoco_driver` 的 launch/驱动、Marvin 运动学和 Wuji retarget。

本地历史源码可见 Teleop launch 的 `base_config` 与 `config` 合成、MuJoCo driver 在 `/tianji/joint_command` 与左右手命令通道订阅、发布左右臂关节与仿真手状态。topic 拼写和类型以执行时冻结图和 MCAP schema 为准。无授权运行、真机驱动进程、硬件传输通道、控制发布多 owner，或仅替换成简化 adapter 的结果，均无 G-D1 资格。在收到可能误接的输入后停止仿真发布并保留图和日志。

## 3. 指定录包与数据身份

| 必选回归记录 | 用途 |
|---|---|
| `/home/ubuntu/teleop_ws/data/s0_sim_20261119_172809/s0_sim_20261119_172809_0.mcap` | `TELEOP_DEPLOYMENT_REGRESSION` |
| `/home/ubuntu/teleop_ws/data/s0_sim_20261117_181022/s0_sim_20261117_181022_0.mcap` | `TELEOP_DEPLOYMENT_REGRESSION` |

设计阶段只读取 Ubuntu 文件和外部 metadata。两文件可读。172809 首读为 132,641,216 字节、69,260,778,050 ns、103,707 条消息，SHA-256 为 `ffd5353918e38e6527b534aa3263baa20d7cb3f8a868d91f33b8644cbd437817`。181022 为 229,433,886 字节、119,700,535,066 ns、182,620 条消息，SHA-256 为 `cbac62bb6d58ce2fa159d547408cf3534a29c32ca8b0bba1cddc8c4bc99ba1f5`。metadata 列出左右 MANUS、`/vive/tracker/{left,right}_wrist`、工作流和控制 phase、策略日志及机器人命令与反馈候选。语义、录制内容和登记时间仍须在正式 preflight 按实际消息核验。若 wrist 输入已注册，须确认其坐标、版本和适配资格，禁止重复注册或把派生列冒充原输入。该首读未解码消息或启动软件链路。

只读打开每包并确认实际字节与可解析性。录取日志含文件/分片清单、SHA-256、字节数、开始/结束/时长、message count、schema/版本与 QoS、录制与 header 基准、原始记录 ID、消息类型和用途。分开标注 MANUS/VIVE 原输入、模式/START/重定位/停止/生命周期、有资格的初态依据、原机器人关节命令、原反馈及诊断。检查运动方向、左右身份、背景模式及记录期脚本。以实际可验证的初态或预注册公共初态启动，录包不能佐证原初态时记录语义，公共初态的受限运行列不能冒称原始精确重现。不能得出必要初态或安全事件方案、或者任一必要人端通道无法重建时，将对应包标 `BLOCKED` 或 `INCONCLUSIVE`，披露可验证部分和缺项。单包通过不能补另一包。

验证各来源、别名、转换链、近重复段及训练、验证、选型、回放使用记录。已有历史实验曾登记 181022 回放难段/专项恢复和 172809 派生产物，实际与 Teleop HMB/RMB 是否重叠依逐源查证。任何明确或疑似重叠都进入 lineage 台账并限制独立性主张。禁止把指定包或其衍生切片、名义轨迹、原反馈或筛选案例新增到训练池。对已封存历史 HMB/RMB 中的来源关联，保存原物件与已有用途，单独登记暴露、核对与后续采样隔离，无法判定谱系时阻塞新增训练准入。两包及冷启动复跑属于工程回归，独立 Final Test 仍另办。[历史用途](../HISTORICAL_DATA_USAGE.md)

## 4. 因果回放、运行图与公平对照

`Recorded MANUS/Tracker + 可核验事件 → 实际 Teleop Mapping → Marvin IK + Wuji SDK → Actor/零残差旁路 → Safety/Servo → 现有 MuJoCo backend → 本次反馈`。隔离回放器只发布 allowlist 中的人端 raw 与必要事件。原录机器人命令、反馈及 cached RMB 只可供审计/分析，不挂载给命令和状态接收线程。原模式/发生时点证据明确的事件按源顺序发布。消息同戳用不可变原记录索引破同，按重复原采样、迟到、失联和关闭规则记录；回放器发布时钟或 backend 仿真时钟只能有一个权威，记录 origin 与换算。保持原相对间隔，host monotonic 用于运行耗时；发布不早于原资格时刻，禁止未来插值、事后最佳时移、多次重配准或中途重新 reset。故障后进入原状态机，按全部剩余计划拍继续登记。

| 组 | 控制 | 允许的比较 |
|---|---|---|
| T0 | 用户原始 Teleop action/mapping/控制器，关闭历史 Policy 及所有学习残差，启动日志与故障负控证明。 | T0/T1 同评分标尺，归因限映射及接口多因素诊断。 |
| T1 | S0-5 兼容 mapped intent、现有在线名义、共同安全/Servo/时钟，残差为零。 | T1/T2 的唯一主参照。 |
| T2 | 和 T1 同一图、输入、状态机、初态、时钟、模型、Servo、评分及故障计划，仅启用现有命令钩子处的 54D Actor。 | T2 − T1 给出冻结条件下残差增益。 |

配对前冻结计划 tick、原几何完整域、共同可改善域和功能/停止/难段掩码。录包域用冻结的与 Policy 无关的 nominal sidecar、独立合法残差见证和模型规则生成。可用另一个 T1 回放或独立名义计算锁证据后再读 T2 输出，且不得读取待评组的 closed-loop 执行状态。评分员从冻结源与映射配置算共同目标，三组共用 mapped 列。T0 可附原 SDK FK 兼容列，T1/T2 的原 SDK FK、离线诊断和本次在线名义各有独立列。若 T0 的后端约定与同构比较无法调齐，登记差异和局限，禁止填充 T1/T2 的 Policy 增益结论。

## 5. 测量和实际接受

每拍绑定 source/index、计划时刻、实际设备进程单调时标、外生标签、deadline、世代、命令 owner、`q_nominal_online`/`q_requested`/`q_issued`/`q_applied`、feedback ID 与状态，记录两臂两手各组。保留每指世界误差、max-ten P50/P95/P99、3/5/10/15 mm 全计划同拍率、局部手型、Pinch vector/distance、Cross-hand、Relative wrist SE(3)、动态与停止/稳定窗口、关节幅值与饱和、J6/J7 与碰撞、安全事件、完成/回退/控制失败、推理和周期/deadline、缓存/在线名义及观测分布。观测差异报告有名 feature 的 mask、age、分位数、近邻覆盖、归一化前后差异，按阶段和左右手分层。后端新反馈必须以时间前缀与命令边界证明，历史反馈按只读审计列。

D1 和 D3 各保留两包各方法至少两次冷启动配对。每次独立建进程、唯一 owner、一次声明的初始 reset，以及固定 replay plan。运行顺序在可用预算内做冻结的成对交错，避免运行顺序和负载被方法固定混淆。若预检表明覆盖/预算无法满足，记 `INCONCLUSIVE` 或 `BLOCKED`。稳态统计按每袋、每次冷启动及重复区间分列，按冻结规则汇总；重复或窗口不可当独立母源。主门共用 [完整评估协议](../EVALUATION_PROTOCOL.md)。T2 相比 T1 的共同可改善域复合动态 P95 改善 ≥10%，全有效动态域增加 ≤5%（近零见 E4），功能、静止速度及控制失败非退化，硬违规和迟到发令均为 0。合法残差旁路与同 T1 及故障回退一致。任何一个必选包缺证、边界偏移或重复不可接受时不得推荐。

## 6. D2 诊断矩阵与改动顺序

| 因素 | 隔离比较与处置 |
|---|---|
| START/映射/坐标 | 同人端事件前缀重复锁存，核对源 ID、T_WU/T_UTCP、手性、单位和原映射目标。只修确认的接口身份或时序，冻结评价目标和映射不得按成绩更动。 |
| 缓存/在线 IK 与 SDK 状态 | 同输入同启动、分侧 seed、会话初始化/恢复、名义合法性与历时。对比离线 CACHED 与实际 online 分布，记录训练覆盖，无有效输出时留原失败。 |
| Observation/归一化/采样和截止 | 同一实际 trace 对 Builder 错序、mask、history、age、周期、延迟回退与归一化器逐字段重算。固定时基与发令 owner。 |
| Servo/安全和到达 | 比较 requested/issued/applied/feedback，追溯投影压缩、夹紧、J6/J7/穿透、丢失、目标滤波、推理超时及回退；共同物理条件和动作权在 T1/T2 同步修复。 |
| WM/Policy/覆盖 | 在先排除上述接口错误后，运行安全探针、动作输入一致检查、真实 vs imagination、观测覆盖和剩余误差对照。确认能力缺口可在训练桶内修补。参考运动覆盖不足时按 E0 申请条件支线。[EgoPro 资格](V1-E_conditional_egopro_augmentation.md) |

每轮冻结 `refinement_id`、原 checkpoint 与证据 SHA、失败类型、假设、仅一类可归因的变动、实际 transition 和资源收据、新 SHA、独立复算及回退条件。先修接口再复验；后端、Builder、Servo、映射或安全身份变动使受影响的上游收据失效，按依赖再过门。任何训练用数据只可来自既有获准 Train 源。每个实际修订候选提交完整冻结离线集与两包正式回归，未胜出或产生新退化则恢复原合格 checkpoint。只为早期诊断用的短探针单列。最多两个有界修订轮次及既有验证重试上限，失败和 WM 重资格同预算。重复安全/因果故障或预算耗尽停止推进并封存。

## 7. D3 封存、测试与预算

最终固定源码、二进制与配置、Actor/归一化器、名称、评分、数据及谱系，先复算 G-D0 再在相同软件链路实跑，更新合法收据。复核软件图完全标识、时间完整、零残差及冷启动、命令/反馈来源、禁训练污染和复合分母。单元覆盖单位、schema 黑白名单、同戳裁决、非法 reset 和 role 映射；集成负控禁以独立 runner 充数，故意给历史机器人命令/反馈、RMB 参考、真机驱动、未来消息、额外 owner 和二次重置时拒绝。对照正控是完整该软件图的 T1/T2 配对。保留运行命令/退出、真实环境控制拍计数、容器和线程/CPU/GPU、进程监控与实际回放时钟。诊断低等级结果保留其证据和退出原因。

D1 预算 150,000 实际评估 transitions，D2/E 共 100,000，D3 200,000；另有 D0 200,000。阶段预算含全部真实 MuJoCo 实际控制拍，跨臂不重复计。按解码后的真实时长和 50 Hz 候选时基预估冷启动配对的成本，附初态、故障窗口、复跑和最终回归余量；资源不足禁止用降采样冒充原速或减袋。不挪用 D3 的预留。D2 补充训练最多 1M transitions，独立于 E2 的 1M，均从 C3 总 10M 留配，已消耗历史不归零。[参数登记](V2_parameter_registry_evaluation.md)

各 Run 归档 `v1/<campaign_id>/V1-D1|V1-D2|V1-D3/<run_id>/`，含 `gate_receipt.json`、`graph_manifest.json`、`mcap_inventory.json`、`split_overlap.json`、`replay_plan.json`、`trace.jsonl`、`paired_scores.json`、`timing.json`、`failures.jsonl` 和 D2/D3 的 `refinement_receipt.json`/`release_manifest.json`。独立 Reviewer 按锁定身份重放样本、重算包含失败的主门、审查作业授权和回退证据后才可提交技术候选。若最终只满足完整现有链路功能，可提交 `TELEOP_SIM_VALIDATED`；D3 联合性能满足才到 `TELEOP_SIM_RECOMMENDED`。
