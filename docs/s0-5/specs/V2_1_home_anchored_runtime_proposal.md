# Home 锚定的现有 MuJoCo Teleop 运行时提案

状态为 `PROPOSAL_ONLY_PENDING_HUMAN_REVIEW`。本文件定义待审的仿真名义链路和 Pilot 设计。已批准执行范围、活动技术门、M0/M1 评分目标、数据 split、DCR-008 状态及 Final Test 状态沿用现有记录。Pilot、全量重建、RMB 准入、Policy 开发和真实硬件执行各需其原有授权。 [几何提案](V2_1_home_anchored_geometry_proposal.md) · [数据与准入提案](V2_1_home_anchored_data_gate_proposal.md) · [运行时契约](V2_runtime_contract.md) · [DCR-008 独立评审](../campaign/DCR-008_independent_review.md)

## 1. 源码与已观测运行时

| 层 | 实际依据及设计义务 |
| --- | --- |
| 启动与所有权 | 真正观察到的系统是用户 `teleop_ws` 的 `4sensor` 源快照 `30cf13bdc496c43b3016462cfa0a206334d1ff7f`、已安装 MuJoCo driver 和现有 bringup。源码快照、用户工作区 diff、overlay、提供运行服务的二进制和实际参数逐项绑定。DCR-008 的 `6d0b0b5b2601618f1b0052d93ef2403eea9b064a` core ABI 是离线候选依据。它以代理反馈驱动，无法证明实际 ROS 发布、运行世代和 Servo 的行为。 |
| 目标 seam | Shared 03 overlay 在 `TeleopCore::map_targets` 的腕姿映射处替换 native START-relative 目标。它读取一套旋转与六个固定侧别桥接，在会话 `SessionReference` 上用双掌中点求共享平移，保留 `process_targets`、原生 Marvin 解算及控制器发布。隔离版本限制 localhost，完整回放版本使用域 206–213，并拒绝 t-heading 与该 seam 混用。正式提案需要可验证配置与世代收据，环境变量诊断 hook 仅为历史实现依据。 |
| Native arm | `TeleopCore::process_arm` 先构造工作空间目标，再形成受限步目标及实际 IK 请求。`process_targets` 和 `finalize_command` 使用本次快照的反馈、上一条已接受命令与 START 锁存的 NSP 状态。仿真 overlay 配置为 250 Hz ManiMux Marvin 位置命令、平移限速 0.2592 m/s、旋转限速 1.085734 rad/s、nominal dt 4 ms 和最大 dt 16 ms，重复尝试尺度为 `[1, 0.5, 0.25, 0.1]`。上游 YAML 保有 0.5 mm 工作空间 reserve 和 5 mm 平滑区。两臂可独立 HOLD，共享故障仍使双臂停止。按实际启动查询冻结全部生效值，包括 native 修改差分，区分 arm control shaping 与物理执行。 |
| Native hand | `WholeBodyTeleopNode` 把左右 `RetargetNode` 的发布交给 `commit_hand_command`。每侧使用独立 Wuji session、进入 Wuji 的原生 MANUS SDK-common XYZ 点和有名手指/关节顺序。SDK-common 输入与 HMB 解剖掌系几何各存身份。源码检查手侧、设备、正有限 ROS 源时间和 freshness，展平 21 个点为 63 个 float，默认左/右 Z 旋转 ±15°，执行 SDK、输出平滑，再复查源龄并尝试发布。默认输入上限 100 ms。拓扑变化、SDK 输出拒绝、命令生成和 native publish 均须独立记录。发布 gate 的世代检查不能自动证明 SDK 内部/平滑状态跨世代已清空。 |
| Plant | 回放证据使用 500 Hz MuJoCo 更新、2 ms 物理步及 120 Hz 反馈发布候选。源码 arm driver 对消息检查模式、session、递增序列、寿命、14 个有名轴与控制范围；手部检查手侧、顺序和 100 ms 的候选命令寿命，并可施加 actuator clamp 及 Enable 后 0.5 s hold + 1.0 s ramp。实际 target buffer、物理子步应用、simulated encoder 状态是三种不同观测。用实际运行参数和有名轴复核，保留与 50 Hz HMB 研究时钟的异步关系。 |
| 发布与回退 | `publish_command` 在同一锁内检查状态、session、generation、sequence、两臂追踪源龄和反馈，提交后才更新上条命令。带独立单臂 HOLD 的 14 轴包仍可能产生新序列。`retain_previous_command` 也可重发旧 q。原诊断 `arm_command_accepted` 和 ROS 观察者收到的包是不同事件，driver 另做验收。每次报告 HOLD、保护回退和拒绝，不给主动跟踪覆盖。 |

固定 03 overlay 执行证据来自 `TELEOP-SHARED-RIGID-START-HANDOFF-03-01` 和完整 `TELEOP-SHARED-RIGID-START-HANDOFF-FULL-01`。完整回放 32 根均通过先前测得的 START handoff，运行时间 1.733–3.305 s，最长 Home 关节偏差 0.004969 rad。旧 runner 选择每根的首个因果四输入有效拍，本提案增加独立稳定窗口条件。32/32 仅支持旧条件下的仿真干运行证据，且不含原操作者 START 复现或新映射验收。 [实际源码审计](../../../scripts/s0_5_campaign_teleop/ACTUAL_TELEOP_SOURCE_AUDIT.md) · [overlay 代码](../../../scripts/s0_5_campaign_teleop/prepare_shared_rigid_overlay.py) · [旧 handoff](../../../scripts/s0_5_campaign_teleop/replay_start_handoff.py) · [完整运行结果](../reports/TELEOP_full_start_handoff_layer_audit.md)

### 1.1 模型、标定与时钟身份

证据中实际 live 模型 `be69f387f5bc25c8011a992a88ac2c2449c2c0419e09318b740fcdcd1402871d` 和 P02 签名模型 `939b3651e2426e596eed482e70a68c2382720df1fd55b124aa5f0893b640e92d` 均为 54 轴。105 个有名配置上腕 TCP 位姿一致，世界地标坐标最大差 6.271 mm，局部手地标最大差 6.354 mm，两个 MJCF 的碰撞属性和资格须分开。其各自 XML 和包括项、mesh、关节与 actuator transmission、符号、零点、边界、world/base/tool、Home key frame、palm 安装、地标定义和实际有效 Servo 都要独立锁定。模型不一致时只发布已经验证的同名量，指尖不得直接跨模型当作毫米级误差。 [Native 模型和 kinematics 核验](../reports/TELEOP_native_nominal_replay.md) · [P02 模型证据](../reports/P02_acceptance.md)

历史七块共享变换的 SHA-256 为 `33cbc93c4a2db925986f596b4b1730c7eaca026a9facfdf26aa4aaa8032c8f94`。该标定基于 2026-11-18 留存的腕安装、MANUS anatomy 及模拟模型桥接，录制当天佩戴外参和机器人世界物理配准证据仍未知。每侧 14 个 Home 扰动只校验了相应模型到 Marvin native TCP 桥接，不能承接外部物理精度。未来 Pilot 的 `R_cal` 从几何提案所述独立、适用且受审标定取得。若仅复用历史变换，其标签为 `HISTORICAL_CAL_DIAGNOSTIC` 并保持几何资格 `UNQUALIFIED`。同步发布原 `M0/M1`、共享目标、可见 runner/控制器时间和每侧安装标识。 [七块变换生成](../../../scripts/s0_5_campaign_teleop/prepare_shared_rigid_frames.py) · [HMB 软件标定历史](../reports/P01_C_acceptance.md)

原始录制回放的主键为 `(root, partition, tick, cutoff_ns)`，时钟使用相邻 `SampleObservation` 的 collector monotonic、writer instance、序列与物理记录索引。记录器到达是历史因果代理，原控制器到达时刻不可恢复。按签名 HMB `tick_ns == cutoff_ns` 及每流 `arrival_hi_ns ≤ cutoff_ns` 执行原始支持判定，保留原时钟资格、源龄、skew、失效和分母。原始 native ROS header 用于该流的时序校验。每次隔离回放最多使用一个明确记录的新 ROS epoch 平移，保留原相对时间、顺序和未修改 CDR 摘要。回放 observer 的 `receipt_mono`、控制器 cutoff、driver steady 和物理子步时间有独立单位及身份。需要端到端的应用时钟转换时先插入真实单调时钟同步探针，epoch 跳变、漂移超窗或因果关系不清保留 `CLOCK_UNQUALIFIED`。 [回放输入册](../../../scripts/s0_5_campaign_teleop/hmb_cohort.py) · [时基与原生服务](../../../scripts/s0_5_campaign_teleop/nominal_live_replay.py) · [同拍数据契约依据](../../../scripts/s0_5_campaign_teleop/compare_start_handoff_layers.py)

## 2. Home、稳定输入和 START 的提议时序

使用 `T_AB` 把 B 坐标映到 A，W 为实际仿真世界，H 为有身份的人体 Tracker 世界。`p_W,Home,s` 从已测 `q_home` 的同名解剖掌根求得，`p_H,START,s` 来自选定的两只人手同时刻、应用固定安装后的解剖掌根。每次 START 在同一世代只构造一次 `t = m_W,Home − R_cal m_H,START`，其中 `m` 为配对掌根中点。共享 `(R_cal, t)` 推进两侧的独立目标层，保留起始相对方向、间距和随时间的实际意图。左右手固定安装及 wrist/palm/native/base 转换遵循同一有名桥接表。Home 双手物理向下用于核对安全初态。START 姿态直接来自该人端拍，可掌心向上、相对或不对称。原始姿态始终保留，拒绝因求解困难改尺寸、加单侧偏置、自行估计 yaw、动态重锚或把 START 记成 Home。 [共享映射历史实现](../../../scripts/s0_5_campaign_teleop/prepare_shared_rigid_overlay.py) · [起步实测报告](../reports/TELEOP_start_handoff_03.md)

### 2.1 稳定 anchor 规则候选

候选参数为 `PRE_REGISTERED_CANDIDATE`。项目用户确认后登记 Pilot 计划，独立 Reviewer 先锁定规则、Train 母源、数据角色、起算事件和允许对照，在打开 Validation 或评分前验证定义。若规则失败，保留该拍和完整原因。调整只允许在另审 Train-only 协议中进行。

| 项目 | 主候选 A | 预登记灵敏度 B |
| --- | --- | --- |
| 事件与搜寻窗 | 如存在可辨认且经核验的显式 START，取其相同 session 的事件截止拍，要求当拍及此前连续稳定窗已满足条件。若不存在历史事件，在每根原计划开始后的 2.0 s 内选择最早已证的稳定窗末拍并标 `RECONSTRUCTED_START`。 | 使用同一事件身份及 2.0 s 搜寻窗。两个候选各有独立状态；对照结果不得用于逐根混选或替换主规则。 |
| 稳定时长 | 0.20 s，连续 11 个 50 Hz 签名拍含两端。 | 0.40 s，连续 21 拍含两端。 |
| 有效性与时间 | 全窗 4 路有限、命名拓扑不变、合法 `safety_permission` 和连续 segment；维持签名 HMB 8 ms 代理、60 ms 源龄、35 ms skew 的既有身份。额外要求每路在窗内至少 4 个不同源序列，输入 receipt 上界全部先于各自 cutoff。重复 hold 样本不能增加支持。 | 同主候选，至少 8 个不同源序列；保持同一 freshness 和边界语义。 |
| 左右腕稳定性 | 以非重复源时戳求每侧平移和 SO(3) 有限差分，全窗最大速度 ≤0.05 m/s、≤0.35 rad/s；各侧相对首帧最大位移 ≤8 mm、姿态变动 ≤4°。 | 最大全窗速度 ≤0.035 m/s、≤0.25 rad/s；各侧相对首帧最大位移 ≤6 mm、姿态变动 ≤3°。 |
| 手型稳定性 | 同一侧掌局部五指末端有限差分，每侧任一指速度 ≤0.08 m/s，任一指对首帧偏移 ≤10 mm。 | 任一指速度 ≤0.06 m/s，对首帧偏移 ≤8 mm。 |

纳入稳定窗的唯一新 anchor 拍为最后一个已观察有效拍。用实际各源最近时戳对照并加 `support` 掩码，禁止导数跨来源割裂、世代、分区或输入失效。对明确存在但不合资格的事件记录 `START_UNQUALIFIED`，不把该事件当作缺省事件回退。没有可核验历史 START 且窗内未取得重建依据时也记录 `START_UNQUALIFIED`。报告 `explicit_event_status`、首尾时戳、各路 unique 计数、速度/位移上界、拒绝原因、阶段时钟与完整未选分母。若实际在线配置记录了已合格的显式事件则记 `EXPLICIT_START`。合成 dry-run 时不得伪造历史事件，用独立 evidence 标明为运行操作。 [HMB 源龄条款和统计](../reports/P01_E_acceptance.md) · [历史 anchor 校验](../../../scripts/s0_5_campaign_teleop/prepare_start_handoff.py)

### 2.2 生命周期及 watchdog

以下阶段是拟加在现有 workflow 之上的审计状态。命令发布和 Enable/Stop 由现有 Teleop workflow、控制器及驱动承担，审核组件仅观测和提出拒绝。所有边界必须采集同 session/generation 的完整证据。

| 阶段 | 进入与退出条件 |
| --- | --- |
| `SIM_PREFLIGHT` | 冻结软件、模型、映射、launch 合并和有效参数。在 localhost 隔离冷启动，核实唯一 Teleop 命令 writer、唯一 MuJoCo feedback owner、driver 输出闭环以及 hardware/bridge 禁入。原始人端允许话题只包含两路 MANUS 和两路 Tracker。预检失败中止本次会话。 |
| `HOME_MEASURED` | 连接后经原服务 Home，等待新的四反馈组同 stamp、命名 54 轴、全手 `connected && online && valid && !stale`。候选为 Home 服务返回后 3 s 内，连续 0.20 s 的测量都满足 40 ms 反馈 freshness、相邻样本间隔 ≤40 ms、与已核验的模拟 palms-down Home 全轴最大偏差 ≤1°、每轴速度 ≤0.16 rad/s。用 live 模型从实测 q 复算两掌法向距世界向下各 ≤5°。失败记 `HOME_UNQUALIFIED` 后 STOP。 |
| `START_ARMED` | 按 2.1 选择源输入，借助 native Home/Start/Enable 的顺序和专项 guard 冻结本会话 `T_WH`、`P_home`/`P_start`、原始 MANUS 和实际腕样本、SDK 会话及全部身份。记录真实服务发起和响应，清除旧控制历史。需要预热时，重复同一锚点的已选输入并将其标为 preposition。这些合成刷新不增加稳定窗支持。 |
| `PREPOSITION` | 仅允许现有 Marvin/Wuji 追踪冻结的人端 START，使用 native workspace shaping、限速、机体约束和相同 driver，禁止瞬移和直写 `qpos`。当姿态求解、硬约束、软回退或 freshness 失败时不放松相应条件。将原始请求、限制、命令、应用、测量和超差按步保存。 |
| `START_REACHED` | 初始候选沿用历史诊断的实际测量尺度，Enable 后至少 1.5 s 并有持续 0.20 s 的实测 dwell。两腕对未限幅的固定 START 各误差 ≤10 mm、≤5°，每手对已发且仍有效的同世代名义目标各最大轴误差 ≤0.10 rad，全身速度 ≤0.16 rad/s。同 session ACTIVE/TASK 下双方必须有已通过本周期发布 guard 的合法跟踪，允许符合目标容差的静止保持；持续拒绝、非跟踪 HOLD、限位停滞或保护制动不积累 dwell。所有核验数据 required freshness ≤40 ms，累计用互异反馈 header 而非 polling 次数。10.0 s 到期无满足条件时记 `START_NOT_REACHED`、保留 intent 并 STOP。 |
| `HISTORICAL_INPUT_ACTIVE` | 历史仿真回放一次从已封存 anchor 对应的原时间起点进入，使用原四流 CDR 和一条显式 epoch 映射。anchor 前规划帧保留为未回放，正式计分不使用外部 START 导致的热启动过渡。只发布人端输入，不回放录制的关节命令、机器人反馈或名义缓存。记录每根的输入顺序和 `historical_input_seal`。在线无回放同理，应接受当下实际合法的人端数据并保留 START 世代。 |
| `HOLD`、`FAULT_STOP` 和再进入 | 当样本源龄或反馈超时、左右关系不一致、保护回退、安全证据不完整时先拒发不合格 proposal。若 native 安全权威明确允许保持已验证目标，以更严的现有 cutoff/候选 60 ms 上限持有，记录依据并逐周期检查。硬安全、模型、关节映射、时钟或 ownership 故障立即转由 native STOP 清空有待处理的 arm/hand 命令并撤回 Enable，不等待 hold 用完。恢复的 candidate 路线为显式新 Home/START、generation bump 并重置 Wuji/filter/NSP/history，再采新输入。Pilot 每根每个条件只允许一次 START，失效后结束该次运行，禁止中途再锚定来提高评分。 |

现有软件 `ACTIVE/TASK` 可能早于观察到的到位阈值，必须分别记录 native ACTIVE 与 `START_REACHED` 两道事件。离线输入在到位前仅可走严格冻结的 warmup 四帧，签名负控演示 TIMEOUT 仍发历史输入会被拒。若 driver 的手部 watchdog 在物理子步进入 HOLD，应报告终点与实际失败。真实安全权威仍需后续独立验证。 [严格回放状态机](../../../scripts/s0_5_campaign_teleop/run_start_handoff_cohort.py) · [完整运行收据](../campaign/evidence/teleop_full_start_handoff_layer_receipt.json)

## 3. Nominal、issued、applied、measured 追溯

在已发布组的实际 owner 内增设只读审计记录。Pilot 补丁预检要求同一 build 的日志关闭与开启模式在合成同样输入的离散安全决策、命令值和序列一致，另验证吞吐、observer backlog 与掉样。Arm 250 Hz 控制周期、手输入回调、MuJoCo 物理子步和 50 Hz HMB 截止拍不得以线性插值或最近帧拼成同一时间点。

| 层 | 时间和身份字段 |
| --- | --- |
| 独立 intent | `root/partition/tick/cutoff_ns`，source instance、原 MCAP/CDR/hash、各侧 calibration、输入年龄和 `T_WH` 版本。保留 `M0`, `M1`, `shared_hmb_raw`, `native_shared_raw`, `native_filtered`, `workspace_target`（XYZ；无可靠姿态时保留该 mask）、`native_rate_limited`, `native_ik_requested`。 |
| `q_nominal_proposed` | 每臂和每手实际 provider 当拍给出的有名容器、原始 SDK 值和 solver/finalizer 结果、prev accepted、拒绝原因与 input keys。命令契约记录在哪个步骤施加 shaping、joint/coupling/碰撞检查或较保守的门。`q_nominal_proposed` 尚未经真实 publish guard，保留每组 mask；缺组可审计，禁止补零当合法命令。Pilot residual 恒为零，仍跟踪 proposal 到下游实际权威的变化。 |
| `q_issued` | 审核唯一发布 owner 真正发布的 payload，绑定 `writer_id/session/generation/group/sequence/cycle_id`、header、publish mono、observer mono、有效寿命、side 和精确有名 q。接受独立 consumer 的精确诊断 join 与 recovery reason，拒绝使用晚到诊断为历史已 observer 行补 active。仅 `published` 证明发布，含单臂 HOLD 或双臂 retain 的包保存其原 q 但不给覆盖。 |
| `q_applied` | Pilot 新增诊断证据记录 driver callback 对同一 command key 的 accept/reject，目标缓冲内 clamp/ramp 后 q、首次/末次使用的物理子步与超时保持/失效。不存在明确 driver 观测时标 `APPLICATION_UNOBSERVED`，同时保存接近 command 的独立反馈而不赋予 ACK。 |
| `q_measured` | 来源为当前实时 MuJoCo 实际四组反馈。每组 header 与 receiver mono 记录，完整 54 轴要求四组同 header、同一 epoch、完成收件的最晚 mono ≤ cutoff；腕 FK 仅可在合法 14 轴有效时计分。保留时钟缺口、冲突重复、状态字段及真实测量 qd。 |

横向配对时先校验 `header ≤ cutoff_ros`、`receipt_mono ≤ cutoff_mono`、生命周期相同、包未超时且 freshness 合格。独立手流的序号不能与臂 seq 合并为新原子 seq。必要时读出每组最后一条可用且未过期的数据及其年龄；任何一组缺少立即标 `PARTIAL`。计算 `qd_nominal` 或 `qd_issued` 时仅在同世代、同分区、相邻两拍有相同合法覆盖条件且时标有效时用后向差分。未来消息删除或替换、状态切换后的迟到包、重复同 stamp、滚动/重用序列、未来 header 和 STOP 后回包不得改变前缀结果。前缀和全 run 证据比较需要清楚区分实际在线当拍可见记录和之后用 driver step ID 查出的审计关系。 [现有真实 cutoffs 评分器](../../../scripts/s0_5_campaign_teleop/compare_start_handoff_layers.py) · [部署执行契约](V2_runtime_contract.md)

每臂、每手分别记录 `TRACKING_ACTIVE`（对应正常主动跟踪）、`LIMITED_ACTIVE`、`HOLD`、`FALLBACK`、`FAILED`、`UNKNOWN`，另保留 SDK exact pose 合格辅助列。`TRACKING_ACTIVE` 需合法下发，且更新源确有持续跟踪或已达到未修改的 provider 目标。进展候选要求关节变化超过 1 µrad 且后一条命令对同一拍可审核跟踪目标的几何或有名关节误差确有下降。静止相符候选要求未裁切目标的腕误差 ≤1 µm、≤10 µrad，手指目标误差 ≤1 µrad。`LIMITED_ACTIVE` 还需明确 shaping 原因与实际指令沿可追溯路径推进。上述门槛为防止数值抖动计成覆盖的诊断候选。Pilot 单独报告真实重复命令与达到较粗 task 容差的帧。若 native 准备返回成功但没有发布、故障姿态仅制动或最新合法目标是重发保持，则按照其证据分类。不能解析同周期状态的候选记 `UNKNOWN`。将 active 覆盖以原计划和 HMB 有效分母分别计算，冻结的原 85% 和 70% 门照原口径并列报告。新分子在人工批准前只做诊断。 [DCR-008 主动覆盖口径](../campaign/DCR-008_independent_review.md)

## 4. 79.11、76.74 和 18.66 mm 的隔离诊断

以下均是已归档的既有回放数值，单位为双腕较差侧的 world 位置误差 P95。表中的支持分别来自配对统计和四层共同域。

| 同拍链路 | 已观测 | 待解问题及 Pilot 辨别试验 |
| --- | --- | --- |
| `native_shared_raw → issued_fk` | 79.11 mm、44.14°。四层共同域 99,562 拍。`M1 → native_shared_raw` 同域为 693.28 mm、87.33°。 | 先分标定/目标定义，再按同一侧逐拍的 `raw → filter → workspace → step → attempt → issued` 签名记录。按初次运动、原速快慢、停止、反向、可达疑似、投影方向、左右约束、HOLD/失败层级分层；需要保留同时间点逐侧向量向下分解。映射差、控制限制和实际误差不能由一个相对改善百分比归因。 |
| `native_filtered → native_rate_limited` | 76.735 mm、44.184°。配对域 99,833 拍。 | 目前该段含工作空间边缘软投影、位置步进和角速度 shaping。记录原生 workspace 状态、投影距离及坐标、前条发令/实测 FK、实际 dt、平移/旋转步上限、seed/NSP、guard 及 backoff。若 workspace 无可靠 orientation，仅对其位置辨别，其余保持联合未知。使用相同实际请求、相同状态做只读复算，保持 guard 参数。按静态/慢速/工作空间内和边界分层。末两段 `step → request` 0.939 mm、0.225°，`request → issued` 1.311 mm、0.441°，反映各自可评分支持，不能解释丢失或拒绝的样本。 |
| `issued_fk → measured_fk` | 18.662 mm、4.029°。四层共同域 99,562 拍。配对域 99,839 拍的 P95 为 18.665 mm、4.029°。 | 在实际仿真补 driver 接纳、clamp/ramp、target buffer 与子步生效的 `applied` 测量，将 `issued → applied` 和 `applied → measured` 分离。仅使用源计划的原时间，独立把 ROS 发布队列、命令超时、控制频率、Servo 和 FK 误差分层。Pilot 在明确合法的小幅阶跃、正反向连续运动和静态保持上报告逐轴响应、速度以及 deadline/丢包；主评分禁止事后时移。 |

世界系位置差 `delta_{A→B}(s,t) = p_B(s,t) − p_A(s,t)` 按固定侧及相同中间层逐拍相加验证闭合。P50、P95、P99 对同一控制拍先选双腕较差侧再跨拍统计。SO(3) 主距离采用 `‖log(R_Aᵀ R_B)‖`，复合变换可顺序相乘，误差角或百分位不可简单相加。配对域、四层共同域、缺失状态和 Train 母源等权结果都要公开。完整 run 99,562 拍结果用于设定研究问题。Pilot 不承担消除已有大差距或证明某个映射符合真实人体的任务。 [逐拍结果与实际支持](../reports/TELEOP_full_start_handoff_layer_audit.md) · [独立分解数字](../campaign/evidence/teleop_full_start_handoff_layer_receipt.json)

## 5. 未来 Pilot 的固定范围、交付及 No-Go

DCR 获人工批准且授权范围覆盖有界 Pilot，才能在合格标定及实施核验后启动。Pilot 技术通过后仍需独立 Reviewer 与用户批准，再依据正式授权门申请全量。6 个互异 Train 母源预登记覆盖至少 1 根掌向向下、1 根向上、1 根相对、1 根混合。手掌尺度和运动分层使用 Train 只读几何、每根配对掌距 P50 的 Train 四分位，以及原始计划窗口内的腕速和方向变化。候选最少 1 根位于掌距下四分位、1 根位于上四分位、2 根含明显启停或反向。掌向以几何提案方向定义计算，速度和转向约束由独立 Reviewer 固定到选择脚本。六根可以交叉覆盖这些类别。无法满足的类别记录 `INCONCLUSIVE`，保留母源和所有原因。禁止看事后 mapping 优劣或控制误差来挑根。固定每根的首段与评估窗口、唯一初始 reset、时戳、条件向量和分母。候选最低证据为每根一次独立 cold start，在预先选定的两根分层样本再各做一次冷启动，最多 8 次实际 MuJoCo 仿真尝试，离线正负控单独记账。每根历史段候选最多 120 s、每次总最长 180 s、累计最多 72,000 个 50 Hz 计划控制拍，物理步和 observer rows 单独记账。超过限制时以 `BUDGET_STOP` 归档并等待审核。Pilot 的采样及导出只附诊断档案，保留原 HMB。

每次交付原人端及实际控制器/plant 的证据身份、Home 和 START 分层证明、动作 group manifest、时间轴、原始日志、状态 mask、容量与退出收据、按根覆盖差异、共同评分矢量及独立复核。定量功能建议为已选根六次主运行模型和来源有效、6/6 通过 Home 和完整严格 START，6/6 STOP 后零晚到指令，安全硬违规为 0；若出现历史缺失真实 START，则仍要按重建路线单列质量与申请范围。各根计划内的完整 issued/measured 与 14/20/20 组支持分别报告，候选共同链路支持 ≥80% 的独立 HMB 有效拍，用于解释 Pilot 分解是否有充足覆盖。保留原 85% 主研究门以及所有后续功能、联合动作测试和部署门，不能用 Pilot 80% 替代。一次不合格或重复冷启无法保持同等资格即报告 `INCONCLUSIVE` 或 `FAIL`；对校准和安全当场硬失败时不轮换到更容易母源。

| 对照 | 判定 |
| --- | --- |
| 因果与身份 | 校验逐流已知名称与 CDR SHA、截止签名、唯一实际 writer、freshness 和有效寿命。删除或替换未来须保持当拍及此前结果。修改来源、跨世代回送、checksum 错配、输入模式改动必须触发规定拒绝。 |
| 状态和控制 | 已知合法小幅动作可产生不同 q 并有真实仿真响应。断网/迟到/NaN/单臂 HOLD/双臂保留/回退/STOP 后包不能取得主动或完成，driver ACK 缺失不取得应用测量。逼近耦合、限位或碰撞时保持安全权威及已有路径检查。 |
| 模型和坐标 | 确认合格右手世界变换、所有 wrist/palm/tool 及有名轴；换 XML、swap 左右、反转旋转、反射、错误长度单位或把已发态当实测必须被拒。对被批准的 `R_cal` 做独立几何覆盖检验。 |
| 入场和退出 | 用缺 source、非稳 START、STOP 被拒、任意硬违规或未验认证的真实节点的合成控制验证 fail closed；清理所有 owned 进程并证明 ROS 图空。按先前签名条件的 32 根历史成功率作为资料，不能当 Pilot 通过。 |

若标定角色不清、模型/坐标或 clock 身份失效、实际 command ownership 超界、未知 driver 应用足以影响结论、安全非法、记录或恢复不可靠、Reviewer 关键意见未解、资源超预算，停止其下游。控制 guard 维持现有优先级。碰撞 path proof 需要 S0-5 独立安全验证，端点通过或已观察到 native motion 都不满足替代条件。历史 root01 的 399 个观测名义端点中 154 个在签名模型失败，重复相同 q 在原 live 模型另有 9 个拒绝；这一结果充分保留为必须校验完整路径和模型的依据。Formal G-A、D0/D1、硬件、Final Test 保持各自门禁。 [整身影子证据](../reports/TELEOP_native_nominal_replay.md) · [最终仿真部署门禁](V1-D_teleop_replay_and_refinement.md)
