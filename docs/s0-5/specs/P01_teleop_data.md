# P01 Teleop-only 数据、谱系与时序

[执行基线与样本预检](../reports/P00_review_fix.md) · [接口契约](../CONTRACTS.md#c1-几何和数据)

## 1. 阶段与版本

P01 v0.2.0，待人工冻结。P00 Review Fix 仅验证环境和少量只读样本。P01-A 至 P01-F 均需 P00 人工 ACCEPTED 后另行授权。v0.2.0 冻结格式、语义和校验方法。数据驱动参数在 P01-D 审核时另冻结。

## 2. 目标

从原始 MANUS 双手和 Tracker 双腕采集建立 Teleop-only Human Motion Bank，以及有独立准入资格的 System Identification / Robot Telemetry Bank。保存来源、时钟、标定、有效性和排除证据，向下游交付可重放的离线和因果数据。

## 3. 与主设计的关系

落实 M1–M3、M5 和 C1/C2/C6。为 P02 提供人体意图输入，为 P03/P08 提供可探查的机器人遥测，为 P04/P07 提供不含未来样本的可用输入流。可行目标、优化 IK 和几何保真验收分别由 P02/P03 负责。

## 4. 覆盖需求

S05-REQ-001、011、012。接口支持 003、004、015、021、023、026 的后续验证。

## 5. 前置条件

P00 已人工 ACCEPTED，P01 另有明确授权。固定 [执行基线](../EXECUTION_BASELINE.md) 的源码提交、解析环境、消息定义和源路径绑定。审核 v0.2.0。Preregister 只读预实验的候选样本、开发池资格、写入位置与资源上限。独立终评来源可以在 P01 暂缺，条件与限制见 `13。

## 6. 现有实现审计

Wholebody S0 的 `bag.py` 是分流录包读取参考。`sync.py::align_episode` 产生以 header 为轴的双端插值 pose。单独保存的 `*_causal_index` 按 `receive_ros_ns` 选最近已到记录，并按 header 检查年龄。此字段的采集语义不能自动当作控制节点到达，插值 pose 本身也不因旁存索引而成为因果数据。P01 从原始读数重建两种对齐。分流和合并 MCAP 的样本及缺项见 [预检报告](../reports/P00_review_fix.md)。历史 `parsed.npz` 与原始 CDR 样本均见 **25 个 MANUS 节点**，raw topology 保留现场原始节点及编号。

## 7. 范围与来源发现

| 根 / 输入 | 发现与核验规则 |
|---|---|
| `/home/ubuntu/dexgym/data/raw/wholebody_s0/{2026-11-12,2026-11-13}` | 跟随 episode 索引符号链接，保存逻辑路径及解析目标。根节段外目标仅允许显式批准的 collector archive，载入 manifest、effective_config、run snapshot、events、分流 MCAP 与 sidecar |
| `/home/ubuntu/MANUS_STEAMVR_ws/dex_data_platform-cpp-collection/data/raw/episodes` | 与采集索引合并计数，校验采集 manifest、分流目录和存储类型 |
| `/home/ubuntu/teleop_ws/data` | 发现带 sidecar 的 ROS2 MCAP episode。按 session 身份组合 write-split parts，查找采集配置、topic 和 recorder 记录。文件名前缀仅作检索 |
| 历史派生目录 | 只注册谱系、使用角色、标签和子片段；证据见 [历史使用设计](../HISTORICAL_DATA_USAGE.md) |
| 后续新增采集根 | 白名单固定、采集者登记、时钟/标定凭证和源权限确认，新增后更新 registry 版本 |

发现输出计数按目录、文件、逻辑录制、确认唯一母源、状态分开。轨迹、控制器试跑录包、仿真、EgoPro 和无法验证生产者的候选进入显式剔除或隔离清单。只读访问原文件，任何新产物均写独立输出根。

## 8. 范围外

人体映射、可行目标 Oracle、生产控制器、训练、机器人命令和 P09 性能评估均为后序阶段。预检或正式 P01 的数据 schema／完整性审计只检查来源和数据质量，隔离区以及 Final Test 的内容不参与评分参数、阈值与训练样本选择。

## 9. 输入契约与 Adapter

Adapter 输入 `source_id/path_segments + recorder provenance + calibration refs`，输出 `stream_id/topic/type/schema_hash + event + clock_fields + source_ref` 的有界批，原始 CDR 可按索引回读。编码未知、type support 缺失、无法解码或 sidecar/hash 不匹配时产生 reason code。每条消息的缓存/索引均不依赖未来样本。

| Adapter | 优先结构 | 源/有效性检查 | 机器人遥测 |
|---|---|---|---|
| `dex-native-v2@1` | Episode manifest，MANUS/Tracker raw 与 registered 分包，观察器元数据和安全状态，按 manifest stream 和 segment 组合 | 核验 manifest schema、录制 run ID、消息和观察器的 stream/sequence 或已证明配对关系。配对歧义保留两条流，接收语义 UNKNOWN。`initial_state/` 由 manifest 标注为 episode 前的缓存，独立索引并保留其原时钟 | 按实际 feedback、cmd 与来源判断，样本分流录制用于人体输入见证 |
| `ros2-merged-mcap@1` | 单 bag 或分片，sidecar 与 MCAP 话题 catalogue，发布和录制域分离，字段名按有版本 schema 映射 | 原始录制需采集配置、操作者或设备日志/标定支持。整包含 raw glove 不足以证明它是一次新的真人采集。当前 ROS 类与包内定义不同则按 MCAP schema SHA 选择录制时的 CDR decoder | 有真实硬件证明且至少有配对 command/反馈的 episode 才可入 telemetry。缺 effort、current、wrench 或时间均单列 |
| 未识别或衍生 | 仿真 producer、replay 运行目录、继承 parent 指纹或来源证据不足 | 记 SIM_REPLAY、DERIVED_ONLY、EGOPRO 或 UNKNOWN_PROVENANCE，附原始父关系与置信证据 | 不进入实机系统辨识 |

读取器能力须明示。`rosbag2_py.SequentialReader.read_next` 的日志时间仅按该 API 与记录器语义记录，发布时间若未暴露写 UNKNOWN。P01-B 使用已锁定的原始 MCAP reader 读取 record log/publish 两字段，并按来源保留 CDR schema SHA。控制器到达仍须单独建立证据。禁止用 header/日志代填未暴露字段。

## 10. 输出契约与字段模型

JSON 使用 UTF-8、稳定键顺序、IEEE 非有限值禁止；数组按一维记录/时间维在前，采用小端 I64/固定版 F64 和 bool，保留消息原字段单位及转换证据。下表的结构和枚举用 JSON Schema 与数组断言双检。数组使用 `.npz`（`allow_pickle=False`）或 schema 等价块格式，跨 run 固定格式和版本。`N` 为流内样本，`K` 为某 episode 当前 topology 中原始节点数，`T` 为控制网格帧数。

| 产品与字段 | Shape、dtype、单位／帧 |
|---|---|
| `raw_registry.v1` | `asset_id, source_id, uri_rel, canonical_uri, format, producer_kind, admission_reason, checksum_state` 为 UTF-8 string；`size_bytes` U64；`file_sha256, sidecar_sha256, recorder_config_sha256, calibration_id, type_schema_sha256` 为 SHA 或 null；`part_order` U32；`recorded_interval_ns[2]` I64 + valid/domain；`catalogue[]` 为 topic/type/message_count/definition hash |
| `episode_lineage.v1` | `episode_id, root_episode_id, parent_ids[], source_asset_ids[], relationship, evidence_ids[], adjudication, usage_roles[], split, final_attestation_id`。裁剪/慢速/重编码另有父采样索引区间、时间映射和 config hash。片段或 augmentation 继承全部父级历史使用标签 |
| `parsed/index` | `source_asset_index[N], part[N], channel_id[N], message_index[N], source_sequence[N]` 整数，plus availability mask。`schema_id, CDR reference, clock_domain, config_id, frame_id` 用定版字典索引。完整离线元数据保留每流样本数和来源 |
| `parsed/MANUS/{left,right}` | `node_id[K], parent_id[K]` I32，拓扑变更切 segment 并另记；`node_position[N,K,3]` F64 米、`node_quat_xyzw[N,K,4]` F64，消息本地／原帧明确；`node_valid[N,K], glove_valid[N]` bool；可选 raw sensor/ergonomics 列独立 schema。保留原消息顺序或反向映射，21 点选择另行派生 |
| `parsed/tracker/{left,right}/{raw,registered}` | `position[N,3]` F64 m、`quat_xyzw[N,4]` F64，`valid[N], safe[N]` bool 值与 known mask，`quality[N]` F64 加具体语义 ID、`frame_id[N]` 字典；raw 与 registered 同时保留，后者绑定 frame/calibration |
| `parsed/clocks` | 每流各时钟 `timestamp_raw[N,C]` I64 + `known[N,C]` bool、单位/域字典及 source 字段名。原始 µs 不丢精度，映射的 `clock_interval_ns[N,C,2]` I64 和 bound known。缺值 JSON null 或数组 -1 + false，raw 零值有独立 mask |
| `aligned_offline` | `tick_ns[T]` I64，pose 每流 `[T,K,3]/[T,K,4]` 或腕 `[T,3]/[T,4]`；`bracket_index[T,2]` I64 和 `bracket_clock_ns[T,2]`、`weight[T]` F64、`valid_mask[T,...]` bool、`reason_code[T,...]`。显式允许双端插值 |
| `aligned_causal` | 同 pose/quat shape；`selected_index[T]` I64、每帧 `capture_lower/upper_ns、controller_arrival_lower/upper_ns、cutoff_ns、age_upper_ns` I64 与 known mask；`tick_ns[T]`，`valid_mask[T,...]、fresh[T]、held[T]` bool，`clock_quality/reason_code/safety/segment_id` 版本枚举字典。每流独立，双手联合 mask 另存 |
| `calibrations.v1` | `calibration_id, version, source_hash, from_frame, to_frame, source_units, valid_interval`；T_to_from `[4,4]` F64 左乘的右手系；原始校准文件及转换脚本 SHA；未知以 null 状态表达 |
| `telemetry_index.v1` | episode/site/controller/config/calibration refs，topic/part/event offset 或可靠序列、硬件渠道、时钟及采集证据。command/q/qd 可索引原消息或物化按名称数组 `[N,J]` F64，q rad、qd rad/s，缺失单列；`wrist_pose[N,7]` m/XYZW、`twist[N,6]` m/s、rad/s，配置坐标。`effort/current/wrench[N,J或6]` 保留单位元数据（N·m/A/N,N·m 在来源支持时）；diagnostics、RobotState、TeleopState、safety events 单流字段索引和 mask |

Human Motion Bank 包含 raw registry、lineage、parsed HMB、两种独立 aligned 产品及 calibration。Telemetry Bank 另发可追溯轻量索引，父源可共用，物理反馈、配置与时钟资格分别检查。可访问的人体专用录制保留 HMB 身份。

## 11. 数学定义、时钟与对齐

| 时间字段 | 语义及证据 |
|---|---|
| `source_capture` | 设备原始捕获时间或计数器，独立域/分辨率。MANUS `source_timestamp_us` 保留原值并依设备说明/录包验证其时基 |
| `sdk_receive` | 记录来自 SDK 的接收时间，例如 `receive_timestamp_us`，无需与捕获等同 |
| `ros_header` | 发布者定义的 header stamp／ROS 时钟域，保留 frame 和发布节点来源 |
| `mcap_log`、`mcap_publish` | MCAP 记录与发布两个字段，各自保留 read source 和 unknown。`reader_log_ns` 默认仅为见证的 API 时间 |
| `collector_receive` | recorder/采集器的接收或 observed 时间。分别记录 `receive_ros_timestamp_ns`、`host_observed_monotonic_ns` 和 `host_realtime_ns` 等原值，名称不证明同域 |
| `controller_receive` | 控制进程收到的单调时间，须有控制侧日志或有证据代理。与 collector receive 分开 |
| `control_tick`、`command_issue` | 调度执行查询时刻及指令实际发布/写入时刻，分别记录单调和同步域 |
| `physical_response` | 实际反馈采样与被定义物理响应事件，独立于命令发布。只有推定对象时标清语义 |

时钟映射按 segment 记录别名、单位、整数 ns 原始范围、估计器配置、启用生效时间和误差界。候选是带不确定度的仿射 `τ=a·s+b`，仅使用已可用的同步锚点；跨跳变/重启不连续，禁止离线全 episode 拟合用于因果。若存在可靠上界，用 $c^+$ 与 $r^+$ 表示规范时钟下最晚可能的捕获与到达。仅纳入 $c^+≤t_k$ 且 $r^+≤d_k$ 的已知有效样本，沿捕获序、合法序列选择最新。年龄上界 $A_k=t_k-c^-$。缺时钟界的样本标 UNUSABLE。四种资格分别为 VERIFIED、PROXY（有验证映射和误差界）、SIMULATED（显式仿真到达生成器）、UNUSABLE（不具备该时序验收条件）。PROXY/SIMULATED 单独报数，不能支持真实端到端延迟主张。

离线位置插值 `p=(1−α)p0+αp1`，旋转对归一化 XYZW 作最短弧 slerp。禁止跨 source reset、标定切换、无效 gap 或安全过渡。因果版本默认零阶保持，控制拍缺样本时不借后续插值，记录 held/fresh。已过去事件更新估计器状态，乱序与晚到包只影响满足条件的之后控制拍，不能修改已发出的结果。重复消息按资产/记录索引和 verified stream-instance sequence 判定，重复源捕获可由 validated 设备 stamp 识别。重复保持不会刷新 age。时钟反跳／序列重启分段，清历史重新取得有效输入。左右手/腕分别判定，跨路 skew 超出冻结阈值时联合 mask 无效。STALE、DROPOUT、UNSYNC、UNSAFE、CALIBRATION_INVALID、SOURCE_UNKNOWN 与缺时钟状态显式传播，P03/P07 执行超时回退。安全未知不自动当成 SAFE。

## 12. 架构与模块

`source registry → adapter/read-only CDR → parsed clocks/topology → QC + lineage → offline/causal builders → manifest + checksum + split`。telemetry index 在 parsed/registry 层独立分支。时钟校准和可用性信息单独版本化。低质量但可解码的姿态保留在诊断层，规范训练视图由冻结规则筛选。

## 13. 实施任务、重复筛查与数据隔离

校验 MCAP 按分片 SHA-256 和组合顺序签名去重。重命名／跨目录相同文件依据内容而非 basename 合并。同源裁剪、慢放、重编码用 provenance 链及同序采样指纹、时基窗口、采集器实例与源时间区间联合比对。语义精确指纹用标定前 node/pose/validity 的固定字节编码与相对时间；变速相似仅生成候选，由区间、设备/采集说明及人工裁定，不因相似否定真实录制。歧义未解决前状态 QUARANTINED。所有版本根 episode 连通分量作为组，对任意父根/共享区间做跨 split 检查。

历史 Train、选型、反复评估、难段分析和污染不确定的谱系可标为开发池。`FINAL_TEST_SEALED` 只给独立录制、完整 attest、未使用记录与访问日志能支持的分组。P01 可核对 schema、标签、签名和完整性，面向 Final Test 的 QC 和阈值不得使用控制表现，访问使用角色隔离，交叉核对 lineage 后密封。数据不足时输出 Train/Validation 和显式 `final_test_status=BLOCKED`，另列新录制、无用途中间池、封存和授权方案，P09 开工必须解除该项。完整设计见 [历史使用与封存](../HISTORICAL_DATA_USAGE.md)。

## 14. 预期文件与接口

生产实现拟在 `source/dexgym/tasks/direct/wholebody_s05/data/`，包括 `source_registry.py`、`adapters/`、`lineage.py`、`synchronize.py`、`telemetry.py` 和 `manifest.py`。拟用入口 `python -m dexgym.tasks.direct.wholebody_s05.data.preflight --config <draft>`、`...data.build --config <frozen> --run-id <id>` 与 `...data.audit --manifest <published>`。这些是 P01 的实施接口，本次交付仅含可执行的有限探查工具与设计。

正式输出根 `/home/ubuntu/dexgym_data/s0-5/p01`。写入 `runs/<run_id>/.staging/<build_id>`，完成每个独立块后 SHA 与输入/配置身份登记。全量验收前可断点恢复仅对原参数/相同 SHA 的完整块；损坏或变更生成新 build，中间产物保留较小诊断。全部文件 close/fsync、全量校验，保证同 filesystem 后原子重命名 publish 至 `products/<contract_version>/<run_id>`，先拒绝已有目标。通过状态由独立 manifest 和事务日志表明。并行 worker 写互斥分片，无重叠更新。原始源及上一个发布版本均只读。

## 15. 单元测试

实施后由 `pytest tests/s0_5/unit -k 'registry or lineage or mcap or clocks or causal or telemetry'` 运行。fixture 两种真实格式由工具生成最短完整消息区间，固定 CDR、真实字段和 source SHA，不把重编码 fixture 当独立源。先验证部分 manifest、缺流、损坏块、原始 topology 非 21、分片顺序与单位 / transform。构建因果负控：未来替换、未来删除、乱序/晚到、重复、stale、unsafe 未知、时钟反跳、左右手缺包。比较已过去输出和索引逐元素一致，所有 derived pose 都能从合法来源索引重算。

## 16. 集成测试

实施后由 `pytest tests/s0_5/integration -k 'teleop_fixture or cutoff or split or atomic_recovery'`。在已选 dex-native-v2 以及旧合并包各取有版本的原始样本和至少一条真实 telemetry fixture。验证 adapter→registry→parsed→两个 aligned→manifest 闭环、未匹配 observation 路径、仿真重放排除、同根裁剪和变速不跨 split、失败中断隔离及哈希验证。控制侧到达时钟未知的场景证明不会被归类为 VERIFIED。无硬件控制。

## 17. 实际运行门顺序

| 门 | 运行 / 交付 |
|---|---|
| P01-A | 只读盘点、注册/谱系裁定、历史 usage registry、读取许可并证实初始写入预算 |
| P01-B | 对每个正式准入 adapter 跑真实小样本，完成 CDR、原始时钟、缺字段和 telemetry 资格记录 |
| P01-C | 用固定开发小样本做标定／单位验证、左右拓扑、两种对齐和未来信息负对照 |
| P01-D | 冻结来源白名单、阈值、标定规则、split、资源、运行命令和证据，人工审核后允许扩大 |
| P01-E | 严格按冻结参数执行原子全量重建，失败或磁盘压力可暂停并留证 |
| P01-F | 独立审计源级隔离、全量计数、重复、运行日志、代表性可视化和放行矩阵 |

P01 Review Fix 的小样本记录不作 A–F 的正式成果。各门由可验证结果推进，D 的人工证据锁定先于 E。

## 18. 定量协议与 QC

每流统计 catalog/实际读取、解码/丢失/重复、原始/映射/接收时间可用率、有效关节拓扑比、帧率及 $Δt_i$ 的分位、有效区间时长、连续失效率。skew 用相同 tick 各流可用捕获区间的保守最大跨度；延迟只按同域或带误差边界的有效配对计算。四元数检验 $|\|q\|_2−1|$ 与有限性，旋转检验 $R^TR$、det>0；骨长取同手父子关节点距离分组分位，速度和旋转角速度只在正常可证时轴按向后差分，异常记录原消息和比较基准。原测噪声和运动可并存，样本异常不直接当来源错误。

按格式、设备配置、episode 和左右侧分层，使用开发 Train/Validation 的录包及独立采集时钟正常/掉包试片。逐 episode 记录经验分位 P50/P95/P99、MAD、anomaly tail，候选阈值以观测分布、设备/控制需求和有标注故障区分率共同选取。样本少时用最保守实测范围或 BLOCKED。P01-D 提交候选表／原始计数、分类边界和影响，人工签核后绑定 config SHA。test 总体审计可用同固定逻辑，阈值不得按 test 派生性能重调。超出适用域或需修订时发起版本变更，冻结 test 访问状态。正常、降级、不可用、来源隔离分层，不得静默筛除困难动作。

## 19. 强制验收矩阵

| ID / 门 | 验证脚本或测试选择器 | 输出证据 | PASS / FAIL / BLOCKED |
|---|---|---|---|
| A1 / A 源谱系 | `data.audit --check source,lineage,historical_usage` | `registry.jsonl, lineage.jsonl, prior_use.jsonl, rejects.jsonl` | PASS 每个候选有处置，准入为原始 Teleop 且谱系可回溯；FAIL 跨源重复漏记、EgoPro/replay 混入；BLOCKED 来源/校准不可证明 |
| B1 / B 各格式 | `pytest ... -k teleop_fixture` | `fixtures_manifest.json, decode_fields.json, attempts.json` | PASS 两种批准格式实际 CDR、数据能力及缺项；FAIL 类型错/丢失原始节点；BLOCKED 无合法原始 fixture 或依赖 |
| B2 / B 遥测 | `data.audit --check telemetry` | `telemetry_index.jsonl, missing_fields.json` | PASS 真机 provenance、指令与实测可回溯，未知显式；FAIL HMB-only 标作遥测；BLOCKED 硬件来源/反馈不具资格 |
| C1 / C 几何 | `pytest ... -k topology_units_frames` | `calibration_report.json, invalid_rows.jsonl` | PASS 字段、25 节点样本和可变拓扑、标定/单位往返符合有据容差；FAIL 单位、拓扑或 frame 错；BLOCKED 标定缺失 |
| C2 / C 因果 | `pytest ... -k causal_cutoff` 和 `data.audit --check sync` | `prefix_mutation_results.json, source_trace.jsonl, clock_quality.json` | PASS 未来替换/删除输出相等且每帧已有可用源，stale/unsafe 等均进 mask；FAIL 任一泄漏或错误有效；BLOCKED 需 VERIFIED 的时钟无真实证据 |
| D1 / D 质量与 split | `data.audit --check qc,split,budget` | `frozen_config.json, preregistration.md, split.json, budget.json` | PASS 人工签核及所有根零跨集；FAIL 把低质数据无记录剔除或使用 test 调参；BLOCKED 没有足够预算/证据。独立 Final Test 待补时开发池可验收，P09 门明确 BLOCKED |
| E1 / E 全量 | `data.build ...`，`data.audit --check content,atomic_recovery` | `build_log.jsonl, manifest.json, coverage.json, output_sha256.json` | PASS 无覆盖历史，文件/计数/哈希对齐、失败块可回溯；FAIL 文件篡改、缺记录或部分发布；BLOCKED 资源和权限不足 |
| F1 / F 独立 | `data.audit --check acceptance --manifest ...`，独立重跑样本 | `P01_acceptance.md, test_results.json, qc_visuals` | PASS 逐项证据及独立抽样，全部必须门完成；FAIL QC 或定量偏差未处置；BLOCKED 必需规则/证据/采集权限缺失 |

上述脚本是 P01 计划名称，由 P01 A–F 的 TDD 交付。当前预检结果只在 P00 Review Fix 报告裁定。

## 20. 失败处理、原因码与数量

每个资产、episode、流和帧同时持有多个 flags，再从版本化优先级选择一个 disjoint disposition。根级由读取错误、格式不支持、来源/谱系未知、派生/仿真/EgoPro、标定或关键流缺失、准入的次序裁决，重复文件作为有主源的 alias；帧由 `DECODE_ERROR`、`CLOCK_JUMP`、`DUPLICATE`、`LATE`、`STALE`、`MISSING`、`UNSAFE`、`UNSYNC`、`CALIBRATION_INVALID`、`VALUE_OUTLIER` 和 `VALID` 等记录原因及 flags，具体优先级在 D 按安全优先冻结。总量必须能复算 `inputs = admitted + excluded + quarantined + duplicate_alias`，以及每流计划/读入、成功/失败/缺失、可评分/不可评分。出错保留输入指纹、最小事件和重试状态。未知来源和损坏不自动重新准入。

## 21. 风险、未定阈值及资源

实时 age/skew、跳变、缺失、骨长/速度容差、物理时钟的映射误差由 P01-C 的分层录包确定候选，P01-D 人工冻结；P02/P03/P07 依赖。当索引 raw MCAP 时逻辑体量和写入体量分别计。先逐格式/样本量测源读取、派生字节/帧、峰值 RSS、写放大和吞吐，空闲状态用 `df -B1`、`statvfs` 与峰值估计。保守峰值 = 已占项目空间 + 预计所有输出未压缩上界 + 可并行的临时副本 + 断点保留 + 审批空闲预留。新输出前比较此值与实时可用空间，走同一文件系统的原子发布。输入数据、已发布产物和实验目录不参与回收。不足、哈希异常、超 RSS／长时 I/O stall 时保存完整块并暂停，向 P01-D 责任人报告，重新核算再恢复。没有要求使用 GPU。实际观察见 [执行基线](../EXECUTION_BASELINE.md)。

## 22. 可复现性

保存输入 SHA/reader/模型无关的配置版本、schema 与标定身份、双时间戳域、split 规则、工具哈希、Python 与 ROS 版本、实际命令、stdout/stderr/退出、资源/失败记录。在同一父源和冻结配置上两次重跑，保证逻辑记录排序、数值 array 与内容哈希一致。若压缩封装带不可避免时间元数据，明确区分容器哈希与规范化数据哈希，前者保留。外部工具身份由执行基线复验。

## 23. 交付物

独立 registry、lineage 和 use registry、两个逻辑产品与离线/因果对齐、split 与封存记录、质量/时延分布、失败清单、证据、资源评估、版本标记及独立验收报告。P01 实施 source 路径在 `14` 计划定义。

## 24. 退出门

P01 A–F 的明确证据齐全并提交人工审核。Final Test 可用性为显式独立状态，若无法证明独立，保留 P09 强制 BLOCKED 条件，允许只交通过验收的开发视图与补录计划。P01 开工须收到授权。

## 25. 下游依赖

P02 使用 HMB/原始标定与原因分层。P03/P08 使用有资格的 Telemetry Bank 并验证实际控制链时钟。P04/P07 使用具有明确类别的 aligned_causal。P09 用独立封存和不可变评分。P10 验证部署源与控制时钟。
