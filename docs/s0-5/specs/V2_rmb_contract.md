# V2.0.0 Robot Reference Motion Bank 契约

`DRAFT_READY_FOR_REVIEW`。版本候选 `s05.rmb.v1` 的字节契约、源时钟、评估目标和窗口资格须在 V1-A 封存。构建规格定义本阶段实测、判门、预算及失败处理。[V1-A 实施规格](V1-A_offline_rmb.md)

## R1 存储、坐标和索引

- 每个发布分片对应一个 HMB partition，保留该分区全部 `T` 个计划拍及严格递增的 `int64 tick_ns[T]`，与批准 HMB 同一 20 ms 代理时轴。`plan_row` 只在索引中按母源、分区、拍定位。分片、训练窗口索引和证据可独立核验 SHA-256。所有数值采用 little-endian，布尔值只允许 0/1。索引和清单使用 UTF-8 JSON，训练数组采用无 pickle 的 NPY 分片或等价只读内存映射格式。规范轴顺序见下表，禁止隐式广播和顺序推断。
- `s=(L,R)`，`f=(thumb,index,middle,ring,little)`，`k=(proximal,intermediate,distal,tip)`。这些 V2 语义名显式映射到批准的 MANUS graph `MCP,PIP,DIP,TIP` 和机器人模型中 4 个有名地标。拇指和其余四指分别核对节点，不能依据 MANUS 或 MJCF 的自然数组次序替换节点。
- 所有关节张量采用 7 左臂、7 右臂、20 左手、20 右手。配置中锁定 `joint_name[54]`、手指次序、SDK 表项到模型关节名称、符号、单位、偏置、`qposadr` 和 `actuator_id` 双射。同名节点、缺名、重复、位置轴/执行器索引碰巧相等均需在实际模型下独立检查。SDK 输出维持原 API 单位和顺序，转换后入 `q_nominal`。
- 列向量公式使用右手系。`W` 为签名仿真模型固定世界，`U_s` 为该模型与 P02 注册明确衔接的腕系，`P_s` 是机器人掌系，`H` 是录音软件注册 frame。`R_AB` 将 B 表达变为 A 表达。四元数存 XYZW，计算后验方向差用 SO(3) 最短测地角。发布规范以 `w>0` 统一符号，`w=0` 时首个非零 x/y/z 为正；差分仍以相邻四元数内积选择半球。各几何数值用 m 和单位四元数，角度/角速度分别 rad、rad/s，时钟 ns。
- 数值字段和字段掩码同存。有效数值必须有限。无效浮点槽统一位级 `+0`，无效四元数槽写 `[0,0,0,1]`；只有掩码赋予语义。来源索引缺失用 `-1`；未知时钟保留独立资格位，禁止把占位解释为零时延。审计层可保留原非法 SDK/IK 字节或错误代码，其加载入口与 policy 张量隔离。

## R2 每拍数组

下表形状均隐含首轴 `T`。公用 `Geometry` 结构为 `wrist_xyz[2,3]`、`wrist_xyzw[2,4]` 在 W，`local_landmark_xyz[2,5,4,3]` 在 P，`world_tip_xyz[2,5,3]` 在 W，`pinch_xyz[2,4,3]` 在 P，`cross_xyz[5,5,3]` 在 W，`relative_wrist_xyz[3]` 和 `relative_wrist_xyzw[4]` 在 `U_L`，全部 `<f4`。每个字段附布尔掩码，shape 与字段去掉末尾坐标维后的前导维一致。各层可存相同结构却须各有版本和来源。对十指 `world_tip_xyz[s,f] = p_WU[s] + R_WU[s] * (p_UP[s] + R_UP[s] * local_landmark_xyz[s,f,tip])`。捏合为同掌非拇指尖减拇指尖，四个索引依次食、中、无名、小指。跨手第 `[i,j]` 项为右手指 j 减左手指 i，左指优先，右指次之。相对腕为 `R_WU[L]^T(p_R−p_L)`、`R_WU[L]^T R_WU[R]`。

| 层/字段 | shape、dtype 与读权限 |
| --- | --- |
| `audit` | `plan_ordinal[] <i8`、`hmb_source_index[4] <i8`、`capture_ns[4] <i8`、`receive_bound_ns[4,2] <i8`、`clock_eligible[4] bool`、`generation[] <i4`、`segment[] <i4`、`cause_code[] <u2`。六路 HMB 的非核心路保留源指针于 sidecar。源根、分区、文件名、拓扑 ID、标定 ID、窗口 ID 和绝对时间不进入 Actor。 |
| `inputs` | `stream_age_ns[4] <i8`、`stream_valid[4] bool`、`pose_joint_valid[] bool`、`safety_permission[] bool`、`source_changed[4] bool`。四路按左右 MANUS、左右已注册腕。前向输入从已验证 Reader 的原始节点/语义拓扑与有资格记录构造。`source_changed` 只供 SDK 调度和诊断，Actor 只接收 V1-B 所列特征。 |
| `nominal` | `q_nominal[54]`、`qd_nominal[54]` 与 `Geometry` 使用 `<f4`。`joint_valid[54] bool`、`velocity_valid[54] bool`、`status[4] <u2`、`limit_applied[54] bool`、`held[4] bool`、`nominal_generation[] <i4`，四组与动作分组一致。`raw_sdk_hand[2,20] <f4`、`raw_sdk_valid[2] bool`，保留 C API 原始输出；`raw_ik_arm[2,7] <f4`、`raw_ik_valid[2] bool` 记录已声明 SDK 单位的成功输出。缓存标识固定为 `CACHED_CAUSAL_NOMINAL_REFERENCE`。 |
| `mapped` | 主 V1 `Geometry`、`M0_audit Geometry` 与由固定输入计算的 `pinch_active[2,4] bool`。`rooted_pose_valid[]`、`target_valid[]`、`geometry_velocity_valid` 对应速度字段。位置速度为 `<f4`；角速度由相邻 SO(3) 对数给出，世界腕用世界外增量、相对腕用该相对旋转前帧坐标表达。字段名必须带单位与表达 frame。原始者及 M0 逐拍关联保留。 |
| `feasible` | 独立合法见证存在时给出 `Geometry` 及 `witness_id[] <i8`、`witness_status[] <u2` 与签名输入。可完全缺失；缺失时另报覆盖与状态。若记录 `q_witness[54] <f4`，仅诊断包持有，与 Actor、Replay Actor 输入和部署包物理隔离。 |
| `domain` | `mapped_valid, nominal_legal, improvable, hard [] bool`，`strictly_excluded_at[4] bool` 对应 3/5/10/15 mm，`unclassified_at[4] bool` 表示该阈值无合法见证及严格排除。保留 `radial_lower_bound_m[2,5] <f8` 和证明资格。`residual_search_status[] <u2` 保留合法见证、超时、尝试后无合法见证、输入不足、不适用。独立评估器使用预冻结域与窗口身份。 |

源换帧只消费已经在 `t` 可用且捕获时间不晚于 `t` 的事件。固定 20 ms 上，仅在相邻合法同世代拍计算差分 `qd=(q_t−q_{t−1})/0.020`；对旋转裁用 SO(3) `Log`，对因果重复样本保留年龄与原速度掩码而非推断新测量速度。`HOLD` 可携带上一合法名义命令用于回退，`held` 和原因必须可见于审计，`velocity_valid=false`，其数值不提升目标或训练资格。

## R3 M1 连续输出与 M0 审计

共享锁定的 `R_WH,t_WH` 作用于双腕，`p_WU[s] = R_WH·p_H[s]+t_WH`、`R_WU[s] = R_WH·R_H[s]·C_s`。`C_s` 为签名分侧安装；模型腕、TCP、基座变换另列。关节名义 IK 通过同一目标腕和锁定 `U_s → TCP_s → base_s` 适配，记录适配前后程序，并验算反向到 W。局部手型沿签名版本的人端方向、机器人指节长度、掌系和固定捏合规则构造，几何生成器的入参不含 Wuji SDK 输出、名义 q 或被评机器人状态。M0 使用其原固定软件注册和规则，记录 M1−M0 位移与功能几何变更，沿用历史九对跨手指作为审计列；V2 全 25 对是从相同尖点派生的量。M2、Minimax 及未冻结的平滑捏合不进入 V1 主目标。

V1 默认保留 M1 已签署局部手型和 20 mm 活跃捏合门，用严格 `<=` 和同边界浮点实现复现。记录阈值进/出、去重后原事件、每侧每指变化、实际 mapped 目标以及非法原因。连续资格须在 A 的独立窗口验证。异常只按事先冻结的失效规则作标记和保持，保留计划拍与原目标。变换或生成规则若有变更，必须取得修订批准、生成新候选、全量重做域与公平对照资格。[阶段验收](V1-A_offline_rmb.md)

## R4 域、评分和 Actor 白名单

全部计划拍来自原 HMB 索引。有效 mapped 目标、合法名义命令和各阈值严格排除均为保留在同一索引上的独立列。静态可达证据为某个已重算与验证的合法姿态。严格径向下界可排除给定阈值，未排除且缺合法见证的为 UNKNOWN。共同可改善使用冻结残差范围和与名义关节 FK 相比的预注册合法见证。存在残差改善仍可能受绝对 3 mm 不可达界约束，因此两列允许同时为真。困难域与可改善域的交集、差集和连续窗口长度均独立报告。课程序号、初始化姿态、见证、全轨迹派生标记和最终分母只供采样与评估使用。

Replay 的外生目标由环境逐时提供。每个 Actor 调用只接收 V1-B 列出的当前反馈与当下/既往参考及年龄、方向正确的因果速度和可用掩码；训练第 `t+k` 步外生目标须到对应想象步才进入。历史次序固定为最老到最新，同一有效 segment、时钟/标定/拓扑/名义世代，复位或边界切断历史。在线新世代用全零 masked padding 建新历史，当前帧合格以前走安全 hold；训练主窗口五帧全有效，另有显式 padding/恢复课程。禁止从右侧补帧、双端插值、Episode ID、绝对偏移、未来、`q_witness`/Oracle、课程或可改善域形成 Actor 特征。负控在完全相同的输入前缀下替换/删除未来及元数据，比较当拍 reference、历史和 Actor 输入字节。[环境观测契约](V1-B_residual_environment.md)

部署无 RMB。缓存与在线名义路径需复算同前缀，报告 arm/hand q、qd、有效性、HOLD、START/世代、Servo 命令、FK、时龄和 P50/P95/P99 漂移，报告各域覆盖。相同样本和初始化条件的共用适配代码须优先满足零残差等价，生产图、异频率及真实时钟资格另作结论。[仿真部署规格](V1-D_evaluation_deployment.md)
