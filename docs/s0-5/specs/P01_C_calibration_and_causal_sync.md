# P01-C 几何、标定与因果同步实施规格 v0.1.0

适用已签核 [P01 v0.2.1](P01-data-v0.2.1.md) 和 DCR-001。范围为 P01-C 有界开发样本的几何、时间与可复用同步核心。技术结论按 C1、因果算法 C2 和历史实机时钟资格分别判定。真实证据不足时给出 BLOCKED。P01-T、全量 Motion Bank、正式切分、P02、控制发令及训练继续在授权范围之外。

## 冻结样本和预算

采用 [冻结计划](../evidence/P01_C/sample_plan.json) 的七个已认证 native 母源，覆盖 11-12、11-13、11-20，arm_motion、hand_pose、arm_hand、bi_hand、fruit，双侧全流。各来源分别复验 A1 registry、P01-B-HMB 实解资格、writer/collector、逐文件和采集期配置 SHA，记录每个录制的实际配置和条件依赖。只接纳与 manifest 已证的配对规则及相应 schema；其他录制需单独审核。所有样本源用于开发诊断，Final Test 仍未封存。

窗口锚定 recorder host realtime。头部从 episode 起点后 1 秒开始，中间及尾部取居中或结束前的同长度窗口，按 manifest 范围裁剪，前后各保留 250 ms 辅助上下文。安全流在预算内找首个记录的 unsafe 状态，若存在且与三窗不同则按同规则加入，实际缺测和安全事件逐一报告。选择脱离姿态、残差和 coverage。所有处理只覆盖计划中的连续短窗及上下文。原文件可顺序校验/枚举物理索引，正文 CDR 只在有界窗口及安全预览中解析。原包上限和运行预算见计划。超限停止并留记录，禁止用更多帧冲淡异常。

## 输入和物理几何

沿用原 raw events 的 recording_id、part_id、物理索引、asset/CDR/schema SHA、消息字段、13 个原始时钟字段和 collector observation。MANUS 保留 node_id、parent、joint/chain、顺序、XYZ、XYZW、frame 与证据化的 validity。根可以指自身或使用 -1。根唯一，其余父节点指向同图且沿边收敛到根。不同拓扑使用版本边界。若 SDK 初始化为 world coordinates，位置/姿态是共同 SDK 空间的全局 node 量。通过采集时 publisher 编译配置、SDK 头以及位置和骨段观测校验，禁止把 SDK 全局标签映射为 OpenVR 或 robot base。掌系为手根单独派生空间，p_palm = R_rootᵀ(p_i - p_root)。同一全局表达下骨长为 ||p_i - p_parent||。局部 pose 如出现，需按已有合同单独审定。记录双侧骨架和掌系图、数值分布、足够和缺失的证据。

采用 T_A_B 把 B 中列坐标变到 A，T_A_C = T_A_B T_B_C。右手系，位置以 m，ROS 四元数 XYZW；录制原值与归一化结果分开。验证正向、逆向、RᵀR 和 det R。按采集时确切左右 wrist 外参重建 T_W_wrist，并用同 side、同原始 header stamp 且同 frame 的独立已录注册腕作为对照。绕同一输入的两个计算路径不作独立真值。按录制/侧报告平移及 SO(3) 通用角分位和未匹配计数。估计仅适用于已哈希且在当期有效的配置，左右 frame 不能互换。有记录的 OpenVR/world/base 组件逐个标注；缺独立 robot-base 证据不得宣称已物理对齐。

## 时钟、事件身份和资格

分别保留 source_capture、sdk_source、sdk_receive、ROS header、MCAP log/publish、collector receive/monotonic/realtime、controller_receive、control_tick、command_issue、robot_feedback。域、单位、角色、来源和缺值显式记录。SDK 压缩整数按已锁定 SDK 组件 API 解码，非 timecode 的 fraction 为组件毫秒。保留 raw 和组件，禁止直接以压缩整数计算普通间隔。数据和独立时钟锚点联合验证前，不给 capture→controller 的真实上界。

MANUS 重复指同 source instance、相同 SDK source 与 SDK receive、相同原始拓扑与骨架的载荷，publication/collector 事件继续保存。同姿态不同 proven receive/sequence 可以是不同真实采样。相同源值而载荷相异需分开诊断，乱序或时钟回退按序列、发送和到达证据分段。无匹配证据不把 recorder adapter 序号当设备序列。时钟估计只吸收可用的外部锚点。设斜率证据区间 [a_lo,a_hi]，锚点设备时刻 x_i 与目标区间 [l_i,u_i]，则设备时刻 x 的目标区间为各 [l_i+min(a·(x-x_i)),u_i+max(a·(x-x_i))] 的交集。冲突、未获得锚点或域不明会阻塞真实映射。此模型不从单向日志识别传输延迟。

历史真实性路径仅限有独立证据的 VERIFIED。PROXY 仅用于注明域和用途的观测间隔。SIMULATED 使用显式模型，当前窗口实验用 collector monotonic 等价索引取代理样本时刻并加固定候选到达延迟；该合成域不表示真实 capture→controller 时延。UNUSABLE 阻止真实消费。无不确定度上界时，空间预算只给分位速度乘假设误差的敏感性分析。

## 分离的离线与因果算法

离线只在时间递增、边界一致且有效/Safe的两条同流事件内作分段线性和最短弧 SLERP，输出两端物理索引、插值权重和时间区间。重复源不作为独立输入。无效样本、安全变化、无证明的校准或时钟重置、拓扑切换、超过候选 gap 及无前后端的窗口边界阻止插值。

在线核心按事件可用性时间序逐条处理。只有 capture_upper ≤ tick 且 arrival_upper ≤ cutoff 才能输出有界状态。查询严格消耗已可用事件，并按流保存当前事件、source identity、记录索引、版本和 age_upper=tick-capture_lower。零阶保持不刷新年龄，旧包、重复包或超时 safety 不恢复失效。跨流在共同可用的输出上计算 left/right 与 wrist/hand skew。初始化缓存只在有真实时间、来源、安全和 age 证据时作为明确的缓存类型使用，本轮输入仅使用 BODY 事件。速度为可用历史的后向差分。mask 和 reason 包含 VALID、HELD、STALE、DROPOUT、DUPLICATE_SOURCE、OUT_OF_ORDER、CLOCK_JUMP、UNSYNC、UNSAFE、SAFETY_UNKNOWN、CALIBRATION_INVALID、SOURCE_UNKNOWN。必要的源和时间条件不足时输出不可用状态。

## 模块和数据产物

在 wholebody_s05/data 内建立 geometry、calibration、clock_domains、sample_identity、synchronize、sync_audit；capture 审计调用固定 adapter/decoder 和已核采集配置。窗口按 NPZ 保存连续时间、空间和掩码。流内目标维度为 7（XYZ 和 XYZW）；MANUS 保留全部节点并采用各窗口的最大 N，腕 N=1，其余元素用 float64 NaN 和 mask 表示。轴为 [tick,stream,node,field]，时间、年龄、索引 int64，类别用版本化整数码及 JSON 元数据。JSONL source trace 将每拍每流有效状态的真实记录引用及无效理由与 NPZ 对应。两个 alignment manifest 分立，保留语义等级与受控边界。产生用户规定的 geometry、topology、coordinate、calibration、clock、sample identity、coverage、validity、候选阈值、误差预算、测试和哈希报告。

## C1 和 C2 验收

| 门 | 验证 | 失败与阻塞 |
|---|---|---|
| C1 | 真包验明 source/sha/身份、树和原始单位、四元数/旋转、按配置的独立 raw/registered 对照及左右分层残差 | 非法图或确定变换错误 FAIL。来源、单位、配置或配对不足的记录隔离。物理 registration 缺证单列 BLOCKED。 |
| C2 算法 | 固定真包窗的代理/模拟前缀修改及删除、未来锚点和安全变异、批流等价、索引/时钟状态可复验，以及乱序/超时/安全边界负控 | 任一未来泄漏、错 mask 或不可追溯的输出 FAIL。暴露降级并按完整窗口分母。 |
| C2 真实时序 | 独立捕获和控制节点到达证据，报告界与误差，禁止把 SDK/collector 当控制器 | 只有代理或模拟时继续 BLOCKED。整个阶段不得因模拟算法通过就自动验收。 |

执行单元数学/边界测试以及真实集成测试。重复、重启、两手异步、无可靠到达、跨失效插值和拓扑/标定切换均应有负控。未来事件变异需要比较 pose、速度、mask、age 和索引，不允许历史回填。对比同拍 offline 与 causal 的几何和后向速度，以及带分母的动态/静态代理差异。报告安全事件和失效是否见于真包；人工边界测试不作真包证明。P01-D 冻结 age/skew、gap、clock、quat、标定、骨长、速度和 safety timeout 等最终阈值及覆盖与资源策略。本阶段只生成候选、分布、覆盖敏感性和风险，报告全部未满足项。
