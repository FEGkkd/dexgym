# P01 v0.2.1 Topic Dependency Matrix

阶段审核采用 [冻结的机器规则](../evidence/P01_B_review/topic_dependency_matrix.json)。规则依据 P01-B 的实际原生 manifest、schema SHA、原始 payload／SampleObservation 配对、追踪器消息及命令/反馈数据路径。未审定通道以目录证据保留。每条录制先完成来源和证据依赖判定，之后才能确定其可用的字段。

| 类别 | 消息及声明 | 实际用途与失败处置 |
|---|---|---|
| HMB_REQUIRED | 左右 MANUS 原始骨架；左右 RAW wrist PoseStamped；左右 registered wrist TrackerPose；tracker safety | 保留骨架顺序、父关系、关节/chain 类型、raw/registered 坐标、原始时钟以及显式 valid/safe。必要 CDR、字段或 schema 错误使对应审核失败。缺采样或必要配对使录制隔离，不能生成训练准入。 |
| HMB_CONDITIONAL | 配套 SampleObservation；采集时期标定与侧边站点；CollectorStatus/CollectionStatus、TopicCatalogue、WriteSplitEvent 作为某一证据 | 原生 writer 的独立证明由已哈希的 manifest、run snapshot、collector ID、标定文件和实际 BODY 记录组成。它必需每个输入的已证 recorder 观测和索引。旧合并包若依赖状态、目录或 split 事件证明来源、完整性、有效性、安全或同步，则该字段在那条录制纳入必需证据。它缺失或失败时持续隔离。 |
| TELEMETRY_REQUIRED | 双侧 arm/hand 的 JointCommand、JointState、HandCommand、HandState、真实 driver／teleop state 和对应 safety | 这些流服务 P01-T 的独立真实指令与测量责任。数据存在证明观察到消息。设备身份、发令和反馈语义、顺序、动态覆盖及误差区间依赖实证。 |
| AUXILIARY_DIAGNOSTIC | diagnostics、EE pose/twist/wrench；无唯一依赖的 CollectionStatus 等采集显示 | 记录、定位和修复版本差异。若某支线以其中数据作为唯一响应或安全证据，则按该主张升级为必需。任何失败和被隔离的样本均保留。 |
| OUT_OF_SCOPE | camera/image、视觉 obs，以及未审定的其他通道 | 保留 catalogue 及 SHA，不解码视频。后续若作为来源或合法性证明须提请审定。 |

本轮七个候选录制的来源、完整性及观测证据来自原生记录协议与独立侧边材料，均无 CollectionStatus 依赖。四条受影响 merged 录制来自诊断队列。恢复其 Collector 状态只改善兼容性和排障。未确认的人类来源及已知仿真仍分别隔离或排除。新录制必须声明自己的条件依赖，不能复用七条原生样本的证明。
