# P02 硬件前置证据与就绪度

**物理 robot-base 注册、C2-RT、P01-T 均为 BLOCKED。** 本次完成源码取证和离线合同对照。没有新的实物测量、现场采集、系统辨识或机器人操作。P03 启动仍需上述三项与 P02 分别 ACCEPTED，再取得 P03 启动授权。独立 Final Test 按原封存要求保留。[阶段依赖](../dcr/DCR-002_simulation_and_real_clock_qualification.md) · [就绪计划](../specs/HARDWARE_SHARED_MEASUREMENT_PLAN.md)

## 1. OBSERVED 只读取证

读取来自 Ubuntu /home/ubuntu/teleop_ws 的 19 个选定源码或消息定义。来源工作树分支 4sensor，HEAD / SHA-1 为 30cf13bdc496c43b3016462cfa0a206334d1ff7f，工作树有修改。对全部 19 个文件复算 SHA-256 和字节数，与快照取证清单一致。其中 5 个与该 Git HEAD 不同。ZIP SHA-256 为 5ff7f9a7dd6d6a29074ab96b12171d29ef935442b2b2558d13b4b8211ed7e4c0，取证清单 SHA-256 为 8d03a59fa99e10508f8020abfe2eee0466b143d47d7c0109f7902b3dc1763c39。下文源码位置取该实际选定工作树字节。现场加载二进制、参数、消息 type hash、固件和源码对应性仍未核实。[文件 SHA 与复演命令](../evidence/P02/final_handoff/hardware_evidence.json)

本地 Whole Body Teleop 镜像用 git rev-parse HEAD 和 git status --porcelain=v1 --untracked-files=no 做了独立读取。该镜像同 HEAD，跟踪文件干净，包含控制器、重定向、Wuji C++ 封装和仿真驱动，可为函数意图提供辅助源码。Ubuntu 快照额外提供实物臂和手驱动、MANUS 与 VIVE 源码，优先用于当前审计。被选定的控制器及臂驱动诊断源码与 HEAD 有差异，不能用本地精简镜像推断这些运行能力的实际装载。

## 2. CODE 现有信号语义与断点

路径相对于选定的 Teleop 源码工作树。源码断点按函数和行号标记，接口出现仅证明代码具有对应字段。

| 环节 | 源码见证 | 当前可用证据与缺口 |
|---|---|---|
| MANUS 捕获及发布 | src/manus_ros2/src/ManusDataPublisher.cpp:370–375、594–607 和 src/manus_ros2_msgs/msg/ManusGlove.msg | 记录 SDK stream publishTime.time 原值、回调中 host steady 接收，以及发布时的 ROS header。设备实际物理采样时间、独立锚点、设备到 host 的传输或采样上界仍缺证。原 packed source 字段不能直接换算成跨域物理时间。 |
| VIVE 捕获及注册 | src/vive_tracker_ros/src/vive_tracker_node.cpp:173–205，tracker_pose_transformer_node.cpp:62–76，world_adapter_node.cpp:8–11 | 查询 OpenVR 后以节点时间填 header。注册副本继承该 header 并换坐标系，固定 TF 声明 OpenVR 到 teleop_world。TrackerPose.msg 只有 header、pose、role、valid 和 quality。需证实设备采样或曝光区间、稳定源序列和重定位监控。固定软件变换没有外部机械基准的独立测量。 |
| 手部重定向 | 本地辅助源码 src/whole_body_teleop/src/retarget_node.cpp:136–168 和 Ubuntu 的 HandCommand.msg | 本地参考在 callback 取得 steady 值，基于发布 header 查新鲜度，发布含 20 轴名称、位置、side、sequence 的命令。当前 Ubuntu 重定向二进制和源映射需复核。命令需要与 MANUS 实际采用 UID 和 driver trace 增补联结。 |
| 控制器接收和周期 | src/whole_body_teleop/src/teleop_controller_node.cpp:1004–1018、1507–1545、2008–2022、2032–2051、2595–2610 | 两侧 Tracker 当前取锁后再读接收时间。快照持锁复制状态。周期滚动诊断具有 steady 开始、状态、世代、序列和是否发布。缺单独入口和缓存可见区间、四路实际采用 UID 与 tick/cutoff 的同一证据，以及准确的实际 publish 调用时界和设备答复。周期批次 header 单独是诊断发布时间。 |
| 双臂指令及反馈 | src/tianji_real_driver/src/real_driver_node.cpp:644–698、1036–1049、1067–1094、1523–1534、1595–1617、1643–1660，及 include/tianji_real_driver/command_trace.hpp | 代码支持有界接收、覆盖、过期、提交、失败和读回 trace，保存 session/sequence、14 轴裁剪、SDK frame、反馈、reported command 和原始动态字段。接收 trace 在准入逻辑内产生。读回行借用最后提交的序号，原作者明确其为软件关系。SDK position/velocity 按配置转换到逻辑 rad、rad/s。current/torque/friction/disturbance 保持原始量，其单位和语义未验证。JointState header 取发布时刻，实际设备采样和接收需另存。设备应用/同步锚点、实际响应的强连接仍缺证。 |
| 双手指令及反馈 | src/wuji_hand2_real_driver/src/node.cpp:89、209–231、634–648、708–721、803–817，及 src/manus_wuji_interfaces/msg/HandState.msg:14–32、HandDiagnostics.msg:8–16 | 命令检查命名/序号，驱动接收取 steady。SDK frame 在接收通路使用 ROS/steady 并可上报源时钟、序列和有效位。applied_command 是驱动送出内容的镜像，dry-run 也会发布，其 JointState 缺上游 sequence 和设备 ACK。各字段位置、速度、effort、在线、stale 和年龄语义独立，effort 实物单位可能为 sdk_unspecified。设备真实应用语义、采样界、生效 ID 和动态覆盖待证。 |
| 反馈腕与机械基座 | src/whole_body_teleop/src/teleop_controller_node.cpp:1145–1197 | 腕位姿从关节反馈做 FK，速度由相邻帧推导。几何外测、独立的基座机械 datum、TCP/掌装配与实际工作范围误差仍缺证。 |

将 54 轴的命令链分成原始请求、最终发布、driver 接收、SDK 提交、实际已应用及独立设备反馈。表中闭环 trace 已具有可利用的工程起点。尚无真实设备端可验证的命令应用时间或所有必要链条的序列语义，故不能从相近的 ROS stamp 推导真实控制延迟或完成 Servo 辨识。运行时代码、wrapper 和协议的物理含义各自由现场证据确认。[P01-T 合同](../specs/P01_T_telemetry_acquisition.md)

## 3. OBSERVED 合成检查

在 Windows 只读副本上用 Python 3.14.3、NumPy 2.5.4 执行，返回 0。完整手工复演命令、原代码、19 个文件哈希及实际输出见机器记录。试验只验证确定性数学和证据连接示例，数值不属于 P03 资格参数。[离线机器记录](../evidence/P02/final_handoff/hardware_evidence.json)

| 项目 | 实际结果 |
|---|---|
| 几何 | 合成保留点最大误差 7.73e-17 m，拟合旋转行列式约 1。留出目标施加正负 3 mm，恢复 +3.000 / −3.000 mm 有符号残差并超出夹具阈值。镜像数据的 proper 刚体拟合仍有约 227.44 mm 最大残差。 |
| 时序 | 8 个用例。原值、−30 ns、等于 cutoff 均通过合成检查。+30 ns 超 cutoff、捕获上界晚于 tick、换代、缺样本和反向时间边界按原因阻止。 |
| 事件连接 | 6 个用例。强 ACK 且前向匹配通过。相同时戳错序号、缺 ACK、应用晚于反馈、只有 SDK 提交和旧世代均被拒绝。 |

## 4. 三个独立验收项

| 门 | 已有证据 | 关闭 BLOCKED 所需证据 |
|---|---|---|
| 物理 robot-base 注册 | 软件世界变换和仿真 FK。没有独立测量数值。 | 现场机械 datum 及同一外测参考下的左右基座、Tracker、TCP/掌变换，留出姿态和跨 session 的残差、不确定度、覆盖与装拆影响。独立技术报告及现场／项目审批。 |
| C2-RT | 已签核历史设计仍为 0 个 VERIFIED 控制拍。本轮无新物理计时。 | 人端四路实际采样及控制器可见上下界、所用 input UID 与 tick/cutoff、独立配对锚点、下发／反馈和误差区间。给出异常时钟世代、不可观察链段和各源真实覆盖。 |
| P01-T | 既有 37 个去重视图缺真实硬件来源证明，动态辨识资格为 0。本轮新证据是源码。 | 真实硬件、SDK/固件和轴语义，逐层有效命令、可证应用／反馈关联与时间证据，54 轴静／动态覆盖和独立保留批的辨识验证。 |

## 5. DESIGN 下一步申请的具体范围

先由现场负责人和项目用户独立审批无运动工位勘查与被动记录。绑定执行人、监护人、物理急停、工位、工具与量测仪器、访问窗口、次数及停机准则。对应计划包含独立机械测量、事件锚点和数据选择。若后续需动态响应，另提交 A3 的试验包络和执行许可。所有门先独立实测，再人工签核。本报告及合成检查不产生硬件许可或验收。[联合测量计划和申请模板](../specs/HARDWARE_SHARED_MEASUREMENT_PLAN.md)
