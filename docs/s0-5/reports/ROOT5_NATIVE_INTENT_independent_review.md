# 第 5 根原生事件与人体功能手型独立复核

**固定腕部原生恢复与独立功能手型诊断通过复核。完整 54 DoF 保持 `HOLD / NOT_RUN`，按用户要求在七个冻结窗口与独立审查后停止。** 范围为 `DIAGNOSTIC_ONLY`，覆盖计分资格为 `false`。复核基准为 DexGym `891aa5cffdf095c2669cfb17d3a6d7ef29a5615a`、原生 Teleop `6d0b0b5b2601618f1b0052d93ef2403eea9b064a` 和 Train 根 `capture:b63bc72d2e8b2ee395e966f6`。

## 原始输入与采样时钟

复核重新计算了上轮 25 个归档成员的 SHA-256，六个原生采样单元检查全部通过。全根沿用 13 个原始分区，计划／联合有效／五帧候选分别为 4,844／4,831／4,741。START 后分别为 4,804／4,794／4,708。采集器单调时钟与固定 HMB 8 ms 可用延迟沿用原定义。历史 RetargetNode 调度、DDS 到达、实际控制许可与机器人回读的状态为 `UNOBSERVED`。 [上轮输入回执](../campaign/evidence/teleop-parity-root05/event-audit.json) · [上轮独立复核](ROOT5_TELEOP_PARITY_independent_review.md)

只读验证程序在隔离 Ubuntu 运行并返回 0。它重新计算两份 MCAP 的哈希、从 13 个存储分区的附属账本恢复原事件字节，并核对 23,242 条全部写入。重复 SDK 源时间戳的不同 writer 事件各进一次原生回调。C 以两个新会话加载固定 SDK 和归档 A 的已核验编译探针，会话连续跨越普通分区。A/C 的 21 点选取和单精度 Z 旋转（L2）、SDK 原始关节（L3）、输出滤波（L4）逐值相等。两份完整输出逐字节一致。验证程序在 48,435 个 500 Hz 预驱动网格点重算左右独立零阶保持，再核对全部 4,844 个签名 50 Hz 行。左右各有 4,842 个可用样本，签名时刻相对所取 500 Hz 网格的最大延差为 96 ns。 [独立核算结果](../campaign/evidence/native-intent-root05/independent-recompute.json) · [原写入分区回执](../campaign/evidence/native-intent-root05/native-writer-event-partition-summary.json) · [全事件重放回执](../campaign/evidence/native-intent-root05/native-event-replay-summary.json) · [独立验证程序](../../../scripts/s0_5_campaign_teleop/verify_native_intent_replay.py)

## 固定腕部执行与安全

A 使用归档的密集原生事件输出，C 使用重新执行得到的密集输出，两者进入同配置的独立驱动和签名 Servo 实例。B 使用归档的 HMB 选帧原生路径，D 对 C 增加现行 V2 准入。驱动适配器编译固定 Teleop 对象里的接纳、保持和 watchdog 方法。它以签名 Home 开窗，按原写入时标接纳命令，执行有名执行器裁剪、0.5 s 保持、1.0 s `smoothstep` 渐入和 0.1 s 命令超时。首次预检用例把右手超时锁存错误地绑定到晚一拍的实测状态。修正侧别独立的测试断言后，裁剪渐入、侧别独立、重复／过期／未来命令和一次性实测保持均通过。复核重新计算了三个 C++ 提取片段与适配器的哈希。物理执行使用签名 500 Hz Servo、0.500 mm 净空、0.002 mm 证明数值余量及原速度／力矩门。 [初次预检记录](../campaign/evidence/native-intent-root05/driver-preflight.log) · [原生接纳](../campaign/evidence/native-intent-root05/native_hand_accept.inc) · [原生 watchdog](../campaign/evidence/native-intent-root05/native_hand_watchdog.inc) · [固定窗口执行回执](../campaign/evidence/native-intent-root05/fixed-arm-native-summary.json)

本次复核逐行重算了 4,804 个 START 后行和每个 2 ms 时标。窗内为 615 拍，窗外的 4,189 拍记为 `NOT_EXECUTED`。重建控制截止比精确签名 HMB 截止晚 904–1,000 ns。重建控制采样需要此前整段十个无故障实测子步。精确签名采样读取同窗、时间不晚于该签名拍、年龄小于 2 ms 的最新无故障实测端点。 [逐拍紧凑账本](../campaign/evidence/native-intent-root05/fixed-arm-status.jsonl.gz) · [固定窗口采样逻辑](../../../scripts/s0_5_campaign_teleop/run_native_event_hand_parity.py)

| 路线 | 完整拍／部分拍 | 实测物理步 | 精确签名反馈拍 | 手轴速度峰值 (rad/s) | 实测最小间隙 (mm) |
|---|---:|---:|---:|---:|---:|
| A | 564／2 | 5,652 | 561 | 3.575565 | 0.802463 |
| B | 583／3 | 5,851 | 580 | 3.668398 | 0.781112 |
| C | 564／2 | 5,652 | 561 | 3.575565 | 0.802463 |
| D | 564／2 | 5,644 | 560 | 3.575565 | 0.802463 |

四路的物理事件、手轴速度超限、命名接触和穿透均为 0。手轴速度门为 5 rad/s，容差为 0.00002 rad/s。窗内 6,150 个 A/C 计划子步的状态和共同字段一致。5,654 个可比较的来源与驱动目标相等，在 5,652 个实测子步上下发、应用、回读位置、速度和力矩均数值相等。C 同一快照上的共同／额外 V2 结果为 5,440 次 `PERMIT/PERMIT`、3 次 `PERMIT/SWEEP_UNPROVEN`、148 次 `COLLISION/COLLISION`、22 次 `COLLISION/SWEEP_UNPROVEN` 和 41 次 `SWEEP_UNPROVEN/SWEEP_UNPROVEN`。额外门新增的三个拒绝均为查询范围内未获证明。D 自己的递归轨迹留作闭环结果，独立比较发现 28 个可配对的实时快照已与 C 分岔。 [逐子步独立核算](../campaign/evidence/native-intent-root05/independent-recompute.json)

六类动作代理各有九个冻结观测拍，A/B/C 同拍完整物理比较实际覆盖左右不对称、自然张手和展开各九拍。握拳、单指弯曲和拇食对捏的代理拍保持原计划状态及缺测记录。相应的静态功能手型见证另有账本。 [实测动作代理配对](../campaign/evidence/native-intent-root05/motion-comparison.json)

相邻且同窗的实测回读速度以 2 ms 差分。A/C 的 5,645 个手轴最大加速度幅值样本一致，P95 为 27.14 rad/s²，峰值为 62.76 rad/s²。加速度结果的口径为描述统计。视频哈希、2,402 帧记录和 A/B/C/D 的 283／293／283／283 帧实测画面计数与运行回执相符。 [加速度描述统计](../campaign/evidence/native-intent-root05/hand-acceleration.json) · [视频回执](../campaign/evidence/native-intent-root05/video.json)

## 人体功能几何与可行路径

人体目标来自上一轮已逐点与签名 HMB 核对的 `raw_manus_local`，由原始四点指链计算单位骨向、指尖差向量和间距，用行列式为 +1 的掌系旋转重表达。原尺度点距由刚体旋转保持。冻结形态按原评分文件读取。独立拟合用原 MANUS、签名 FK、模型关节和驱动范围，以签名 Home 为种子，其输入与 Wuji 输出无关。复核逐行核对了全根 4,844 行，并按原几何键重建全部 187 条右手拇食近捏合行。 [原始几何回链](../../../scripts/s0_5_campaign_teleop/audit_root_five_geometry_layers.py) · [独立拟合与证明代码](../../../scripts/s0_5_campaign_teleop/functional_hand_intent_candidate.py) · [几何账本](../campaign/evidence/native-intent-root05/functional-witnesses.jsonl.gz)

实际运行显式选择 `--pinch-proof-samples 187`。187 个近捏合和六个动作代理中心有一个重叠，共 192 个不同姿态，签名 MuJoCo 3.13.0 路径均获许可。187 个右手见证均满足评分点距 ≤20 mm 且向量误差 ≤5 mm。右手拇食向量误差 P95 在冻结形态、独立拟合、许可见证上分别是 29.415、0.545、1.593 mm。签名证书的最小端点间隙是 0.646382 mm。中间留证包括 120 次端点不许可和七次查询耗尽，均保留为 `UNKNOWN` 尝试。另 40 个代理姿态完成拟合，状态为 `NOT_SELECTED_FOR_PROOF`。拇中捏合与连续快速开合保持 `INSUFFICIENT_INPUT_EVIDENCE`。 [功能目标与证明回执](../campaign/evidence/native-intent-root05/functional-summary.json)

本地补充复核以相同哈希的模型和依赖在 MuJoCo 3.17.0 重新验证 192 个端点、全部 12,621 个原包络盒的关节范围和腕耦合条件，以及本地具名 FK 评分点。再取九个等距近捏合索引、六个代理中心、最低间隙端点和最长路径，共 16 个不同姿态，重新证明其中 983 个包络盒，全部获许可，最低证明采样间隙为 0.642798 mm。该复核以静态签名几何为对象，3.13.0 原证书仍是本轮资格依据。评分点距、碰撞几何间隙和 500 Hz 实测接触分别保留。 [完整见证尝试与路径](../campaign/evidence/native-intent-root05/functional-witnesses.jsonl.gz) · [签名模型清单](../campaign/evidence/dcr009-stage-two/model_frame_receipt.json)

## 保护边界

基线到本轮的变更限于独立诊断代码、测试和证据。签名模型依赖、原运行回执、HMB 与原 MANUS 数据按原哈希核对，官方 85%／70% 覆盖契约以原签名来源为准。本轮所有实际运行指向隔离工作树，日常工作区的前后快照状态为 `UNOBSERVED`。 [运行修订与输入输出哈希](../campaign/evidence/native-intent-root05/runtime-checksums.txt) · [主结果报告](ROOT5_NATIVE_INTENT_results.md)
