# 第 5 根原生事件恢复与独立手型功能几何

范围为 `DIAGNOSTIC_ONLY`。本轮工作基线为 DexGym `s0-5` 的 `891aa5cffdf095c2669cfb17d3a6d7ef29a5615a` 和原生 Teleop `codex/whole-body-sim-phase1` 的 `6d0b0b5b2601618f1b0052d93ef2403eea9b064a`，两者均由远端分支核对。保留第 5 根 `capture:b63bc72d2e8b2ee395e966f6` 的 4,844／4,831／4,741 原始分母及 START 后的 4,804／4,794／4,708。物理运行范围为上一轮冻结的七个固定腕部窗口，共 615 拍。既有签名模型、Home、0.500 mm 净空、物理 5 rad/s 手轴速度上限、硬关节限制和 85%／70% 契约均保持原值。

[上一轮 Parity 证据](ROOT5_TELEOP_PARITY_results.md) · [上一轮独立审查](ROOT5_TELEOP_PARITY_independent_review.md)

## A．原生事件和驱动行为

使用上一轮直接从已签名 MCAP 解码的两侧原始写入记录。完整帧记录含 source instance、writer sequence、源时间戳、采集器单调时钟、原始消息标识、侧别、完整骨骼、拓扑及有效标志。附属的分区映射按签名 HMB 存储边界连接录制事件，每个原事件保留原始序列。普通分区边界只提供存储身份。历史 RetargetNode 的真实回调和 DDS 到达顺序仍无观测，执行身份为 `RECORDED_EVENT_REPLAY`。

独立重建的左右 Wuji session 各消费 11,621 条原始写入事件，保留相同 SDK 源时间戳对应的不同写入。原 C++ 选点、21 点顺序、float32 的 ±15° Z 旋转、原 SDK 与原 `CommandSmoother` 全部复用。两个手的 L2、L3、L4 在每侧各 11,621 个对应事件上与 A 逐值一致，完整恢复输出文件与 A 按字节一致。50 Hz 签名采样从原生事件输出已有的 500 Hz 零阶保持状态取样，配对行还保留了原 HMB 身份。

固定腕部执行时，A 的原生参考和 C 的独立完整事件重放分别供给隔离实例，B 保持 HMB 缩减的原生 Wuji 历史，D 在 C 路线上增加当前 V2 检查。驱动适配器从固定 Teleop Git 对象提取原 `MujocoBackend::accept_hand`、`hold_range_locked` 和 watchdog，直接编译原实现。隔离装配在每窗首个物理控制拍设置模拟使能及签名 Home。每个物理子步按原写入时间消费当时已可用的输出。原生驱动进行命名执行器限幅、0.5 s 保持、1 s 平滑渐入和 100 ms 超时锁存。固定腕部路径移除了原 C/D 手部 Python 外部滤波和每 4 ms 的限幅处理。每条记录保留 SDK raw、SDK filtered、driver target、下发、应用、测量及实测速度。

共同的物理裁决仍对签名全身状态使用严格连续证明，任何拒绝需对保持指令复证。额外 V2 在 C 的同一候选和同一状态快照上留独立影子判定。允许继续的物理步在 500 Hz MuJoCo 上执行并验证碰撞、速度和力矩。输出分开记录 1 µs 重建控制截止与精确签名 HMB 时刻的反馈可用性。精确采样只使用时间不晚于该签名时刻、年龄小于 2 ms 的已观测前序端点。

## B．独立人体功能几何

原始 MANUS 局部手型继续作为独立人体目标。只使用正交掌系旋转重表达单位尺度下的原指尖相对向量、指间距离、掌系骨段方向、握拳和张开变化。刚体转轴保留原点距。冻结形态目标作为对照保留，SDK 输出与工作流 A 继续独立。诊断型机器人形态候选在签名关节和规定范围上通过有限任务空间拟合求取，分别计算局部方向、指尖向量、张开尺度和形态误差。

目标有三个独立语义。`HUMAN_FUNCTIONAL_INTENT` 保存未缩放的人体功能数据。`ROBOT_MORPHOLOGY_TARGET` 在本轮诊断账本中由 `ROBOT_FUNCTIONAL_CANDIDATE` 表示，冻结旧版仍以 `FROZEN_ROBOT_MORPHOLOGY` 追溯。`ROBOT_FEASIBLE_TARGET` 仅对应 `KINEMATIC_WITNESS.status=SIGNED_HOME_PATH_PERMITTED` 的具名关节与目标几何。独立的 fit 掩码和见证状态为下游提供明确的可用范围。

有限搜索的机器人向量与连续 Home 路径见证分开保存。证书使用签名整身模型，保留所有碰撞对、净空、驱动范围、腕耦合和明确的数值余量。限定 20 mm 是原先的对捏评分点描述阈值。点距、实体几何净空和实测接触分别记录。未获证明或有限搜索未找到满足阈值的见证依旧保留原目标、几何差异与失败原因。无历史解剖腕掌标定的情况下，这些结果属于独立工程功能意图与签名仿真模型的范围。

## 固定腕部实跑

| 本轮对照 | 完整／部分物理拍 | 实际 2 ms 子步 | 手轴速度最大值 | 最小观测净空 |
|---|---:|---:|---:|---:|
| A 密集原生参考 | 564／2 | 5,652 | 3.576 rad/s | 0.802 mm |
| B HMB 原生参考 | 583／3 | 5,851 | 3.668 rad/s | 0.781 mm |
| C 原生事件恢复 | 564／2 | 5,652 | 3.576 rad/s | 0.802 mm |
| D C 加当前 V2 | 564／2 | 5,644 | 3.576 rad/s | 0.802 mm |

四组实际子步均无实测手轴超速、物理约束事件、接触或穿透。A、C 的 5,654 条进入裁决的 driver 和 source 记录逐值相同，已执行的 5,652 个子步的 issued、applied、q 和 qd 也逐值相同。独立复算从 4,844 个签名取样拍中为每侧找到 4,842 个有因果可用原生输出的拍，并从 C 模式的实际窗口找到 561 个精确签名时刻的因果反馈样本。前者属于整根 L4 原生名义流，后者属于限定窗口的已执行反馈。

同窗连续实测 2 ms 速度差分给出手轴最大绝对加速度。C 的 P95／峰值为 27.14／62.76 rad/s²，和 A 一致，B 为 54.78／172.79 rad/s²。各组以自己实际观测到的连续子步为统计域。加速度单独报告，未新增正式验收阈值。 [实际速度差分与每轴统计](../campaign/evidence/native-intent-root05/hand-acceleration.json)

C 的同状态候选中，共同证明和旧 V2 同时许可 5,440 条。另有 3 条共同证明许可但新增 V2 未获证明。C、D 都有窗口因为保持未获证明或连续 16 次未获证明而终止。D 最终观察的子步比 C 少 8 个。候选状态的净空／穿透判定、证明搜索未完成、已执行的物理净空分别记录。未执行的区间保留 `NOT_EXECUTED`。

密集恢复后 C 有 897 个非零下发子步，稀疏 B 有 377 个。六个原动作代理的冻结代表片段中，自然张手、张指和左右不对称各有 9 个共同完成拍可以成对看后端 FK。自然张手的右手非拇指平均卷曲代理误差中位数由 B 的 0.111 变为 C 的 0.064。其余三个代表片段没有 B/C 共同完成拍。拇中对捏和快速反复开合仍缺满足既有代理标签的源窗口。运动细节改善的结论限于同拍实际可观察区域。

## 静态手型见证

保留 4,844 行原始手型和旧几何掩码。对 187 个右拇食近捏合拍以及六种有源证据的动作代理运行独立功能几何拟合。明确使用 `--pinch-proof-samples 187`，对全部 187 个近捏合拍和六种动作的中心拍尝试连续证明，共有 192 个不同拍。所有 187 个近捏合拍均取得完整签名模型的 Home 路径见证，且评分点距离 ≤20 mm、向量偏差 ≤5 mm。右拇食向量偏差 P95 从冻结旧形态的 **29.42 mm** 变为普通几何候选的 **0.55 mm**、已取得证书的见证 **1.59 mm**。六个有源证据的动作中心也分别得到满足各代理定义的合法静态见证。每个非证明拍的 witness 掩码、未证明的中间候选及搜索结果分别保存。

这是独立人与机器人局部手型之间的静态几何证据。完整连续 54 DoF 跟踪、物理解剖掌系和人与机器表面接触的资格仍需各自验证。两条工作流在视频和逐拍索引对齐，B 的几何输出没有进入 A 的控制器。

## 状态与复现

本轮执行止于固定腕部试验和独立审查。第 5 根完整 54 DoF 回放保持 `NOT_RUN`，等待人工审核。进入该回放仍要核验完整 START、双腕动态、模型与同步输入，并保留已观测到的证明拒绝。六根和全量 RMB 分别依赖后续实际资格及批准。本轮没有正式训练、真实部署或正式 85%／70% 计分。

Ubuntu 运行目录为 `/home/ubuntu/dexgym_data/s0-5/campaign/S05-V1-20261127-490770e/runs/NATIVE-EVENT-INTENT-ROOT05-01`。完整原始 MANUS 附属文件为 `partitions/native-writer-events-by-partition.jsonl.gz`，已从该文件无损重建原 writer stream 并逐条验证。完整名义 500 Hz、50 Hz 流和物理、功能几何分别存于 `replay`、`physics`、`functional`。这些运行大文件留在 Ubuntu，Git 包括验证回执、紧凑逐拍状态、几何见证和视频。启动使用固定 Python 环境、签名资产及只读原始归档，实际命令与各工作树、原始数据及输出 SHA 随证据发布。

### 执行命令和代码身份

使用 `/home/ubuntu/miniconda3/envs/manus_wuji_real/bin/python`，原生工作树为 `/home/ubuntu/dexgym_worktrees/teleop-sim0-study`。以下在各自目录以该工作树的 `source` 为 `PYTHONPATH`。 `OLD` 指向同一 Campaign 的 `TELEOP-PARITY-ROOT05-01`，`B0` 指向 `DCR010-OFFLINE-RMB-ROOT05-02`。每个输出路径以新的空目录开始运行。

```bash
# 事件恢复在 59ce18a 的 dexgym-event。
"$P" -Bu scripts/s0_5_campaign_teleop/native_event_replay.py \
  --teleop "$T" --events "$OLD/input/manus-events.jsonl" --ticks "$OLD/input/hmb-selections.jsonl" \
  --event-audit "$OLD/input/event-audit.json" --archived-a "$OLD/wuji/wuji-a-updates.jsonl" \
  --archived-a-summary "$OLD/wuji/wuji-summary.json" --bridge "$OLD/wuji/build/libparity_wuji_probe.so" \
  --out "$NEW/replay"

# 固定腕与几何在 478520b 的 dexgym-windows。
"$P" -Bu scripts/s0_5_campaign_teleop/run_native_event_hand_parity.py \
  --baseline "$B0/receipt.json" --teleop "$T" --model "$M" --model-receipt "$MR" \
  --windows "$PAR/window-plan.json" --sdk-summary "$OLD/wuji/wuji-summary.json" \
  --sdk-ticks "$OLD/wuji/wuji-bc-ticks.jsonl" --native-a "$OLD/wuji/wuji-a-updates.jsonl" \
  --native-c "$NEW/replay/wuji-c-updates.jsonl" --out "$NEW/physics"
"$P" -Bu scripts/s0_5_campaign_teleop/functional_hand_intent_candidate.py \
  --baseline "$B0" --geometry "$PAR/geometry.json" --window-plan "$PAR/window-plan.json" \
  --input-ticks "$OLD/input/hmb-selections.jsonl" --model "$M" --model-receipt "$MR" \
  --pinch-proof-samples 187 --out "$NEW/functional"
"$P" -Bu scripts/s0_5_campaign_teleop/link_native_raw_event_partitions.py \
  --events "$OLD/input/manus-events.jsonl" --ticks "$OLD/input/hmb-selections.jsonl" \
  --event-audit "$OLD/input/event-audit.json" --out "$NEW/partitions"
```

原生驱动初次预检的测试断言要求已超时的右侧再次锁存测量值。原生实现只在该侧超时触发时锁存一次。保留失败日志，修订测试以核验先锁存的右侧保持其值和随后超时的左侧锁存新测量值。实际仿真前的七项原生驱动控制及六项回放单元控制均通过。

[完整原始 sidecar 回执](../campaign/evidence/native-intent-root05/native-writer-event-partition-summary.json) · [原生 Wuji 等价回执](../campaign/evidence/native-intent-root05/native-event-replay-summary.json) · [七窗物理回执](../campaign/evidence/native-intent-root05/fixed-arm-native-summary.json) · [逐拍状态](../campaign/evidence/native-intent-root05/fixed-arm-status.jsonl.gz) · [配对动作细节](../campaign/evidence/native-intent-root05/motion-comparison.json)

[功能几何摘要](../campaign/evidence/native-intent-root05/functional-summary.json) · [全部见证和掩码](../campaign/evidence/native-intent-root05/functional-witnesses.jsonl.gz) · [四栏视频](../campaign/evidence/native-intent-root05/root05-native-event-and-intent.mp4) · [视频回执](../campaign/evidence/native-intent-root05/video.json) · [运行和数据 SHA-256](../campaign/evidence/native-intent-root05/runtime-checksums.txt) · [归档 SHA-256](../campaign/evidence/native-intent-root05/evidence.sha256)

[全量逐行独立复算](../campaign/evidence/native-intent-root05/independent-recompute.json) · [独立审查](ROOT5_NATIVE_INTENT_independent_review.md)
