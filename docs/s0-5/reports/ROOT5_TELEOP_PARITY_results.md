# 第 5 根原始 Teleop 与离线 RMB 等价性审计

范围为 `DIAGNOSTIC_ONLY`，固定 Train 母源 `capture:b63bc72d2e8b2ee395e966f6`。启动时核对 DexGym `s0-5` 为 `70963eb6d41a4716a4f3fe519bbdb50a67feb5ce`，默认分支 `main` 为 `e7f747e16e83aea4f8e6549e216cdd751bede229`。原生运行基准采用 Teleop `codex/whole-body-sim-phase1` 的远端 HEAD `6d0b0b5b2601618f1b0052d93ef2403eea9b064a`，该仓库默认分支 `main` 为 `f197d5eaf11bafbaa63cdafde1d589128f36cca6`。Ubuntu 的原生隔离 worktree 与工作分支一致，日常 `/home/ubuntu/teleop_ws` 的 HEAD 为 `30cf13bdc496c43b3016462cfa0a206334d1ff7f`，存在原有改动。审计只读日常工作区。

录制母源的采集质量记录显示控制器为 `IDLE`、硬件为 `DISCONNECTED`，没有原运行的臂手命令、回读或 RetargetNode 实际回调时钟。A 使用原生代码及采集器观察顺序重建输入，经隔离的签名 MuJoCo 对照。结论不延伸到历史 DDS 到达顺序、真实驱动状态与真机表现。

## 信号链

| 层级 | 原生路径 | 当前离线路径 | 配对语义 |
|---|---|---|---|
| L0 | MANUS 左、右话题，原消息及语义元数据，发布约 120 Hz | HMB 50 Hz 因果选帧，保留原 source instance、writer sequence 及签名可用边界 | MCAP 内容、观测和 HMB 原位姿逐条绑定。采集器观察时钟与 ROS callback 分开保存。 |
| L1–L2 | 按节点元数据构造腕及五指 21 点，先转 float32，再使用左右 +15°／−15° Z 旋转 | 由 `order_for_sdk`／`sdk_keypoints` 获取对应点，float64 旋转后 cast | 相同 writer 消息比较有序节点和逐点坐标，保留无效 mask。 |
| L3 | 左右独立 `RetargetSession::step()`，通过输入侧别、时钟、新鲜度等条件时对该回调调用一次 SDK | `OfflineTeleopNominal.refresh` 对合格 HMB 拍中新 writer 身份调用生产 ctypes Wuji | 原二进制、初始状态和序列控制独立于 A/B 时序实验，不跨批次共享状态。 |
| L4 | CPP `CommandSmoother`，每次可消费回调更新，首条和非正时间差或至少 100 ms 的中断后重种 | Python alpha 0.30，首条播种，后续以新合格 source 更新，没有同一间隔重种规则 | 两条路径各有一个 SDK 外部平滑器。是否有额外延迟由输入调用节拍、初态和状态机共同决定。 |
| L5 | Teleop 驱动按 actuator alias 校验、裁剪和受启用混合／看门狗约束 | 从上一实际下发状态向 L4 推进，手轴每 4 ms 最大 0.02 rad，同时受关节限位和原生配对状态限制 | 固定臂实验保留原 L4 与离线裁剪，独立签名仿真器共同实施安全门。完整的原生驱动启用和 DDS 边界仍待核对。 |
| L6–L7 | 原生手部命令经控制权和驱动后执行 | 按有名传动定位到 54 轴，完整向量过共同连续证明；D 再加当前 V2 深度 18 准入 | C 同一快照上的 D 判定与 D 分支自己的闭环反馈分别记账。新目标与历史保持不混记。 |
| L8–L10 | 原生 MuJoCo 写应用目标及采实际反馈，录包未留对应事件 | 每 4 ms 下发，500 Hz 签名 MuJoCo，采应用、q/qd、命名接触及局部／世界 FK | FIXED ARM 为相同双臂 Home 命令。报告实测腕偏差，区分 20 ms 前缘目标和区间末反馈。 |

原 Teleop 源码检查涵盖 `retarget_node.cpp`、`command_smoother.hpp`、`wuji_sdk_cpp`、MuJoCo 和实际驱动的命令接口。这次执行仅使用隔离仿真和 SDK 编译适配，真实驱动和真实机器人不启动。

## 实验边界

A 在记录的写入节拍上重建 native callback 和输出，模拟按共同 4 ms 物理下发网格提取结果。B 用签名 HMB 样本、相同的 C++ 路径及相同 SDK 二进制。C 复用生产离线 refresh、去重、滤波和步幅接口。固定臂使手部比较不依赖 Marvin IK。D 使用同样的 C 提案规则和额外 V2 门。C 每个判定另记从相同状态盒得到的 D 影子结果，用来归因安全层，D 的递归轨迹只作实际反馈比较。共同执行门是有界连续证明，净空为原 0.500 mm，另保留原 0.002 mm 数值余量。每个 MuJoCo 子步检查接触、合法速度和力矩。所有阶段保留 HOLD、NOT_EXECUTED、原 4,844／4,831／4,741 以及 START 后 4,804／4,794／4,708。

## H3 的几何构成

原始 MANUS 点距对共同刚体旋转和平移不变。离线 `SimulationIntent` 额外替换了五指各自的掌部起点和三段骨长，沿人体记录的方向重建到机器人模型。右手食指尖相对拇指尖的向量包含两个变化项。第一项为两指近端起点的相对向量变化，第二项为各节长度变化乘对应的记录方向。掌心坐标注册后的刚体摆放不会改变这两个尖点之间的距离。正式独立目标和碰撞规则保持原值，四个离线对照只隔离计算误差来源。

在 4,801 个共同几何拍上，原 MANUS 有 187 拍右拇指–食指点距 ≤20 mm。只替换起点、只替换骨长或两者均替换时，该计数均为 0。联合变换重建归档目标的最大坐标偏差为 7.69 nm。SDK 的 21 点非拇指序列使用 PIP/IP/DIP/TIP，独立形态重建使用 MCP/PIP/DIP/TIP；本项点距比较只用两条路线都定义的 TIP。点间距离和实体碰撞网格净空分别计分。

上述 187 拍的右拇食点距范围分别为原始 12.20–19.94 mm、只换起点 58.37–71.04 mm、只换骨长 41.12–61.07 mm、同时替换 22.14–40.53 mm。两项改变的是向量，组合距离无须是两个独立距离增量之和。现有目标重建保留骨向，未保留该范围内的实际近捏合点距。

签名模型和 Teleop 当前已核验的模型之间，两套 54 关节名称、轴方向、范围和 40 手部执行器别名一致，碰撞目录分别有 2,542 和 3,454 对。62 个共同合法的同关节向量上，腕位置一致，单个局部手部指尖的最大模型差为 6.39 mm。这个静态偏差有助于说明几何评分差异的模型依赖性，不能代替手部联调或动态校准。 [形态和模型配对证据](../campaign/evidence/teleop-parity-root05/geometry.json)

## H1 与 H2 实际对照

原录包两侧各 11,621 条写入事件，观测节拍约 120 Hz，writer sequence 无间断或倒退。左、右分别有 757、1,940 个重复 SDK 源时间戳。重复源时间戳表示采集端的采样复用，线上实际控制节点是否因 DDS 调度丢帧未知。拓扑稳定。全母源 HMB 实际选中 4,828／4,677 条不同左／右原写入事件，分别有 6,793／6,944 条原事件未被这一采样链选中。加权同步、配对许可、缺失及 8 ms 模拟可用延迟继续沿用签名 HMB。

用 Teleop `real.yaml` 加 `sim.yaml` 叠加后，手部左右旋转 +15°／−15°，输出 alpha 0.30，输入年龄门为 100 ms。原版 C++ probe 和离线 ctypes 均加载同一份 Wuji SDK 2026.10.26。链接证据指向固定 Teleop 工作树，SDK SHA-256 为 `1bdc160c6c27a9b1bfa727dfb7c33c8b01fbb5acbefc8cfdfa17b2fd84c75282`。左右独立 session 使用原调用语义。测量以每手每拍的坐标最大绝对差或关节最大绝对差统计。

| 实测对照 | 左 | 右 |
|---|---:|---:|
| 相同原消息，语义节点索引或旋转前坐标不一致 | 0／4,794 | 0／4,794 |
| 原生 float32 旋转与 Python 后 cast 的 L2 最大坐标差 | 14.90 nm | 14.90 nm |
| 完全相同 L2 和历史，C++ 对 ctypes 的 SDK 原始输出最大差 | 0 rad | 0 rad |
| 完全相同 L2 和历史，外部滤波输出最大差 | 1.19e−7 rad | 1.19e−7 rad |
| 保留各自 L2 的原生 B 对离线 C，L3 误差 P95 | 1.99° | 1.98° |
| 从相同 START 消息重置，密集对 HMB 历史，同消息 L3 误差 P95 | 11.77° | 11.41° |

最后一行使用全新左右 session，先验证两边各自的 START 消息 SDK 输出按数值完全相等，然后只比较同一 writer 消息对应的原始输出。密集路线在各自 START 后分别处理 11,531／11,530 条消息。0.5×、1×、2× 原事件序列的重新计时控制保留了全部事件。在无超出重种阈值的记录条件下，各次 11,621 条／侧的 raw 和 filtered 输出均复现。因此，这组证据优先支持**输入事件历史差异**。小到一个 float32 量级的 L2 差异也可使有状态 SDK 后续走到不同解。只复现调用库文件，不能保证源点数值和会话历史一致。

## 固定腕部实际执行

首次物理尝试遗漏了原生驱动启用渐入。原样保留前 100 个计划拍的观测后停止了这次尝试。A/B 直接从 Home 请求整幅目标时，首先遇到整身证明的查询预算。C/D 的首个实际约束事件发生在候选已经获得许可之后。复验加入了两套 Teleop 配置中均存在的 0.5 s 保持、1.0 s 三次曲线渐入及 0.1 s 命令超时，在每个诊断窗口采用唯一、明示的使能时间。C/D 仍保留现行离线处理。驱动的历史使能、DDS、命令传输和回调时间仍未观测。A/B 在连续八次未获证明后停止对应窗口，其他安全规则保持原值。

以原始输入几何选出七个互不重叠窗口，共 615 个计划拍。各对照始终按原速执行。每个窗口从同一签名 Home 开始，使用实录窗口内含的前置动作渐入。窗外 4,189 个计划拍保留，均标记 `NOT_EXECUTED`。在既定的原始输入筛查条件下，自然张手、握拳、拇食对捏、展开、单指弯曲和左右不对称有代表窗口。拇中对捏和连续快速开合没有满足代理条件的记录窗口，状态为 `INSUFFICIENT_INPUT_EVIDENCE`。这些几何代理没有人工动作标签。

| 复验对照 | 615 拍内完整执行 | 部分执行 | 实际 2 ms 子步 | 观测穿透 |
|---|---:|---:|---:|---:|
| A，密集事件与原生 Wuji | 565 | 1 | 5,654 | 0 |
| B，HMB 与原生 Wuji | 584 | 2 | 5,852 | 0 |
| C，当前离线手部处理 | 28 | 7 | 323 | 0 |
| D，C 加当前 V2 守卫 | 28 | 7 | 323 | 0 |

A/B 的真实子步均无接触和物理事件，最小观测间隙分别为 0.800 和 0.781 mm。其部分拟议姿态或证明搜索样点出现穿透，原样拒绝并记具体层次。C/D 的实际最小间隙均为 0.922 mm，观测无接触，七个窗口均触发 `PHYSICS_EVENT:64`。在 C 的 165 个同快照候选中，两套证明都许可 151 次、都因查询耗尽未证明 14 次。本次额外 V2 对照未新增拒绝。因此，固定腕部的 C/D 停机在实际物理约束层已经发生。这个结论只适用于此次窗口及其共同模型，不能外推到原 54 DoF 历史执行中的守卫阻塞。

C、D 的全部十四个故障子步均有实测手轴超过 5 rad/s 加 0.00002 rad/s 容差的证据。首次出界关节为右拇指基部外展轴 `r_thumb_cmc_abd`，速度 −5.04186 rad/s，最小净空 1.118 mm。其余故障涉及左右拇指及右中指。结合 C/D 的每 4 ms 名义步幅上界 0.02 rad，可确认现有目标步幅限制不足以保证实际瞬时速度低于对应的执行门。这里保持原速度上限。 [逐故障速度见证](../campaign/evidence/teleop-parity-root05/velocity-witnesses.json)

实际执行中，A/B 的非零下发子周期分别为 896／377，保持子周期为 104／170。A/B/C/D 的命令源采集器年龄及有效样本数由账本独立报告。A 左右 P95 约 7.77 ms，B 左右约 30.76／35.11 ms。这是采集观察至仿真选取网格的时间差，各自包含实际有效执行子周期。真实 DDS、控制器调度和手指机械响应的端到端延迟仍无历史观测。签名 Servo 的手轴目标应用系数为 1，手轴在这一执行层没有新增低通。 [逐拍、逐命令状态与年龄](../campaign/evidence/teleop-parity-root05/trace-audit.json) · [动作代理分母和实测覆盖](../campaign/evidence/teleop-parity-root05/hand-slices.json)

主要条件判断为 **A/B 的原生输出路径可以部分执行，C/D 在这组固定腕部窗口中较早触发执行约束**。HMB 选帧、输入浮点处理、渐入和物理速度响应分别有独立差异。下一次修复提案应从保留原始 MANUS 事件供 Wuji 完整消费、精确复用原生坐标转换和 handoff 开始，把 50 Hz 采样移到原生手部输出之后，并使用原安全设定复查离线 4 ms 下发与实际 Servo 的速度响应。独立形态目标需另外审查其捏合保真度。当前固定腕部未达到开启第 5 根完整双臂运行的条件，完整 54 DoF 为 `NOT_RUN`。

四栏视频在原时间轴上展示 2,402 帧。A/B/C/D 的实测画面分别为 283／293／17／17 帧，缺测右栏为 `NOT_EXECUTED`。任一完整计分拍包含 10 次真实 MuJoCo 子步。窗口中实测腕部最大位置漂移分别为 0.102／0.105／0.682／0.682 mm，足以区分本试验的固定腕部命令与实际闭环腕状态。 [四栏视频](../campaign/evidence/teleop-parity-root05/teleop-parity-root05.mp4) · [视频回执](../campaign/evidence/teleop-parity-root05/video.json) · [原始固定腕部回执](../campaign/evidence/teleop-parity-root05/fixed-arm-parity-summary.json)

全部数值仍为诊断资格，正式 85%／70% 统计未引用此结果。原始数据、评分、硬限位、净空和模型保持既有身份。本轮执行止于第 5 根手部试验和审查，未启动全量 RMB、SuperMimic 或真机。

### 复现入口

原始与派生的原精度 JSONL 均位于 Ubuntu 隔离目录 `/home/ubuntu/dexgym_data/s0-5/campaign/S05-V1-20261127-490770e/runs/TELEOP-PARITY-ROOT05-01`。输入和输出哈希由运行产物绑定。下列命令从对应 DexGym 隔离工作树运行，`PYTHONPATH` 指向该工作树 `source` 和隔离依赖目录。

```bash
python -Bu scripts/s0_5_campaign_teleop/parity_manus_events.py \
  --cohort "$freezed_cohort" --out "$run/input"
python -Bu scripts/s0_5_campaign_teleop/parity_wuji_probe.py \
  --teleop "$native_workspace" --events "$run/input/manus-events.jsonl" \
  --ticks "$run/input/hmb-selections.jsonl" --out "$run/wuji"
python -Bu scripts/s0_5_campaign_teleop/parity_wuji_dense_anchor_control.py \
  --events "$run/input/manus-events.jsonl" --wuji "$run/wuji" --out "$run/evidence/dense-control.json"
python -B scripts/s0_5_campaign_teleop/plan_teleop_parity_windows.py \
  --input "$run/input/hmb-selections.jsonl" --out "$run/evidence/window-plan.json"
python -Bu scripts/s0_5_campaign_teleop/teleop_parity_hand_physics.py \
  --baseline "$diagnostic_baseline" --model "$signed_xml" --model-receipt "$signed_model_receipt" \
  --sdk-ticks "$run/wuji/wuji-bc-ticks.jsonl" --sdk-updates "$run/wuji/wuji-a-updates.jsonl" \
  --sdk-summary "$run/wuji/wuji-summary.json" --window-plan "$run/evidence/window-plan.json" \
  --native-handoff --out "$run/physics-native-handoff"
```

运行源身份分别为事件取证 `6a4784e`、原生 SDK 和形态数值 `eacdbcb`、相同 START 对照 `3cc3a3b`、有界手部执行 `5c08060`。运行与后处理退出状态均记录。首轮未完成的固定腕步进另存程序退出和真实证据前缀。 [原始事件和 HMB](../campaign/evidence/teleop-parity-root05/event-audit.json) · [原生 SDK 数值对照](../campaign/evidence/teleop-parity-root05/wuji-summary.json) · [相同 START 控制](../campaign/evidence/teleop-parity-root05/dense-control.json) · [首轮停止记录](../campaign/evidence/teleop-parity-root05/initial-physics-disposition.txt) · [落盘数据 SHA-256](../campaign/evidence/teleop-parity-root05/runtime-artifacts.sha256) · [归档证据 SHA-256](../campaign/evidence/teleop-parity-root05/evidence.sha256) · [独立审查](ROOT5_TELEOP_PARITY_independent_review.md)
