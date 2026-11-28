# Root 5 Teleop Parity 独立复核

复核范围为 Train 根 `capture:b63bc72d2e8b2ee395e966f6` 的隔离仿真诊断。DexGym 基线为 `70963eb`，Teleop 基线为 `codex/whole-body-sim-phase1` 的 `6d0b0b5b2601618f1b0052d93ef2403eea9b064a`。复核包括原始 MANUS、签名 HMB、回调与 SDK 接缝、固定臂目标的 A–D 运行及手型目标来源。

**54D 保留 `HOLD`。** 同输入的 Wuji L3 接缝通过一致性复核。固定臂物理对照中，C、D 的七个窗口均触发物理约束停止，观察到的同状态请求也没有额外 V2 门区别。两种任务动作在 ROOT05 缺少满足定义的输入窗口。结论限于诊断，覆盖资格为 `false`。

## 录制和因果时间

ROOT05 签名清单在 13 个分区内保留 4,844 个计划拍。原始清单只有两只 MANUS、两只注册 Tracker、两只原始 Tracker 和 Tracker safety 七个数据角色。原始 `SampleObservation` 给出采集端时标和 writer identity。历史 Teleop 回调开始、发布闸门、驱动接纳、历史机器人命令和实测状态均标记为 `UNOBSERVED`。先前 Teleop 回放的命令与反馈由当次仿真生成。 [原始数据清单](../evidence/P01_D/products/source_whitelist.json) · [签名计划](../evidence/P01_E_E1/production_plan.json) · [Teleop 回放记录](TELEOP_shared_rigid_reprocess.md)

原始提取逐文件验证字节数、SHA-256、嵌入 schema 和 MCAP CRC，按物理记录次序保留每个 MANUS 消息和后续相邻 writer observation。HMB 选中端点按原始 token、资产、CDR、writer identity 和时钟回链。有效姿态核对原节点顺序中的 SDK-common XYZ/XYZW、拓扑和因果边界。A 使用可重复的 recorder-monotonic 排序和开放诊断闸门。B、C 以已签名控制拍安排输入，B 在回调接缝即时重盖 ROS stamp，C 在同一截止拍调用离线刷新。固定臂对照使用同一 4 ms 下发网格，物理采样周期为 2 ms。末端反馈对应区间末尾，配对目标对应区间起点。 [原生记录提取](../../../scripts/s0_5_campaign_teleop/parity_manus_events.py) · [回调与 SDK 对照](../../../scripts/s0_5_campaign_teleop/parity_wuji_probe.py) · [固定臂对照](../../../scripts/s0_5_campaign_teleop/teleop_parity_hand_physics.py)

独立核对了初步归档 SHA-256 `805c253af255908c6bef3b8283fc829089057f7e2ee8656e619bafe7e71a12ff`、每个解包成员的字节和 MCAP 审计到 SDK 运行回执的输入哈希链。每侧均有 11,621 个不同 writer 记录和 CDR 摘要，重建回调全部到达发布接缝。两侧的采集端观测速率均约 120 Hz，最大的相邻间隔为 11.145／11.138 ms。SDK source stamp 左有 757 次重复，右有 1,940 次重复。全根计划／联合有效／五帧候选为 4,844／4,831／4,741，START 后为 4,804／4,794／4,708。HMB 的可用边界使用有记录的采集端单调时间加固定 8 ms。 [MCAP 审计回执](../campaign/evidence/teleop-parity-root05/event-audit.json) · [SDK 配置与运行回执](../campaign/evidence/teleop-parity-root05/wuji-summary.json)

## 逐层证据

| 层 | 对照与可观测范围 |
|---|---|
| L0 原始 MANUS 位置 | 原始资产保留全部节点及顺序。每个 21×3 选点可沿资产、物理记录、writer identity 和 HMB 端点回链。 |
| L1 语义选点 | Native 由固定回调的节点 ID 缓存选点。Offline 由签名拓扑按序取值。对比旋转前的同一 21 点位置。 |
| L2 绕 Z 轴旋转 | 对比原生 ±15° 路径与 Offline `sdk_keypoints` 浮点运算。注入同一 float32 输入的对照用于隔离此处。 |
| L3 Wuji 原始关节 | 每手 20 个弧度值。比较选点路径、同一 SDK 输入、会话和采样时序。 |
| L4 输出滤波 | Native 使用原 `CommandSmoother` 与回调接收间隔，Offline 使用自身递推与观察去重。对照单根会话和分区重置。 |
| L5 步长限制 | C、D 调用 Offline 的按手限幅和 4 ms 步长限制。A、B 对回调输出应用签名 actuator 的控制范围，四轨均受共同连续净空证明。 |
| L6 名义 54 维目标 | 固定 14 个 Home 臂命令。SDK 标签按签名模型各 actuator transmission 对应到 40 个有名关节。保留 float32 精确候选以及该次证明的历史下发、Servo 滤波和测量位置。 |
| L7 实际下发 | 记录准入目标或独立复证的既有目标。可观测为隔离仿真内的下发。 |
| L8 物理应用 | 每 2 ms 记录 Servo 应用的目标。臂驱动 `ctrl` 另包含重力补偿。 |
| L9 实测物理 | MuJoCo 每步记录位置、速度、观测间隙、接触对和事件。 |
| L10 几何 | 区间末尾从实测位置计算局部和世界指尖。双手拇指–食指功能点距与 mesh 间隙分别解释。 |

## A–D 对照判定

**A → B。** 两轨使用同一抽取的原生回调和 SDK。A 重放所有采集端 MANUS 事件并在 4 ms 网格取得最新发布值，B 只在签名有效拍推进新的 writer identity。对照包含采样集合、前缀预热、去重、输入闸门与间隔差异。A 的缓存采用诊断零阶保持，历史 ROS QoS、消息 lifespan、驱动 watchdog 和启动斜坡缺少历史证据。分区重置对照可说明人为会话边界的敏感度。 [原生回调接缝](../../../scripts/s0_5_campaign_teleop/parity_wuji_probe.cpp) · [采样与生命周期](../../../scripts/s0_5_campaign_teleop/parity_wuji_probe.py)

**B → C。** 比较相同签名输入的语义顺序、旋转后的 SDK float32 数组、同输入 L3 raw 输出、L4 滤波及 L5 限制。C 的身份使用 writer instance 和 sequence，与基线离线入口一致。不同滤波状态或 SDK 输入带来的滚动差异需与固定臂物理反馈分别解释。 [离线刷新与步长限制](../../../source/dexgym/tasks/direct/wholebody_s05/v1/rmb/offline_nominal.py) · [签名输入身份](../../../scripts/s0_5_campaign_teleop/home_anchored_start_anchor.py)

**C → D。** 每个 C 请求在改变当前 `issued` 和物理状态前，把同一份 float32 候选及同一预下发状态交给旧深度 18 verifier。该 same-snapshot shadow 与共同深度 24 证明可以逐请求比较。独立的 D 物理运行在第一次下发差异后拥有自己的反馈历史。`PERMIT`、数值或预算未证明、采样净空不足、穿透和实际接触分别可查。候选拒绝后需要为 HOLD 自身取证，现有模型碰撞对及净空均参加证明。 [共同及影子证明](../../../scripts/s0_5_campaign_teleop/teleop_parity_hand_physics.py) · [基线盒证明](../../../scripts/s0_5_campaign_teleop/offline_rmb_lockstep.py) · [V2 verifier](../../../scripts/s0_5_campaign_teleop/swept_live_guard_v2.py)

## L1–L4 数值复核

表中每个值先按手、按配对拍取各坐标或关节的最大绝对差，再统计分位数。每侧共有 4,794 个 START 后有效配对。Wuji `2026.10.26` 的共享库 SHA-256 与原生预检一致，`ldd` 指向固定 Teleop worktree 的 SDK 路径。两侧原生配置分别使用绕 Z 轴 +15°／−15°、100 ms 陈旧阈值和 0.3 输出滤波。 [SDK 数值与哈希](../campaign/evidence/teleop-parity-root05/wuji-summary.json) · [动态链接解析](../campaign/evidence/teleop-parity-root05/wuji.ldd) · [基线原生预检](../campaign/evidence/dcr010-offline-rmb/native-nominal-preflight.json)

| 配对量 | 左 | 右 |
|---|---:|---:|
| L1 语义节点与旋转前位置完全相同 | 4,794／4,794 | 4,794／4,794 |
| L2 B–C SDK 输入最大差 | 1.49 × 10⁻⁸ m | 1.49 × 10⁻⁸ m |
| L3 B–C 原始关节差 P95 | 0.03480 rad | 0.03451 rad |
| L3 C 注入同一 B SDK 输入后完全相同 | 4,794／4,794 | 4,794／4,794 |
| L4 同一 SDK 输入的滤波差最大值 | 1.19 × 10⁻⁷ rad | 1.19 × 10⁻⁷ rad |
| L3 A 取最新发布值与 B 的差 P95 | 0.22900 rad | 0.22906 rad |
| L3 A 与 B 恰好同 token 时的差 P95 | 0.21079 rad，781 拍 | 0.22125 rad，669 拍 |
| L3 B 每根一次重置与每分区重置差 P95 | 0.54597 rad | 0.07354 rad |

三档 0.5×／1×／2× 对同一原生事件序列重建会话后，左右全部 11,621 个 L3 和 L4 数组逐值相同。这个对照改动接缝收到的时间戳，保留事件和调用次序。B–C 的同输入正控支持把原始差定位到旋转浮点输入与有状态 SDK 的组合。A–B 的同 token 子集仍存在更大的差，完整 A–B 差异同时含输入选择、相位和会话历史。 [配对与速率对照](../campaign/evidence/teleop-parity-root05/wuji-summary.json)

进一步将密集轨和 HMB 轨重置到每手同一首条记录，双侧首个 L3 关节向量完全相同。密集轨随后处理 11,531／11,530 次更新。按 4,794 个同消息配对重算后，L3 差 P95 为左 0.20541、右 0.19905 rad。原生回调和 SDK 相同，差异在相同首状态下随所处理消息的历史出现。 [同重置锚点的密集／稀疏对照](../campaign/evidence/teleop-parity-root05/dense-control.json)

同输入控制在 `HandTap` 调用中把 `b_sdk_keypoints` 写入 SDK。配对记录里的 `c_exact_sdk_keypoints` 字段保存注入前 C 自身计算的坐标。读取控制的实际输入时，以 B 的 tap 数组为准。 [注入位置与记录字段](../../../scripts/s0_5_campaign_teleop/parity_wuji_probe.py)

## L5–L10 固定臂物理对照

较早的运行尝试从 Home 直接下发完整的 A、B 回调输出。后续按原始手型规划有限窗口，保留前次停止证据。每个窗口从同一个签名 Home 开始，SDK 序列沿用整根回放的历史。A、B 重建 0.5 s 初始保持、1 s 平滑交接、命名驱动限幅和 100 ms watchdog。A、B 的八次连续证明未成立触发本次诊断停止条件。C、D 保留离线手部的 4 ms 限制。初始 Home 保持、未下发候选和已执行动作均有独立状态，窗口之外保留计划行及 `NOT_EXECUTED`。 [窗口计划](../campaign/evidence/teleop-parity-root05/window-plan.json) · [初次运行记录](../campaign/evidence/teleop-parity-root05/initial-physics-disposition.txt) · [物理台账](../campaign/evidence/teleop-parity-root05/row-ledger.jsonl.gz)

已核对最终归档 SHA-256 `0a920beeab66d772bf45642ed56489d378253757e09959da4affdbee84081bec`、本地发布文件与归档的字节一致性以及物理回执到压缩台账的哈希链。独立读完 4,804 拍，每拍每轨五个判定。七个窗口包含 615 个待物理执行的拍，另外 4,189 拍均标记窗口以外。完整拍有十个归档物理子步，包含已执行的保持。变化下发按整个 float32 下发向量计数，动作表现以实测手部几何为准。逐轴完整数据的归档路径和哈希在运行清单。 [物理运行回执](../campaign/evidence/teleop-parity-root05/fixed-arm-parity-summary.json) · [压缩台账](../campaign/evidence/teleop-parity-root05/row-ledger.jsonl.gz) · [完整运行产物哈希](../campaign/evidence/teleop-parity-root05/runtime-artifacts.sha256)

| 轨道 | 完整拍 | 部分执行 | 未执行 | 实际子步 | 变化下发次数 | 观察最小间隙 |
|---|---:|---:|---:|---:|---:|---:|
| A | 565 | 1 | 4,238 | 5,654 | 896 | 0.801 mm |
| B | 584 | 2 | 4,218 | 5,852 | 377 | 0.781 mm |
| C | 28 | 7 | 4,769 | 323 | 151 | 0.922 mm |
| D | 28 | 7 | 4,769 | 323 | 151 | 0.922 mm |

四轨的已执行子步均无归档接触或穿透。共同证明在 A 和 B 各报告 81 和 115 次候选穿透见证，另有 4 和 25 次正间隙低于阈值。A、B 有窗口因 HOLD 也未获证明或连续八次未证明而停。C 的 165 个有变化请求中，151 个在同一记录状态和候选上两门均许可，14 个两门均耗尽查询预算。台账里 C 与 D 的紧凑状态逐拍逐步完全相同，28 个配对完整拍的末端局部差为 0。A–B 有 565 个配对完整拍，B–C 有 28 个。 [证明分类及同状态见证](../campaign/evidence/teleop-parity-root05/trace-audit.json) · [物理和 FK 回执](../campaign/evidence/teleop-parity-root05/fixed-arm-parity-summary.json)

初次无交接归档在 860 ms 拍的子周期 2 保留了同一个 C、D 事件 64。右手 `r_thumb_cmc_abd` 实测速率为 5.04186 rad/s。最终运行的 14 条故障见证与压缩台账的所有事件逐条匹配。每条都有手轴实测速率超过 5 rad/s 上限及 2 × 10⁻⁵ rad/s 容差，超限速率绝对值范围为 5.00014–5.04416 rad/s。14 条均由两门许可下发，观察最小间隙为 1.118–1.408 mm，接触深度为零。速度子判据足以触发事件 64。静态盒证明对这些候选的净空判定与实际 Servo 超速可以同时成立。 [最终速度见证](../campaign/evidence/teleop-parity-root05/velocity-witnesses.json) · [初次运行逐步见证](../campaign/evidence/teleop-parity-root05/no-ramp-first-four.jsonl.gz) · [子步检查](../../../source/dexgym/tasks/direct/wholebody_s05/v1/task/physics.py) · [速度条件](../../../source/dexgym/tasks/direct/wholebody_s05/v1/task/safety.py)

同步视频有 2,402 帧、25 fps、96.08 s。用逐拍台账重新计算，实际反馈帧分别为 A 283、B 293、C 17、D 17。视频中的该类帧包含有实测样本的部分拍。A 用当拍可见的原生 SDK 输出，视频第一栏为四轨共享的 HMB 人手样本。 [视频](../campaign/evidence/teleop-parity-root05/teleop-parity-root05.mp4) · [视频回执](../campaign/evidence/teleop-parity-root05/video.json)

## 八种动作的覆盖

按记录的四点链回溯分类。表中的输入是整根中满足稳定代理条件的去重拍数，完整反馈是这些拍与实际物理窗口的交集。跨类别可出现同一拍。`INSUFFICIENT_INPUT_EVIDENCE` 保留所要求动作的证据缺口。 [动作定义及逐侧窗口](../campaign/evidence/teleop-parity-root05/hand-slices.json)

| 动作 | 记录中稳定代理拍数 | A／B／C／D 完整反馈拍数 | A／B／C／D 实测满足同代理拍数 |
|---|---:|---:|---:|
| 自然张手 | 442 | 175／175／6／6 | 129／123／6／6 |
| 握拳 | 643 | 73／73／5／5 | 10／9／0／0 |
| 拇指–食指捏合 | 240 | 28／34／0／0 | 6／12／0／0 |
| 拇指–中指捏合 | 0 | `INSUFFICIENT_INPUT_EVIDENCE` | — |
| 张指 | 458 | 190／190／6／6 | 109／102／4／4 |
| 单指屈曲 | 1,052 | 88／99／4／4 | 24／18／0／0 |
| 双手不对称 | 1,151 | 139／139／9／9 | 32／32／0／0 |
| 快速反复开合 | 0 | `INSUFFICIENT_INPUT_EVIDENCE` | — |

## H3 工程形态诊断

令 \(h_{f,0}\) 为人手某指的第一个选中点，\(u_{f,k}\) 为旋入模型手掌方向的三段单位向量，\(\ell^H_{f,k}\) 为人手记录长度。把右手食指和拇指记为 \(I,T\)。旋转后的人手点距向量等于 \(R(h_{I,0}-h_{T,0})+\sum_k(\ell^H_{I,k}u_{I,k}-\ell^H_{T,k}u_{T,k})\)。工程目标替换第一项为签名 Home 的 \(o_I-o_T\)，并将各段 \(\ell^H\) 替换为签名 Home 的 \(\ell^R\)。两种替换贡献可以在向量上分别计算，距离是替换后向量的范数。腕代理到世界的刚体放置保持这个同手距离。源代码逐段保留人手方向，形态构造未对尖点间的相对位置施加约束。 [目标构造](../../../source/dexgym/tasks/direct/wholebody_s05/v1/rmb/home_anchored_intent.py) · [模型评分点 FK](../../../source/dexgym/tasks/direct/wholebody_s05/p02/model.py)

原始与工程形态的 20 mm 诊断使用解剖链标签和模型评分 site 的点距。评分 site 及其归属的末端框与 pair 2425 的碰撞网格语义各自独立。点距观察支持判断当前工程代理是否保留同手捏合区间。对可达性、触面接触和人手解剖标定的判断需要各自的证据。 [基线根因审计](ROOT5_RMB_cause_audit.md) · [形态反事实脚本](../../../scripts/s0_5_campaign_teleop/audit_teleop_parity_geometry.py)

ROOT05 的签名拓扑在非拇指链都含 `IP`。因此 Native 和 Offline SDK L1 选用 `PIP/IP/DIP/TIP`，H3 解剖链选用 `MCP/PIP/DIP/TIP`。两处取同一 `TIP`，所以相应的原始指尖间距一致。依赖首关节点的卷曲和张开分类须与实际采用的链顺序一起解释。 [SDK 顺序](../../../source/dexgym/tasks/direct/wholebody_s05/v1/rmb/mapping.py) · [解剖链顺序](../../../source/dexgym/tasks/direct/wholebody_s05/p02/intent.py)

H3 回执在 4,801 个有共同几何的拍上复算冻结目标，坐标最大重建差为 7.687 × 10⁻⁹ m。全域已有 187 个原始右手拇指–食指点距 ≤20 mm 的拍，三种工程替换的计数均为 0。对归档中的 187 条唯一拍 ID 和各方案逐行重算阈值，得到下表。

| 几何构造 | 187 个原始近捏合拍上的右手点距 | ≤20 mm |
|---|---:|---:|
| 原始 MANUS | 12.204–19.943 mm | 187 |
| 保留人手段长，替换为模型 Home 起点 | 58.370–71.037 mm | 0 |
| 保留人手起点，替换为模型 Home 段长 | 41.118–61.067 mm | 0 |
| 同时替换起点和段长 | 22.136–40.527 mm | 0 |

在另一个 62 个同数值关节向量的模型对照中，签名模型和 Teleop 模型的有名关节轴、范围、零位与 Home 一致，腕点位置一致。局部指尖最大差 6.388 mm，世界指尖最大差 6.381 mm。碰撞对分别为 2,542 和 3,454。固定臂物理反馈与净空判定采用签名模型。 [H3 数值、逐拍见证和模型对照](../campaign/evidence/teleop-parity-root05/geometry.json)
