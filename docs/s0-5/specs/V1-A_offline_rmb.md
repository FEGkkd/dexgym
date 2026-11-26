# V1-A Offline Robot Reference Motion Bank v2.0.0

`DRAFT_READY_FOR_REVIEW`。本规格定义后续可执行的 RMB 构建与判门。全部实际运行均等待已批准的设计哈希、连续仿真用途和单独有界 Campaign 授权。[路线](../ROADMAP.md) · [RMB 契约](V2_rmb_contract.md)

## 1. 目标和范围

从已验收 HMB 生成可由 GPU 训练采样器直接读取的参考数据库。以现有 Marvin IK、Wuji SDK 和一致因果规则构造名义参考，以独立几何映射构造学习目标。保留全量规划域、错误原因与可选可行性见证。运行环境是离线工具进程和签名仿真模型，输入为 HMB 中既有真人动作。原 HMB、MCAP、split、QC、代理时钟资格和原批准证据均由只读源提供。

## 2. 前置、输入和保留的资格

| 对象 | 输入身份与许可 |
| --- | --- |
| E1 HMB | SHA-256 `e63df2698feb812bc300593d9bf8d1ad3dcae1d2dc59aa4d3a3dfa96851d8074`。须在独立 Reader 中核对清单、计划、配置锁、分区与来源。 |
| F1 → P02 Handoff Manifest | SHA-256 `cc1f21ead157e576113a184dba48a5d2596f4835ffdf53e554823cb67eb3fb97`。这个身份与 P02 Final Handoff receipt 不同。 |
| P02 最终有限交接 | Run `P02-FINAL-20261126-ecd12ec`。Receipt SHA-256 `f7c874a48d6be5128f2f895c12038c9fb1057f25be57dab91ce2280478fa1458`；Limited Scope Candidate `9f78b4c3087289859d22eefd71a1b567317b3951488159bff174b2657c10ffa7`；Agent Work Manifest `b45344cd15c08a583eaba84b49bd40ef917d1eb844a0720beb1d11d8e08b2c75`。 |
| M1 | Train-only 注册、尺度 1。P02 专项复查的 `fitted_candidates.json` SHA-256 `577195a8eb66699e11cf8bcf910383086672c384723d61fa90c13ed90652baa7`。另锁定原局部手型、腕掌和腕 TCP 变换、模型、范围与代码身份。 |
| 生产参考适配 | 锁定已有 Teleop Marvin SDK 和 Wuji SDK 实际版本和二进制，头文件、源码、模型与所有外部资产逐 SHA 核验。P02 签名模型与原 Teleop MuJoCo 模型如果不同，显式对齐 frame、关节、FK 和约束并各自保留身份。 |

已知 HMB 规模为 32 个真人录制母源，24 Train / 8 Validation、308 分区、116,613 计划拍、114,532 联合有效拍、114,332 个有连续段归属的拍，884 个连续段和约 110,796 个五帧历史候选。联合有效到 RMB 资格之间可能出现拓扑、映射、IK、限幅、连续与采样覆盖损失。表格必须逐层用实际分母列出，不可套用 HMB 计数。静态有限签核原有范围继续有效。V1 连续使用 M1 是本次单独审议的仿真扩展，没有物理标定、真实时钟或动态继承资格。[P01 签核](../reports/approvals/P01_ACCEPTED_P02.md) · [P02 交接](../reports/P02_final_handoff.md) · [P02 历史有限签核](https://github.com/FEGkkd/dexgym/blob/c5af9af94d7b9d3f437c14df128441d1183bec56/docs/s0-5/reports/approvals/P02_LIMITED_ACCEPTANCE_SIM0.md)

## 3. 实际代码入口与工程任务

| 复用入口 | 审计与适配任务 |
| --- | --- |
| `source/dexgym/tasks/direct/wholebody_s05/hmb_reader.py` | `HMBReader(manifest_path,plan_path,config_path,products,split,purpose)`。按批准 split 分别实例化，`keys()`/`read()` 返回因果数组、元数据、源表与 frame。计算 mapped 和名义参考时使用 `HMB_GEOMETRY_DEVELOPMENT` / `HMB_SIMULATED_CAUSAL` 视图。`history()` 需要五拍连续有效、20 ms 间隔和未换世代，诊断视图及身份表不给 Actor。 |
| `source/dexgym/data/wuji_retarget.py` | `WujiRetargetSession.step` 需要有限的 21×3 `float32`，返回 20 个 API 原始量。使用 HMB `sdk_common_pose` 和经拓扑校验的 1+5×4 节点布置，左右手会话独立。SDK C API 初始化和关闭可能有进程级效应，先验证同进程隔离，失败时用分侧隔离子进程并保持顺序。通过有名映射换算关节量，上锁 yaw/filter 参数及 SDK SHA。 |
| `wholebody_s0/native/replay_ik.cpp`、`wholebody_s0/retarget.py` 与生产 Teleop | 复用 Marvin SDK FK/IK、名义限界和有状态求解入口。旧离线工具自身含 START、节拍倍乘和参考行为，`run_hand_sdk` 用双端插值。只复用经验证的底层调用和适配逻辑，V1 的输入构造、到达调度、单边采样由本规格确定。现有 `MarvinKinematicsAdapter` 提供生产算法审计锚点。 |
| `wholebody_s05/p02/intent.py`、签名 P02 与现有 S0 reference bank | 重用有名骨架、独立腕/掌语义和签名 M1 参数，核验世界变换和地标。S0 历史 bank 的采样、GPU 列式加载可复用，经 schema adapter 读取 RMB。禁止继承其离线未来插值、参考重定时及旧 split 的训练资格。 |

预期新增模块为 `wholebody_s05/v1/rmb/schema.py`、`nominal.py`、`mapping.py`、`domains.py`、`build.py`、`reader.py`。单独的 CPU 独立审计入口使用源指针、冻结配置和机器人 FK 重算证据。完成构建和独立检查后原子发布清单。实际运行无完整 ROS 2 图，缓存的状态机只承担离线 QA 与参考生成，部署复用实际实时适配并在 V1-D 验证。[RMB 字节契约](V2_rmb_contract.md) · [现有 Teleop](https://github.com/FEGkkd/whole-body-teleop/tree/6d0b0b5b2601618f1b0052d93ef2403eea9b064a)

## 4. 名义状态机与数学

按 `(root,partition,tick)` 单调递增。每个分区、时钟/标定/拓扑世代改变、失效超过安全时限或明确复位事件时，销毁该世代的 Seed/SDK 状态并以固定合法模型 Home 初始化新世代。合格输入在其可用区间内按 time/schema 解释；重复源 ID 只更新 age 或周期命令管理，禁止向 Wuji 会话重复提交同一个原事件。无视未来合法拍的存在，初始有效命令和后续命令均只响应当时消息。初始 `RECOVERING` 需要两拍连续有资格且过路径/步长检查才进入 `ACTIVE`。生成失败进入 `HOLD`，连续无效期间只可维持上一合法命令或用冻结命令所有权回退。恢复重新查验路径，拒绝不连续命令。原 SDK 错误、原 IK 状态、限位和颠簸分侧记录。

每一拍独立生成 mapped Wrist `T_WU[s]`，名义 IK 同拍目标为 `T_baseTCP[s] = inv(T_Wbase[s]) · T_WU[s] · T_UTCP[s]`。一拍内只允许由当前及先前可得事件驱动子拍名义计算，锁定调度规则和实际步时间；V1 初始候选按 50 Hz 单边更新，同源实时比较时以同节拍和同输入复算。一侧解使用前一合法同侧名义角作 seed，矛盾或超时保留非法状态，实际 `q_nominal` 按两臂两手有名拼接并做名义合法性检查。`qd_nominal` 采用契约的同世代向后差分，无前驱、持有、无效场合独立掩码。缓存名义独立于 MuJoCo 当前状态以及被评 policy，实际执行用 q、qd 和 Servo 状态闭环。

## 5. 意图、阈值切换及域

按签名尺度 1 M1 生成独立的 mapped wrist/world/local/pinch/cross/wrist-relative 几何。M0 原样校核并输出全量审计。每拍复算腕到 TCP、掌和世界点的往返。使用原已签的活跃捏合阈值和局部手型规则。测量阈值进入、退出、同拍多对切换和目标步骤尖峰，以相邻同段 mapped 值和原 MANUS 步量联合报告。原目标、原资格和异常分母保留。动态失效只由冻结的规则和输入证据得出。阈值引起的几何跳变如未过 §9 的门，先找单位、拓扑、阈值浮点、时标、安装矩阵或 SDK 钟问题。任何平滑捏合启用或映射重拟合须独立审议并重新封存数据域。

**共同改善域候选。** 对每个映射有效且名义合法拍，用 `E(q,t) = max_i ||tip_i(q)−tip_i(mapped_t)||₂`。独立径向必要界 `b_i = max(0, ||tip_i(mapped_t)−anchor_i||₂−L_i−10⁻⁹ m)`；`L_i` 来自签名链长度和，仅用于阈值排除。预定有界残差合法集合检查关节名硬范围、残差 ±8°/±5°、耦合、驱动、碰撞及 `q_nominal` 到见证的预先验证路径。用固定迭代上限和正则化局部 Jacobian 步提出低成本见证，独立 CPU FK/路径检查具有唯一资格权。初始门 `E(q_w,t) ≤ E(q_nom,t)−max(0.002 m, 0.10 E(q_nom,t))`，同时局部手型 P95、活跃捏合向量和跨手向量每拍最大误差增量 ≤2 mm，相对腕平移 ≤2 mm、旋转 ≤1°。无见证和预算耗尽仍为 UNKNOWN。见证是针对签名模型、名义命令和有限路径的改善证据，实际 Servo 动态执行另测。与 B0/Actor 的执行状态隔离，域规则、成员和见证证据在 Actor 学习前按 SHA 冻结。

**困难连续域候选。** 同源有效段内按 `5` 拍之前的单边速度估计，定义世界腕线速度、SO(3) 角速度、名义几何十指最大误差与目标/名义捏合误差的 Train 分位阈值，初始各取 P90。任意一项连续 5 拍超界，取其在同段的 [起点前 10 拍，终点后 10 拍] 评估窗，重叠合并，最短 25 拍。窗口属采样和评分索引。硬件限界或模式变化造成失效则保留结束原因与全分母。另冻结停止、反向、单侧、双侧和手型变化标签，规则仅进采样器或评估器。V1-D 完整轨迹按来源连续窗每条只初始 reset，名义失约、Actor 失约和不适用独立计数。[共同计分](../EVALUATION_PROTOCOL.md)

## 6. 输出、证据与复现

建议所有未来实际运行路径统一为 `v1/<campaign>/V1-A/<run>/`，Run ID 与提交、环境、命令、时间及实际资源绑定。输出 `rmb_manifest.json`（输入身份、分片 SHA、schema/关节/坐标及目录表），`rmb_gate.json`（技术证据和独立审查，统一 gate_receipt.json 引用其哈希），`causal_tests.json`、`domain_counts.json`、`threshold_audit.json`，以及 `nominal_bridge.json`、原始及失败日志、`source_hashes.json`、`witness_ledger.json`、`static_windows.json`、`dynamic_windows.json`、`load_profile.json`。全量分片按源/分区原子发布。预留 `diagnostics/` 保存见证和历史 M2/Minimax 链接。保留每次失败目录和计数。固定缺省种子 49 用于样本索引，见证算法使用固定种子和候选步顺序。需记录第三方 `.so`、模型 include/mesh、坐标、运行解释器和可变线程数配置。

训练主入口加载 `Train` 的只读列式分片及合法五帧/连续窗索引。独立验证入口可读取冻结的 Validation 与共享对象身份。完整 episode 计分不使用训练 sampler 的丢弃规则。按源映射审计分母层级，仅在评价任务已经适用的情况下使用审计全部计划域作复合分母。

## 7. 验证与未来实际运行

1. 单元测试覆盖有名拓扑与手指、quat XYZW、SE(3) 符号、左右调换负控、腕/TCP 往返、世界指尖重建、每轴符号/装配、掩码占位、分区/复位/掉包、HOLD 与限界、SDK 错误、时间顺序、阈值等号和多对切换、路径见证、径向界及 UNKNOWN。以独立 CPU FK 和顺序检查注入转轴顺序、单位倍率、重复源和机器人状态读入目标的负控。
2. 集成抽样使用 24 个 Train 各一个连续窗，8 个 Validation 各一个连续窗，每窗最多 250 拍。方案、抽样 seed、独立 reviewer 和阈值在读取 Validation 或 Actor 输出前冻结。核对手/臂 SDK 实际调用、二进制导出、因果步过渡、缓存/等输入流在线组件的型态与数值。未来替换、删除和元数据扰动至少覆盖每根 2 个前缀，其中一个跨失效/世代边界。逐字节比较合法前缀的名义与映射输入和历史，不接触未来索引。
3. 抽样 gate 通过后再对全量 308 个分区顺序处理，验证全计划数量守恒及 SHA。分母按 split、母源、分区、四种域阈值、可选 witness、否决原因和连续窗口汇报。选 ≥512 逐源分层拍进行独立量值复核，含每种主要失效状态的全部或至少 16 例、全部极值阈值切换前 10 例，以及 ≥64 个合法 witness 随机样本与边界 witness。
4. 独立数据消费者随机加载 ≥128 个批次，每批 ≥64 个五帧窗，校验扁平索引、读入尺度、跨根/世代/分区 0 违规。V1-A 以 CPU 结构化数据和可选低开销 loader 预检证明可读。V1-B 可先用相同 schema 的合成分片做实际仿真 smoke；G-B 须以同一封存 RMB 复验真实数据批量接线。C0 再在最终 GPU 学习环境核对 host→device 的输入哈希和数值。

## 8. 预算与故障处置

候选上限为一个 CPU 构建进程池（最多八个 worker）、6 小时墙钟、RMB 和独立证据 20 GiB，清单与源输入保持在只读 HMB 挂载。本阶段计入 Campaign 总墙钟、磁盘及重试预算。先用 Train 约 3,000 拍测 CPU 吞吐、实时可用大小、峰值暂存与 I/O，预估含两份尝试的空间后再全量执行。见证初始最多 8 次局部线性试步/合格拍，CPU 时间不可靠时以固定步数资格与实际超限台账判定。资源和尝试上限以更严格的 Campaign 批准值为准。主要失败允许最多两次有原因的修复尝试，已有分片只有身份和产物 SHA 完全匹配才能复用，禁止覆盖原尝试。错误身份、未处理的因果性问题、坐标系错配、历史条目污染、覆盖低于门、预算耗尽或需修改目标时阻塞下游。

## 9. 技术 Gate 与下游

所有规定的 SHA 一致。全量 116,613 拍来源归档守恒，Train/Validation 根交集为 0。无任何前缀不变、时钟/单位、未知标签、关节/地标顺序、命令有效性或见证合法性负控失败。独立 FK/映射复算建议容差关节 `1e−6 rad`、坐标 `1e−6 m`、SO(3) `1e−5 rad`，只用于同输入同实现精度验证。名义合法覆盖初始要求每 split 不低于其联合有效拍的 85%，完整 5 帧历史覆盖每 split 不低于其 HMB 已有 5 帧候选的 70%。所有资格减少按原因和实际分母交代。共同改善域在 Validation 至少覆盖 4 个根、1,000 拍、20 个至少 25 拍的连续评估窗，上述是功能性 G-A 的最低覆盖。V1-D 性能比较还要求预登记的更高独立支持。若 Validation 只有 4 至 5 个根达域，必须在 Actor 学习前记录 V1 性能初始支持不足及 INCONCLUSIVE 风险，保持冻结可改善域及分母。切换 audit 对所有有前拍的 threshold crossing 报告反向离开及多对并发；初始连续性门为连续同段切换额外世界指尖步长 P95 ≤5 mm、最大 ≤15 mm。额外跳变用同输入下原 P02 `before` 未作 near-pinch 修正的控制列估计，`|| (tip_corrected_t−tip_corrected_{t−1})−(tip_before_t−tip_before_{t−1}) ||`，按每拍最差指统计。若该域无足够切换，追加固定合成阈值扫掠并将真实覆盖不足留在适用声明。

审查需重算来源、冻结目标、严格界、残差改善、因果及分母。通过后封存 V1-A 证据获得 `TECHNICAL_PASS`，以同一 RMB 哈希先交 G-B 实际接口验证，再同该环境资格交 V1-C0。Mapped 目标版本、SDK/模型、合法层、残差界、可改善规则、源索引或 schema 变更须触发受影响门重审，未受影响证据则保留原身份。[V1-B](V1-B_residual_environment.md) · [V1-C0](V1-C0_world_model.md) · [Campaign 规范](../MULTI_AGENT_PROTOCOL.md)

## 10. 风险

SDK 状态或二进制差异可能改变合法覆盖和名义构型。HMB 代理到达与真实输入不同。M1 保持尺度 1，部分域始终受绝对链长限制。近对捏阈值切换能放大目标步长。有限 witness 搜索的空结果只意味着未知。冻结物理模型的关节耦合、碰撞、剩余轨迹覆盖及可改善域样本分布，都可能限制后续结果能推广的场景。各失效按原拍和源保存，V1-B/D 同顺序同模型对照，真实计量和时钟另行资格化。[风险台账](../RISK_REGISTER.md)

## 11. `PRE_REGISTERED_CANDIDATE` 参数登记

| 参数 | 初始及允许范围 | 定标数据及冻结点 | 审查、失败动作和另行批准 |
| --- | --- | --- | --- |
| 名义调度、恢复 | 50 Hz、2 有效拍后路径核查恢复，允许 2–3 拍；内部适配可证的整数子拍调度为 1–5 | Train 预注册 24 窗，记录源 ID 和锁定 SDK/Servo；在全量及 Validation 前冻结 | Data Agent 提议，Independent Reviewer 审核前缀/步长；区间内适配经 Campaign 记录可定标，越界或改变评估时钟需项目用户批准；失败阻塞 A。 |
| 构建负载 | 初始 4 worker，允许 1–8；6 h、20 GiB 为本阶段候选封顶 | 约 3,000 Train 拍，实际机器刷新 CPU/内存/磁盘/吞吐；全量前冻结 | Main Agent 核对剩余资源，Reviewer 复核预算；总预算外需另行批准；资源不足停产不删除输入。 |
| 残差 witness 诊断 | 最多 8 步，允许 4–12；改善门固定 `max(2 mm,10%)`；功能保护候选固定 §5 | Train 分层取 2,048 拍，核对稳定吞吐和独立重算；读 Validation 前封存成员规则、搜索顺序、预算和共同分母 | Reviewer 独立核验。已批步数范围内无需另批，改变改善门、功能容差或成员定义须项目用户审批并重构资格；未达覆盖保留 UNKNOWN。 |
| 阈值连续性及覆盖 | 采用 §9 数值；本轮候选审核时允许用户裁定。执行时数值无自主调整区间 | 首轮仅 Train，统计阈值附近频率、修正增量和极端事件；Validation 前锁定数值及适用口径 | Independent Reviewer 复验；执行时调整、启用平滑捏合或放宽阈值必须另行批准；未通过停止 A 并保留原几何。 |
| CPU 重算容差 | `1e−6 rad / 1e−6 m / 1e−5 rad`，数值入库前最高可为 `2e−6 / 2e−6 / 2e−5`，限 float32 转换误差 | Train 抽查 512 帧，比较 double 精度 FK 与生产序列，记录误差上界；Validation 前锁 | Reviewer 独立计算；越界先修数值和 frame，放宽上限需另批。资格影响随下游受影响重验。 |
| 困难域切分 | Train P90，允许 P85–P95，连续 5 拍，窗口扩展 10 拍且至少 25 拍 | Train 母源等权查覆盖和任务标签，窗口索引在首个 Actor 学习前且读 Validation 前冻结 | Reviewer 复核适用性。批准区间内初次定标无需另批；冻结后改域须设计审议并重审比较。 |
