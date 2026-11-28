# DCR-010 离线 RMB 独立技术审查

状态：`INDEPENDENT_REVIEW_COMPLETE_WITH_EVIDENCE_LIMITS`，根技术结论 **`BLOCKED`**。首轮 START 失败；修正轮完成原历史窗物理执行，但活动覆盖及五帧覆盖远低于门槛。本报告区分源码、已核验的实际回执/START 摘要/审计产物和未独立重算的完整 trace/数组；不授予 G-A、G-B、G-C0、训练、实时或物理部署资格。

## 审查身份与基线

- 审查者：独立子代理 `/root/offline_independent`，与实现者分离。本审查者没有实现或修改名义器、锁步器、映射、合成预检或生产代码，仅拥有本 Markdown 报告。
- 指定基线：`548f04195d9f3815cc522d0f7f76c95b3bee8c46`。初次读取的实现提交：`73d3914`；另外读取了工作区未提交的 `reference.py` 提取和 `synthetic_preflight.py`。
- 独立性限度：审查者与实现者属于同一 Codex 模型家族；结论是独立任务上下文下的技术复核，对科研结论仅为 provisional，不能替代异构 reviewer 或项目用户的最终验收。
- 操作边界：本审查只进行源码、契约、历史 Git 差异和证据文件读取及哈希核验；没有启动 live/ROS、硬件、训练、Validation、全量构建，也没有推送。

## 直接读取的主要证据

直接读取 [DCR-010](../dcr/DCR-010_offline_rmb_first.md)、[v2.2 规格](../specs/V2_2_offline_rmb_track.md)、[DCR-009](../dcr/DCR-009_home_anchored_human_intent.md)、[原 RMB 契约](../specs/V2_rmb_contract.md)，并对照 `git diff 548f04195d9f3815cc522d0f7f76c95b3bee8c46`。

读取实现与依赖：`offline_rmb_lockstep.py`、`build_offline_rmb_diagnostic.py`、`offline_nominal.py`、`home_anchored_intent.py`、`teleop_native.py`、`teleop_candidate.py`、`physics.py`、`safety.py`、`native_sweep.py`、`swept_live_guard_v2.py`、`home_anchored_start_anchor.py`、`reference.py`、`synthetic_preflight.py`、`reader.py`、学习 adapter 所调用的 `train.py`/`catalog.py`，以及两组离线名义/映射测试。

审查期间的工作区文件哈希采样（实现者同时在更新代码，此表不是某次执行的冻结快照；实际身份以各 run 回执及相应 Git 对象为准）：

| 文件 | SHA-256 |
| --- | --- |
| `offline_rmb_lockstep.py` | `8ef68b11f4121ae8abac5ada627fda8cab76f201ed4fc3c84dcdfad893630934` |
| `build_offline_rmb_diagnostic.py` | `c90e14e6671d55445068a5c8ff1951c8a1dc4e984b48677907bb4e8fe325bb64` |
| `offline_nominal.py` | `0ee5fcf3b9ac8038ec75daf6d53d85937ff086110c76f6ee8b129d82062d86af` |
| `home_anchored_intent.py` | `dacdfa84306f8e437145ce7135383d8f631ab24e1717d433ad46edb7a265e7be` |
| `reference.py` | `a1d3f4462d90ade1227bff20a0c3a9677f600ec9fd768d98905a209eecf1137b` |
| `synthetic_preflight.py` | `cdb218c14d12323ec1cb7600648cf9ce72204ec06705c6e6766ba81921e9f04a` |

## 已确认的实现语义

1. 主循环每个 20 ms HMB 拍产生五次 4 ms 原生命令，每条通过连续几何检查后执行两个 2 ms MuJoCo 子步，即 50 / 250 / 500 Hz。当前输入在五次计算中因果保持；没有读取下一 HMB 拍。Wuji 只在所选 MANUS 的稳定 `(source_instance, writer_sequence)` 改变时调用，不因分片局部索引变化伪造新观察。这是预登记的离线调度，不证明 ROS 原约 120 Hz 回调与真实调度等价。
2. START 仅使用稳定窗末端的已可见输入。共同刚体变换在根开始时锁定一次，普通分片没有重锁；公共偏航来自双腕有侧别的基线，并明确为仿真候选。预定位使用冻结 START 目标、先腕后逐指，不消耗未来历史输入。它属于显式预定位，不应计入原 HMB 覆盖。
3. `SimulationIntent` 只接收原人端几何、锁定映射和签名 Home 的形态常数；没有执行 q、SDK 输出或跟踪误差输入。原始 Tracker、腕代理、映射腕与机器人形态目标分层。未知解剖外参和物理校准资格仍为 `UNKNOWN` / false。约 6 mm 的历史模型差异没有因单模型仿真而得到解决。
4. 请求转为 MuJoCo 实际接收的 float32 数值后构造几何包围盒；请求证明失败时不把该请求作为 `q_issued`。原生 raw Wuji/Marvin 与合法名义字段独立。五个子段全部活动、无保持/故障且共十个物理子步，才写 `q_nominal_mask`。保持和回退不计活动资格。
5. 每分片保留全计划行、原 joint-valid 与原五帧候选 mask。历史窗外的行仍保留输入目标而不执行。原五帧候选按原分片计算，没有把跨分片新窗加进旧分母。计数最终与签名计划核对。此 schema 是诊断候选，尚无生产 `s05.rmb.v1` 准入资格。

## 发现与修订复核

| ID | 严重度 / 范围 | 初次读取时发现 | 当前处置 |
| --- | --- | --- | --- |
| R1 | P1，停止与技术门 | `Lockstep.advance` 把 `proposal.hard_fault` 转为可证明保持，但没有锁存故障，仍推进物理并允许后续提案。与明确硬故障停止要求不符，可能在原生硬故障后得到根技术通过。 | `FIXED_SOURCE_REVIEWED`：现为 `NATIVE_PAIR_HARD_FAULT` 锁存并在证明/物理之前返回，无后续物理步；实际控制实验尚待证据。 |
| R2 | P1，独立可复核性 | START 只保存各阶段末摘要；保持证明被丢弃；`Interval` 的净空、深度、事件、力矩等实际物理安全证据未写入结果。代码中确有检查，但原产物不能独立重放核对整个 START 与保持轨迹。 | `DIAGNOSTIC_FIX_SOURCE_REVIEWED`：现有逐 4 ms START 事件、测量与速度，显式保持证明 verdict，历史 interval 的 event/gap/depth、逐物理子步 applied/ctrl/torque。保持搜索 metrics 和历史 500 Hz 实测 q 未逐项归档；现结论仅针对有签名代码与可重放命令的诊断，不授予独立完整物理 trace 的正式资格。 |
| R3 | P2，相邻速度与历史 | `qd_nominal` 仅用 `active_previous` 判断，没有检查相邻拍的 clock/calibration/topology/segment 边界。原五帧 mask 能阻止跨边界五帧计数，但不能修正边界第一拍的错误速度资格。 | `FIXED_SOURCE_REVIEWED`（`62f830a`）：以原字典值锁定 clock/calibration/topology，变化时停止，不重新映射；分片内 `segment_id` 变化显式清除 `active_previous/consecutive`。普通分片边界继续状态，原五帧 mask 独立。此修订的实际执行范围须由各 run 的实现 SHA 确定。 |
| R4 | P1，后续六根的资格绑定 | 六根入口仅检查 root5 的 root、字符串 PASS 和 plan SHA，没有绑定当前代码、模型/Servo、工程帧与原生配置的相同身份。旧通过回执可被用于不同配置。 | `FIXED_SOURCE_REVIEWED`（`c2b5720`）：现检查主要实现、桥/Wuji/frames/config 与完整 model/Servo signature，且纳入 physics/safety/native_sweep/teleop_native/Wuji、mapping/schema、旧守卫基础模块。首轮 `056be67` 仍是修复前较窄的依赖清单，不获得事后身份升级。 |

本表记录的是具体实现或证据缺口，不把有限搜索失败称为几何不可达，也不把 CPU 合成通过称为人类轨迹或正式训练通过。

运行身份还需外部启动记录支持：builder 在结束时从模块 `__file__` 计算源码 SHA，不能单凭这个结果证明运行过程中源目录未改变。如果进程运行时更新同一 checkout，内存中的旧代码可能与结束时磁盘 SHA 不同。执行者已确认将 Ubuntu checkout 保持在 `c2b5720`，直到 run02 回执出现且进程记录结束后，才切到 `62f830a` / `5b6f0b4` 做只读审计。本审查已核对 run02 记录的 13 个源码 SHA 均匹配 `c2b5720`；尚未取得独立的启动 manifest 原件。不把后续源码修订回填为 run02 实际执行代码，也没有证据显示发生了中途改动。

修订还将根技术门收紧为 START 合格、全部计划历史拍各执行十子步、无接触和锁存故障，并要求当前诊断窗达到原联合有效拍 85% / 原五帧候选 70%。这些只是该诊断窗的前置门，不替代全 Train / Validation 的正式覆盖门。统计时钟现明确为「20 ms 区间结束反馈对区间开始时已可用独立目标」，并排除物理执行不足十子步的行。该统计是冻结目标的一拍响应诊断，不能直接充作历史实时同拍评分。

## ACTUAL：合成 V1-B 证据

已直接读取并复算 [V1-B 执行回执](../campaign/evidence/dcr010-offline-rmb-first/v1-b-execution.json) 所列三个产物 SHA；三项均匹配。回执中的 `reference.py` 与 `synthetic_preflight.py` SHA 也与当前工作区相符。

| 已存在证据 | 核验范围 / 结果 |
| --- | --- |
| `v1-b-synthetic-cpu.json` | CPU，2 个环境，40 次完整实际物理 transition，每拍 10 子步，Actor 形状 `[2,8515]`，54 轴、因果五拍，8 步 WM 推理；优化器更新为 0。合成 RMB 及生产 reference 构造器的拒绝检查存在。 |
| `v1-b-native-cpu.json` | 136 次实际 transition，B0 / 零残差 B54 一致性、54 轴正负脉冲、非法动作回退、复位前 trace 检查均记 pass；24 次保持含 240 个物理子步。 |
| `v1-b-regressions.txt` | 实际日志为 `37 passed, 5 subtests passed`。这是记录复核，本审查者未再次执行。 |
| GPU | `NOT_RUN`。回执说明共享 GPU 已在使用，且该 CPU 环境缺 `mujoco_warp`；没有 CUDA/Warp 或规模资格。 |

合成预检以显式虚构的固定几何偏移区分 target 与当前 FK，并核验最新反馈来自当前 MuJoCo rollout。它实例化学习 engine 并执行 collection / WM inference，没有调用 optimizer 更新，不产生正式训练资格。合成 source 的 fit/holdout 仅是两条虚构通道，不能解释成真实 Train / Validation 证据。执行服务器基线为 `14936b54b895f6c2a54e0aad58e9017bfbb1bfda`；本审查者通过本地 Git 对象独立确认，该提交到指定基线在 S0-5 V1 和共享 SuperMimic 目录的差异为空。两个 overlay 文件身份已核验；服务器工作区是否还有其他未提交差异仍由执行回执说明支持。

本审查者另对六个新增/修改 Python 入口执行 `ast.parse` 静态语法检查，全部通过；physics/safety/native_sweep/teleop_native/Wuji/两版守卫在初始审查时相对指定基线没有源码差异。语法检查不等于仿真执行测试。

## 修订候选：START 与控制时钟

`c2b5720` 的代码已独立读取。START 阶段在证明之前把手指请求每 4 ms 限制为 0.015 rad，再转换成实际下发的 float32 值并证明；原人端目标、历史名义步幅及实际硬限制没有因此改变。START 原提案与最终 issued 可不同，不能把原提案当成已执行命令。

控制时钟修订保留原 `tick_ns`，另存 `control_cutoff_ns = anchor + ordinal × 20 ms + 1 μs`。只有原拍与未偏移控制网格相差不超过 1 μs 时才进入该固定周期；在这个条件下原拍不会晚于控制截止，所以没有未来输入。超过容差仍停止，已存在的首个故障不会被后续扫描覆盖。原 HMB 五帧分母未重建。此结论针对源码的因果性；所选根的实际纳秒偏差仍待读取原审计产物核验。

## ACTUAL：root5 首次诊断失败

已读取本地下载的 [run01 回执](../../../artifacts/dcr010-offline-rmb/run01/receipt.json)、[START 摘要](../../../artifacts/dcr010-offline-rmb/run01/start.json)、[末条 START 原始记录摘录](../../../artifacts/dcr010-offline-rmb/run01/terminal-start.jsonl) 和 [远端哈希表](../../../artifacts/dcr010-offline-rmb/run01/raw.sha256)。远端原路径为 `/home/ubuntu/dexgym_data/s0-5/campaign/S05-V1-20261127-490770e/runs/DCR010-OFFLINE-RMB-ROOT05-01`。

回执 SHA-256 为 `de018bbbcf30be809644e8b2bda2cec097c12692f5b36e3a743e631997f9bb4d`，START 摘要为 `d712fc92b666f275923fd219ca7d779345cf1e01b8e09fddf8841296dde248e6`；均与远端哈希表一致。五个已记录实现 SHA 全部与 Git `056be67` 的原始字节相符。该 commit 的 `offline_nominal.py` SHA 为 `bf3aaabd5b83aaf9581f14c361e3c5f60b3b19c44d7ab4d0e65c56ec5b127b8e`，与审查期间另一个工作区修订不同；实际执行身份以 run01 回执及 Git 对象为准。

| ACTUAL 指标 | 结果 |
| --- | --- |
| 母源全量保留分母 | 13 分区、4,844 计划拍、4,831 原联合有效拍、4,741 原五帧候选；回执分片索引行数之和为 4,844。 |
| 历史窗原分母 | 4,804 计划拍、4,794 原联合有效拍、4,708 原五帧候选；各分区窗口计数可相加复核。 |
| START | `START_UNQUALIFIED`。腕阶段合格后，在首个右无名指阶段（stage 1 / tick 3）于仿真 1.198 s 触发 `PHYSICS:64`。 |
| 物理失败见证 | 有名轴 46 `r_ring_finger_mcp_flex` 为 5.019181542738256 rad/s，超过原 5.0 rad/s 限制；该 4 ms 段只执行 1 个 2 ms 子步即停止。 |
| 同段几何证据 | 请求为 `PERMIT`，采样最小间隙 0.875979530620175 mm；实际区间最小间隙 0.9568662499077618 mm，穿透为 0。它是速度约束失败证据，不能称为几何碰撞。 |
| 历史执行 | 4,804 行全部 `START_UNQUALIFIED`；历史物理子步、活动名义和五帧活动历史均为 0。`recorded_physical_seconds=0` 只计历史段，不否认已发生的 1.198 s START。 |
| 根技术门 | `BLOCKED`，独立分类为 **FAILED**。没有六根展开资格。 |

另确认 P2 故障归因缺陷：首个物理事件在 `start.json` / 末条事件中为 `PHYSICS:64`，随后原版历史索引的无条件赋值把顶层 `physical_fault` 改成了 `UNBRIDGED_HMB_TIME_GAP`。不能用这个覆盖后的字符串否认真实 START 物理失败；后续候选已改成保留首个故障。正式结果陈述必须同时保留该代码缺陷与原始物理失败。

审查范围：本地获得完整分片**索引**和单条 terminal trace，不是全部 NPY 数据或完整 `start-events.jsonl`。本审查已复算上述本地文件 SHA、Git 源码 SHA 和索引守恒；尚未独立复算远端全部数组/完整 trace 的内容。零历史执行与 FAILED 分类由摘要、终止事件和索引一致支持，不据此宣称已验证整个动态历史。

## ACTUAL：修正后 root5 完整执行、覆盖失败

已读取 [run02 回执](../../../artifacts/dcr010-offline-rmb/run02/receipt.json) 与 [几何对诊断](../../../artifacts/dcr010-offline-rmb/run02/pair-diagnosis.json)。本地 SHA 分别为 `d6d83d40b44d29447e1d4a1e357bc0424b123dbabfcc113eef8218d330c605df`、`11fc5a7a8592f9e0613d4cf1aa60b10940f9b2149ba3f376d4c4e48fa6d7d6ea`，与执行者先前给出的远端 SHA 相符。随后恢复的 [raw 哈希表](../../../artifacts/dcr010-offline-rmb/run02/raw.sha256) 与 [bundle 哈希表](../../../artifacts/dcr010-offline-rmb/run02/bundle.sha256) 已核验，本地 raw 表 SHA `853c182577fb3809118a0c0aa85ebd5c9963f24546e26ac3310780d0b41e41ef` 匹配 bundle 项，receipt/pair 条目也与本地文件相符。远端独立目录为 `/home/ubuntu/dexgym_data/s0-5/campaign/S05-V1-20261127-490770e/runs/DCR010-OFFLINE-RMB-ROOT05-02`。模型 XML 与原计划 SHA 仍为签名值；13 个已记录实现文件 SHA 均与 Git `c2b5720` 字节一致。

| ACTUAL 回执字段 / 独立复算 | 结果 |
| --- | --- |
| 全根保留 / 历史分母 | 仍为 13 分区、4,844 全计划行；历史窗 4,804 / 4,794 / 4,708（计划 / 联合有效 / 原五帧候选），与 run01 和签名窗口一致。 |
| START | 已恢复摘要并核验 SHA 与回执一致：`START_REACHED`，阶段编号 0–10 共 11 个，每阶段 dwell 为 5、末条 proof 为 PERMIT、无 fault/hold。各阶段耗时合计 189 个 20 ms 拍 = 3.780 s，与最后记录一致。最终左右腕误差 0.280471 / 0.252090 mm、0.003304° / 0.002180°，四组均 ACTIVE。这里复核的是阶段摘要，未从完整 START 事件重新计算这些测量值。 |
| 历史物理执行 | 4,804 / 4,804 拍完成十子步，共 48,040 子步，即 96.08 s 仿真；`physical_fault=null`、`contact_ticks=0`。该表是回执及整数守恒核验，完整物理数组尚未本地重算。 |
| 活动名义 | 146 / 4,794 = **3.0455%**；85% 门至少需要 4,075 拍，未通过。 |
| 活动五帧 | 139 / 4,708 = **2.9524%**；70% 门至少需要 3,296 个，未通过。 |
| 逐拍分类 | `ACTIVE=146`、`INPUT_HOLD=10`、`NOMINAL_OR_PROOF_HOLD=4,648`，合计 4,804。 |
| 4 ms 请求分类 | `PERMIT=1,115`、`SWEEP_UNPROVEN=22,905`，合计 24,020 = 4,804 × 5。请求 PERMIT 的计数不是所有下发保持命令的证明总数；保持另行证明。 |
| 统计范围 | 4,801 个具有独立目标且完成物理区间的拍。左右腕位置 P95 为 76.864 / 83.675 mm，姿态 P95 为 59.253° / 64.402°；是全部可观察拍的区间末反馈对区间初目标，包含保持，不能解释成活动子集或正式同步评分。 |
| 根技术门 | **`BLOCKED`**。完成物理时间轴不等于活动名义合格。 |

几何诊断文件给出 pair 2425：`r_finger1_tip_sensor_col` / `r_thumb_tip_sensor_frame` 对 `r_finger2_tip_sensor_col` / `r_index_finger_tip_sensor_frame`，首条列出的记录是分区 `00000` 的 tick 253、HMB 时间 5.060 s。文件报告 previous measured / retained issued / nominal proposed 的 float32 端点净空分别约为 2.907 / 2.726 / 2.461 mm。端点净空没有证明整个请求盒，`SWEEP_UNPROVEN` 也没有证明碰撞、不可达或目标无解。

### 已恢复的 START 与审计原件

已直接读取 [run02 START 摘要](../campaign/evidence/dcr010-offline-rmb/run02-start.json) 与 [run02 配对审计](../campaign/evidence/dcr010-offline-rmb/run02-verified-audit.json)。两者 SHA 分别为 `c5057912e3346f8aee7132f1806ae47190e29588c7ae66774a2bccb45d84f403`、`134e2c639702935cb668e07762919ddca536cf990df7a0f6d9601b4442ca4a56`，与恢复的 raw/bundle 清单相符；START SHA 也匹配回执，审计的 `run_receipt_sha256` 绑定同一个 run02 回执。

审计产物记录 22,905 个 `SWEEP_UNPROVEN` 的 termination 均为 `MAX_SUBDIVISION_DEPTH`；配对检查为 24,020 个下发子段有对应 PERMIT，其中 22,955 个是已证明保持。历史已采样最小净空为 1.170595409 mm、最大已采样穿透为 0，没有短物理拍；原分片内跨 segment 的名义速度 mask 计数为 0。首次 HOLD 在历史开始后 2.86 s（分区 `00000` / tick 183）；这与几何诊断列出的 pair 2425 首条记录（tick 253）是不同的记录，不能合并为同一个首次失效时刻。

本审查已核对审计产物的分区分类之和与回执一致、四组各有 24,020 条原生状态、请求证明类别与回执一致、24,020 个已证明下发与 4,804 × 5 相符，并已读取 `5b6f0b4` 的审计源码：该程序校验事件/数组 SHA，检查 `q_issued_mask` 对应的 active/hold PERMIT、未记录下发和物理区间计数。**这是对已恢复审计原件、源码及计数守恒的独立复核，不是本审查者从远端完整原始事件与 NPY 再执行一次审计。**“全部未知都涉及 pair 2425”仍由执行者的额外诊断说明支持；当前已恢复审计 JSON 没有逐拒绝 pair 清单，独立读取的 pair 文件只明确给出上述一条位置。

审计使用“每拍十指最大误差”时，全可观察 4,801 拍的 P50/P95/P99 为 148.698 / 234.072 / 318.826 mm；活动 146 拍为 54.740 / 57.564 / 76.599 mm。两种样本域及这一 max-ten-tip 指标与回执中逐指分位数分开，不能以活动子集替代完整分母。

### 已恢复的媒体元数据

[视频元数据](../campaign/evidence/dcr010-offline-rmb/run02-video.json) 与 [容器探测记录](../campaign/evidence/dcr010-offline-rmb/run02-video-probe.json) 的 SHA 分别为 `e717146c71f20a85e9288d82e2dc0cb5a547d639d46eaad59e277f4cd3bf1c91`、`c55e192cc692696a9c96579a80c540d04a1329382a700758a97b76656c6b7d82`，均匹配 bundle 清单。元数据绑定同一回执，记录 1,202 帧、每帧有实测反馈，探测为 H.264、1440×640、12.5 fps、96.16 s。该媒体时长与回执的 96.08 s 物理累计时间分列，不改变原分母。视频文件预期 SHA 为 `b140f252fb6a6a51c6156b031cbf23db10fdb7b049ed548d0edf01451d2ffd23`；本轮审查未读取或观看完整视频，媒体元数据不能替代物理 trace 证据，也不表示视频已向用户完整交付。

## 后续门与剩余取证

六根展开的前置 root5 技术门没有通过；本轮没有据此授予六根、全量 RMB 或训练资格。`62f830a` 的 segment 速度遮罩和 rejected-arm 占位处理已做源码复核，但不能把这些后来提交的行为归给 `c2b5720` 的 run02。START 阶段摘要和配对审计原件已恢复并复核；完整 START/历史事件、NPY 数组的独立重算，以及启动执行身份 manifest（命令/工作区快照）的独立取证仍未完成，不能用视频替代这些证据或改写已观察到的覆盖失败。

全量 RMB、正式全 Train / Validation 85% / 70% 门、正式评分层、G-C0、Actor 训练、实时及硬件资格均仍需各自证据与批准。审查结论是有范围的实际诊断与代码复核，科研层面的独立性仍为同模型家族 provisional。
