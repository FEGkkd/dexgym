# Root5 RMB 数据链独立质量审查

独立审查结论为 **`BLOCKED`**（2026-11-28），资格为 **`same-family provisional`**。两轮 Root5 都停在同一 20 ms 控制拍。第二轮仅多执行 12 ms 的已证明 HOLD，完整 ACTIVE 和五帧覆盖仍为 3.0038% 和 2.9524%。

## 身份、范围与实际操作

本报告保留前审查者 `/root/rmb_independent` 的阶段一和首次实跑前记录。接续审查由独立任务上下文中的 `/root/rmb_review_sol`（GPT-6.2 Sol，xhigh）完成，负责复核归档和签署结论。两位审查者均与实现和执行分离，编辑范围限于本报告。指定基线为 `22db972a84986bdf8b9a7adaf5c17c5b6ac701f5`；首次源码审查读取到 `b337f597e7abf9be24435cf3c86a8b010b69a644`（新增两个只读审计）及其父提交 `5bdebe19903399f89d0af3a452d202911d7f8822`（issued-only 重建）。前审查者没有启动仿真、名义器、训练或硬件，没有更改生产实现、原始数据或阈值，没有推送。审查属于同模型家族的独立任务上下文，结论为 **provisional**。证据范围覆盖归档核验和源码审查，异构模型复核、外部实验室复现和用户最终验收另属各自环节。

已直接读取旧 [root5 结果](DCR-010_offline_rmb_root_five_results.md)、[旧独立审查](DCR-010_offline_rmb_independent_review.md)、run02 回执与已发布审计、[Lockstep](../../../scripts/s0_5_campaign_teleop/offline_rmb_lockstep.py)、[builder](../../../scripts/s0_5_campaign_teleop/build_offline_rmb_diagnostic.py)、[proof inputs 重建](../../../scripts/s0_5_campaign_teleop/reconstruct_rmb_proof_inputs.py)、[pair ledger](../../../scripts/s0_5_campaign_teleop/audit_root_five_pair_ledger.py)、[geometry layers](../../../scripts/s0_5_campaign_teleop/audit_root_five_geometry_layers.py)，及实际调用的 `offline_nominal.py`、`home_anchored_intent.py`、`physics.py`、`swept_live_guard_v2.py`。

## 前审查者记录

| ID | 级别 | 发现 | 状态 |
| --- | --- | --- | --- |
| IR-01 | P2，审计完整性 | `audit_root_five_pair_ledger.py` 将 `physical_contact[tick, sub]` 当作 4 ms 子周期接触。实际该字段形状是 `[N,13]`，由 builder 汇总整拍十个子步的 13 个接触组；`sub=0..4` 只读取前五组。它不足以独立声称 whole-robot `any_contact=false`。应对整行取 `any` 并注明是 20 ms 聚合，或另核验实际每子周期 trace。 | `FIXED_AND_EVIDENCE_REVIEWED`：`b33897f` 改为整行 `.any()`。修正后摘要、39 项数组身份和独立 13 通道交叉检查均已核验；全部为零。旧字段不回填升级。 |
| IR-02 | P2，模型身份 | `audit_root_five_geometry_layers.py` 校验主 XML SHA 和关节名，未核验完整递归模型资产与 MuJoCo 版本。单独主 XML 不绑定被引用的 mesh/include 内容；同轮 pair ledger 已有完整资产核对。 | `FIXED_AND_EVIDENCE_REVIEWED`：`b33897f` 用 `asset_files` 全清单及 MuJoCo 版本核验；审查发现并修正的中间 str/Path 类型问题也已消除。补充 asset closure 将 71 项资产、MuJoCo 3.13.0、原回执及原几何结果绑定，独立逐项核对无差异。未把旧脚本 SHA 改写为修订后身份。 |

## 已确认的证明与统计边界

4 ms 请求的实际几何盒公式为

`S_k = stack(q_issued[k-1], servo_filtered[k], q_measured[k], float64(float32(q_proposed[k])))`，

`low_k[j] = min_i S_k[i,j]`，`high_k[j] = max_i S_k[i,j]`。

`q_measured[k]` 是本次提案前的当前物理状态，不能用本拍末端 20 ms 导出的 `q_measured` 或前一拍实测代替。重建程序从 Home 重放原 START 与原已下发命令，每条执行两个 2 ms 子步；START 每个 4 ms 端点位置/速度按 f64 相等核验，历史每个实际子步的 applied/ctrl/torque 及每 20 ms 反馈按原 f32 导出精度相等核验。历史中间的 f64 precontrol 状态来自确定性重建，**不是当年直接归档的原始 f64 测量**。实际完成证据见下文；前审查者没有再启动物理重建。

`--prove` 在没有原 START 证明缓存的情况下建立新检查器。因此它是对重建盒的**新几何证明**，不是原证明树、查询序列或时限行为的复现；fresh proof tally 应逐项保留与历史 verdict/leaf 的差异。未证明、预算/深度用尽、发现阈值内见证和真正穿透必须分别报告。`unresolved_pair_indices` 在深度停止时保存的是停止叶的 pair 列表，不能推出所有待处理盒都只有该 pair。

已通过 Git 对比确认，当前 `physics.py`、`safety.py`、`native_sweep.py`、`p02/model.py`、Lockstep 与两版 guard 相对于实际 run02 的 `c2b5720e663d54d30ae937ecf710baac20112271` 无差异。保持 54 轴、原关节/耦合界、0.500 mm 净空及 0.002 mm 证明数值余量。模型资产的实际一致性仍以每次执行检查为准。

原 builder 的 full ACTIVE 要求同一 20 ms 内五次 4 ms 的四组均为 `ACTIVE` 或 `LIMITED_ACTIVE`，无 hold、fault，并完整执行十个物理子步；原五帧候选掩码和连续五个 full ACTIVE 拍同时成立才计五帧。部分组推进、hard fault、hold、恢复/rejoin 过渡不能记成 full-54 ACTIVE 或五帧覆盖。分组候选必须分别对整个机器人状态盒检查，并在选中多组后对**最终同时提交的 54 轴向量再证明**；单组独立许可的并集不自动构成最终同时许可。已读静态 E 对照满足这一结构，新分组动态运行尚不存在。

## A–F 数据链检查记录

核对采用 `aris:experiment-audit` 的 Step 2 清单。各项 PASS 仅覆盖其注明的证据范围。第 5 根的技术门为 `BLOCKED`。

| 项目 | 已核验内容 | 当前证据限度 |
| --- | --- | --- |
| A，GT 来源（PASS，明确 proxy） | `home_anchored_intent.py:114–170` 的 `SimulationIntent` 从签名人端几何、一次锁定的共同 START 变换和固定模型 Home 形态构造工程目标，不读执行 q、SDK 输出或跟踪误差。原 Tracker、注册腕代理、工程形态目标分层。 | 解剖腕掌外参仍 `UNKNOWN`；工程目标不是人体世界真值，也未证明目标可达。全行几何审计已到件；固定模型 Home 属形态先验，而非被评价动作的输出。 |
| B，评分归一化（源码 PASS） | `shape_metrics` 直接取几何差的范数并乘 1000 转成毫米，姿态差用角度；`stats` 输出原始 P50/P95/P99/max，未除以预测自身 max/min/mean。覆盖率沿用原 HMB 分母。 | 人体骨段方向的基变换、固定模型骨长/指根替换属于形态映射，不是分数归一化，更不等于训练归一化器资格。归档 raw Wuji 为 f32，重建 filter 必须标明 derived；hybrid FK 只能归因局部手型与指对。 |
| C，结果文件（PASS，已核验到件） | 已存在 run02 回执/旧审计，历史分母为 4,804 计划拍、4,794 原联合有效拍、4,708 原五帧候选；活动 146，五帧 139。新逐请求 ledger、重建回执、geometry JSON、A–E 和 witness 均实际存在，关键数字及 SHA 在下文逐项闭合。 | 完整重建 NPZ/原始 NPY 未由前审查者本地重跑；不能把执行者回执复核写成 reviewer 独立动态重演。 |
| D，实际到达代码（WARN，低影响死 helper） | 真实调用链为 builder → native nominal → Lockstep whole-robot proof → CpuServo。`audit_root_five_geometry_layers.py:42,259,278,319` 的指标有实际调用及 JSON 输出；`reconstruct_rmb_proof_inputs.py:124–140` 的新证明由非空 fresh tally 确认实际执行。 | `quat_error` helper（geometry 脚本:34）定义后未调用，也未支撑任何结果声明，属无声明影响的死代码。E 只执行静态反事实路径，不声称动态代码已到达。 |
| E，范围与阶段门（WARN，适用域有限） | 一根、一次已有主执行的完整归档诊断，三例 A–E、23 个不同静态 witness 样本；不是多根、多种子或正式训练。原始数据、目标、时轴和安全阈值保留。旧 root5 覆盖门仍失败。 | 不授予六根展开、正式 RMB、G-A/G-B/G-C0、Actor/WM 训练、实时或物理部署资格。下一阶段新实跑不能从三例静态证据推断全域收益。 |
| F，数据分类（PASS） | `simulation_only`，并使用明确的工程形态 proxy。原 HMB/原 trace/原回执属于归档事实；重新 FK、filter、状态盒和新证明属于 derived/recomputed；单组候选是反事实诊断，若未物理执行则不是新的执行轨迹。 | 不属于有真实解剖标定的绝对 `real_gt` 机器人精度评价；没有 human_eval。全部保留 `DIAGNOSTIC_ONLY`、`formal_coverage_credit=false`；UNKNOWN、NOT_RUN 与失败不可转写为通过或训练可用。 |

### 已直接到件：全拒绝请求静态端点交叉检查

已直接读取 早期摘要 `summary.json`（SHA `2500ffd11906ad2c3474c88b5da7dbe52904ec55cc060054d4ea3eb1f2217114`）与 [完整状态校核](../campaign/evidence/dcr010-rmb-quality/full_state_crosscheck.json)（SHA `60e4fc3d0933b996636b27c545e58d272ba911a7bf680ab87b7e7accfb5bb2a9`）；各自记录的执行脚本 SHA 均与本地归档脚本一致。它们对应 IR-01 修正前版本，旧 `any_contact` 完整性不据此验收。

22,905 个原始未证明请求的停止叶均为 `[2425]`。对这些归档候选端点复算该 pair 的静态距离，21,950 个达到 0.500 mm、85 个在 `[0,0.500)` mm、870 个为负。完整 54 轴设置相对于八个相对关节轴计算的分类差异为 0，最大距离差为 `6.010816844259637e-16 m`，870 个负距离均另有 `mj_fwdPosition` 静态 pair contact 见证；没有 `mj_step`。最坏端点净空约 −1.065901 mm。

因此，“全部未证明只是数值保守性”没有证据支持；至少这些 955 个候选端点不满足固定净空。相反，21,950 个合法端点也不证明整个请求盒安全。静态候选碰撞不等于原执行轨迹碰撞，因为这些候选当时没有获准下发。这个结果还不能推出原人体目标不可达或不存在其他合法名义解。

随后取得 [完整压缩台账](../campaign/evidence/dcr010-rmb-quality/request_ledger.jsonl.gz)，前审查者独立解压并复算 SHA `53d9359928f753b826af0a5a7065d5d5e27b924fed8ac1670748ecdb2d99ede6`，逐条解析得到 22,905 行及同数唯一请求，870/85/21,950 类别一致。深度 18、停止叶 `[2425]`、原拒绝标签、对应 hold PERMIT、`control_cutoff_ns >= time_ns` 均无不一致。这里复核了完整 derived ledger，而非前审查者再次运行 MuJoCo 几何求值。

修正后摘要 SHA `a768bc2b6c296804e2d346c3d8521e8cf4bea66490278885ecf0dbe9c36dce43` 绑定当前 pair ledger 脚本 SHA `09646afd3528115df5532901d74ce7efe0640d3d54874901012c5815be252a72`，两者已独立核验。其全部 13 通道检查为 false；另一个 contact crosscheck SHA `b34ac3aa6e4a8e1f517c3c7bc4165478d48524e789bd7d99c06f6800106ed134` 的 39 个 contact/tick/substep 数组 SHA 与 run02 回执逐项一致，4,804 拍、48,040 子步、13 个接触通道计数全部为 0。 [修正后摘要](../campaign/evidence/dcr010-rmb-quality/corrected-summary.json) · [13 通道交叉核对](../campaign/evidence/dcr010-rmb-quality/contact-crosscheck.json)

完整资产核验由已审 ledger 的运行路径提供，实际资产数是 71；crosscheck 中名为 `runtime_assets_sha256` 的字段实际为主 XML SHA，不可误当全部资产清单的合并 SHA。

### 已直接到件：issued-only 重建与新证明

[重建回执](../campaign/evidence/dcr010-rmb-quality/reconstruction-receipt.json) SHA `732a27eac4e56c0243b0610c0e022252fd3d032c27534e0e85762bb348bb4093`，其执行脚本 SHA 与已审源码一致。实际完成 945 个 START 命令的 f64 状态匹配、24,020 个历史命令、48,040 个 applied/ctrl/torque 子步及 4,804 拍末反馈匹配，13 分区守恒。重建 NPZ 身份为 `d1d9228d0a6b1fb8dd13daaa5fa41d0783b271f12277787d5707dc134e28afe4`。

这次确实执行了新证明：1,115 条 PERMIT 仍为 PERMIT，22,905 条仍为 depth-18 `SWEEP_UNPROVEN`，停止叶全部相同。它增加了**相同状态盒的新证明结果一致性**证据，没有恢复原始未保存的 proof tree/cache/全查询路径。未获准下发的 proposal 没有在重建中驱动物理；故该重建不能被称为这些被拒候选的执行测试。

### 已直接到件：三例 A–E 有界对照

[A–E 结果](../campaign/evidence/dcr010-rmb-quality/envelope_comparison.json) SHA `a9da716aaabb177973c0be6c86ef57205472578c818be3c74d94adf361103ebe`，脚本 SHA 与本地归档源码一致，绑定上述重建 NPZ、台账及模型。三个病例由首拒绝、候选端点最小/最大 gap 固定选出，均无 `mj_step`。

| 病例 | A，原全身盒 depth 18 | D，同盒最大 depth 24 | E，左臂/右臂/左手/右手 | E，最终同时全身重证 |
| --- | --- | --- | --- | --- |
| 首次拒绝 | UNPROVEN | PERMIT，实际深度 21 | PERMIT / PERMIT / PERMIT / UNPROVEN | PERMIT |
| 最坏候选端点 | UNPROVEN | UNPROVEN | PERMIT / PERMIT / PERMIT / UNPROVEN | PERMIT |
| 最大候选端点 gap | UNPROVEN | UNPROVEN | PERMIT / PERMIT / PERMIT / UNPROVEN | PERMIT |

源码明确从同一个 issued/filtered/measured 快照开始，按照固定次序叠加已接受组，对每个候选以及最终组合调用完整全身 V2 证明。它没有把四个单独许可简单求并集。补充 `group-proposal-deltas.json`（SHA `1d747029ed2f404aa83b69c288e0942ecda66ceeba42cb18ff6a460ae65dbc1c`）已读且核验哈希，三例中左臂 7、右臂 7、左手 20 轴均有非零候选增量，右手 20 轴候选被拒；支持“34 个非零轴组成的候选获证”，不是机器人已实际推进。同盒 depth-24 是固定三例的诊断预算对照，不修改历史 depth-18 或下游生产配置。

B 的 issued→proposal 线段和两个 Servo 参考线段在首例、最大端点例返回 true，最坏端点例返回 false；它使用 NativeSweep 的 0.500 mm 线段界，与 A/D/E 的 V2 0.500+0.002 mm 证明界不是完全同口径。B 未约束真实 measured 动力学，不能替代准入。C 只引用原已下发轨迹通过的物理相邻步，不能对未下发 proposal 推断安全。最大端点例虽端点 gap 7.706 mm，枚举的一个盒角点仅 0.128 mm，直接说明该盒存在不满足净空的混合状态；合法端点不蕴含合法全盒。

### 已直接到件：几何层误差

[几何层 JSON](../campaign/evidence/dcr010-rmb-quality/root-five-geometry-layers.json) SHA `ad91c665a932d2d022f721021804ba4c9a11d91846b66d510e563164f454bbdf` 的 script SHA 与归档脚本相符。复核其 4,844 全行、4,804 历史行、4,801 有目标完整执行行、4,794 Wuji 可评分行及 146 ACTIVE 域，均与原分母语义一致。

其可观察域实测 world max-ten-tip P95 为 234.072062 mm，ACTIVE 为 57.563945 mm；ACTIVE 原 Wuji/滤波 Wuji 的每手局部最大指尖 P95 为 47.9826/54.1033 mm 与 47.9942/54.1689 mm，说明滤波前已存在工程目标与名义手型差异。ACTIVE issued→applied 的 world max-ten-tip P95 为 0.578610 mm，applied→measured 为 8.721050 mm；这些分别计算的分位数不可相减为因果百分比。该审计的 nominal endpoint 分类只取每 20 ms 最后一次提案，因此 174/17 个负 gap/净空不足与完整 4 ms 台账的 870/85 个属于不同统计域，无矛盾。

补充 [模型资产闭合](../campaign/evidence/dcr010-rmb-quality/signed-asset-closure.json) SHA `18eaef1fbca7b444a6967fd642370145224a9d111152486016920729e1986b05` 绑定该几何产物、原回执及 MuJoCo 版本。前审查者将 71 项资产 SHA 与 run02 清单逐项对照，全部一致，IR-02 证据缺口关闭。

已读 [手部静态 witness](../campaign/evidence/dcr010-rmb-quality/hand_witness.json)（SHA `97c63796402f8a2bdee9a33b138ee67314a495d9fed4516a0edb1007869db79e`）和完整执行脚本，SHA 匹配。样本按 ACTIVE/OBSERVED 各 12 个等距序位选取、合计 23 个不同原拍；每样本固定臂，手限于原 issued±5°，最多 30 次 least-squares evaluation 和五个固定 line-search 比例。目标仍读原签名数组，未由 witness 重定义。

前审查者从全部 23 行 `issued_q_rad` / `witness_q_rad` 独立计算：臂变化为 0，手最大变化为 5°且无越界，23 个局部最差指尖误差均下降，各行有原静态合法性与全身直线配置路径 `PERMIT`。这里的 NativeSweep 明确设 0.502 mm，与 B 的旧 segment 设置不同。它仅提供有限邻域静态改善见证，不证明 4 ms 速度/力矩、Servo 跟随、完整时间序列或新 nominal 资格。

另读 [原 MANUS 与形态目标的指对对照](../campaign/evidence/dcr010-rmb-quality/right-pinch-endpoint-comparison.json)：4,801 拍中 raw 右拇/食指尖距离≤20 mm 有 187 拍，冻结形态目标为 0。174 个末子周期候选负 gap 对应的原手点中 30 拍≤20 mm、144 拍>20 mm，目标距离为 35.11–54.20 mm。20 mm 是描述性点距分箱，不是接触许可或正式阈值；形态映射没有保留这一近捏合标签，不能把工程 proxy 当完整人体意图真值，也不能据此豁免实体碰撞几何。

## 阶段一结论

既往根技术门为 **`BLOCKED`**：146/4,794=3.0455%，139/4,708=2.9524%，均未达到原 85%/70% 门槛。完成历史物理时间轴不等于有效数据覆盖合格。

阶段一证据足以支持同时存在三类问题：原预算下可证明性不足、某些请求盒/候选本身不满足固定几何约束、工程形态目标与名义手型不一致。它支持按已批准边界准备最小分组实现和有界 root5 对照；**不支持直接通过 root5、展开六根或继续正式训练**。新实跑须保留最终同时全身证明、完整状态/接触记录、原始全分母，并将 partial/hard/hold/rejoin 排除在 full ACTIVE 与五帧分子外。此处为前审查者在阶段一结束时的验收范围。

## 首次实跑前源码审查

基线为 `91c9d01`；审查到实际准备运行的 `a1ec267b06f149a951f4f441f89cd840b95b7c38`。前审查状态为 **`PRE_RUN_SOURCE_REVIEW_NO_P1_BLOCKER_PROVISIONAL`**，当时仍待 QA 原件和实跑结果。已直接读取 `group_admission.py`、builder 差异及 `test_group_admission.py`，本结论没有改变上文的根技术门。Reviewer 仍只写本报告，没有运行仿真或改动生产源码。

| ID | 级别 | 实跑前发现 | 当前处置 |
| --- | --- | --- | --- |
| IR-03 | P2，结果审计接口 | `blank(grouped=True)` 初始保留的 `hmb_source_index[N,4]` 被后续完整 `source_index` provenance 复制覆盖。现几何审计脚本:151 仍核对 `[N,4]` 原字段，分组结果将无法沿用这一审计。 | `FIXED_SOURCE_REVIEWED`：`80d6e7c` 已新增 `hmb_source_index_full`，原字段保持。 |
| IR-04 | P2，证明归因 | 分组新 `proof` 保存最终组合结果，原 `nominal_proposal` 仍是原完整请求。初始代码没有另外记录该原完整 54 轴请求的 baseline verdict；不能把最终组合 PERMIT 解释成原 nominal 全量 PERMIT。 | `RESOLVED_BY_EXPLICIT_SCHEMA`：receipt 新增 `proof_semantics`，明确 combined candidate before fallback，nominal 保持 pre-admission。没有原 A 新证明时不得报告该结果，实际最终准入与原请求归因分开。 |
| IR-05 | P2，后续技术门审计 | 新 grouped auditor 初版直接累加 `history_eligible`，full active 仅重复相信 `group_status` 数组，尚未从 native 状态、EXACT 提交、rejoin streak 和原 segment/五帧 mask 独立重算，也未逐项核对 event group status 与数组。 | `FIXED_AND_ARCHIVED_AUDIT_REVIEWED`。接续审查核对了 `9b823d4` 中实际取证路径及首轮审计产物。审计从 native 状态和实际提交重算 ACTIVE、10 次 rejoin 门和五帧分子，并核对每次获准向量和 EXACT 组的原提案。 |

源码目前已确认：同一 measured/issued/filtered 快照贯穿组候选；每组保留先前获证组后做全身 `_proof`，最终再证明同时组合，失败后独立证明完整上一 issued，仍失败则不执行。Marvin pair hard fault 在几何与物理前全局否决。候选先归一到实际 MuJoCo 接收的 float32 值再按 float64 证明；分组 branch 只有最终组合获准且对应臂 `EXACT` 时更新。

物理实现每个 4 ms 最多两次 `CpuServo.interval(count=1)`，使用同一个 issued；第一次物理故障后不再执行第二步。新的 2 ms q/qd 与接触/事件记录逐样点保存。恢复组的手步幅限制 0.015 rad，半幅/四分之一幅或尚未完成 10 个 exact 子周期时一律 REJOINING；full ACTIVE 要求全部五个子周期四组 ACTIVE、最终 PERMIT、无 hold/fault、十物理子步完整。

已直接读取后续扩充的 13 个 unittest 方法。新增测试实际调用 `_physical_steps`（替换底层 physics seam），核对同一 float32 issued 连续两次 count=1、首物理故障只调用一次、结果顺序及聚合；并调用真实 builder `record_execution` 核对十个 f64 q/qd 顺序与 mask、8/9 子步、partial/rejoin/hold 均不授 full ACTIVE。原 HMB NaN payload 独立写入 `raw_diagnostic_only` 数组，增加按位往返测试；新 finite RMB actuation 字段使用显式有效 mask。这些与原来的不安全组合、最终 veto+独立保持、hard fault、反馈失效及 rejoin streak 正负控共同构成启动前合成检查。

执行者已报告 Ubuntu 对精确 `a1ec267` 完成 13/13 controls、CLI、13 个签名 HMB 分区/4,844 行及首尾分区往返检查，原件路径为 `DCR010-RMB-ROOT05-GROUPED-01-QA-02`。前审查者尚待直接读取这些日志，未将执行者消息当作独立复跑。前审查者尝试在 Windows 执行纯合成 unittest，第一次在 import 阶段缺 numpy，接入已有 `marvin54_runtime` 后仍因缺 gymnasium 在 import 阶段退出；测试数为 0，没有新仿真或运行失败证据，不用此环境问题否认已报告的 Ubuntu 检查。


## 接续审查与首次分组实跑

接续审查在 Windows 对本地归档做只读核验。阶段一清单 11 个文件均通过 SHA-256，解压后的 22,905 条请求均有唯一的分区、拍、子周期键。按候选端点重新计算分箱得到 870 条穿透、85 条低于 0.500 mm、21,950 条达到端点净空，最小值 −1.065901 mm。所有拒绝的截止时戳、原标签、深度 18、停止叶 2425 和原 HOLD 许可记录一致。首轮分组回执的 71 项模型资产与已签资产闭合逐项一致，MuJoCo 版本同为 3.13.0。 [证据清单](../campaign/evidence/dcr010-rmb-quality/phase1-evidence.sha256) · [逐请求台账](../campaign/evidence/dcr010-rmb-quality/request_ledger.jsonl.gz) · [签名资产闭合](../campaign/evidence/dcr010-rmb-quality/signed-asset-closure.json)

将 `a1ec267` 的 builder、分组模块、Lockstep 和两版 sweep 的 Git 字节与首轮回执逐一核对。已收到 Ubuntu 13 项合成检查实录、CLI 输出、13 个 HMB 签名分区共 4,844 行的预检和首尾分区磁盘无损记录，QA 与运行进程退出码均为 0。结论的技术门取自回执及按实际数组重算的审计。运行与审计回执、START、停机摘要和同盒复证已核对相互绑定的哈希，本地紧凑副本与归档对应。三份命令文本只有归档末尾换行差异。完整分区数组及全事件 trace 保存在 Ubuntu 执行目录，本轮接续审查独立重算本地归档的派生指标和停机快照。 [QA 原件](../campaign/evidence/dcr010-rmb-quality/grouped-01/qa.log) · [运行回执](../campaign/evidence/dcr010-rmb-quality/grouped-01/receipt.json) · [后置数组审计](../campaign/evidence/dcr010-rmb-quality/grouped-01/grouped-audit.json)

| 首轮完整分母和执行结果 | 实际值 |
| --- | ---: |
| 全数据计划／HMB 联合有效／五帧候选 | 4,844／4,831／4,741 |
| 历史计划／HMB 联合有效／五帧候选 | 4,804／4,794／4,708 |
| 全部 2 ms 物理子步完成的历史拍 | 215／4,804 |
| 完整 ACTIVE | 144／4,794 = 3.0038%（门槛 85%） |
| 合格五帧 | 139／4,708 = 2.9524%（门槛 70%） |
| 组活动不足但完成物理的拍 | 34 个 PARTIAL_GROUP，37 个 REJOINING |
| 位置样本、接触／穿透／物理事件子步 | 2,152，0／0／0 |
| 历史实际采样的最小几何间隙 | 1.170595 mm |

首轮由 `START_REACHED` 进入历史录制第 40 拍。分区 00000 第 255 拍是历史索引 215，录制时刻 5.100 s，相对 START 为 4.300 s。该拍子周期 0 获准并完成两个物理子步，子周期 1 在 4.304 s 停在 `HOLD_INFEASIBLE:SWEEP_UNPROVEN`，新组合和完整上一 issued 均未获证明。累计 1,076 次获准组合均对应两个实际物理步。停机拍有 2 个物理步，随后 4,588 个计划拍没有物理推进。最后一次提交仍包含左臂／右臂／左手变化 0.003619／0.003187／0.002227 rad，右手保持。所有 13 个停机分区数组身份与回执一致。 [停机归档](../campaign/evidence/dcr010-rmb-quality/grouped-01/terminal_archive_summary.json) · [完整运行日志](../campaign/evidence/dcr010-rmb-quality/grouped-01/runtime.log)

用停机摘要中的上一 issued、当前 Servo filtered 与实测 54 轴逐元素重建保持盒，上下界与复证归档逐元素一致。256 个已归档相对轴角点中的最小间隙为 0.653361 mm。保存的全身复证在深度 18 用 126 次子盒查询停在 2425，在上限 24 用 146 次查询、实际深度 19 通过。点距和角点统计是分开的几何见证，连续许可来自同盒全身复证。原停机运行保留原判定。 [同盒连续复证](../campaign/evidence/dcr010-rmb-quality/grouped-01/terminal_hold_reproof.json)

首轮相同 215 个完整观察拍的实测世界 max-ten-tip P95 为 97.7624 mm，原 run02 同拍为 97.7628 mm。144 个完整 ACTIVE 拍的同拍 P95 均为 57.1036 mm。ACTIVE 的目标到名义局部每手最差尖点 P95 为 47.97／54.12 mm，应用到实测局部残差为 3.07／3.03 mm。分组准入维持了部分平台运动，已观察的同拍误差仍近似相同。工程手型误差维持独立未解决状态，冻结形态目标与真实解剖映射的误差继续为 `UNKNOWN`。 [同拍几何和层间统计](../campaign/evidence/dcr010-rmb-quality/grouped-01/grouped-audit.json)

23 组冻结目标的静态手部见证在接续审查重新逐轴复算，臂关节变化为 0，手关节变化最大 5°，双方静态合法性与全身直线配置证明均有归档，最小的局部最差尖点改善为 1.528 mm。原 MANUS 4,801 拍中右手近捏合点距 ≤20 mm 为 187 拍，冻结工程形态目标为 0 拍，二者是按相同归档拍重新相加。这里的 20 mm 是描述性分箱。实体净空由签名表面定义，评分尖点是另一工程几何层。静态改善和签名仿真运行各有各自的适用范围，实物触觉、传感器形变、绝对解剖精度和跨根手型跟踪的资格待验证。 [手型静态见证](../campaign/evidence/dcr010-rmb-quality/hand_witness.json) · [近捏合对照](../campaign/evidence/dcr010-rmb-quality/right-pinch-endpoint-comparison.json)

## 第二轮限定 HOLD 的实跑前审查

`bb37fde` 新增的引擎只在最终组合被否决后调用，参数逐元素等于上一实际 issued。常规 HOLD 要明确返回 `SWEEP_UNPROVEN` 且停止原因为 `MAX_SUBDIVISION_DEPTH`，才额外证明一次。相同 issued、滤波、实测盒使用相同限位、耦合、0.500 mm 净空和 0.002 mm 数值余量。子盒查询上限为 8,191，每次证明时限为 2 s。额外深度上限为 24，START、四组提案和最后组合的常规上限为 18。已知间隙不足、穿透、反馈无效、超时和查询耗尽均继续走原失败路径。最终组合的否决仍然有效，额外复证通过的旧命令结果保留为 HOLD，排除在完整 ACTIVE 和五帧合格分子外。额外复证失败会锁存故障。该变更无递归重试或再次运行的入口。源码审查支持一次限定的第二轮诊断。 [限定 HOLD 规格](../specs/V2_2_group_admission_diagnostic.md) · [分组源码](../../../scripts/s0_5_campaign_teleop/group_admission.py) · [同一盒证明入口](../../../scripts/s0_5_campaign_teleop/offline_rmb_lockstep.py)

接续审查直接加载当前源码的 HOLD 守卫、分组事务和共用盒检查，以 NumPy 和替代证明器做纯合成核验。98 组开关、原因和终止状态组合、相邻数值的向量拒绝、加深后拒绝仅调用一次、三个组合及保持提交路径、四次同盒及越界检查均通过。这些控制覆盖调用语义，签名 MuJoCo 实跑的行为由实际产物另行核验。


## 第二轮实际分组诊断

实际运行和 14 项合成 QA 均绑定 `bb37fde`，输入清单中两轮独立工作目录的模型和模型回执路径不同，对应 SHA 相同。运行脚本有独立命令和 SHA。两轮的根、原始分母及 Servo 配置一致，计划和 cohort、Anchor、Bridge、Wuji、Frames、配置、71 项模型资产及 START 两份产物的 SHA 均相同。压缩台账的 SHA 与摘要匹配，4,804 行按原分区／拍查验，逐行控制截止不早于样本时刻。重新统计得到 1,076 个同时组合许可、4 个最终组合未证明、3 个后备保持许可和 1 个终止保持未证明。实际数组审计对三个后备许可逐项核对了下发命令与上一 issued。接续审查再次将四次精确见证关联到同一事件、回执和 10 个源数组 SHA，并对向量逐位比较。 [运行回执](../campaign/evidence/dcr010-rmb-quality/grouped-02/receipt.json) · [QA 原件](../campaign/evidence/dcr010-rmb-quality/grouped-02/qa.log) · [紧凑台账](../campaign/evidence/dcr010-rmb-quality/grouped-02/step-status.jsonl.gz) · [数组审计](../campaign/evidence/dcr010-rmb-quality/grouped-02/grouped-audit.json)

| 同一历史拍的 4 ms 子周期（零基） | START 相对时刻 | 常规 HOLD 查询数／深度 | 额外查询数／实际深度 | 额外 HOLD 判定 |
| --- | ---: | ---: | ---: | --- |
| 1 | 4.304 s | 126／18 | 146／19 | PERMIT |
| 2 | 4.308 s | 157／18 | 194／21 | PERMIT |
| 3 | 4.312 s | 192／18 | 264／23 | PERMIT |
| 4 | 4.316 s | 228／18 | 286／24 | SWEEP_UNPROVEN，停止叶 2425 |

四次均位于历史索引 215、分区 00000 第 255 拍。台账核对四次最终组合均否决、常规 HOLD 均耗尽深度、四组均无新提交。三个许可各完成两个物理子步。三个成功 HOLD 的实际下发均与同一个上一 issued 按位相同。四次待证组合也等于该向量，首个额外复证的四态快照与第一轮同一时刻逐位一致。此后实测仍在演化。碰撞对八轴上，实测相对保持命令的最大偏移从 0.01562 rad 增至 0.01730 rad。最后一次下发和物理区间的掩码均为 false，该 4 ms 子周期的两个物理样点均缺失。最后一个实测子步采样间隙 1.233370 mm，两个臂组峰值速度限位占比 18.65%／18.47%，两个臂组峰值力矩占比 18.81%／19.30%，左右手同类比例更低。整个观察前缀的最小采样间隙为 1.170595 mm，2,158 个子步中接触、穿透和事件均为 0。终止证明记录的最小探测间隙为 0.682169 mm，认证在深度 24 用尽后仍缺少覆盖全部保持盒的连续界。该判定和离散已执行样本来自不同检查。 [精确向量和掩码](../campaign/evidence/dcr010-rmb-quality/grouped-02/exact-proof-witnesses.json) · [停机快照和数组身份](../campaign/evidence/dcr010-rmb-quality/grouped-02/terminal_archive_summary.json)

第二轮共完成 215 个历史拍，停机拍执行了 8 个 2 ms 子步，后续 4,588 个计划拍保持无物理推进状态。仍为 34 个 PARTIAL_GROUP、37 个 REJOINING，完整 ACTIVE 为 144／4,794，五帧覆盖为 139／4,708。两个门槛分别至少需要 4,075 与 3,296 个合格拍。两轮 215 个完整观察拍及 144 个完整活动拍的全部已归档几何统计一致。两轮视频的 25 fps 抽样帧计数为 2,403，其中 108 帧有实测反馈。第二轮多出的子步在同一 20 ms 拍内，两个视频的 SHA 一致，回执各自独立绑定。 [第二轮视频回执](../campaign/evidence/dcr010-rmb-quality/grouped-02/video.json) · [共同视频](../campaign/evidence/dcr010-rmb-quality/grouped-01/root-five-offline.mp4)

## 独立签署

| 阶段 | 审查结果 |
| --- | --- |
| `ROOT5_CAUSE_AUDIT` | 原拒绝归因、已下发归档重建、几何分层、原分母复核完成 |
| `GROUP_ADMISSION_CANDIDATE` | 已检查初版和一次限定 HOLD 复证的源码、合成控制、实际签名输入 |
| `ROOT5_FULL_DIAGNOSTIC` | 实跑证据已到件，两次技术门均为 `BLOCKED` |
| `ROOT5_INDEPENDENT_REVIEW` | **`BLOCKED`**，同模型家族的独立任务上下文 |
| `SIX_TRAIN_ROOT_DIAGNOSTIC` | **`NOT_RUN`**，要求 Root5 实际技术通过和独立通过 |

签署者 `/root/rmb_review_sol`（GPT-6.2 Sol，xhigh）。审查范围为 `DIAGNOSTIC_ONLY`，`formal_coverage_credit=false`。几何追踪限制与本次 HOLD 证明上限均仍可观察。第二轮的完整运行、无执行故障及 85%／70% 覆盖条件均未满足。独立结论停在 Root5 审核门。
