# 第 5 根 RMB 分组执行诊断

本轮使用 DCR-010 下固定的第 5 根 `capture:b63bc72d2e8b2ee395e966f6`，从远端 `s0-5` 基线 `22db972a84986bdf8b9a7adaf5c17c5b6ac701f5` 开始。结果为 `DIAGNOSTIC_ONLY`。编译模型、原生 Marvin IK 和 Wuji SDK、原 HMB、START 注册、评分层、0.500 mm 净空及 0.002 mm 数值余量在候选和基线上相同。控制按 50 Hz 人体时间轴、4 ms 原生子周期及 500 Hz MuJoCo 执行。

[完整根因审计](ROOT5_RMB_cause_audit.md) · [分组执行规格](../specs/V2_2_group_admission_diagnostic.md) · [独立审查](ROOT5_RMB_quality_independent_review.md)

## 分组实现与对照

从签名关节和传动命名核验左臂 7、右臂 7、左手 20、右手 20。对每个累计候选和最后一次同时下发使用同一物理快照做完整机器人连续证明。组拒绝时，仅经复证的组合可动。保持和重新接入单独记账。每个 4 ms 子周期保存原名义提案、累计组合、原 issued/Servo/实测快照、实际下发、证明细节，及两个实际 2 ms 子步的 q、qd、应用目标、接触、净空和事件。原 HMB 数组和缺失值按位保留，独立于命令数据。无反馈拍持续保存原目标、原分母和标记。

首次试验在 START 后 4.300 s 所在的控制拍、其 4 ms 子周期索引 1 停机。最后一个获准子周期，右手保持时左臂、右臂、左手分别移动了最大 0.003619、0.003187、0.002227 rad，最后完整组合获得连续证明。紧接着的新组合及上一保持指令均在深度 18 未获证明，最早停止叶为右拇指与食指碰撞对 2425。这次拒绝没有下发，没有对应物理子步。

| 首次分组候选的分组状态（4 ms） | 左臂 | 右臂 | 左手 | 右手 |
|---|---:|---:|---:|---:|
| 活动 | 735 | 889 | 855 | 839 |
| 组保持 | 127 | 51 | 25 | 27 |
| 重新接入 | 214 | 136 | 196 | 210 |
| 故障未提交 | 1 | 1 | 1 | 1 |

以相同原保持盒离线重算，获得证明所需的实际深度为 19、查询 146 次。原深度 18 在 126 次查询后耗尽。这个复算只对应原保持命令及其快照。拟议右手新动作仍使用原预算。 [停机前各组与状态](../campaign/evidence/dcr010-rmb-quality/grouped-01/terminal_archive_summary.json) · [同一保持盒复证](../campaign/evidence/dcr010-rmb-quality/grouped-01/terminal_hold_reproof.json)

首次试验完成 215 个历史控制拍，合计采集 2,152 个物理子步，包括停机拍先执行的两个子步。这个区间的最小观测几何间隙为 1.171 mm，接触、穿透和物理故障子步均为 0。最终组合证明耗时 P50/P95/P99 为 0.280/0.363/1.363 ms（1,076 个获准子周期）。13 项合成检查在 Ubuntu 通过，覆盖跨组冲突、组合否决、保持、因果恢复、故障、原分母和完整 ACTIVE 掩码。 [执行回执](../campaign/evidence/dcr010-rmb-quality/grouped-01/receipt.json) · [逐层统计和组账](../campaign/evidence/dcr010-rmb-quality/grouped-01/grouped-audit.json) · [合成检查记录](../campaign/evidence/dcr010-rmb-quality/grouped-01/qa.log)

在这 215 个完成的同拍上，原执行和首次分组的实测世界十指最大误差 P95 分别为 97.7628 和 97.7624 mm。分组候选在个别子周期能安全推进其他组，本次同拍整体精度未见实质性改善。冻结形态目标到名义 FK 的局部手型 P95 已达左 68.24／右 66.26 mm，实测为左 69.15／右 66.55 mm。目标到名义腕位置 P95 为左 0.31／右 9.43 mm，目标到实测为左 11.84／右 32.45 mm。以世界十指最大误差衡量，名义到下发 P95 为 0.00 mm，下发到应用 4.71 mm，应用到实测 39.28 mm。各层不同的分位数不相加。这一短区间的局部手型误差在 Servo 之前已经出现，同时世界几何存在实际执行误差。 [同拍比较和误差分位数](../campaign/evidence/dcr010-rmb-quality/grouped-01/grouped-audit.json)

## 限定复验和实际技术门

第二次诊断使用 `bb37fdeff64c90f7243ad3535de1b8aac7f1c61e`，隔离工作树通过 14 个合成控制和 13 个签名分区的 4,844 拍数据检查。只有原指令和上一条下发值完全相同、且常规保持证明耗尽细分深度时，才用同一全身盒作一次深度上限 24 的补充证明。整身活动命令和各组候选使用原深度 18。三个保持复证分别在深度 19、21、23 通过，新增 6 个实际 2 ms 子步，全数记保持。第四次保持仍在 pair 2425 耗尽深度 24，286 次查询，最小查询采样间隙 0.682 mm，未获全路径证明。最终组合和保持都未下发，停止发生在 START 后 4.316 s。此后无额外重试。 [精确下发与未执行见证](../campaign/evidence/dcr010-rmb-quality/grouped-02/exact-proof-witnesses.json) · [14 项合成检查](../campaign/evidence/dcr010-rmb-quality/grouped-02/qa.log)

| START 后指标 | 原 DCR-010 | 首次分组 | 限定保持复证 |
|---|---:|---:|---:|
| 计划拍 | 4,804 | 4,804 | 4,804 |
| 原联合有效／原五帧候选 | 4,794／4,708 | 4,794／4,708 | 4,794／4,708 |
| 完整物理拍 | 4,804 | 215 | 215 |
| 实际子步／物理历时 | 48,040／96.080 s | 2,152／4.304 s | 2,158／4.316 s |
| 完整 54 DoF 活动 | 146／4,794（3.05%） | 144／4,794（3.00%） | 144／4,794（3.00%） |
| 有效五帧 | 139／4,708（2.95%） | 139／4,708（2.95%） | 139／4,708（2.95%） |
| 观测物理穿透子步 | 0 | 0 | 0 |

全母源原始分母仍为 4,844／4,831／4,741。第二次诊断保留全部计划拍及 4,801 个独立形态目标，215 个有十个子步的完整实测拍，另一个部分物理执行后停止的拍及 4,588 个停止后无物理执行拍。2,158 个子步最小观测间隙为 1.171 mm，最后采样为 1.233 mm，全段已观察前缀的物理接触、穿透和事件计数均为 0。未获证明的包围盒和候选端点保持各自的证据定义。原审计的 870 个静态穿透端点均来自未执行的旧提案。

原始模型腕掌解剖外参仍未知。本轮证据只支持工程腕代理、冻结机器人形态目标和签名仿真结果。手部问题可分别验证两个候选。其一，用独立形态重定向保留原手型和捏合向量的功能信息，保留冻结评分层和现有碰撞规则作配对。其二，以已取得的 ±5° 有界静态见证为起点，对有限连续片段测试使用原目标和全身证明的名义手部改善，单列实测跟踪、接触、净空和恢复。原 MANUS 与形态目标中，20 mm 内的右手拇指–食指间距分别为 187 和 0 拍，说明形态目标本身还需要独立复核。正式目标、接触语义或评分变更须另行 DCR。

阶段结果为 `ROOT5_INDEPENDENT_REVIEW / BLOCKED`，保持 `DIAGNOSTIC_ONLY`。第 5 根没有完成目标历史的实际物理回放，85%／70% 技术门对应的计数远未达到。六根验证停在门前；全量 RMB、正式训练和真实机器人均未启动。

## 复现和产物

首次分组实现 Git SHA 为 `a1ec267b06f149a951f4f441f89cd840b95b7c38`，后处理为 `9b823d4`。原生 Teleop Git SHA 为 `6d0b0b5b2601618f1b0052d93ef2403eea9b064a`。签名 MuJoCo 模型 XML SHA-256 为 `939b3651e2426e596eed482e70a68c2382720df1fd55b124aa5f0893b640e92d`。模型资产、HMB 来源与目标身份见原始和分组审计回执。原始分区完整数组和逐拍 `diagnostic-events.jsonl` 存在 Ubuntu 上回执对应的运行目录。回执逐项记录数组 SHA 以及事件文件 SHA，Git 归档包含审计结果和视频。 [实际运行命令](../campaign/evidence/dcr010-rmb-quality/grouped-01/command.txt) · [实际输入 SHA](../campaign/evidence/dcr010-rmb-quality/grouped-01/input.sha256) · [读后审计命令](../campaign/evidence/dcr010-rmb-quality/grouped-01/audit.command) · [首次三栏视频](../campaign/evidence/dcr010-rmb-quality/grouped-01/root-five-offline.mp4) · [视频回执](../campaign/evidence/dcr010-rmb-quality/grouped-01/video.json)

第二次三栏视频按原时间轴显示人的原始动作、独立映射和机器人形态目标、实际机器人反馈。右栏仅绘制有实际反馈的时刻，停机后显式显示缺测。25 fps 共有 2,403 帧，其中 108 帧具备机器人反馈。追加的 12 ms 落在同一 20 ms 控制拍内，两次导出的视频内容逐字节一致。以下各运行元数据及逐拍台账分别归档。 [第二次实际命令](../campaign/evidence/dcr010-rmb-quality/grouped-02/command.txt) · [第二次回执](../campaign/evidence/dcr010-rmb-quality/grouped-02/receipt.json) · [第二次物理统计](../campaign/evidence/dcr010-rmb-quality/grouped-02/grouped-audit.json) · [首次逐拍台账](../campaign/evidence/dcr010-rmb-quality/grouped-01/step-status.jsonl.gz) · [复验逐拍台账](../campaign/evidence/dcr010-rmb-quality/grouped-02/step-status.jsonl.gz) · [复验视频回执](../campaign/evidence/dcr010-rmb-quality/grouped-02/video.json)

台账保留 4,804 行卡点状态和子周期证明来源，归档输入和输出可按原回执核验。重建每份压缩台账的命令是 `python -B compact_grouped_trace.py RUN NEW_LEDGER_DIRECTORY`。精确向量转存器另对来自运行分区的原数组做 SHA 校验。 [台账脚本](../campaign/evidence/dcr010-rmb-quality/compact_grouped_trace.py) · [精确见证脚本](../campaign/evidence/dcr010-rmb-quality/grouped-02/extract_grouped_proof_witnesses.py) · [阶段二及复验产物 SHA-256](../campaign/evidence/dcr010-rmb-quality/grouped-evidence.sha256)
