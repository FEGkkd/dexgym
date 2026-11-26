# P01-F 独立 HMB 资格验收

**F1 技术门 PASS，P01-F READY_FOR_REVIEW。** Run `P01-F-20261126-b22f8be`。E1 已获人工签核，P01 整体保持 IN_PROGRESS。P02 仍待授权。[运行凭据](../evidence/P01_F/review_receipt.json) · [E1 人工签核](approvals/P01_E_ACCEPTED_P01_F.md)

## 独立完整性与来源

校验批准的 E1 提交 `b22f8be3c35bf1b5ad9bae98335ecd1510cacc1e`、manifest、计划、运行凭据、A1 Registry、生产索引与完整台账、D1 配置及 E0 环境锁。复验 42 个锁定生产源码文件、全部 1,612 项 E1 产物及日志，其中有 1,540 个正式分区文件。正式归档为 7,121,396,746 字节。E1 保持原文件内容和 SHA。

从 int64 原始区间及正式数组重算 32 个真人母源、308 个分区、116,613 个计划及发布拍，隔离为 0。复验 224 个原始 MCAP 分片 SHA 和采集清单、快照、原母源父级与冻结角色。按 A1 既有 lineage 和用途事件登记每个母源的历史风险。正式产品范围、索引和来源校验排除了试片、旧 schema、未知来源、EgoPro、仿真回放及同源副本。Train 24 个根、Validation 8 个根、交集 0。[独立完整性与谱系登记](../evidence/P01_F/independent_integrity.json)

## 真包重建与误差尾部

先冻结元数据抽样规则、seed、Top-3 及容差，再读取对应几何。88 个样本目标覆盖全部 32 个根、两个 split、五类任务、八类事件场景和 33 个极值见证。复验 4,790 对原 CDR 与 SampleObservation，来自 224 个原始文件，与 E1 见证物理索引重合为 0。正常、分区边界、尾段、重复发布、STALE、UNSAFE 和恢复各三例，DROPOUT 两例。[预注册计划](../evidence/P01_F/sample_plan.json) · [逐源见证](../evidence/P01_F/raw_reconstruction_report.json)

| 独立验证项 | 分母 | 结果 |
|---|---:|---|
| 原始关联与观察时钟 | 4,790 对 | CRC、物理索引、嵌入 schema/CDR SHA、实例、序号及观察时标一致 |
| 25 节点拓扑、SDK common、掌系和腕姿态 | 1,003 个有效逐流位姿 | 原始节点与 Frame 一致。平移复验最大残差为 0，旋转最大约 5.64e-16 rad |
| 软件 Tracker 配准 | 64 对 | 左右逐源按相同 raw/registered header 重建。最大平移约 1.24e-16 m，旋转约 4.79e-16 rad |
| 时间、因果最新可用样本与离线插值/速度 | 1,001 个速度值资格，归属上面的 1,003 个姿态 | 原始前驱及离线两端与掩码一致，差分数值最大复验残差为 0 m/s |
| 无效与安全 | 53 个逐流无效资格，176 个拍/模式安全核验 | 无效位姿没有启用速度。许可使用可追溯安全事件。三对重复 SDK 见证的原始节点相同 |

独立计算使用四元数乘积和旋转公式，以及线性平移和最短弧插值。冻结数值容差为 5e-12 m、2e-7 rad、速度绝对和相对各 1e-8。浮点重建只验证软件与记录语义。采集器时钟及 8 ms 模拟到达没有物理采样时钟资格。

| 极值 | 实测 | 见证结论 |
|---|---:|---|
| 左掌系节点因果速度 | 6.355 m/s | 采集时轴相邻间隔 65.706 μs，约 0.418 mm 位移。时间间隔主导该速度，保留代理时序不确定标记 |
| 右掌系节点因果速度 | 6.238 m/s | 采集时轴间隔 8.338 ms，邻域有联合无效拍。实体动作成因未确认 |
| 右注册腕与右 raw Tracker 因果速度 | 12.473 / 12.873 m/s | 同分区同一控制拍，原始差分间隔 8.286 / 8.304 ms。两路记录均有位置突变，邻域有联合无效拍。快速运动与跟踪/重新定位仍待外部判据区分 |
| 左 / 右掌系原始指尖对齐差 | 58.79 / 61.04 mm | 重建确认模拟保持与双端插值的几何差 |
| 双腕相对平移最大对齐差 | 58.37 mm | 样本两套独立历史对齐均具对应原始索引 |

Top-3 复查覆盖 11 个指标列。33 个局部窗口中有 9 个观察到邻近联合无效。每例保留 ±4 拍上下文、原始身份和索引、位置与旋转差、采集时间差以及安全与资格。样本计算与记录一致，当前证据不足以将最高速度认证为物理量或认定为某种传感器故障。处置为保留冻结 QC 及原值，后续可单独评估极短间隔和双路腕突变的质量影响。该诊断不改变 E1 资格。[尾部索引与处置](../evidence/P01_F/tail_anomaly_review.json)

代表性实测图包括左右骨架、左右腕轨迹、正常双侧运动、重复与时龄、UNSAFE 和恢复，以及速度、几何差、STALE 和缺测窗口。各图标注母源、分区、时刻和数据等级。

[25 节点骨架](../evidence/P01_F/manus_geometry.png) · [腕部轨迹](../evidence/P01_F/wrist_trajectories.png) · [双侧运动](../evidence/P01_F/normal_bimanual.png) · [重复与时龄](../evidence/P01_F/repeat_and_age.png) · [失效](../evidence/P01_F/unsafe_and_recovery.png) · [恢复](../evidence/P01_F/recovery_transition.png) · [腕速度](../evidence/P01_F/wrist_speed_tail.png) · [指尖差](../evidence/P01_F/tip_error_tail.png) · [STALE](../evidence/P01_F/stale_age_boundary.png) · [缺测](../evidence/P01_F/dropout_boundary.png)

## 全量窗口与 generalist 覆盖

因果联合有效 114,532 拍，884 段共 114,332 拍。离线联合有效 109,965 拍，2,061 段共 108,504 拍。同拍共同联合有效 109,574 拍。

| Split | 五拍姿态历史 | 五拍含独立速度 | 五拍 causal/offline 共同有效 |
|---|---:|---:|---:|
| Train | 82,559 | 82,421 | 74,459 |
| Validation | 28,237 | 28,201 | 25,519 |
| All | 110,796 | 110,622 | 99,978 |

| 产品 | 1 s 窗口 / 可容纳的段 | 2 s 窗口 / 可容纳的段 | 5 s 窗口 / 可容纳的段 |
|---|---:|---:|---:|
| 因果 | 79,711 / 537 | 57,899 / 363 | 20,192 / 177 |
| 离线 | 47,580 / 634 | 26,654 / 281 | 6,145 / 64 |

全量划分记录按母源、任务、split、日期与配置提供窗口计数。窗口相互重叠，计数表征覆盖。原捕获时长 2,331.957 s，来自 2026-11-12、13、20 三个日期（14 / 16 / 2 个母源）。采集配置快照有 32 个指纹，实际冻结软件腕外参 SHA 为 1 个。操作者与独立 session 均无证明。任务标签含 11 个手部、11 个臂部、6 个臂手、2 个双手、2 个 fruit 录制。

冻结位置速度诊断中静态 6,796 拍、动态 107,857 拍，不可判定 1,960 拍。左运动右保持 9,962 拍，右运动左保持 11,112 拍，双侧同时运动 86,783 拍。双手同时改变手型 45,831 拍，双腕同时平移 61,011 拍。在同一离线有效段内以 0.1 rad/s 作旋转诊断阈值，左右腕为 78,840 / 81,907 拍，同拍均旋转为 68,630 拍。观察到 157 次启动、165 次停止、2,436 次联合恢复和 71 次安全许可恢复。后续补采优先增加不同操作者、独立 session、双手任务、左右角色互换和包含纯腕旋转的轨迹。[全量分层统计](../evidence/P01_F/history_reference_statistics.json)

## Reader、放行与 F1

独立实现了 E1 archive Reader。对 Train/Validation 各执行三种用途的实际读取，共六次，并复核实际源 SHA。模拟因果视图在两个 split 上验证未来数组变异、删除及诊断元数据变异后，供 Actor 读取的五帧逐字段相同。还验证了分区开始、无效 segment、拓扑/标定/时钟变更、越 split 以及不独立的速度掩码等负控。历史窗口统计只准入相同分区的合格连续段。分段聚合和离线插值留在采样与诊断接口。[Reader 契约](../specs/HMB_reader_contract.md) · [Actor 白名单](../evidence/P01_F/actor_feature_allowlist.json) · [真包读取证据](../evidence/P01_F/reader_check.json)

| F1 技术条件 | 结果 |
|---|---|
| 生产归档完整、输入版本未变 | PASS |
| 源级 split、数量守恒与历史谱系 | PASS |
| 独立真包几何、时间和速度重建 | PASS |
| 预注册异常尾部复验及处置 | PASS（代理时序，物理成因待证） |
| Reader、历史边界及 Actor 因果输入 | PASS |
| 人工放行范围、依据和阶段隔离 | PASS，等待签核 |

GEOMETRY_DEVELOPMENT、SIMULATED_CAUSAL 和 OFFLINE_DIAGNOSTIC 三项技术资格已验证，等待人工批准相应历史仿真用途。HMB_REALTIME_VERIFIED、Robot Telemetry Bank、独立 Final Test 和物理 robot-base 均为 BLOCKED。P02 移交指向同一批准 E1 manifest，启动状态仍为 NOT_STARTED。P03 须同时满足 P02、P01-T、C2-RT 和物理 robot-base 的独立 ACCEPTED。[用途矩阵](../evidence/P01_F/data_qualification_matrix.json) · [P02 移交清单](../evidence/P01_F/p02_handoff_manifest.json)

## 命令、失败与复验

Ubuntu 原始产物位于 `/home/ubuntu/dexgym_data/s0-5/p01/runs/P01-F-20261126-b22f8be`。F1 用同一 E0 锁环境。全部 98 项单测通过。早期全量审计程序曾错误地假设诊断报告属于 D1 配置锁的 sidecar，保留了失败退出。早期旧 E0 授权负控依赖当前阶段，改为冻结阶段 fixture 后通过。首轮原包重建通过数值检查后图形渲染失败，后续修正图坐标与文件名并保持样本和阈值。所有早期日志和原结果分别保留。

```bash
cd /home/ubuntu/dexgym_worktrees/s0-5
python3 -B docs/s0-5/evidence/P00/check_documents.py --require-history
bash docs/s0-5/evidence/P01_F/run_f.sh archive 02
bash docs/s0-5/evidence/P01_F/run_f.sh tests 02
bash docs/s0-5/evidence/P01_F/run_f.sh raw 03 02
```

命令记录首次成功会话。复验应使用独立输出编号，正式 E1 归档只读。首次封存清单在启动日志写完前生成，最终封存在日志完成后从独立目录生成并重新核验全部 SHA。技术报告与正式手工签核区分。

[测试结果](../evidence/P01_F/test_results.json) · [完整输出 SHA](../evidence/P01_F/output_sha256.json)
