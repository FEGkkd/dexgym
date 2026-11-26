# P01-E 正式 HMB 全量构建验收

**E1 技术门 PASS，P01-E READY_FOR_REVIEW。** Run ID `P01-E-E1-20261126-3f90652`，正式输出范围 `E1_PRODUCTION`。32 个母源的 308 个分区全部发布，经独立反读及原包见证复验。P01 整体为 IN_PROGRESS，E1 等待人工签核。[运行凭据](../evidence/P01_E_E1/review_receipt.json)

## 批准的执行身份

项目用户批准 E0 审查提交 `3f906526edccc108a1ff08e523da9559b16abcb0`、原 E0 凭据和正式 Run ID。实际生产启动提交为 `cb0483a0b1de466dd67f5f6f01a92da1db09487e`，该提交在审核基线上登记授权并修复独立日志脚本。独立审计执行提交为 `93e026c36121569c44fa132661569f6b4e970bde`。原 D1 配置、D1 凭据、E0 候选、42 个锁定实施文件及实际依赖均通过复验。实际环境使用对应 Ubuntu worktree、Python 3.12.3、ROS Jazzy 与两个有序 overlay、锁定版本的 SDK、NumPy、SciPy 和 MCAP 解码库。[人工授权](approvals/P01_E_E0_ACCEPTED_P01_E_E1.md) · [配置](../evidence/P01_E_E0/candidate_config.json) · [依赖锁](../evidence/P01_E_E0/execution_lock.json)

| 对象 | SHA-256 |
|---|---|
| 获批执行配置 | `6aeaa203e39bea392bc20d5fceee26ea5f85752d219824f98f96b2c852846317` |
| 获批执行依赖锁 | `da3e1f0759ac010a989d9431f0337371df6fe208914139bd711be15354a453af` |
| E0 原凭据 | `4dd4c3ea4c1b22a669c9b6d67ccbb3ec49b7653128ea88c728ed9562319337a4` |
| E1 全量预注册计划 | `0f94b842e2558ec04da257aff94d469ae9b965968fff1a4593382fc2eea09404` |
| E1 HMB manifest | `e63df2698feb812bc300593d9bf8d1ad3dcae1d2dc59aa4d3a3dfa96851d8074` |
| E1 运行凭据 | `0930e732e92e6213742e8b48cd1921d980c1315c7c108bda1ea24c0c59de483b` |

启动前从 32 份采集清单、A1 Registry 和白名单独立核对 int64 时间区间，复验 224 个原始 MCAP 分片共 2,275,871,930 字节。按八秒分区和 20 ms 控制拍计算的 Train 228 分区 / 86,338 拍、Validation 80 分区 / 30,275 拍均与审核参考值相同。8 ms 模拟到达、60 ms age、35 ms 左右及腕手 skew、60 ms 离线 gap、100 ms safety timeout、最短段 5 拍、四元数容差 1e-6 保持冻结。[全量计划](../evidence/P01_E_E1/production_plan.json) · [资源预检](../evidence/P01_E_E1/resource_preflight.json)

## 数量守恒与质量

| Split | 母源 | 原始时长 s | 分区（发布 / 隔离） | 计划拍 | 发布拍 | 隔离拍 | Causal 有效 / 无效 | Offline 有效 / 无效 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 24 | 1,726.535 | 228 / 0 | 86,338 | 86,338 | 0 | 85,382 / 956 | 81,940 / 4,398 |
| Validation | 8 | 605.423 | 80 / 0 | 30,275 | 30,275 | 0 | 29,150 / 1,125 | 28,025 / 2,250 |
| All | 32 | 2,331.957 | 308 / 0 | 116,613 | 116,613 | 0 | 114,532 / 2,081 | 109,965 / 6,648 |

逐分区、逐母源及 split 核验计划拍等于发布拍加隔离拍，两个产品各自的发布拍等于联合有效拍加联合无效拍。Train/Validation 的母源交集为 0。308 个分区均为 ELIGIBLE。TAIL_INHERITED、INSUFFICIENT 和 HARD_FAILURE 均为 0，因此本次历史仿真样本没有短尾继承或额外隔离裁定。[源级处置](../evidence/P01_E_E1/source_disposition.json) · [分区索引](../evidence/P01_E_E1/build_index.json) · [split 审计](../evidence/P01_E_E1/split_isolation.json)

| 产品 | ≥5 拍连续段数 | 段内帧数 | 段长 P50 / P95 / P99 / max（拍） | 左 / 右手速度有效拍 | 安全许可拍 |
|---|---:|---:|---|---:|---:|
| Causal | 884 | 114,332 | 70.5 / 400 / 400 / 400 | 115,243 / 114,863 | 115,335 |
| Offline | 2,061 | 108,504 | 29 / 188 / 398 / 400 | 115,268 / 114,743 | 115,371 |

D1 外推为 112,733 个可用帧，E1 全量实测因果段内 114,332 帧。联合有效率为因果 98.22%、离线 94.30%。实际保留 1,199 个因果 UNSAFE 拍、272 个 STALE 流拍和 6,993 个 HELD 流拍。各流独立姿态、速度、许可、原因码和所有段都按索引复验。[覆盖及失效原因](../evidence/P01_E_E1/full_coverage.json) · [有效段](../evidence/P01_E_E1/segment_statistics.json)

### 运动、时钟与几何诊断

- 使用 D1 诊断阈值在离线位置速度上标记 107,857 个动态拍、6,796 个静态拍、1,960 个运动不可判定拍。双手同时改变手型 45,831 拍，双腕同时平移 61,011 拍，双侧同时运动 86,783 拍。已计 157 次启动、165 次停止及 2,436 次同分区联合有效性恢复。纯旋转另由姿态统计表征。按真实标签分层包括 11 个 `arm_motion`、11 个 `hand_pose`、6 个 `arm_hand`、2 个 `bi_hand`、2 个 `fruit` 母源。独立 `bi_hand` 标签的覆盖仍有限。
- MANUS 两侧原始发布率约 119.96 Hz。按连续 writer sequence、SDK source/receive 原始值以及有效性计的独立身份候选率为左 106.07、右 95.12 Hz。相邻且 SDK 时间字段对相同的发布分别有 32,337 和 57,885 条，左右各一个样本对从原始 CDR 复验出相同骨架。上述速率来自录制代理时轴。
- 模拟因果姿态年龄 P95 左手 / 右手为 22.09 / 27.20 ms，左右注册腕为 15.99 / 15.99 ms。双手、双腕、左腕手、右腕手 skew P95 分别为 16.66、0.12、9.89、16.85 ms。全量报告保留包括超过联合有效上限的逐流诊断配对。离线 bracket span 和模拟因果 age 分别记录。
- 共同有效样本的因果保持与离线插值诊断中，原始掌系指尖节点差 P95 左 / 右为 2.38 / 2.31 mm，最大为 58.79 / 61.04 mm。注册腕平移差 P95 为 2.85 / 3.14 mm，双腕相对平移及旋转差 P95 为 9.44 mm / 0.037 rad。SDK common、掌系地标、指尖、各腕旋转及速度差均在 QC 中单列。因果左右手速度超过审阅标记的节点值为 1,806 / 2,518 个。离线骨长异常标记每侧 4 个边值，均保留。以上几何差异为两种历史对齐方式之间的诊断，机器人世界系十指跟踪需要独立实验。[全量 QC](../evidence/P01_E_E1/full_qc.json)

## 原包回读、资源与失败记录

全量独立复验对每个正式分区加载 causal 和 offline 并校验 schema、掩码、版本、时间界、拓扑、来源索引、段和内容哈希。原始回读从 14 个 MCAP 文件复验 33 对载荷与 SampleObservation，覆盖左右 MANUS、左右 raw/registered Tracker 和 Safety，且覆盖 Train/Validation 正常拍、UNSAFE、恢复、自然尾段、左右 MANUS 重复发布。按记录物理索引重取消息，核对 CDR/schema SHA、观察器序号、时钟和实际可用姿态值。两个重复发布见证的各自原始几何相同。[原包回读](../evidence/P01_E_E1/raw_traceability.json)

正式单 worker 构建历时 28:32.27，独立反读历时 1:58.75。成功分区逻辑读取 88,842,585,706 字节，含重复原包校验以及发布与语义反读。按分区重复访问的 MCAP 文件计 23,476,430,258 字节，与 D1 同口径估算相同。Linux 分区阶段物理读取计 887,537,664 字节，受缓存影响。CDR 解码含配对观察器计 4,135,920 条，累计 269.24 s；同步累计 312.16 s。逻辑写计 7,202,639,309 字节，双产品合计 7,116,839,133 字节，含索引和台账的归档为 7,121,396,746 字节。最大单分区 24,663,998 字节，构建峰值 RSS 506,748,928 字节，观测最低可用磁盘 675,174,309,888 字节，剩余临时分区 0。[资源报告](../evidence/P01_E_E1/runtime_resources.json)

会话 1 的独立日志脚本预检退出 127，未进入 Python 生产入口。保留了原退出证据并修正脚本，该脚本独立于批准的执行依赖锁。会话 2 重新验证同一批准身份后，预检、91 项单测和正式构建均退出 0。正式台账记录 308 PLANNED、308 RUNNING、308 PUBLISHED，生产 FAILED、PAUSED、RESUMED 和 QUARANTINED 均为 0。两轮真包 smoke 与全量独立审计均退出 0。授权范围、批准类型及两项哈希的篡改负控均拒绝。[测试与命令](../evidence/P01_E_E1/test_results.json) · [生产台账](../evidence/P01_E_E1/build_log.jsonl)

## E1 验收矩阵

| 条件 | 实际 | 状态 |
|---|---|---|
| 冻结来源、原始区间、plan、配置和实施锁 | 32 个母源、224 个分片和 42 个实施文件复验一致 | PASS |
| 数量守恒及根级 split | 308 个分区及 116,613 拍均有处置，逐源、逐分区、逐产品守恒，根交集 0 | PASS |
| 独立的双产品及首次发布前检查 | 308 个分区的双产品通过 staging 校验和完整独立反读 | PASS |
| 安全、有效性、位置与速度 | 来源、掩码、速度资格、时间界、有效段边界完全复验 | PASS |
| 时间和允许用途 | Causal 为 SIMULATED，Offline 为 OFFLINE_PROXY，VERIFIED 为 0，数据供历史仿真开发 | PASS |
| 代表性实际原包重建 | 33 对原载荷与 SampleObservation 成功，所有必需场景具见证 | PASS |
| 生产台账、失败及哈希完整 | 原启动失败可追踪，正式生产无隐藏处置，所有正式分区及最终索引有 SHA | PASS |
| C2-RT、物理 robot-base、P01-T、Final Test | 分别保留独立未满足条件 | BLOCKED |

## 复现与停止点

原包、正式数据分区、完整逐分区处置、stdout/stderr 和运行副证据托管在 Ubuntu 的 `/home/ubuntu/dexgym_data/s0-5/p01/runs/P01-E-E1-20261126-3f90652`。正式 manifest、索引、台账、汇总统计及完整产物 SHA 目录保存在仓库。分段及未来诊断仅服务诊断和数据组织，后续 Actor 使用独立因果输入契约。[正式 HMB manifest](../evidence/P01_E_E1/hmb_manifest.json) · [产物及日志 SHA](../evidence/P01_E_E1/output_sha256.json)

```bash
cd /home/ubuntu/dexgym_worktrees/s0-5
python3 -B docs/s0-5/evidence/P00/check_documents.py --require-history
bash docs/s0-5/evidence/P01_E_E1/run_e1.sh 2
bash docs/s0-5/evidence/P01_E_E1/run_review.sh review 1
```

上面记录本次实际会话及审核序号。对已完成归档复验使用新的审计序号。生产续跑按原授权使用相同 Run ID、配置和执行锁以及未占用的会话序号。P01-F 继续 NOT_STARTED，须等待人工签核并获得新授权。
