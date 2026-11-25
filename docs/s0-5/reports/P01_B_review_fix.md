# P01-B 独立审核修复与依赖复验

| 状态或门 | 结论 |
|---|---|
| 历史 P01-B / B1 / B2 | BLOCKED / FAIL / BLOCKED。原产物保留。 |
| 当前 B-HMB 技术门 | PASS，限七个来源已核实的原生录制 |
| P01-B-HMB | READY_FOR_REVIEW，等待人工验收 |
| 辅助 schema 兼容性 | RECOVERED，旧类型精确复验 119／119 |
| P01-T / B2 | BLOCKED，动态系统辨识候选为 0 |
| P01 / P01-C–F | IN_PROGRESS / NOT_STARTED |
| 独立 Final Test | BLOCKED |

审查基线 `f32d6bbfc4f535a5c53db1025a484e48fc5f2ce3`。新版 [P01 v0.2.1](../specs/P01-data-v0.2.1.md) 按 [DCR-001](../dcr/DCR-001_p01_hmb_telemetry_dependencies.md) 更新依赖。项目用户批准本轮 Review Fix 和所述依赖调整，新技术门待人工验收。

## CollectionStatus 根因与影响

原 attempt-04 的 119 个失败都属于嵌入定义 `b8ded4b4…` 和录包 DDS 类型 `RIHS01_ce543f80…`。当前两个 overlay 的类型为 `RIHS01_56af1b62…`。有限调查覆盖历史 .msg、嵌套定义、生成工件和两仓库触及这些接口的有限 Git 记录。

在采集器旧 `ros2_ws/install/dex_collection_interfaces` 中找到匹配的类型。重新计算历史生成类型描述的 RIHS，并核对旧 Python 生成类与 C Fast RTPS typesupport 的加载路径及 SHA。历史 `StreamStatus.msg` 也匹配采集器 Git 提交 `2fc969e94eab939d4525e080cb627a5a1e169fe0`。顶层 wire 字段一致，录包嵌入的 StreamStatus 比历史发送端少 10 个嵌套字段，包括 `received_bytes`、`queued_samples`、`source_sequence_available` 和 gap 计数。CDR 载荷与嵌入的嵌套定义之间的版本差异解释了原解码失败。

| 录制 | 原物理消息索引范围 | 原失败 | 历史解码恢复 |
|---|---:|---:|---:|
| `s0_sim_20261117_153201` | 105–6548 | 32 | 32 |
| `teleop_no_camera_20261016_133930` | 16–15563 | 32 | 32 |
| `bag_no_camera` | 391–13177 | 23 | 23 |
| `head` | 15811–16351 | 32 | 32 |

索引范围包含其他消息。逐条列表保存在 `legacy/legacy_raw_cdr.jsonl`，含原始 CDR、索引、录制/分片 SHA、CDR SHA、嵌入 schema SHA、原失败和类型哈希。恢复值来自隔离的历史编译接口。另用有 Git 和生成描述佐证的历史 .msg 重组 ros2msg，首条恢复值与旧编译接口一致。保留原 P01-B、B1 和失败日志。[恢复凭据及来源](../evidence/P01_B_review/review_receipt.json)

### 对 HMB 准入的影响

按 [Topic Dependency Matrix](../specs/topic_dependency_matrix.md) 逐源判定。通过的 7 个原生见证具有 A1 来源、writer manifest 与快照、采集器身份、标定文件 SHA，以及同一 BODY part 的 payload–Observation 配对。注册腕位保留 explicit valid/quality，单独 tracker safety 保留 safe/code。CollectionStatus 没有承担这些录制的来源、完整性、有效性、安全或同步证明。

受影响的四条 merged 录制属于诊断样本，其中一条已证实仿真，三条持续隔离。若未来某条录制以状态消息证明来源、完整性、安全、有效性或同步，必须核验与原包和相应物理信号的绑定。解码恢复不授予准入。32 个真人母源候选、89 个隔离视图及 100,071 条未完全解析历史用途记录的口径不变。

## HMB 技术复验

对 14 个冻结录制逐一裁定，7 个通过、7 个保持隔离。重新从 7 个原始录包读取有界前缀，798 条新读事件的分片、索引、CDR、schema、字段、时钟与配对均和历史证据一致。保留 raw/registered wrist、原始骨架顺序、节点、父关系、joint_type、chain_type，并校验独立标定文件 SHA。P01-C/D 的物理标定、左右/Tracker 同步、年龄和 QC 均未运行。

39 项 P01 单测通过，其中 9 项为本轮。必要 MANUS／Tracker 解码失败使门 FAIL，缺安全字段不得通过，缺同步或标定证据使门 BLOCKED。来源未知仍隔离。辅助消息没有准入依赖时失败独立报告。辅助字段作为必要证明时，同一失败会阻塞。无新增训练资格。

## MANUS 时间前置事实

固定 `libManusSDK-amd64.so` 构建。Git blob 为 `363be71c6609a9624002e076c7d5e819d231fc28`，SHA-256 为 `b4e98f04144585cf51428c41e6af202dc664a599fb70a61adbcfe6ff0958825b`，ELF build ID 为 `2099bc6e34d0e330445e47da5c80e1a9e3e36b77`，已核对 Teleop commit `30cf13bdc496c43b3016462cfa0a206334d1ff7f`。厂商发行版本未验证，SDK 版本查询需连接 Core。本次使用 attempt-04 的原值及离线组件结果，未启动 Core。

2,336 条 SDK 原始压缩整数由 `CoreSdk_GetTimestampInfo` 返回 year/month/day/hour/minute/second/fraction，均为非 timecode，fraction 表示日历内毫秒，组件差中位数为 8 ms。压缩整数的差不能作为普通微秒。左侧 100 次、右侧 206 次相邻重复，共 306 次具有相同 SDK receive 和骨架且 ROS header 变化，符合缓存骨架重复发布。真实捕获身份、跨设备时钟及控制器接收关系均待验证。P01-C 需验证新捕获与重复发布区别、有效年龄、映射误差、左右手与 Tracker 同步及前缀不变性。

## P01-T 审查与补录

对 37 个缺硬件证明的去重视图定向检查绑定的目录和归档。发现 36 个 merged metadata，以及一条 native 的 manifest、effective config、events，无匹配的录制期运行快照、独立设备身份证明或控制器/反馈时间文件。完整清单在 `telemetry_provenance_scan.json`，SHA-256 为 `4eb03334cb0b9ec51b46ebbcc2724bb588f5bd7b0123c22773a7e9a34aafcb12`。未登记的外部原始来源仍待补证。原 sample plan 无 attestation。1,824 条关节轴记录均缺已证发令时钟，1,504 条缺已证实际反馈时钟。

`verify_attestation` 核对 SHA、artifact 角色和包绑定，`validate_bindings` 核对索引、记录角色、轴名称集合及外部声明。这些检查无法确认测量事实、实际顺序、丢帧、时钟误差和各组动态覆盖。[P01-T 采集协议](../specs/P01_T_telemetry_acquisition.md) 规定专项补录和实测验收。真机动作、采集或硬件变更均未运行。

## 运行复验与退出条件

正式 Run ID 为 `P01-B-REVIEW-20261126-f32d6bb-r02`。Ubuntu 独立 `s0-5` 与 GitHub 开工 SHA 一致。复核原 P01-A 11 个输出、P01-B 15 个输出、15 个模块、9 个诊断文件及 21 个运行时 pin。初版输出清单先于 stdout 完成，保留初版。正式复跑分别封存稳定输出及进程日志，核验 20 项稳定 SHA。环境预检、单测、历史解码及 HMB 复读进程均退出 0。

Windows 稀疏镜像的辅助预检因历史 Git 对象缺失、CRLF 转换和缺 gymnasium 无法复验原快照与训练包导入。正式验证使用 Ubuntu 独立完整运行 checkout。新增阶段依赖校验分别拒绝提前启动 P01-C、无 F 签核退出 P01，及无 P01-T 签核启动 P03。

```bash
cd /home/ubuntu/dexgym_worktrees/s0-5
run=/home/ubuntu/dexgym_data/s0-5/p01/runs/P01-B-REVIEW-20261126-f32d6bb-r02
old_a=/home/ubuntu/dexgym_data/s0-5/p01/runs/P01-A-20261126-09c1c9e-r03/registry
old_b=/home/ubuntu/dexgym_data/s0-5/p01/runs/P01-B-20261126-46ed5e2/attempt-04
bash docs/s0-5/evidence/P01_B_review/run_review_fix.sh "$run" "$old_a" "$old_b"
python3 -B docs/s0-5/evidence/P00/check_documents.py --require-history
```

复现时使用新的 Run ID。完整 HMB 判定、topic 分类、遥测审查、历史逐条 CDR/恢复值、命令、日志及 SHA 保存在 Ubuntu。GitHub 发布设计、脚本和证据摘要。[运行凭据](../evidence/P01_B_review/review_receipt.json) · [进程 SHA](../evidence/P01_B_review/process_sha256.json)

P01-C 启动须人工接受 B-HMB，确认 CollectionStatus 逐源处置和历史失败、DCR 批准、P01-T 独立门槛、运行基线、源及 Final Test 隔离，再另行批准新版范围中的标定、时钟与因果测试。目前停在 B-HMB 人工审核。P01-C、全量重建、P02 和训练均未开始。P01-T 仍阻塞 P03。
