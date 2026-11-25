# P00 Review Fix 与 P01 Readiness Preflight

Run ID `S05-P00-REVIEW-FIX-20261126-CC05F09`。原始发布基线 `cc05f09847254ddae3b76581fb9e1feed4f31065`，GitHub `s0-5`。实际只读预检从同 SHA 的独立 Ubuntu checkout 加载源码。以下 PASS 针对六项修订交付，技术实测证据与设计验收分别记录。

## 修订验收矩阵

| Review ID | Required Fix | Actual Result | Status | Evidence | Remaining Risk |
|---|---|---|---|---|---|
| R01 | 唯一开发与执行基线 | 隔离 clone、实际解释器/overlay/模块/依赖 hash、当前资源、独立输出根均见证 | PASS | [执行基线](../EXECUTION_BASELINE.md) · [环境](../evidence/P00_review_fix/environment.json) | 外部 overlay 变动重验，P01-D 全量预算，P02/P03 模型与 Servo |
| R02 | Teleop 准入与双数据产品 | 两种原包有限 CDR 成功，见证 25 节点、旧类型漂移和遥测字段。定义 HMB、Telemetry 与排除规则 | PASS | [样本](../evidence/P00_review_fix/samples.json) · [P01 Spec](../specs/P01_teleop_data.md) | 原始录制、硬件版本与全量谱系须逐源认证 |
| R03 | 真正因果同步 | 核对双端 pose 与单独 index，定义两个分产品、时钟域、资格与前缀负控 | PASS | [源码](../evidence/P00_review_fix/sync_source.txt) · [时钟契约](../specs/P01_teleop_data.md#11-数学定义时钟与对齐) | 控制端到达未证，阈值由 P01-C/D 确定 |
| R04 | 独立划分及 Test 污染 | 建历史使用登记设计，查 5 个家族的入口或归档。Final Test 适格性未证，P09 设强制门 | PASS | [历史用途](../HISTORICAL_DATA_USAGE.md) · [现场证据](../evidence/P00_review_fix/historical.json) · [Marvin54 归档](../evidence/P00_review_fix/marvin54_archive_witness.json) | 旧 Marvin54 Ubuntu 目录缺失，全量谱系和封存待办 |
| R05 | 可执行 P01 Spec | v0.2.0 含两个 adapter、确切字段、质量分层、预算、拟实现验证入口及 A–F 定量门 | PASS | [P01 Spec](../specs/P01_teleop_data.md) | 人工冻结，驱动阈值与峰值预算在 P01-D 另签 |
| R06 | 证据一致性与历史可复验 | 结构、引用、哈希、实际退出和状态转换检查，原始 Git blob 回放及负控通过 | PASS | [检查器](../evidence/P00/check_documents.py) · [历史快照](../evidence/P00_review_fix/p00_snapshot_manifest.json) · [验证](../evidence/P00_review_fix/review_validation.txt) | 技术结论及授权真实性由人工审核 |

## 实际版本、命令和证据

- Ubuntu 通过 `git clone --depth=1 --single-branch --no-tags --branch s0-5` 建立 `/home/ubuntu/dexgym_worktrees/s0-5`，探查时 HEAD 与 GitHub 一致，初始 status clean。原 `~/dexgym` 和 `~/teleop_ws` 的 tracked diff 指纹均与原始 P00 快照一致。输出根 `/home/ubuntu/dexgym_data/s0-5/p01` 当轮为空且可写。实测约 637.4 GiB 可用，约 67.1 GiB RAM available。[执行基线](../EXECUTION_BASELINE.md)
- 系统 Python 初次探查因缺 `gymnasium` 退出 1。首次 PyPI wheel 下载读超时退出 2。重试下载、按 SHA 锁定后离线安装均退出 0。隔离 ROS/Python wrapper 三个模式 `environment`、`samples`、`historical` 均退出 0。保存结果、退出文件、stderr 与实际解析路径。[命令与退出](../evidence/P00_review_fix/commands.json) · [失败和复验证据](../evidence/P00_review_fix/README.md)
- 修改均在 `docs/s0-5/`。原始 `reports/P00_acceptance.md` 和历史命令日志留存。检查器验证历史 38 文件及当前结构、证据、状态，技术结论仍由人工审核。[逐文件变更](../evidence/P00_review_fix/review_change_list.txt)

## 录包、时间与来源的新发现

1. **分流样本。** Manifest 实际为 `dex-native-v2` 版本 2，`initial_state` 是带原时钟的 episode 前缓存。有限 CDR 读到左右 MANUS、左右 raw/registered Tracker 和 tracker safety 共七流，每流两个话题。MANUS CDR 直接看到 **25 个节点**，含 `node_id`、`parent_node_id`、`joint_type`、`chain_type` 和 pose。旧 `parsed.npz` 双手 `[5500,25,7]` 和双腕 `[5500,7]` 与此吻合。生产契约保留可变拓扑。
2. **合并样本。** 当前 ROS 类型读到双手 glove、双腕与手命令。两侧旧 `HandState` 按当前类解码失败。使用录包内嵌的 `ros2msg` 后，在 18 个选定遥测话题取得样本，每话题至多 12 条。见双臂 14 维 command、每手 20 维 command/state、腕 pose/twist、wrench、RobotState、TeleopState 与 safety。两臂 `JointState` 首条 position/effort 各 7、velocity 为空。手 `HandState` 首条 position/velocity/effort 各 20，嵌入类型无 current。每个源的缺失率、硬件身份与配对留给 P01-B/C，样本解码不授予整包系统辨识资格。
3. **模拟重录。** `s0_sim_20261117_153201` 的 metadata 也含左右 MANUS 和 Tracker。准入另核验采集来源、配置与母源指纹。
4. **时间。** 读取到两种格式各自 MCAP wire log/publish。分流每个首 32 条中两字段相等，合并样本首 32 条均不同。这只说明该读数。MANUS 捕获与 SDK 接收、ROS header、采集器 `SampleObservation` 的观测与接收字段保持各自时钟域。`sync.py:73–80` 按 `receive_ros_ns` 到达与 header 年龄求索引，`127–133` 的 pose 为双端插值。实机 controller receive 和物理响应时钟未证，代理及模拟时钟只支持对应等级。[源码见证](../evidence/P00_review_fix/sync_source.txt) · [原包证据](../evidence/P00_review_fix/samples.json)
5. **历史用途。** S0、S0-1、S0-2、S0-4 分别发现 8、8、9、8 个抽样 manifest。实际 S0-4 bank 含 train/validation/test。旧 Marvin54 Ubuntu 原名数据目录现不可见，Windows 既有归档有三种分集角色与 EgoProStandard-body 用途证据。本轮只读盘点不替代 P01-A 逐源使用登记，缺证来源不能视作独立 Final Test。[历史用途与封存](../HISTORICAL_DATA_USAGE.md)

## 后序约束与剩余工作

- P02/P03 将 `mapped_intent`、`feasible_scoring_target`、`q_oracle_offline` 与 `q_nominal_online` 分开，下发独立记录 requested/issued/applied/measured。冻结评分目标不依赖被评估执行，在线名义以可用人端输入和过去状态运行。
- P03 比较 50/500 Hz 等候选下的完整链路 P50/P95/P99、deadline miss、回退和精度。P02/P03 固定共同目标和 P09 主指标、非退化门。端到端意图、执行、投影、对捏及双手几何分报。主目标和精度挑战保留。
- **FAILED（已定位并留证）** 系统 Python 首次导入失败、首次 wheel 下载超时、当前 ROS 类型读取旧两手状态失败，已分别通过隔离环境、重试和嵌入 schema 恢复有限见证。
- **BLOCKED（后序证据）** 独立 Final Test 尚未证明，阻塞 P09。历史包缺真实 controller receive／物理时间链证据，相关真实时延验收需补测。旧 Marvin54 live root 缺失，使用历史归档启动登记。Windows 旧 checkout 缺原始 P00 Git 对象，历史回放在 Ubuntu 隔离 checkout 验证。
- **NOT_RUN** P01-A 至 F 正式执行、全量谱系/QC/split、生产因果对齐及其动态前缀验证、P01 峰值试验、P02–P10 生产功能、任何训练及完整控制链性能／真机试验。证据检查负控与生产算法测试分别验收。

## Gate Decision

**P00 保持 READY_FOR_REVIEW。P01 NOT_STARTED。** 受控预检支持申请 P01-A。实际启动还需 P00 人工 ACCEPTED、明确 P01 授权、Spec v0.2.0 人工冻结，并复查源、解析环境和小样本访问范围。P01-C/D 再实测并锁定阈值、split 和资源，P01-E 在前置门完成后才执行。独立 Final Test 资格为单独的 P09 前置门。[项目状态](../PROJECT_STATE.md)
