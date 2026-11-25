# P01-A 来源、谱系与历史用途验收

| 项目 | 实际状态 |
|---|---|
| P00 | ACCEPTED，见 [人工签核](approvals/P00_ACCEPTED_P01_A.md) |
| P01 / P01-A | IN_PROGRESS / READY_FOR_REVIEW |
| P01-B–F | NOT_STARTED |
| 批准基线 | `s0-5`，`09c1c9e3c63797083a82493e2a40dd8bb4b1db6d`，P01 Spec v0.2.0 |
| 正式审查 Run ID | `P01-A-20261126-09c1c9e-r03` |
| 独立 Final Test | BLOCKED，P09 前需新的或可证独立的母源和封存证据 |

## 执行基线与真实命令

Ubuntu 在 `/home/ubuntu/dexgym_worktrees/s0-5` 开工时 HEAD、GitHub 分支和审查对象一致，工作树干净。运行时使用该 worktree 的 `.venv/bin/python -P`（Python 3.12.3、MCAP 1.6.0、mcap-ros2-support 0.6.0）。加载 ROS Jazzy、Teleop install、采集平台 install，按此顺序 source。ROS MCAP 存储插件版本是 `0.26.11-1noble.20260903.070458`。复查 `sys.path` 及模块 `__file__`，旧 S0 bag/sync 均解析到独立 worktree。预检记录消息定义、type-support 与包版本，校验固定运行时 SHA。输出目录可写且独立于原始数据，预检有约 637 GiB 磁盘和 67 GiB 可用内存。实际资源和外部工作树修改指纹见 [运行凭据](../evidence/P01_A/review_receipt.json)。

```bash
cd /home/ubuntu/dexgym_worktrees/s0-5
git rev-parse HEAD
git branch --show-current
git status --porcelain=v1 -uall
git ls-remote --heads origin refs/heads/s0-5
run=/home/ubuntu/dexgym_data/s0-5/p01/runs/P01-A-20261126-09c1c9e-r03
bash docs/s0-5/evidence/P01_A/run_p01_a.sh "$run" 01
python3 -B docs/s0-5/evidence/P01_A/summarize_run.py "$run"
python3 -B docs/s0-5/evidence/P00/check_documents.py --require-history
git diff --check
```

第一次运行前，在清洁环境下使用既有 `readiness_probe.py environment`，并执行 `dpkg-query`、`df -B1`、`free -b`、`nvidia-smi` 和 checkout Git 查询。环境命令、退出码、stdout/stderr 与哈希保存在 `resource_preflight.json`。上述 registry runner 依次执行 unittest unit suite、`python -m dexgym.tasks.direct.wholebody_s05.data.source_registry` 以及真实集成 suite。展开的 CLI 参数见 `attempt-01/registry-command.txt`，日志和退出值见 run 内的 `attempt-01/`、`test_results.json`。复验需要新 Run ID、同版本宿主依赖和可访问的现有根目录，使用同一 runner 和资源预检，新产物写入独立目录。

## 真实盘点与分母

| 对象 / 分母 | 数量 |
|---|---:|
| 发现的录制目录视图（原生 / DexGym 原始索引 / Teleop） | 152（33 / 30 / 89） |
| MCAP 文件实例（直接 / ZIP 成员） | 651（622 / 29） |
| 同字节合并后的 MCAP payload / 重复实例 | 382 / 269 |
| 多资产录制视图 / 此类视图引用的资产实例 | 63 / 533 |
| 多分片的单 writer 组 / 该组分片总数 | 0 / 0 |
| 逻辑录制视图 / 字节去重后的录制 | 181 / 122 |
| 录制视图处置（准入候选 / 重复别名 / 明确排除 / 未确认隔离 / 其余待解码） | 32 / 59 / 1 / 89 / 0 |
| 可确认真人捕获母源 / 已确认 replay 录制 / 来源不明录制 | 32 / 1 / 89 |
| 排除为原始采集的历史衍生元数据对象 / ZIP 容器 | 810 / 28 |

计数中的资产实例包含 symlink 访问及归档成员。完全相同的分片组指定 canonical 视图，重复资产单列，在捕获母源计数中只记一次。分流包的多个 stream 或 episode 前状态文件与同一 writer 的连续分片分开计数。排序优先采用 sidecar，缺失时保留文件名编号推定与候选原因。ZIP 成员流式校验，不解压整包到源目录。资产的 `ADMITTED_CANDIDATE` 仅表示资产完成字节校验，训练资格在录制层判断。

逻辑录制凭采集器身份、episode/单调窗口、运行 profile、实际发布者、必要四路流及文件指纹认定。捕获快照中的 tracker 启动配置按录制时登记的 SHA 校验。该证据只支持记录来源和配置身份，标定几何、控制节点时钟及逐帧有效性在 P01-B/C 验证。无来源证明且有相同 topic 的旧合并包依然隔离。已证实的 replay 排除。非同字节的复杂关系依据父级 manifest、原始区间、共享分片或捕获窗口登记；没有足够证据的旧包留在待验证索引。

32 个母源的元数据总时长约 38 分 52 秒。日期为 2026-11-12（14）、11-13（16）、11-20（2）。四路 MANUS/Tracker topic 均为 32/32；任务标签为 arm_motion 11、hand_pose 11、arm_hand 6、bi_hand 2、fruit 2。采集 session 标签均未能独立确认。累计时长按录制窗口计算，解码后的有效覆盖率待验证。

## 历史用途与两个产品的候选资格

历史元数据对象共 810 个，S0 91、S0-1 357、S0-2 220、Marvin54 1、S0-4 141。记录有 `DECLARED_SPLIT` 103,967 条、`RUN_STATE_AND_BANK_SPLIT` 8,652 条，另有 744 条未知使用记录。100,071 条历史事件尚不能完整解析至母源，计入待核。已定位的已知母源中，18 个有 TRAIN 标签，6 个有反复测试和模型选型证据；类别可重叠。

既有 manifest 的 split 声明与实际运行证据区分登记。S0-4 的 run state 及绑定的 bank 用于识别训练、反复评估和选型记录。已知来源的角色按父边传播到全部已知母源。缺少原始索引、裁剪链或真实评估日志的请求保留为未知；尤其 S0-2 和 Marvin54 的部分对象只能确认历史存在。历史未匹配的引用数是事件数，含多个产物对同一源或片段的引用。训练集和验证集比例本轮未冻结。

Human Motion Bank 目前有 32 个原始真人元数据候选。按录制视图计，59 个同时列出命令和反馈 topic，但缺独立实机来源证明；121 个无完整命令/反馈配对，另 1 个 replay 排除。已认证为 System Identification / Robot Telemetry Bank 的录制为 **0**。标定已核的真人录制仍可作为 HMB 候选。

## 新增时钟及消息问题

- 已有 P00 样本中的左手 20 条消息，ROS header 间隔 P50 为 8.334 ms，MCAP log 为 8.349 ms，SDK receive 字段为 8416 个原始微秒单位。名为 `source_timestamp_us` 的源字段间隔 P50 为 8 个原始单位，右手 20 条样本有 7 次相邻重复。
- 本轮读查 Teleop `src/manus_ros2/src/ManusDataPublisher.cpp`。第 372 行将 SDK `publishTime.time` 直接复制到源字段，第 373–374 行将接收 `steady_clock` 时间转换到微秒。源字段的物理单位、与 ROS/MCAP 的时基关系、重复成因仍为 UNKNOWN。P01-A 沿用历史样本核对源字段，未重新解码，也没有对源字段乘除 1000。
- 采集器 `recorder.cpp` 保留录制观察者携带的时钟；控制节点到达须用独立证据核实。旧 HandState 需按嵌入定义重新解码。P01-B 保留 MANUS 原有节点数、`joint_type`、`chain_type`、父节点和原始顺序，P01-C 验证语义、坐标和因果时间。见 [I17/I18](../OPEN_ISSUES.md)。

## A1 审核矩阵

| 条件 | 预期 | 实际 | 状态 | 证据 |
|---|---|---|---|---|
| 执行环境 | 使用获批 checkout，留存消息类型、依赖、源码和资源凭据 | 复验 exit 0，关键文件 SHA 通过，历史 checkout 追踪差异哈希一致 | PASS | `resource_preflight.json`、运行凭据 |
| A1 目录盘点、去重及数量守恒 | 已发现对象均有处置和原因，重复不挤占母源计数 | 三个源根、文件/视图/母源分开核算，已校验文件哈希及全量计数 | PASS | `registry.jsonl`、`rejects.jsonl`、`source_inventory.json` |
| A1 准入原始真人、谱系可追溯 | MANUS/Tracker 四路和捕获来源可证，校准来源有指纹，replay 与未知来源不可准入 | 32 个真人母源和 32 个启动标定配置 SHA 通过。89 个未知旧合并/归档录制保留隔离 | PASS（元数据门） | `lineage.jsonl`、`pending_verification.jsonl` |
| A1 历史用途登记和 Final Test 防污染 | 覆盖历史家族，原始母源继承已知用途，未证独立全部拒入 Final Test | 有证据的角色已登记，缺历史需补证，Final Test 候选为 0 | PASS（隔离规则） | `prior_use.jsonl`、`final_test_eligibility.json` |
| A1 可复验 | 先通过语义单测，再走真实盘点和固定取证 | 15 项单测、4 项真实集成测试通过（退出码 0），记录字段、产物 SHA 和数量守恒均已检查 | PASS | `attempt-01/`、`test_results.json`、`output_sha256.json` |
| 真实人手几何、遥测资格与控制器到达 | 真实消息、设备及标定/时钟证据 | 仅登记 topic 和元数据候选，未签发系统辨识资格 | NOT_RUN（P01-B/C） | `telemetry_candidates.json`、`timing_source.json` |
| 完整历史、封存独立评估源 | 已知母源互斥、独立采集及访问记录 | 三处历史 Marvin54 路径缺失，无独立封存证据 | BLOCKED（独立 Final Test） | `final_test_eligibility.json`、`pending_verification.jsonl` |

首轮 `P01-A-20261126-09c1c9e` 的 13 项单测和 4 项实测通过，但 A1 返回 BLOCKED，首版仅查 `sources`，漏读 tracker 启动配置。第 2 次写到 `…-r02`，补入实际启动配置哈希检查、重复运行单测及区间字段。第 2 次的 A1 返回 PASS，15 项单测与 4 项集成通过。第 3 次写到独立目录验证冻结的登记字段和引用。历史运行的原始输出及结果保留，预检阶段的调试查询曾因假设 `sources` 是字典失败，正式预检退出码为 0。

## 存储、交付与下一步

Ubuntu 运行根为 `/home/ubuntu/dexgym_data/s0-5/p01/runs/P01-A-20261126-09c1c9e-r03`。以下 12 个文件位于 `registry/`。根目录里的 `test_results.json` 包含真实集成测试，`registry/test_results.json` 记录盘点前单测，二者分别保存其 SHA。

```text
registry.jsonl                     lineage.jsonl
prior_use.jsonl                    rejects.jsonl
pending_verification.jsonl         source_inventory.json
dataset_coverage.json              telemetry_candidates.json
final_test_eligibility.json        resource_preflight.json
test_results.json                  output_sha256.json
```

本次提交文件按用途分组。原始录包和详细索引留在 Ubuntu。

- 审批与状态 `reports/approvals/P00_ACCEPTED_P01_A.md`、`phase_state.json`、`PROJECT_STATE.md`。
- 审计与追踪 `reports/P01_A_acceptance.md`、`HISTORICAL_DATA_USAGE.md`、`TRACEABILITY_MATRIX.md`、`OPEN_ISSUES.md`、`DECISION_LOG.md`、`RISK_REGISTER.md`。
- 运行证据 `evidence/P01_A/{host_registry_config.json,run_p01_a.sh,summarize_run.py,review_receipt.json}`。
- 实现 `wholebody_s05/__init__.py`、`wholebody_s05/data/{__init__.py,source_registry.py,inventory.py,lineage.py,historical_usage.py}`，位于 `source/dexgym/tasks/direct/`。
- 测试 `tests/s0_5/unit/test_p01a_registry.py`、`tests/s0_5/integration/test_p01a_real_registry.py`。

P01-B 待人工审阅及后续授权。P01-C/D 的标定、因果质量门、数值阈值与处理预算，P01-E/F 的全量构建和验收，以及所有训练均未执行。
