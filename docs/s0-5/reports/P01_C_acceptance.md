# P01-C 验收报告 · BLOCKED

| 项目 | 记录 |
|---|---|
| 执行日期 | 2026-11-26，Asia/Shanghai |
| 运行基线 | GitHub `s0-5` 与 Ubuntu 独立工作区均由 `8b0f39916e2fd1104ab39f8a6336c9196fc1c943` 开始，变更后的执行文件逐一记录 SHA |
| 规格 | P01 v0.2.1、C 实施规格 v0.1.0。样本与规格于 11:16:06 +08:00 封存 |
| Run ID | `P01-C-20261126-8b0f399`，正式证据 `attempt-03` |
| 授权 | P01-B-HMB 由项目用户签核 ACCEPTED。P01-C 为本次唯一执行子阶段 |

[人工签核](approvals/P01_B_HMB_ACCEPTED_P01_C.md) · [实施规格](../specs/P01_C_calibration_and_causal_sync.md) · [冻结样本](../evidence/P01_C/sample_plan.json) · [运行凭据及 SHA](../evidence/P01_C/review_receipt.json)

## 验收矩阵

| 门 | 强制验收项 | 实际 | 状态 | 证据及余项 |
|---|---|---|---|---|
| 前置与来源 | HMB 已签核，复验录制身份、文件、writer/collector、schema、配置和窗口边界 | 7/7 选中母源、49 个核心 MCAP 分片通过，读取上限内。复验 A / B / B Review Fix 产物 11 / 15 / 20 个，另核验 B 的 15 个执行文件和 9 个诊断哈希 | PASS | `fixtures_manifest.json`、`resource_preflight.json`。其他 25 个候选未做本阶段几何及同步验证 |
| C1 几何 | 原始节点、拓扑、左右身份、单位、XYZW、frame | 13,798 条窗口内 MANUS 载荷含上下文，14 个录制×侧分组均为 25 节点。原始 SDK 空间与派生掌系分别保存。腕部为右手系米制 | PASS（有界） | `geometry_contract.json`、`manus_topology_report.json`、`coordinate_frame_report.json`。空 ROS MANUS frame ID 的外部参照保留 UNKNOWN |
| C1 标定 | 使用录制期双侧外参，从原始包重建注册腕并与独立记录比较 | 13,640 对同侧同 header 的 raw/registered 腕位。未配对 raw 3 条、registered 4 条 | PASS（有界） | `calibration_report.json`。独立机器人基座的物理注册仍 BLOCKED |
| C2 因果算法 | 双时钟上界、前缀不变、批流一致、遮罩和每拍索引追溯 | 23/23 窗通过变异、删除、锚点和批流对照。检查 1,173 个过去输出拍，变异或删除共 23,984 条后续窗口事件。逐拍复建 valid 状态通过 | PASS（SIMULATED） | `prefix_mutation_results.json`、`source_trace.jsonl`、`test_results.json`。连续真包载荷使用同一个在线核心及明确模拟的到达时间 |
| C2 真实时序 | 独立证明设备物理捕获及控制节点到达上界，并建立跨时钟映射 | 无可证锚点，0/2,300 拍具备 VERIFIED 联合时序资格 | BLOCKED | `clock_mapping_report.json`、`clock_quality.json`。collector 时间只能诊断，不能补写 controller receive |
| P01-C 退出 | C1、C2 强制项均满足并等待人工审核 | C1 通过，C2 算法通过，C2 真实时序阻塞 | BLOCKED | 保持 P01 IN_PROGRESS，提交本报告审核。P01-D–F 仍 NOT_STARTED |

## 真包覆盖与几何

计划固定使用 11 月 12、13、20 日的 7 个原生 `real_no_head` 母源。任务标签为 `hand_pose`（2）、`arm_motion`（1）、`arm_hand`（1）、`bi_hand`（2）、`fruit`（1）。各使用头、中、尾窗，并在 `arm_motion`、`fruit` 母源按预注册规则加入两个安全事件窗。共 23 个 2 秒窗口、46 秒计分时间和 47,983 条有界正文载荷（含上下文），核心所涉原包体积 490,239,228 bytes。读取窗口是该批开发诊断的分母。计划在样本几何及覆盖数据可见之前封存。

按捕获快照恢复 Teleop 源码与配置。`ManusDataPublisher.hpp` 使用 `m_WorldSpace = true` 与 `XFromViewer/PositiveZ/Right/1.0`。`ManusDataPublisher.cpp` 传入 SDK 配置并原样发布各节点 transform。`ManusSDK.h` 区分 global/local node transform，`ManusSDKTypes.h` 定义 `unitScale` 的米制含义。以根旋转逆变换和根平移构建掌系，用共同空间的两点差计算骨长。根节点自身作父节点可接受，非根环、缺父和左右错误会拒绝。左右拓扑、关节/链类型和原节点顺序都进入报告和窗口索引。存有两张真包示例图。[原始与派生掌系示例一](../evidence/P01_C/manus-01.png) · [示例二](../evidence/P01_C/manus-02.png)

`vive_tracker_node.cpp` 直接读 OpenVR device-to-absolute 变换。`tracker_pose_transformer_node.cpp` 一路计算 raw 至 registered，保留源 header。全部选中录制使用同一份已哈希的 tracker 外参快照，捕获期 Teleop 组件配置存在不同版本。计算采用 `T_target_parent T_parent_raw T_raw_wrist`，对左右外参、正反向、正交性和行列式做过检查。记录的目标 frame 为 `openvr_world`。自带的 `world_adapter_node.cpp` 旋转是坐标约定证据。物理 robot-base 相对位置和方向缺独立外部测量。

以下是各侧录制分层统计的最大值，完整每录制 × 侧统计及索引见 `calibration_report.json`。这些量描述按配置的软件重建一致性。

| 侧 | 匹配（raw 未匹配，registered 未匹配） | 平移 P50 / P95 / P99 / max（m） | SO(3) P50 / P95 / P99 / max（rad） |
|---|---:|---|---|
| 左 | 6,744（1，2） | 3.47e-18 / 1.11e-16 / 1.24e-16 / 1.57e-16 | 2.31e-16 / 4.55e-16 / 5.92e-16 / 7.32e-16 |
| 右 | 6,896（2，2） | 0 / 1.11e-16 / 1.24e-16 / 1.67e-16 | 2.81e-16 / 6.33e-16 / 8.33e-16 / 1.12e-15 |

## 时钟、重复发布与有效性

使用已锁定的 MANUS SDK 二进制解压压缩字段。左、右各有 6,899 条可解压 calendar 记录。两侧各窗口的中位组件步长中位数均为 8 ms，未见组件时间回退。左手 562 条、右手 1,230 条相邻发布满足同 source instance、同 packed 值、同 SDK receive、同骨架证据。窗口内发布速率约 120 Hz，按此判据估计的独立身份速率左侧约 97–116 Hz、右侧约 76–112 Hz。相同姿态、新采样可独立计入，writer 序号保留为传输身份。SDK 压缩字段、解码组件以及发布器、采集器和 MCAP 时钟均分列。真实设备身份和跨设备映射仍待证。

历史资料没有物理 capture、controller receive 或真 tick 的有证边界。collector monotonic 仅提供 PROXY 观测。本轮以该观测建立 SIMULATED 实验，在候选 50 Hz 上给定 8 ms 合成到达间隔，并按 capture_upper ≤ tick、arrival_upper ≤ cutoff 消费事件。ROS header、MCAP log/publish 或 collector receive 均未回填真实控制端时间。真实端到端时延和真实时钟误差上界维持 BLOCKED。

因果近邻保持和离线插值各得到 2,248/2,300 个联合有效拍（97.74%，44.96 秒，各有 52 拍失效）。两个产品各自统计。因果输出六路共 13,800 个 stream-tick，其中 13,413 VALID，87 HELD，300 UNSAFE。真实窗口内含 153 条 unsafe 事件记录，旧缓存未被当作 BODY 输入。单位测试另外覆盖晚到、乱序、跳变、失联、overlap、世代/标定切换、过期或未知安全。真实窗口未证明所有人工负控情形均出现。

## 同步误差与 P01-D 候选

下表为每窗 P95 的跨窗范围，离线诊断参考取同窗、有效的分段插值。指标覆盖 SDK 派生掌系的原始地标与 raw/registered 腕，保留各流独立统计。当前不进行 retarget，也未使用外部精度真值。

| 指标 | 窗口 P95 范围 |
|---|---:|
| 模拟因果输入年龄（全路） | 15.07–21.90 ms |
| 双手 MANUS skew | 8.22–24.97 ms |
| 双注册腕 skew | 0.069–0.134 ms |
| 左腕–手 / 右腕–手 skew | 0.42–16.83 / 1.10–25.17 ms |
| Causal vs offline 位置 | 0.131–5.683 mm |
| Causal vs offline 旋转 | 0.0037–0.1161 rad |
| Causal vs offline 速度 | 0.0168–0.2350 m/s |
| 静止（代理速度 ≤ 0.02 m/s）位置 | 0.094–0.447 mm |
| 运动位置 | 0.333–7.678 mm |

各窗 P95 代理速度最高为 0.429 m/s。按该窗口速度，假设 5 ms 时间不确定性消耗约 2.15 mm，10 ms 约 4.29 mm。运动滞后仍限于模型及同拍对齐差，真实传感器、控制器和执行器延迟未测。这组敏感性直接关系后续 3–5 mm 目标。

单参敏感性以 2,300 拍为分母。年龄从 60 ms 降至 30 ms 得 2,183 个联合有效拍，从 60 ms 加到 90 ms 保持 2,248。将左右 skew 或腕手 skew 从 35 ms 分别收紧至 20 ms 得 2,224 或 2,222。离线 gap 从 60 ms 降至 30 ms 得 2,161。测得 safety age 60 / 100 / 150 ms 在本窗口集合均为 2,248。记录了分层覆盖率探针、原始四元数范数、骨长偏差 1/2 mm 比对、独立腕位标定残差，以及由每窗 P99 考察的速度和正 clock gap 风险。`qc_threshold_candidates.json` 保留各分布、通过数及解释。本轮没有冻结最终门槛、训练切分或资源配额。

## 复现、产物及运行状态

命令在 Ubuntu `/home/ubuntu/dexgym_worktrees/s0-5` 中执行。干净环境按 Jazzy → `teleop_ws` → collector 本地 overlay 顺序加载，解释器为该工作区 `.venv/bin/python -P`（Python 3.12.3）。预检时约有 684 GB 可写空间，保存依赖及可读资产的 SHA。使用此审阅提交时，选择新 Run ID 和 attempt，以新目录复验。以下为首次证据的命令形式（各入口完整实际参数、退出码、标准输出/错误 SHA 见凭据）。

```bash
cd /home/ubuntu/dexgym_worktrees/s0-5
base=/home/ubuntu/dexgym_data/s0-5/p01/runs
run="$base/P01-C-20261126-8b0f399"
bash docs/s0-5/evidence/P01_C/run_p01_c.sh "$run" \
  "$base/P01-A-20261126-09c1c9e-r03/registry" \
  "$base/P01-B-20261126-46ed5e2/attempt-04" \
  "$base/P01-B-REVIEW-20261126-f32d6bb-r02" attempt-03
python3 -B docs/s0-5/evidence/P01_C/seal_p01_c.py "$run/attempt-03"
python3 -B docs/s0-5/evidence/P00/check_documents.py --require-history
python3 -B docs/s0-5/evidence/P01_C/verify_p01_c.py "$run/attempt-03"
```

| 执行 | 实际结果 |
|---|---|
| attempt-01 | 单测通过，实际审计 CLI 路径守卫层级判断错误，退出 2，集成 NOT_RUN。失败记录原地保留 |
| attempt-02 | 修复路径守卫后单测、审计和真包集成均退出 0。结果原地保留 |
| attempt-03 | 55 项单元测试、3 项真实集成测试通过，0 跳过，三个执行进程均退出 0。新增独立左右/腕手 skew 灵敏度、随机未来扰动、过期帧控制和证据封存 |
| 准入和生产 | 真实时钟资格及独立 robot-base 物理注册 BLOCKED。生产同步、正式 split、训练与硬件控制 NOT_RUN |

完整产物在 `$run/attempt-03/products/`，约 144 MB。所有要求的 JSON/JSONL 报告、`fixtures_manifest.json`、三套有界窗口 NPZ、真包可视化和 `test_results.json` 均在该处。`output_sha256.json` 指向 92 个产物，SHA `a02e34a71410f7edf7e8cc310cf8a02e9831cdcf370ddaa8fc5d1dea7f5f7434`。窗口 NPZ 用 float64 XYZ/XYZW 和速度，int64 原始时钟、tick 和年龄，uint8 reason，bool mask，索引通过窗口 manifest 和 `source_trace.jsonl` 核对。Git 保存配置、证据摘要、代码和代表性图。原 MCAP、完整 trace 和窗口数组留在 Ubuntu。

## 停止点

P01-B-HMB ACCEPTED。历史 P01-B BLOCKED、B1 FAIL、B2 BLOCKED 与原始 119 条 `CollectionStatus` 失败保留，独立恢复证据仍有效。P01-T、Final Test 持续 BLOCKED，P03 正式启动仍要求 P02 和 P01-T 双签核。当前 C2 缺独立时钟锚点和控制端接收证据。等待人工审核当前结果及后续补证方式，P01-D–F、P02–P10 未启动。
