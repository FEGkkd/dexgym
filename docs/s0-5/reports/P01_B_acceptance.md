# P01-B 原始 MCAP 解码与遥测资格审核

| 项目 | 结论 |
|---|---|
| P01-A | ACCEPTED，人工签核范围为 A1 元数据门 |
| P01 | IN_PROGRESS |
| P01-B | BLOCKED，等待审核和补证 |
| B1 | FAIL |
| B2 | BLOCKED |
| P01-C–F | NOT_STARTED |
| P01 Spec | v0.2.0，SHA-256 `662ffd475dc40eb0769c0d2c474b5c526916d168c2413c67b98c4fecb624013a` |
| 开工基线 | `46ed5e2ae51b2e735897d79205409e4ccd6c8382`，父提交为获审 P01-A `f0340124352fcd9f4cdcb68d98e761b42865c3c0` |
| Run ID | `P01-B-20261126-46ed5e2`，正式证据 `attempt-04` |

[人工授权](approvals/P01_A_ACCEPTED_P01_B.md) · [运行凭据及产物 SHA](../evidence/P01_B/review_receipt.json) · [技术状态](../phase_state.json)

## 执行基线、命令与比对

在 Ubuntu 独立 checkout `/home/ubuntu/dexgym_worktrees/s0-5` 确认开始时 GitHub 和本机 HEAD 相同且工作树干净。Python 3.12.3，先加载 ROS Jazzy，再加载 Teleop 和采集器 overlay。复验通过 21 个固定解析模块／包文件 SHA、实际 import 路径及 11 个原 P01-A 产物 SHA。可用磁盘 684,107,431,936 字节。P01-A r03 文件及证据保留，合同版本未修改。

```bash
cd /home/ubuntu/dexgym_worktrees/s0-5
git ls-remote --heads origin refs/heads/s0-5
old=/home/ubuntu/dexgym_data/s0-5/p01/runs/P01-A-20261126-09c1c9e-r03
run=/home/ubuntu/dexgym_data/s0-5/p01/runs/P01-B-20261126-46ed5e2
bash docs/s0-5/evidence/P01_B/run_p01_b.sh "$run" "$old" attempt-04
python3 -B docs/s0-5/evidence/P01_B/sdk_timestamp_components.py "$run/attempt-04/products/raw_events.jsonl"
python3 -B docs/s0-5/evidence/P01_B/summarize_run.py "$run/attempt-04"
python3 -B docs/s0-5/evidence/P00/check_documents.py --require-history
```

开工运行 `readiness_probe.py environment` 并保存 stdout/stderr。runner 先执行 unittest，再调用 `python -m dexgym.tasks.direct.wholebody_s05.data.decode_audit` 盘点后解码，再执行真实集成测试。实际展开参数保存于 `attempt-04/decode.command`。使用同一清洁 ROS/Python 环境运行 `collection_status_probe.py <products>` 对旧类型做独立交叉验证。复跑使用新 Run ID/attempt，单独保存环境、时间源码、退出状态和 SDK 组件输出。原 P01-A 哈希目录为只读输入。

## P01-A 局部修复

新的独立检查读取真实 registry、lineage 和计数，按 A1 字段、源状态、准入、父边重解、重复 payload 及产物哈希复验。正控与 BLOCKED、缺字段、非真人源、数量不守恒的负控通过。处理器退出值只表明进程执行结果，技术门和人工签核分别记录。

完整合并包签名要求 sidecar 顺序、所有分片和每片 MCAP footer 消息数与声明一致。不同元数据、配置、标定或来源声明遇到相同已见 payload 时单列冲突，等人工裁定。以下是对旧结果的并行补记，保留历史统计分母和身份。

本次补记中有 1 个合并录制的完整分片签名证据不足，历史状态已是 QUARANTINED，未影响任何旧 alias 或 32 个准入候选。相同已见 payload 的声明冲突为 0，短样本比对未找到相同原始前缀的跨录制候选。历史时段较长的重编码、变速和裁剪仍无完整比对。

## 盘点和真实覆盖

保留 P01-A 的 651 个 MCAP 文件实例、382 个相异 payload、181 个逻辑视图及 122 个字节去重录制。仍有 32 个具备原始真人采集凭证的母源候选，元数据录制时长约 38 分 52 秒。仍有 89 个录制隔离、1 个已知仿真回放排除、59 个重复视图。没有新增可证真人母源或正式训练资格。100,071 条历史用途事件仍未完整反查至母源，独立 Final Test 保持 BLOCKED。

382 个 payload 的摘要全部读取，得到 38 个不同 schema 身份。语义范围内实测 33 个格式–版本组合，32 个获得成功解码见证。抽样覆盖 14 个录制和 89 个分片，其中 8 个原生、6 个合并，得到 21,440 条事件，21,321 条解码成功（99.44%）。已核实原生录制有 6,272 对 payload–Observation，另保留 24 条初始缓存事件（含诊断录制）。7 个来源已核实的原生录制四路共同窗口合计 7.405 秒。两个 `bi_hand` 录制皆取到约 1.058 秒四路重叠。两段的左右腕峰值位移分别约为 0.019/0.211 和 0.0022/0.0074（原位置单位）。两段窗口都有左右手节点变化，反映不同程度的双侧活动，完整双手协同覆盖仍未知。

抽样窗口按真实 MCAP part、物理消息索引及包内 schema SHA 定位。分析各包 schema 定义、各流字段和缺失率。相机部分只有 catalogue 记录。重复采样或派生覆盖不增加独立时长。

## 两种 Adapter、拓扑和时钟

- 分流包以原始 manifest 挑选流和 writer 分片。已核实的记录器代码在同一 stream 先写 payload 后写 SampleObservation。配对同时要求同一物理 part、相邻 record、正确话题、stream/source/collector 身份、严格递增的源序列和 episode 索引，以及相符的写入 clock。episode 前 `initial_state` 缓存分单独域保留。缺运行快照的诊断样本允许原始 CDR 解码，其配对和来源资格仍显示未知。
- 合并包的 ROS2 类型按 `name + schema encoding + message encoding + embedded SHA` 识别。MCAP schema ID 只在单文件内有效。旧 HandState 的左右字段以录包内嵌 `ros2msg` 解码，保留 `home_sequence` 等原字段。无效 CDR、未知 schema 和无样本的版本分别记录。
- MANUS 按真实原始顺序保留位置、四元数、frame、`joint_type`、`chain_type` 和父节点。25 节点样本中根节点 `0` 的父值也是 `0`，原样保留。解码支持可变拓扑并记录拓扑变更。未给定传感器有效性时显示 UNKNOWN。
- I17 的 ROS 字段 `source_timestamp_us` 保存为原始整数。SDK 使用压缩 `ManusTimestamp.time`，`CoreSdk_GetTimestampInfo` 在匹配历史 Git blob 的 SDK 库离线执行。非 timecode 模式将 fraction 表达为毫秒。ROS header、MCAP log/publish、SDK receive、collector 的三个时钟互相分开。没有验证 controller receive 或真实设备到响应的映射。

2,336 条 MANUS 样本的 SDK 非 timecode 组件均可读，各有界样本序列的组件间隔中位数为 8 ms。原始整数仍全部保留。左侧 100 次、右侧 206 次相邻源值重复。这 306 次同时具备相同 SDK 接收值及骨架、变化的 ROS header，符合发布端重复发布缓存骨架的证据。两侧样本均无源值回退或缺值；上游真实捕获序列和跨时钟关系仍未知。各录制的跳变诊断统计见 `raw_timestamp_report.json`。

## 真机遥测资格

继承的 59 个命令/反馈话题能力记录按视图计数。正式资格分类为 59 个重复 alias、1 个仿真排除、84 个无完整命令/反馈候选、37 个缺硬件 provenance。旧包样本测到 14 个命名 arm command、左右臂各 7 个命名位置、左右手各 20 个命名命令与反馈。所读旧 HandState 有速度和 effort 数组；数据中没有 current，effort 单位未记。该旧包的臂速度数组为空。按全部诊断样本汇总，每侧臂 288 条反馈中 128 条速度为空，两侧各 256 条手反馈均缺 current。EE pose、twist、wrench、driver、diagnostics 和 safety 保留原 topic/schema 与索引。21 个目录的录包日志及标定窗口已做哈希登记。已证静态、相对响应或动态系统辨识资格均为 0。

有订阅日志或标定窗口的历史录制可以追踪录包流程。上游发布者、原始硬件、标定配置及命令—反馈因果关系仍未绑到这些字节。B2 补证需要原始捕获 SHA、捕获期配置、硬件设备和 runtime driver 身份、独立命令/反馈序列、实测关节顺序和配对的时钟参考。本轮不发出任何机器人动作。

## B1 和 B2 审核矩阵

| Gate | 必须满足的条件 | 实际证据 | 结论 |
|---|---|---|---|
| A1 局部回归 | 字段、计数、谱系、重复关系、准入和哈希独立核查 | 11 个前序 SHA 复验，独立检查技术 PASS；保留原始历史 | PASS |
| B1 两种录包及原始字段 | 真实 CDR、真实索引、旧 HandState、25 节点及初始状态分离 | 正常的原生 MANUS/Tracker/Observation 和旧合并包 HandState 已实测 | 已验证子项 |
| B1 完整版本覆盖 | 适用的 CDR 与消息定义可验证解码，未知版本或内容故障有明确裁定 | 旧合并包 CollectionStatus 有 119 条 CDR 解码失败，保留原文件 SHA、物理索引和 schema。对同一条记录用包内 decoder 与当前 ROS Fast CDR 复核，二者均失败。原发送端相容的定义或序列化条件尚未证实。 | FAIL |
| B2 硬件和配对 | 可追溯硬件、命令、实际反馈和用途对应的时钟 | 诊断了命令和反馈字段，真实动态系统辨识候选为 0 | BLOCKED |
| P01-C/D | 因果同步、坐标／标定和最终 QC | 授权范围以外 | NOT_RUN |
| P01-E/F、P02、训练 | 全量构建、独立验收、控制器或学习 | 授权范围以外 | NOT_RUN |

## 失败记录与审核停止点

- `attempt-01` 的通用 observation 名称筛选误带相机伴随数据，停止本审计子进程，退出码 143。随后以实际 raw payload topic 约束 stream，并添加相机回归负控。
- `attempt-02` 记录了隔离 native 包缺运行快照导致的 adapter 错误及旧 CollectionStatus CDR 失败。下一次允许无运行快照的诊断解码，保持未知配对；修正 footer 有统计时未写消息的通道计数为 0。保留两次日志和已有产物。旧 CollectionStatus 的结果见正式交叉验证。
- `attempt-03` 成功完成正式流程，技术 B1 仍为 FAIL。发现版本汇总把已尝试但从未解码成功的版本写成 NOT_RUN。`attempt-04` 加入负控并明确标记 DECODE_FAILED，保留旧失败及交叉验证记录。
- 最终执行的 30 项单测与 5 项真实集成测试均通过，审计进程退出 0。两次完整诊断的技术门、计数、谱系与覆盖一致。B1/B2 技术结论见上表。独立 Final Test 仍缺封存证据。等待人工审核。若需分开推进 HMB 与 Telemetry 补录，由用户另行决定。

旧包该记录的 ROS 类型哈希以 `RIHS01_ce543f80` 开头。当前两套 overlay 的已生成 `CollectionStatus.json` 都给出 `RIHS01_56af1b62…`。这个差异支持进一步寻找原发布端构建产物，当前证据无法确定实际 CDR 原因。完整哈希、文件路径和快照 SHA 已登记。[类型哈希交叉验证](../evidence/P01_B/type_hash_comparison.json)

## 产物与变更位置

正式目录为 `/home/ubuntu/dexgym_data/s0-5/p01/runs/P01-B-20261126-46ed5e2/attempt-04/products`。产物清单如下。

```text
schema_registry.json           adapter_contract.json
fixtures_manifest.json         decode_fields.json
decode_attempts.json           source_eligibility_updates.jsonl
telemetry_index.jsonl           telemetry_eligibility.json
missing_fields.json            raw_timestamp_report.json
manus_topology_report.json     lineage_followup.json
dataset_coverage.json          test_results.json
output_sha256.json             raw_events.jsonl
```

同级 attempt 还保留完整单测／集成结果、展开命令、SDK 组件与 CollectionStatus 交叉验证。Git 提交包括 `wholebody_s05/data/` 的 A1 检查、两个 adapter、schema/cache、原始事件、遥测和诊断，`tests/s0_5/` 的单测和真实包测试，以及 `docs/s0-5/` 的授权记录、状态、追踪、报告、采样计划与运行摘要。GitHub 保留报告及审计所需哈希，原包和完整诊断切片保存在 Ubuntu。
