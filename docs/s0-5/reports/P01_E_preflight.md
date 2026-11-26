# P01-E E0 生产预检

2026-11-26 · Run `P01-E-E0-20261126-34e44b6` · 通过轮次 `attempt-06`。

**E0 技术门 PASS，提交人工审核。P01-E 保持 IN_PROGRESS，E1 NOT_RUN。** P01-D 已由项目用户签核。原 32 个真人母源、24/8 开发 split、HMB simulation v1 和 QC 保持冻结。全量启动需单独批准下列两项 SHA-256 并登记正式 Run ID。[人工批准](approvals/P01_D_ACCEPTED_P01_E_E0.md) · [运行凭据](../evidence/P01_E_E0/review_receipt.json)

| 审核对象 | SHA-256 |
|---|---|
| 新配置候选 | `6aeaa203e39bea392bc20d5fceee26ea5f85752d219824f98f96b2c852846317` |
| 新实施依赖锁 | `da3e1f0759ac010a989d9431f0337371df6fe208914139bd711be15354a453af` |
| 原 D1 候选 | `f7f2eeba2cfc8adc9d1649618f353956470eb477a6a99448bdbe7332dd5a6d81` |
| 原 D1 凭据 | `40bf56f01337d0ceb32f8977efc80154e40144bbc949becbb7050d12963bed93` |

[新配置](../evidence/P01_E_E0/candidate_config.json) · [新实施锁](../evidence/P01_E_E0/execution_lock.json) · [原配置](../evidence/P01_D/products/frozen_config.json) · [原 D1 凭据](../evidence/P01_D/review_receipt.json)

## 基线与约束

实际执行工作区为 Ubuntu `/home/ubuntu/dexgym_worktrees/s0-5`，运行基线为用户审核提交 `34e44b6debcd4bb6f10be53a36dc8fec8bc4e7b2` 加新锁中逐文件 SHA 所绑定的 E0 实现。复验 D1 的 14 项仓库交付物、60 项原运行产物、既有 Registry 与 224 个原始 MCAP 分片（2,275,871,930 字节）。同步器继续使用 P01-C 最终修订源码，SHA-256 为 `13550058a3114e062c6f7ba2e92b8149659dc5870ae9a97ec256679b7ed571b5`。构建读取原始 MCAP、采集清单及配准快照。[前置实证](../evidence/P01_E_E0/prerequisite_verification.json)

实际使用专用 venv 的 Python 3.12.3，ROS Jazzy → teleop_ws → dex_data_platform-cpp-collection，工作区源码优先导入。NumPy 1.26.4、SciPy 1.11.4 与其他实际依赖取自运行环境。系统包按已安装文件清单取哈希并核对 import 来源。锁还绑定 Python 二进制、包内容、SDK、ROS overlay 加载脚本、Topic Dependency Matrix、消息计划、Registry、白名单和前述实现全树，记录了实际 Python 模块解析路径。[实施依赖锁](../evidence/P01_E_E0/execution_lock.json)

QC 继续使用 20 ms 控制拍、8 ms 模拟到达、60 ms age、35 ms 左右及腕手 skew、60 ms offline gap、100 ms safety timeout、至少 5 拍的 segment、四元数容差 1e-6。骨长和速度异常主要保留为审阅标记。E0 的 1.25 秒短尾继承界仅用于校准证据资格。发现本段实际超限或配置冲突仍然隔离。

## 预检验收矩阵

| 门 | 标准 | 实际 | 结果及证据 |
|---|---|---|---|
| E0-1 | 原始来源与 schema 正确，几何、标定、拓扑、时钟资格有结构化结论，区别数据不足和可保留失效 | 验证 8/8 分区，8 个因果 UNSAFE 拍保留。405.375 ms 尾段直接用每侧 48 对配准证据获得资格。同绑定继承、残差失败优先及不合格来源用负控检验 | PASS · [分区索引](../evidence/P01_E_E0/build_index.json) |
| E0-2 | 双产品 staging 反读，核对 shape、dtype、节点值、掩码、时间、来源与标定/拓扑身份、离线端点和 segment | 各 8 个 causal/offline 文件发布前通过语义校验；注入错误前已完成 staging 校验。非有限值与重算 SHA 的语义破坏均被拒绝；恢复复验已发布分区；从原始 MCAP 回读三个分区的 CDR/schema SHA | PASS · [验证结果](../evidence/P01_E_E0/verification.json) |
| E0-3 | 锁定现场依赖及授权，保留计划、故障、暂停、恢复和最终发布状态 | 8 PLANNED、9 RUNNING、1 FAILED、1 PAUSED、1 RESUMED、8 PUBLISHED。续跑验证前次 1 个分区，最后一轮复验全部 8 个。全量入口持独立阶段授权校验 | PASS · [生产台账](../evidence/P01_E_E0/build_log.jsonl) · [续跑](../evidence/P01_E_E0/.session-2.json) · [幂等复验](../evidence/P01_E_E0/.session-3.json) |

资格门读取结构化源、时钟、配准和几何证据。缺证明的数据保留隔离状态，可信的失效、缺测和保持拍继续写掩码及来源引用。发布先反读 staging 中两个完整产品，通过内容与语义检查并 fsync 后才原子 rename。正式恢复重新检查物理引用、清单身份、语义和台账计数。两种历史仿真产品都拒绝 VERIFIED 时钟资格。

## 有界真包结果

冻结样本计划在 2026-11-26 14:07:57 +08:00 写入。用纯元数据在 5 个候选中选择 Train 母源 `capture:57223e1ea4fd73f5aa16c512`（`recording:c59e65d7a17e7253670ba0cf`）。处理全长 56.405375246 秒，含七个 8 秒分区、一个 405.375246 ms 自然尾段。原始区间首尾和 2,821 个计划拍守恒。[预注册方案](../evidence/P01_E_E0/plan.json) · [样本计划](../evidence/P01_E_E0/fixture_plan.json)

| 产品 | 联合有效 / 无效 | 独立速度有效流拍（六流） | 有效段数 |
|---|---:|---:|---:|
| Causal | 2,809 / 12 | 16,860 | 11 |
| Offline | 2,798 / 23 | 16,865 | 15 |

尾段计划 21 拍，causal 联合有效 21 拍，offline 有效 20 拍。公开路径下 8 个分区的两个产品通过反读。三次原包回读覆盖第一个分区、故障后恢复的分区及自然尾段。三个见证分别重取了原物理消息索引 988、2876、13454，重新核对 CRC、话题与 CDR/schema SHA。该规模覆盖一个母源和一个 Train 实例。32 个母源的全量覆盖、Validation 实测和 E1 数量守恒均待正式运行。[分区索引](../evidence/P01_E_E0/build_index.json) · [回读校验](../evidence/P01_E_E0/verification.json)

成功分区构建计时合计 38.25 s（含扫描、解码与发布，不含前置锁复验及独立见证反读）。分区阶段累计逻辑读取 1,901,769,964 字节；按分区重复访问的 MCAP 文件规模合计 440,734,304 字节，唯一输入包文件合计 55,091,788 字节。CDR 解码 100,548 条 / 5.91 s，约 17.0k 条/s；同步 2,821 拍 / 7.08 s，约 399 拍/s。峰值 RSS 500,011,008 字节，观测最小可用磁盘 682,817,835,008 字节。已发布分区合计 173,343,904 字节，单分区峰值 24,630,695 字节；注入失败保留隐藏 staging 24,619,879 字节，约为已发布总量的 14.2%。这些是单 worker 有界实测，D1 的全量保守预算继续作为请求启动 E1 时的资源约束。[分区测量](../evidence/P01_E_E0/build_index.json) · [D1 资源预算](../evidence/P01_D/products/resource_budget.json)

## 测试、失败记录与复现

前序 P01 单测和 E0 新负控共 91 项通过。真包流程依次执行候选锁生成、故障注入、续跑、幂等复验与原包索引见证。故障注入预期退出 1，其余步骤退出 0。台账确认故障发生于第二个分区完成语义校验且 fsync 之后，目标发布路径当时不存在。[命令、退出码和日志 SHA](../evidence/P01_E_E0/test_results.json)

| 轮次 | 实际结果 |
|---|---|
| attempt-01 | 单测负控正确拒绝损坏数据，测试的错误消息断言不匹配，整组退出 1。修正断言后通过 |
| attempt-02/03 | 准备阶段拒绝缺少 wheel 文件清单的 NumPy 或 LZ4。改用安装清单并核对模块来源 |
| attempt-04 | 资格隔离返回值的 `status` 与台账位置参数冲突，退出 1。显式用 `classification` 保存资格结论 |
| attempt-05 | 拓扑比较把序列化前的 tuple 与原清单的 JSON 数组当成不同拓扑，预检未触发预定故障。将完整拓扑转换为同一 JSON 表达后复验 |
| attempt-06 | 91 项单测和真包试跑通过；预期注入故障、中断恢复、幂等与原包反读均经核验 |

以审核对应的代码进入 Ubuntu worktree，复用同一外部原始 Registry 和内容冻结计划，取新的 attempt 目录名复跑。命令记录、stdout/stderr、试片、失败 staging 和两种数组产品都保存在 `/home/ubuntu/dexgym_data/s0-5/p01/runs/P01-E-E0-20261126-34e44b6`。Git 保存报告、台账、执行候选和 SHA 清单。

```bash
cd /home/ubuntu/dexgym_worktrees/s0-5
run_base=/home/ubuntu/dexgym_data/s0-5/p01/runs
bash docs/s0-5/evidence/P01_E_E0/run_e0.sh \
  "$run_base/P01-E-E0-20261126-34e44b6" \
  "$run_base/P01-A-20261126-09c1c9e-r03/registry" attempt-07
```

[试跑脚本](../evidence/P01_E_E0/run_e0.sh) · [输出 SHA 目录](../evidence/P01_E_E0/output_sha256.json)

## 停止点

E0 已有实际有界证据。启动正式全量构建需要项目用户审核新配置与依赖锁的精确 SHA、确认独立全量 Run ID 并给出 E1 启动授权。待 E1 完成后再生成正式 HMB manifest 和全量验收报告。C2-RT、物理 robot-base 注册、P01-T 和独立 Final Test 保持 BLOCKED。P01-F 及其后阶段均未启动。
