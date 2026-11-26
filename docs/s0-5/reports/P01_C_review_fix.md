# P01-C 独立审查修复报告

**提交人工审核。** C1-GEO、C2-SIM 在固定真包上通过。READY_FOR_REVIEW 仅限 P01-C 的历史 HMB 仿真用途。C2-RT、物理 robot-base、P01-T 和 Final Test 继续 BLOCKED。

| 项目 | 实际记录 |
|---|---|
| 审查基线 | `0d9d351b73e543eab07cfe6fef8f55a7d12d912d`，Ubuntu 独立 `s0-5` 工作区 |
| 执行环境 | `/home/ubuntu/dexgym_worktrees/s0-5`，Python 3.12.3，ROS Jazzy → Teleop → collector，使用 .venv 和固定 SDK |
| 独立 Run ID | `P01-C-REVIEW-20261126-0d9d351`，正式证据 `attempt-03` |
| 冻结 | 2026-11-26 11:49:34 +08:00，实施规格 v0.2.0、DCR、补证规格及原样本计划 |
| 历史证据 | A/B/B Review Fix 原有 11/15/20 项产物复验。B 的 15 项源码和 9 项诊断哈希、B Review Fix 5 项过程哈希通过。原 C 的 92 项产物、原执行文件及日志在修改前复验，attempt-03 和 BLOCKED 结论归档 |
| 授权 | 用户仅批准依赖调整、审查修复和离线验证。下一阶段及任何硬件采集均另行批准 |

[实施规格](../specs/P01_C_simulation_review_v0.2.0.md) · [DCR-002](../dcr/DCR-002_simulation_and_real_clock_qualification.md) · [C2-RT 补证规格](../specs/P01_C_RT_clock_qualification.md) · [运行凭据](../evidence/P01_C_review/review_receipt.json)

## 五项修复与反例

| 项 | 原实现的实际反例 | 修复与验证 |
|---|---|---|
| R-C01 失效与安全 | 可信新 UNUSABLE 输入后沿用旧 pose；未知安全仍沿用许可；unsafe 和恢复状态同落在两拍之间时可继续输出旧 pose | 先按来源/世代和序号筛旧包，新失效撤销旧状态。可信迟到且首次发现的 unsafe 仍触发闭锁，其副本不会重复触发。未知和 unsafe 需要新安全许可及新鲜观测才恢复。拍间撤销锁存到下一拍。pose、velocity 和许可独立 |
| R-C02 时钟与乱序 | 前进序号加回退 source 时钟被判断为旧包。额外从原 Git 对象复现同时到达的两序结果，姿态相同而 input reason 分别是 VALID 和 OUT_OF_ORDER | 区分迟到旧序、新源 reset、映射 generation、新 source instance 与无资格域。撤销跨段缓存并重建历史；按稳定事件键排序。完整状态结果的正反入队一致 |
| R-C03 速度与新鲜度 | 原来跨 69 ns 的测试间断仍生成速度（该反例的 age 为 60 ns）。跨不确定时差也会计算差分 | 单独输出速度有效位、原因、年龄和 SIMULATED/PROXY。任何失效或版本边界撤销历史，首个合格恢复姿态没有速度。HELD 仅短时保留已有同段差分。MANUS 首位缓存以连续 writer 前驱及源身份变更核验新鲜度 |
| R-C04 时域 | 两个虚构 proof 字符串能授予旧代码 VERIFIED。原 grid 的 realtime 偏移与事件 mono 偏移直接混用 | 固定 realtime 窗口仅索引。用窗口前实测同一 observation 的 realtime/monotonic 成对值定相，整个回放使用 monotonic。锚点模型仅给 MODEL_BOUND。已覆盖偏移、斜率、跳变、重启、未来锚点等负控 |
| R-C05 分母及误差 | 原 pooled 统计共同分母混合了腕部与人体点位。原 offline 把所有 bracket 标成同一 tick，不能判断端点跨度 | 四路在线与 raw 诊断分开。按左右 MANUS SDK common、掌系、五个有名指尖、腕平移/转角、相对腕与手内形状统计。保留离线两端跨度、联合跨度及来源索引 |

原冻结的七项反例中六项失败，最初的同到达测试只比较姿态而通过。新增历史 Git 重放定位了该测试遗漏。最终 **70 项单测、3 项真包集成及历史同时到达复验**均退出 0，零跳过。所有 23 窗均通过随机改写和删除未来载荷、未来锚点隔离与批流全状态等价。该检查比较 1,173 个前缀拍，并改写或删除 23,984 个后续窗口事件。真实源边界不足的负控用离线合成输入验证，当前不作实机安全资格。

## 同窗配对回归

原样本为七个已审真人母源、23 个原有 MCAP 窗口、46 秒、47,983 条有界正文载荷（含上下文）、153 条 unsafe 状态记录。重新核对源、采集时配置、writer、包 SHA、schema、物理索引、原计分范围及 13,640 对双侧腕位软件重建。每窗保持 100 拍、20 ms 间距和相同 ordinal。窗口内控制网格统一使用单调时轴，起点相对于旧格点移动 59–187 ns。窗口观测时钟对相对锚点的最大差值 4,773 ns，低于本轮 1 ms 代理回放一致性检查上限。该上限用于判定历史索引与代理时轴是否可配对，最终 QC 不在本轮冻结。这些结果描述录制主机双时钟一致性。设备物理捕获和控制端到达的误差界仍 UNKNOWN。

| 指标 | 原 P01-C attempt-03 | 审查修订 attempt-03 |
|---|---:|---:|
| 四路在线因果联合姿态有效 | 2,248/2,300 | 2,248/2,300，逐拍掩码 0 减少、0 增加 |
| 四路离线联合有效 | 2,248/2,300 | 2,211/2,300，计入原始 bracket span |
| 每路在线 pose 有效（四路核心与两路 raw 各自统计） | 各 2,250/2,300 | 各 2,250/2,300，无姿态原因迁移 |
| 每路在线速度有效（六路各自统计） | 各 2,250/2,300 | 各 2,249/2,300，恢复拍各撤销一次 |
| 离安全变化/观测边界 ±40 ms 的 9 个计分拍 | 6 拍联合有效 | 6 拍联合有效、3 拍 UNSAFE |
| 真控制端 VERIFIED 时序 | 0 | 0 |

配对有效拍中各流年龄变化 59–187 ns，与网格重新定相一致。使用修订核心在原格点复算仍为 2,248 拍，候选 age 60 ms、skew 35 ms、safety 100 ms、插值 gap 60 ms 和模拟到达 8 ms 均保持。左右各 23 个窗口首 MANUS 观测缺前驱而不算新鲜。加上窗口中的重复发布，左、右分别有 585 和 1,253 条 freshness 标记未成立。实际独立身份候选速率约左 97–116 Hz、右 76–112 Hz，发布约 120 Hz。逐流时间字段、频率、原包复验日志和物理时间 UNKNOWN 在产物中分列。

## 分层空间与时间指标

以下为每窗 P95 的跨窗范围，比较修订因果结果与同窗离线插值。每项各自保留有效样本计数。五个有名 MANUS TIP 对应原 ID 4、9、14、19、24，按拇指至小指编码。双手局部相对指标比较左右各自掌内的同名 tip-pair 长度差；双腕相对为同一已录注册坐标中的 `T_L^-1 T_R`。本阶段未计算世界系机器人指尖。

| 分层 | 左 | 右 |
|---|---:|---:|
| SDK common 原始地标（mm） | 0.084–5.253 | 0.097–6.400 |
| 掌系手指地标（mm） | 0.085–5.455 | 0.100–6.483 |
| 原始 TIP 节点（mm） | 0.100–10.206 | 0.107–10.379 |
| 注册腕平移（mm） | 0.141–4.985 | 0.192–7.327 |
| 注册腕转角（rad） | 0.00044–0.0283 | 0.00165–0.0517 |
| raw Tracker 平移，仅诊断（mm） | 0.188–5.866 | 0.333–7.626 |
| MANUS 离线单流 bracket（ms） | 10.214–25.066 | 16.671–41.675 |

相对腕平移 P95 范围为 0.350–13.210 mm、相对腕姿态为 0.00166–0.0710 rad、左右末端点对局部形状长度差为 0.148–4.279 mm。模拟因果 MANUS 左右 skew 为 8.22–24.97 ms，注册腕左右为 0.069–0.134 ms，左、右腕手分别为 0.42–16.83 ms 与 1.10–25.17 ms。离线两个手源的联合 bracket 跨度为 16.73–41.71 ms。各拍的 bracket 端点、插值权重和源跨度均可追溯。

## 门与停止点

| 门 | 结果与边界 |
|---|---|
| C1-GEO | PASS（七个受审源）。保留原 25 节点几何和采集时软件 Tracker 注册结论 |
| C2-SIM | PASS（有界历史回放）。技术证据等待人工审阅 |
| C2-RT | BLOCKED。原 C2 的真实时钟缺证、VERIFIED 为 0。真实采样/控制节点接收与误差界独立补证 |
| 物理 robot-base | BLOCKED。SDK common/OpenVR 注册空间未获独立物理基座测量 |
| P01-T | BLOCKED。仍缺通过资格的真实硬件时钟、命令、反馈与动态数据 |
| Final Test | BLOCKED。独立来源和封存需另验 |

P01 仍 IN_PROGRESS，P01-A/B-HMB 保持 ACCEPTED，历史 B1 FAIL、B2 BLOCKED 和原 P01-C BLOCKED 记录原样保留。P01-D 的开始条件为修订门人工签核、单独启动授权以及只使用明确标记的历史仿真候选。P03 的独立前置为 P02、P01-T、C2-RT 及物理 robot-base 均 ACCEPTED。P01-D/E/F、P02、P03、训练及任何硬件采集和机器人控制在本轮均 NOT_RUN。

## 复现与证据

```bash
cd /home/ubuntu/dexgym_worktrees/s0-5
base=/home/ubuntu/dexgym_data/s0-5/p01/runs
run="$base/P01-C-REVIEW-20261126-0d9d351"
bash docs/s0-5/evidence/P01_C_review/run_review.sh "$run" \
  "$base/P01-C-20261126-8b0f399/attempt-03" \
  "$base/P01-A-20261126-09c1c9e-r03/registry" \
  "$base/P01-B-20261126-46ed5e2/attempt-04" \
  "$base/P01-B-REVIEW-20261126-f32d6bb-r02" attempt-03
python3 -B docs/s0-5/evidence/P01_C_review/seal_review.py "$run/attempt-03"
python3 -B docs/s0-5/evidence/P01_C_review/seal_review.py "$run/attempt-03" --verify
python3 -B docs/s0-5/evidence/P00/check_documents.py --require-history
python3 -B docs/s0-5/evidence/P01_C_review/verify_governance.py
```

原冻结反例执行记录、命令/退出码/stdout/stderr SHA、第二轮运行和第一轮回归均原地保存。发布提交可直接执行 --verify 审计现存封存件。重跑时用新 Run ID 和 attempt，保留冻结文件与原负控前置记录。若新封存沿用原审查基线字段，须在独立工作树复现 0d9d351 加对应实现的执行文件 SHA。正式结果在 `$run/attempt-03/products/`，含逐窗 `paired_regression.json`、物理来源 `source_trace.jsonl`、`technical_gates.json`、`test_results.json` 及产物 SHA 清单。清单 SHA 为 `cdf543b86078660c0bae92fcbbd303aced54ad0c3c133734b3b542f69a8de186`。GitHub 保存实现、冻结文档和证据摘要，原 MCAP 与完整 trace 留在 Ubuntu。
