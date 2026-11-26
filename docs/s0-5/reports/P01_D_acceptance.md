# P01-D 数据冻结候选与验收

**D1 PASS，P01-D READY_FOR_REVIEW。** 本报告申请人工冻结开发 HMB 白名单、QC、split、schema 和生产构建预算。原 P01-C 的 READY_FOR_REVIEW 与历史 BLOCKED 记录保留，历史 HMB 仿真资格已按用户授权登记 ACCEPTED。[人工签核](approvals/P01_C_ACCEPTED_P01_D.md) · [实施规格](../specs/P01_D_dataset_freeze.md) · [运行凭据](../evidence/P01_D/review_receipt.json)

| 项目 | 实际记录 |
|---|---|
| 冻结及执行基线 | Ubuntu 独立 checkout `/home/ubuntu/dexgym_worktrees/s0-5`，`eb00f254dc71ac7f6ceef9e2c76a661ac3e912bd`。样本计划、规格及签核文件于 2026-11-26 12:39:43 +08:00 封存 |
| 运行 ID | `P01-D-20261126-eb00f25`，正式证据 `attempt-02` |
| 运行环境 | 工作区 .venv 的 Python 3.12.3，Jazzy → Teleop → collector overlay。Python、MCAP、NumPy、SciPy、消息组件解析路径保存在 source_qualification |
| 同步锁 | C 最终补丁源码 SHA `13550058a3114e062c6f7ba2e92b8149659dc5870ae9a97ec256679b7ed571b5`。正式入口拒绝其他源码版本 |
| 历史复验 | A/B/B Review 11/15/20 个文件，C 原运行 92 个产物，C Review 和 C 最终补丁产物也逐项核验。用对应发布提交复验历史源码哈希，工作结果不覆盖历史目录 |

## D1 验收矩阵

| 门 | 预先定义的规则 | 实际执行 | 状态 |
|---|---|---|---|
| 逐源资格与白名单 | 32 个原始候选逐一见证、只用认证原生真人录制，未证来源隔离 | 采集期配置、writer/collector、分片与消息 SHA、MANUS 拓扑、safety/validity 和软件配准在各源窗口验证 | PASS |
| 开发 split | 原始母源零重合，固定种子，保留历史用途，终评不分配 | D1 资格后的名单按任务、日期、配置分层；检查同根继承和已知历史用途。session 身份标记 UNKNOWN | PASS |
| QC、segmentation 与用途 | 真实敏感性、Train 决策、Validation 冻结结果；困难运动保留 | 五轴敏感性、逐流有效性、时间与动态几何回归，跨失效状态、时钟世代和版本中断分段 | PASS |
| 生产格式 | schema round trip、记录索引、真实性、速度与姿态独立、前缀不变 | 有界试片分开 causal/offline，按实际 MCAP 物理索引回读 CDR 与 schema SHA；全量入口需独立审批 | PASS |
| 资源、失败恢复 | 实际吞吐和空间、故障隔离、独立发布 | 单/双工比对文件哈希，同一物理试片在 fsync 后故障注入，复验未暴露残片、恢复及幂等 | PASS |

## 来源、split 与覆盖实测

32 个候选均取得独立的有界开发资格，其中 25 个完成本轮新验证。读取 135,378 条有界正文载荷，复验 38,487 对软件注册腕位。89 个未知来源保持隔离，已识别的仿真回放、EgoPro 及衍生副本继续排除。完整见 [资格登记](../evidence/P01_D/products/source_qualification.json) 和 [白名单](../evidence/P01_D/products/source_whitelist.json)。

| 开发集 | 母源 | 台账计分拍 | 因果四路联合有效 | 离线四路联合有效 |
|---|---:|---:|---:|---:|
| Train | 24 | 4,320 | 4,278 | 4,055 |
| Validation | 8 | 1,440 | 1,320 | 1,264 |
| 合计 | 32 | 5,760 | 5,598 | 5,319 |

母源没有重叠，分层涵盖三个日期、三类已验证采集配置和五类任务。按任务统计 6 个 arm_hand、11 个 arm_motion、2 个 bi_hand、2 个 fruit、11 个 hand_pose，五类均分到 Train 和 Validation。会话和操作者身份仍缺充分证据，不能据此宣称跨会话泛化。历史用途记录附在 split，Final Test 母源数为 0。保留曾用于旧训练与重复测试的录制在开发域标记。

96 个窗口实测 115.20 秒，原包合计时长 2,331.96 秒（38 分 52 秒）。观察到 123 个因果 UNSAFE 拍和 162 个因果联合无效拍。五拍最短条件产生 115 个因果段，共 5,575 拍（111.50 秒）。长度 5–60 拍，中位数 60 拍，受窗口分区上限影响。按每个母源的有界有效段比例推算，50 Hz 下约 112,733 拍、2,254.66 秒；全量可用帧和连续段分布等待 P01-E 逐包实测。

离线代理速度标签获得 4,767 个动态、828 个静态、165 个运动状态未知拍。分别观测两手形态/双腕运动，左右手地标运动拍为 2,727 / 3,127，左右腕为 2,703 / 2,784。双手同时改变 1,858 拍、双腕同时移动 1,992 拍、双侧同时有手或腕运动 3,205 拍。按三拍驻留规则观察 30 次动态到静止转变。仅有两个 bi_hand 与两个 fruit 母源，后续宜优先补充其独立会话、操作者、双手异步及启动/停止分布，并为终评单独封存新来源。[覆盖明细](../evidence/P01_D/products/dataset_coverage.json) · [开发 split](../evidence/P01_D/products/split.json)

## QC 选择与实际差异

Train 单因素候选依次得到 age 45 ms、左右 skew 25 ms、腕手 skew 35 ms、离线 gap 45 ms、safety timeout 40 ms。组合使因果联合有效由 4,278 降至 4,208、动态有效由 3,675 降至 3,617；离线联合有效由 4,055 降至 3,597、动态有效由 3,498 降至 3,136。离线全量和动态保留比例未达到预注册门槛，且 arm_motion、arm_hand 的联合覆盖低于每任务 90% 下界。按预注册规则回退到冻结诊断组。最终候选为 **age 60 ms、左右 skew 35 ms、腕手 skew 35 ms、gap 60 ms、safety timeout 100 ms**，50 Hz、模型到达 8 ms、最短段 5 拍。Validation 结果只作描述。

Train 最大四元数范数偏差 5.98e-7，候选容差 1e-6，全部窗口没有原始姿态因此被标无效。Train 父边 P99 残差最大 4.13e-8 m，保留 1 mm 的审阅标记线。Train 窗口/流 P99 代理速度最高为 0.747 m/s，速度与骨长报警用于审核。高速度和困难动作继续保留。Train host 双时钟对的窗口最大残差最高 70,645 ns，沿用已审 1 ms 一致性上限。新时钟世代、负时间步和时钟失锁切段。真实设备至控制端时序仍未校验。

动态有效样本中，各窗 P95 的左/右 SDK TIP 差异范围为 0.086–8.453 / 0.101–8.555 mm，注册腕平移为 0.082–7.091 / 0.052–7.967 mm，旋转为 0.00002–0.0575 / 0.0004–0.0603 rad，双腕相对平移为 0.199–27.001 mm。逐侧、逐流、逐任务的全量分母、age、skew、重复、缺测和速度资格可回查。空间差值为同一窗口的模拟因果与诊断离线差异，保持其输入数据含义。[QC 阈值与动态证据](../evidence/P01_D/products/qc_threshold_report.json) · [敏感性](../evidence/P01_D/products/qc_sensitivity_results.json)

## 容量、运行和失败处理

| 测量或预算 | 本轮结果 |
|---|---:|
| 必要 MCAP 文件读入与索引检查 | 2.276 GB；约 147.34 MB/s，包含 MCAP 枚举和校验，剔除独立计时的 CDR 部分 |
| CDR 解码及字段读取 | 270,756 条，约 12,966 条/s |
| 最终同步与前缀复验 | 约 274.17 拍/s |
| 进程峰值 RSS | 3.374 GB |
| 三份串行真实试片及有效因果帧 | 13,757,691 bytes；179 拍 |
| 实测派生体积 | 3.822 MB/原始秒，约 76.86 KB/有效因果拍 |
| 临时失效残片及试片总写放大 | 4,593,637 bytes；2.87×，含串/双工、故障暂存和恢复 |
| 全量保守归档及含暂存峰值 | 71.61 GB / 143.22 GB |
| 核验时可用磁盘与运行保留线 | 683.04 GB 可用；保留至少 20 GB 空余及约 13.50 GB 可用 RAM |

八秒分片估计重新枚举逻辑读取 23.48 GB，预算另计 SHA 读取。超磁盘、RAM、单分片或记录数上限立即暂停。双工发布试片的产物与串行哈希一致，故障发生在实际 fsync 后、发布前，未暴露目标目录。恢复产物与基线一致且再次运行不写入；原暂存文件保留供核查。源错、校准/版本漂移、语义不明与资源异常均保留显式错误。候选的 32 个来源没有额外失败或被筛除。

最终 84 项单测、3 项真包集成测试全部通过，检查 raw CDR 重读、双模式独立 schema、prefix 和安全/资格字段。首轮单测、数据研究成功，首轮的独立真包 integration 为 NOT_RUN，正式 attempt-02 补齐。P01-E 全量构建与其他后续阶段均 NOT_RUN。[资源与试片](../evidence/P01_D/products/resource_budget.json) · [实际测试与命令](../evidence/P01_D/products/test_results.json)

## 冻结配置及 P01-E 复现入口

[冻结候选目录](../evidence/P01_D/products/) 收录白名单、全部 32 个逐源证据、QC 选择与敏感性、schema、全部原因码、split 和重合报告、覆盖、资源、构建协议、测试记录及 SHA。冻结候选的 SHA 为 `f7f2eeba2cfc8adc9d1649618f353956470eb477a6a99448bdbe7332dd5a6d81`，完整运行清单 SHA 为 `84e8e64716299430b2469762d1643f9e26c664b6e0e830d7216d4df05fd4b225`。试片、真实物理记录引用及临时失败残片原地保存在 Ubuntu 的 `$run/attempt-02/products/`。

生产 schema 为 `s05.hmb.simulation.v1`。分开保存 causal.npz 和 offline.npz，以及分区所属 episode、parent、block、split、校准和拓扑字典、原始 MANUS SDK common/派生掌系、raw/registered 腕位、独立 pose/velocity/safety 掩码与时钟等级。因果模式保存两组时间界、tick/cutoff/age 和可追溯单原记录，离线模式保存端点对、权重、跨度和整个 bracket 安全性。物理索引 sidecar 记录资产、CDR、schema、原时间与身份。每个连续段单独编号，诊断 raw 腕独立于四路联合门，旧 NPZ 拒绝作为生产输入。

正式入口在独立批准后读取冻结候选的哈希及生产阶段授权，核对原始 registry 与具体源证据，按八秒分区，局部上下文用于断点前后的时序识别，分区不强行拼接。写入目录使用 fsync 和原子发布。相同配置可从已核验的不可变分区继续。同一路径的配置或内容不符、节点预算漂移、未知 schema、校准或时钟越界时暂停。下列构建命令仅为交接文档，**本次全量入口 NOT_RUN**。

```bash
cd /home/ubuntu/dexgym_worktrees/s0-5
base=/home/ubuntu/dexgym_data/s0-5/p01/runs
run="$base/P01-D-20261126-eb00f25"
bash docs/s0-5/evidence/P01_D/run_study.sh "$run" \
  "$base/P01-A-20261126-09c1c9e-r03/registry" \
  "$base/P01-B-20261126-46ed5e2/attempt-04" \
  "$base/P01-B-REVIEW-20261126-f32d6bb-r02" \
  "$base/P01-C-20261126-8b0f399/attempt-03" \
  "$base/P01-C-REVIEW-20261126-0d9d351/attempt-03" \
  "$base/P01-C-REVIEW-FINAL-20261126-aecfd2d/attempt-02" attempt-02
python3 -B docs/s0-5/evidence/P01_D/seal_study.py "$run/attempt-02"
# P01-E 单独获批后，在同一 Jazzy/Teleop/collector overlay 与 .venv 下运行
PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}" "$PWD/.venv/bin/python" -P \
  -m dexgym.tasks.direct.wholebody_s05.data.p01d_freeze build \
  --config docs/s0-5/evidence/P01_D/products/frozen_config.json \
  --old-a "$base/P01-A-20261126-09c1c9e-r03/registry" \
  --approval-state docs/s0-5/phase_state.json \
  --output "$base/P01-E-<approved-run-id>/products"
```

每个过程的实际命令、环境、stdout/stderr 和退出码/哈希随运行凭据。复跑时使用独立 Run ID 与 attempt，按固定计划重建，旧结果不覆盖。

## 剩余义务和停止点

已核验源的资格仅限有界观测及历史仿真开发。真实 C2-RT、物理 robot-base、P01-T 和独立 Final Test 继续 BLOCKED。P01 整体仍 IN_PROGRESS。待人工接受 P01-D 冻结候选并单独授权 P01-E 后再执行全量构建。P01-E/F、P02、P03、训练和实际机器人操作均未执行。
