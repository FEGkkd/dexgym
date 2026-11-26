# V1-D 离线精度、实际 Teleop 与联合发布实施规格 v2.0.1

状态 `DRAFT_READY_FOR_REVIEW`。本阶段把独立实际 MuJoCo 物理评估与用户现有 MuJoCo Teleop 软件链路联成一次有界交付。首版目标技术等级为 `TELEOP_SIM_RECOMMENDED`。正式外部科研终评、真实硬件、真实时钟和物理计量保持独立门禁。[评估协议](../EVALUATION_PROTOCOL.md) · [实际 Teleop 回放规格](V1-D_teleop_replay_and_refinement.md)

## 1. 前置、输入及输出

设计经人工批准及单独 Campaign 授权后，可用版本化脚手架准备 scorer/export。正式 D0 以 G-C3、当前 G-C0 资格及 A/B/C1/C2 有效技术收据为前置。输入锁定 HMB/RMB、源谱系和掩码、低误差见证与 membership、首态和扰动配对、B0/Actor/归一化/Builder、模型和 Servo、安全配置、计分协议 SHA。输出阶段收据、离线和软件闭环的原始记录、始末候选与差异、全部分母及最终等级。低资格诊断导出沿独立支路核验，证据不计入正常通过。[阶段依赖](../ROADMAP.md)

## 2. D0 离线精度验收

在首个正式 Actor 更新前，从已冻结的完整计划、独立 mapped intent、合法名义状态、统一残差和安全约束得到独立样本域。A/B 准备并冻结预定窗口，物理见证用未来 G-B 确认的同 profile MuJoCo 与冻结动作序列，一次初始 reset。在 C1 前冻结低误差证据及域。[V1-A 合法见证和域](V1-A_offline_rmb.md)

| 证据 | 口径 |
|---|---|
| 共同可改善域 | 独立见证相对名义可改善。按原条件复验。 |
| S5 静态低误差域 | 单个姿态独立 CPU FK 与合法路径见证 max-ten ≤5 mm。这个存在证据不代表闭环末段实际误差。 |
| D10 动态低误差域 | 整个预定连续窗固定合法残差的参考见证序列，在同 Servo、模型、频率、初态规则的实际仿真中得到 max-ten 复合 P95 ≤10 mm、无跳拍或硬失败。将合格窗口的全部计划 ticks 封存并按身份去重。逐拍静态见证或比名义更准的结论均不能代替整窗动态资格。 |

对同一冻结 Teleop Validation 运行 B0/B54，实际 MuJoCo Warp 或通过 CPU/Warp 确认的锁定独立物理端。静态全程按计划保持，D0 主分取末窗 S5 复合 P95 ≤5 mm。动态原速、一次初始 reset，D10 取复合 P95 ≤10 mm。无尖峰截尾、事后时移或窗口挑选。功能、安全及 E4 相对门同时通过。报告总计划、mapped 有效、低误差域、共同可改善、困难域、不适用、严格排除与 UNKNOWN 分母，另报覆盖不足原因。S5 独立静态目标 ≥40 且 ≥6 个 Validation 根；D10 唯一合格计划拍 ≥1,000 且 ≥6 个 Validation 根，并核对评估协议 E4.1 的 S5/D10 最小比例候选。原有数量和功能/困难层下限继续适用。若低误差域支持不足登记 `INCONCLUSIVE`；门可评分而越界登记 `FAIL`。主指标复合算法与原始目标来自评估协议。[评估协议](../EVALUATION_PROTOCOL.md)

## 3. 候选选择与复验

开发轮最多 6 个候选，仍受 D0 绝对评估预算。先排除缺资格、数据身份、因果、安全或 WM 失效，随后逐个同时检查 D0 绝对、相对、完整域及功能门。满足者优先，以静态与共同可改善动态相对收益两者最小值降序、完整域 P95 升序、较早 checkpoint、SHA 排序。排名在读结果前冻结。独立选定候选固定 SHA 后复跑。仅导出诊断的候选单独排序并登记失败原因。资格以同一冻结候选的最近完整通过闭环为准。部署后确认能力或覆盖问题时可以按 D2/E 有界修订，最终仍按同一冻结规则回归。

## 4. D1 现有 Teleop 仿真集成

D1 从 D0 通过的同一 Actor 启动实际现有 Teleop 软件图并执行两个指定包的原始人端输入回放。Stage Gate G-D1 核验真实图、授权、闭环、两包/冷启动与完整反例、安全、回退、时序和 T1/T2 配对的计分完整性。实回放收益未达推荐时可产生带失败项的合格诊断输入给 D2；G-D1 的功能失败需修复复验，不能视为已完成正常集成。现有名义 SDK 按原接口在线推理，运行目录独立，无训练 bank 或真实环境设备。 [两包和对照](V1-D_teleop_replay_and_refinement.md)

## 5. D2 部署诊断、有界修订和条件支线

D2 前置 G-D1。接口类问题依序定位到 START、映射、在线 IK/SDK、Builder/归一化、采样/时钟、Servo、投影/延迟、WM 与残差能力。为工程问题保留最小修复及复验；补充训练前必须有清楚的数据/能力归因、资格、独立审阅和剩余预算。登记原 checkpoint、问题、假设、配置 diff、transition、候选 SHA 和完整离线/部署回归。可在已经满足各门、无需变动时签署有原因的 `NO_REFINEMENT_REQUIRED` 收据。可触发 E0 评估 EgoPro 是否能扩充已证实的参考分布缺口。E0–E3 各自有门；支线无合格候选时以原有效 Policy-T 回到 D3。D2 重做的收据绑定父产物哈希，失效旧材料单列保留。[条件支线](V1-E_conditional_egopro_augmentation.md) · [自主协议](../MULTI_AGENT_PROTOCOL.md)

## 6. D3 最终回归和发布

D3 从 G-D2 输出和当前 G-D0/G-D1 资格开始。若选 Policy-E，则要求 G-E3，同时保留原 Policy-T 和对应完整回归。锁定真正发布的 Actor SHA 与原始比较、software graph/SDK/Servo、score/domain/data manifest；独立实际完整 B0/B54 和两个包 T0/T1/T2 冷启动重播，复验离线 DG 指标与线上 TG 指标。交付可执行仿真手册、排障复现命令、trace、评分、失败/回退、边界、依赖和 release manifest。任何关键身份和支持缺陷保持 `BLOCKED` 或 `INCONCLUSIVE`。发布技术候选由用户最终验收。

## 7. 低资格诊断导出

当 C1/C2/C3 由于中间性能门失败或 D0 未达绝对/相对门而保留了安全的真 checkpoint，允许有界独立导出诊断包。它必须绑定真实 checkpoint、父阶段 `FAIL` 或 `INCONCLUSIVE`、有效数据谱系、学习与当前 WM 功能、因果、独立功能和安全门、基础跑批和独立 Reviewer 核验证据。独立诊断导出收据的最高等级为 `EXPORTABLE`。如另有已授权的正常独立运行验证，且同一 SHA 确实满足全部独立仿真功能证据，可在独立资格收据登记 `SIM_DEPLOYABLE`。根目录隔离到 `v1/<campaign_id>/diagnostic_export/<run_id>/`。receipt 类型 `DIAGNOSTIC_EXPORT`，入口只读已授权的日志和模型，无部署/真实权限、无正常通过位、无奖励或评分规则修改权，也不能签发后续正常 D1/D2/D3 收据。可在授权与预算范围内回原失败阶段修正。故障关键门未过时只保留不可发布 checkpoint 及证据。

## 8. 架构、测试及复现

计划 `v1/eval/paired_rollout.py`、`v1/eval/score_release.py`、`v1/deploy/export_actor.py`、`v1/deploy/verify_release.py`、`v1/deploy/sim_runner` 和 `v1/deploy/teleop_replay.py`。离线评估、独立 runtime 和实际 Teleop 图分开取证。独立 scorer 不读被测控制状态来改目标。负控须包含带低离线精度请求正式推荐、用简化仿真代替 D1、原录制命令或反馈、RMB 同期望替代在线名义、直接训练部署录包、无历史-真机分离、诊断资格冒用、非法 reset、非符合 deadline 的发令。正控核对相同配置和冻结 trace 的零残差旁路、可证明的实际软件命令与反馈、同域离线和部署独立重算。每阶段保留代码/环境/资产/模型/参数和配置 SHA、PID、启动命令与退出码、独立检查记录及含预算的 `gate_receipt.json`。证据路径 `v1/<campaign_id>/V1-D0|V1-D1|V1-D2|V1-D3/<run_id>/`。

## 9. 防止超预算与交付等级

总评估 2M 中 D 池 650k，D0/D1/D2 含 E/D3 为 200k/150k/100k/200k。按实际物理环境拍计，任何失败/重试/预试/诊断均计相应用途桶。D2 与 E2 分别最多用 C3 原 10M 中各自预留的 1M 训练 transitions，初始 C3 8M。WM 协议重资格、失败和重试照计。现有总资源上界、GPU 与时间规则不变。D3 预留不被前序消耗；无余量则停止、保留原最佳合格候选及证据。[资源参数](V2_parameter_registry_evaluation.md)

| 等级 | 资格与独立证据 |
|---|---|
| `EXPORTABLE` | 合法已训练 Actor 与固定契约、实际导出、哈希及离散/数值匹配通过。 |
| `SIM_DEPLOYABLE` | EXPORTABLE 加独立推理、因果、流式运行、硬安全与回退、连续独立 MuJoCo 通过。资格由独立正常验证收据赋予。诊断导出收据最高为 EXPORTABLE。 |
| `SIM_RECOMMENDED` | SIM_DEPLOYABLE 加 v2.0.0 Functional/Performance Gates 与 V2.0.1 G-D0 Offline Precision Gate。资格范围为独立仿真。 |
| `TELEOP_SIM_VALIDATED` | 同一候选通过独立运行基础及现有软件链路的功能、安全、接口、时序、回退、两包和实际反馈。D1 验证的 SHA 可凭其证据登记；换权重后须取得新 SHA 的独立实际回放功能证据。D3 的性能未通过可保留已证实的功能资格。 |
| `TELEOP_SIM_RECOMMENDED` | TELEOP_SIM_VALIDATED 加 G-D0 初始准入和 G-D2 归因证据，最终 SHA 在 G-D3 独立重跑并满足离线、指定录包性能及功能门。 |
| `REAL_DEPLOYABLE` | 真实基座、Servo、独立时钟/遥测、外测、安全、现场授权、Shadow/HIL/实机逐级取得证据，当前 `BLOCKED`。 |

先验收到的导出、离线和回放通过是不同收据；等级按实际证据核验。任何遗留差异、未完成部分和限制随发布候选交付。
