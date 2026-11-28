# DCR-010 Offline RMB First and Deployment Safety Decoupling

S0-5 v2.2 设计候选，`PENDING_HUMAN_REVIEW`。审计基线为 `548f04195d9f3815cc522d0f7f76c95b3bee8c46`。本轮授权覆盖设计提案、隔离仿真实现、限定 Train 诊断及合成验证。正式主评分层、V1-A 分子、全量构建、正式训练和真实部署各有独立批准。

## 背景与用户决定

Marvin IK 和 Wuji SDK 提供名义动作，人类运动意图提供独立目标，SuperMimic 学习机器人实际动力学及 54 轴控制残差。现有实时守卫 V2 对 24,433 条原始请求恢复了 817 条连续证明。现场回放于 3.529 s 停机，三个终止请求使用更高离线预算后各获得双模型连续证明。腕掌解剖外参仍为 `UNKNOWN`，两模型手部几何仍有约 6 mm 差异。 [DCR-009](DCR-009_home_anchored_human_intent.md) · [V2 结果](../reports/DCR-009_swept_guard_v2_results.md) · [独立审查](../reports/DCR-009_swept_guard_v2_independent_review.md)

## 总体设计及阶段依赖

| Track | 依赖与交付 |
|---|---|
| A 离线仿真与 Policy | HMB → 单次 Home-Anchored START → 独立仿真意图、原生命令提案和离线连续证明 → 固定仿真时序的 MuJoCo 执行 → 候选 RMB → G-A/G-B → G-C0 → 获授权的静态、动态和混合训练 → 独立仿真验收。V1-B 合成接口与几何审计可以并行。 |
| B 实时与物理部署 | 实时整身证明、250 Hz 原生期限、ROS 调度、真实通信与命令应用、Servo 辨识、解剖外参和物理基座标定、Shadow Mode、HIL、现场硬件安全和部署。各项在实际系统上独立验证。 |

Track A 的离线机理沿用 Swept Guard V2 连续几何证明，每请求和每母源均设预算。实际碰撞、净空不足、未知、回退、保持和失效都作为原时间轴中的独立结果。Track B 继续保留历史双模型守卫及实时资格路径。 [候选离线实施规格](../specs/V2_2_offline_rmb_track.md) · [原路线](../ROADMAP.md)

| 资格 | 证据和适用范围 |
|---|---|
| `OFFLINE_GEOMETRY_QUALIFIED` | 对具体模型、轴范围、净空及路径有连续证据。按已证明请求授予。 |
| `SIMULATION_EXECUTION_QUALIFIED` | 固定模型和 Servo，真实 MuJoCo 子步、命令及实测证据、同域硬边界和时间轴检查。 |
| `REALTIME_CONTROL_QUALIFIED` | 控制器、驱动和异频反馈在已冻结实时负载、期限及 STOP 语义下独立合格。 |
| `PHYSICAL_DEPLOYMENT_QUALIFIED` | 独立真实坐标和 Servo 辨识、实物计量、时序、安全与现场授权全部满足。 |

## 意图、模型和兼容性

保留 `RAW_HUMAN_GEOMETRY`、`REGISTERED_WRIST_PROXY`、`SIM_MAPPED_HUMAN_INTENT`、`ROBOT_MORPHOLOGY_TARGET`、`FEASIBLE_GEOMETRY` 和原 `M0_TARGET`、`M1_TARGET` 可追溯引用。单次 START 锁定双腕共同刚体变换，保留原始目标及相对 SE(3)，普通分区延续状态。仿真轴和工程安装桥有签名证据，人体解剖腕掌外参与正式物理精度保持 `UNKNOWN`。固定方向及公共偏航为明确候选。策略训练前由 Train 验证与人工审查确定数学规则和主评分层。

由同一 DexGym 签名模型计算离线反馈、机器人形态目标、参考层与后续训练的 Servo/Reward，版本和依赖单独哈希。原 Teleop 模型和约 6 mm 差异作为独立部署回归证据保留。离线近似迁移到 Teleop 后要按相同输入、相同状态及零残差验证。 [模型与坐标证据](../campaign/DCR-009_calibration_evidence.md) · [原 RMB 契约](../specs/V2_rmb_contract.md)

## 原门槛与批准

32 个母源、24 Train／8 Validation、308 分区、116,613 个计划拍、114,532 个联合有效拍、110,796 个原五帧候选守恒。原 85% 名义主动覆盖及 70% 五帧门槛各自适用。保持、未知、拒绝、恢复边界和失败均留在原分母。第 5 根技术验证、六根同构验证及独立审查是申请全量构建的前置证据。正式 G-C0 须有可训练 RMB、G-B 实际集成和批准预算。正式 Actor 训练还须通过 G-C0 并有训练授权。原冻结候选、DCR-009 批准、历史评分及报告按原 Git 对象核验。 [DCR-009 阶段二批准](../campaign/approvals/DCR-009_stage_two.md) · [现行评分协议](../EVALUATION_PROTOCOL.md)

## 备选方案与风险

完整 ROS 锁步需要时钟注入、独立订阅调度及驱动步进。实际定时器和异步回调缺少已验证的锁步入口。候选采用直接复用已有 Teleop C++ 内核和 Wuji C SDK 的方案 B，并在 Python 本地以固定仿真时序交错执行 MuJoCo。该设计便于独立复算和按来源并行。待验证风险为离线抽取对 Wuji 更新的影响、几何代理差异、Servo 跟踪、较宽证明盒引起的保持以及 START 未达。
