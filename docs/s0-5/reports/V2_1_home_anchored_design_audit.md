# S0-5 v2.1 阶段一设计审计

2026-11-28。基线为 GitHub `s0-5` 的 `cf07a88bc93d1aa9737b41fb83b9392921449142`；用户给出的 `324d771e29d436b206211f444a6d800338f871ce` 是其直接前序证据提交。当前交付身份为 `DCR_PENDING_HUMAN_REVIEW`。批准和运行收据尚未产生。[DCR-009](../dcr/DCR-009_home_anchored_human_intent.md)

## 按证据层复核

| 审计层 | 核查结果 |
| --- | --- |
| 现行设计及批准 | 从 `490770e1df45832d0693380563152bd19eefa3f4` 读取正式 v2.0.1 冻结候选，SHA-256 为 `cd5bff29a5b5f683fae328898aa9060b40fc4be00e193a22d9bf18cb2d10b60b`。核对 MASTER_DESIGN、PROJECT_STATE、CONTRACTS、ROADMAP、V1-A、RMB 与残差任务契约、评估协议和 P02 冻结对象。[原冻结清单](../design_freeze_candidate.json) |
| P02 与 M1 | 固定单位尺度静态 M1 有 P02 有限用途。独立物理旋转、MANUS 安装证据和连续 START 仍欠资格。保留固定 M1、M0 与早期合法不可达见证。[P02 映射](../evidence/P02/mapping_contract.json) · [历史评分候选](../evidence/P02/refinement/scoring_freeze_candidate.json) |
| Ubuntu Teleop | 实际源工作区 `/home/ubuntu/teleop_ws` 在 `4sensor`，HEAD `30cf13bdc496c43b3016462cfa0a206334d1ff7f`，相对上游领先一提交且有用户修改。读了实际 workflow 服务、Teleop core 和安装资产身份，核对已归档 controller、RetargetNode、Marvin adapter 及运行轨迹。源码树、已安装程序、历史专用 overlay 需各自签名。Home/Start/Enable 有 native 服务门，Stop/Hold、反馈新鲜度和两臂两手命令所有权须另行验证。 |
| 现有执行层 | Ubuntu 安装 MuJoCo driver SHA-256 `03a3119b94359ff6d429cfb18c5e16d2eb612866f8d21a25f679a18186cf6d7f`，其安装 XML `be69f387f5bc25c8011a992a88ac2c2449c2c0419e09318b740fcdcd1402871d`，DexGym 签名 XML `939b3651e2426e596eed482e70a68c2382720df1fd55b124aa5f0893b640e92d`。历史共享 Teleop overlay 程序 `d6b6cf85c761abd73c5a92b668210cc51aefb455a7dd1c2fc5f71a5105f8b109`。两 XML 有不同字节身份，须核验关节、执行器、site、FK、接触和 Servo 差异。 |
| 历史诊断输入 | 全量分层索引 `2bc6a63922e9f7790b4366690401c1d3934bd405b1a135cab1266c068fb1acb4`，实测 Home 收据 `65517e1786fe2799184e7e61905671d820159854d7a9e52b6c1aa3897f18f3a6`，共享外参 `33cbc93c4a2db925986f596b4b1730c7eaca026a9facfdf26aa4aaa8032c8f94`，外参说明 `dcee0d59cdee8448741e0d9c9b1e131ea999dc0215448b3622db382f4c87be08`。这些哈希来自服务器上的只读文件。[全量分层](TELEOP_full_start_handoff_layer_audit.md) |

历史回放 32/32 根的 Home→START 使用首个联合合格拍。其全量 99,562 个共同拍中，腕位置差 P95 按层为 M1→共享目标 **693.28 mm**、共享目标→下发 FK **79.11 mm**、下发 FK→实测 FK **18.66 mm**。另外 99,833 个共同拍上的滤波→求解步差 P95 为 **76.74 mm**，该层同时含工作空间 shaping 和步长限制。各项使用各自有效配对集合，分位数不可相加。报告中高误差三根的共享坐标对照保留间距与掌面相对角。旧 M1 与历史共享旋转差约 95.73°，轴假设和正式物理标定的正确性留待实测。[高误差同拍见证](TELEOP_high_error_mapping_witnesses.md) · [双手几何](../campaign/evidence/teleop_full_start_handoff_mapping_geometry.json)

## 分母和当前资格台账

| Split | 母源 | 分区 | 计划拍 | 联合有效拍 | 原签名五帧候选 | 85% / 70% 最小整数分子 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 24 | 228 | 86,338 | 85,382 | 82,559 | 72,575 / 57,792 |
| Validation | 8 | 80 | 30,275 | 29,150 | 28,237 | 24,778 / 19,766 |
| 合计 | 32 | 308 | 116,613 | 114,532 | 110,796 | 两个 split 各自过门 |

| 项目 | 状态及处置 |
| --- | --- |
| 现有 Teleop 路径差异的独立归因 | 部分可观察。已知的原生 workspace、限速、IK、命令与反馈差分保留。其独立因果归因、当前路径检查覆盖和实际生效拍状态仍有 `UNKNOWN`。 |
| 物理 `R_cal`、安装偏置与两模型身份 | 独立标定缺失，资产等价未资格化。用户批准 DCR 后按提案取证并限定适用域。 |
| v2.1 锚点、START 到达、54 轴合法跟踪、残差见证 | `NOT_RUN`，未来通过六根 Train Pilot 验证。新目标的失败、HOLD、回退、不可达、反馈缺失与未知计数尚未产生。 |
| 全量、策略训练和实物 | `NOT_RUN`，对应权限尚未产生。 |

拟议 Pilot 最多六个 Train 母源、八次冷启并含失败，最多 72,000 个 20 ms 仿真规划槽。历史全量含 116,613 个原计划拍，50 Hz 下约 38.9 分钟轨迹跨度，不含预定位、复核、进程启动及 IO。用 Pilot 实测的慢例、磁盘字节/拍和失败比例估算全量资源，预算上限和 START 安全证据提交下一阶段审查。Validation 只使用冻结参数。[运行与资源提案](../specs/V2_1_home_anchored_runtime_proposal.md) · [RMB 证据和门槛](../specs/V2_1_home_anchored_data_gate_proposal.md)

## 复核命令及阶段入口

```text
git ls-remote origin refs/heads/s0-5
git rev-parse HEAD
git show 490770e1df45832d0693380563152bd19eefa3f4:docs/s0-5/design_freeze_candidate.json
git -C /home/ubuntu/teleop_ws rev-parse HEAD
git -C /home/ubuntu/teleop_ws status --short --branch
sha256sum <signed asset, installed binary, and historical audit paths>
py -3 -B docs/s0-5/evidence/V2_1_START_PROPOSAL/check_proposal.py
py -3 -B -m unittest discover -s docs/s0-5/evidence/V2_1_START_PROPOSAL -p "test_*.py" -v
```

设计与人体几何代理复核坐标、历史评分和标定。Teleop、模型与安全代理复核控制状态机、运行时桥、Pilot 和路由。主代理集成契约和检查器，独立 Reviewer 核对证据链、冻结内容、坐标和通过条件。阶段一无实测新结果。检查命令结果、输入及输出 SHA-256 与下一阶段进入条件随 Git 提交封存。[设计检查收据](../evidence/V2_1_START_PROPOSAL/check_receipt.json) · [文件哈希](../evidence/V2_1_START_PROPOSAL/proposal_artifacts.json) · [独立审核](V2_1_home_anchored_independent_review.md)
