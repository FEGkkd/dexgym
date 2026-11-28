# DCR-009 阶段二人工授权记录

设计提交 `6808935d7588834ad875789a2ca6bbb779d2caa9`。设计候选 SHA-256 `8f10a7c6dd2c98358ae4f5f02fbe5908acdfbf82b9e9507a59573467f6652100`。登记依据为该提交交付后项目用户在同一任务中的明确回复。

> 我批准 DCR-009，开始阶段二。继续多 Agent，先核验 R_cal 的独立标定，这个仿真标定可能在teleop-ws有，如果有问题就用mujoco重新算一个和 Teleop 与签名模型的一致性，再实现 Home→START 并跑 6 个 Train 母源的 MuJoCo Pilot。保留原始目标、分母和 85%／70% 门槛，提交独立审查和可视化对照后暂停，等我批准全量重建。不要启动训练。

| 授权或技术状态 | 登记 |
| --- | --- |
| DCR-009 设计 | `APPROVED`。旧阶段一提案和其封存状态按原 Git 对象保留。 |
| 阶段二 | 核验现有标定，必要时取新的 MuJoCo 仿真标定证据，验证模型，实现和测量 Home→START，执行六 Train 根的有界仿真 Pilot 和预登记冷启复验。 |
| 仿真限定 | 已验证的仿真坐标资格可限定用于本轮 Pilot。历史佩戴、实物、时钟或外测精度资格独立登记。 |
| Pilot 技术通过／人工验收 | 将独立登记技术证据和项目用户后续验收。 |
| 全量 RMB、正式 SuperMimic、真实机器人 | 本记录对此三类授权均为 `false`。 |

原始 M0／M1、人体几何、计划拍、已签 HMB 五拍候选、Train／Validation 和两道数值门不变。需要进一步测量或未通过的技术条件仍显式记录。[DCR](../../dcr/DCR-009_home_anchored_human_intent.md) · [试验设计](../../specs/V2_1_home_anchored_runtime_proposal.md)
