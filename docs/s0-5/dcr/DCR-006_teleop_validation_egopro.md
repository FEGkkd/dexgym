# DCR-006 | Teleop Deployment Validation & EgoPro Distribution Expansion v2.0.1

| 字段 | 变更 |
|---|---|
| Original Design | v2.0.0 将 B0/B54 仿真比较与无 Bank 独立部署列为首版交付。EgoPro 排除在首版研究范围。 |
| Discovered Problem | 改善比例无法确定低误差，独立部署不能证明现有 ROS Teleop 行为，Teleop 参考覆盖可能约束泛化。旧 M1 阈值切换和中间性能门失败也需要有界处理。 |
| User Decision | 项目用户确认 Policy-first、54 DoF、SuperMimic 和 Multi-Agent 方向，要求离线绝对精度、现有 Teleop 指定录包闭环、部署反馈迭代，允许有条件扩充 EgoPro。 |
| Supporting Evidence | 原 v2.0.0 候选 Sha `dda0c8254d0c8e637bbe1ee14d1bdfbf8b6b7309a8209263fc350a0c3e4561ea`。两份指定 MCAP 实际可读，有历史开发/选型线索。原 P02 保有各 M1/M0 几何、硬阈值和未冻结平滑候选。 |
| Proposed Change | D0 离线绝对精度，D1 现有 MuJoCo Teleop，D2 诊断与有界修复，D3 离线与 Teleop 联合回归。E0–E3 只在证实覆盖缺口、Teleop-only 已有效且有余量时执行。 |
| Alternatives | 延后至无 Bank 导出后另立集成项目，或无条件混入 EgoPro，均无法在首版区分策略、系统接口及数据分布贡献。保留独立仿真等级和 Teleop-only 合格回退。 |
| Expected Benefit | 同条件 B0/B54 比较量化离线收益，T1/T2 区分现有链路中的残差贡献，T0/T1 供映射或接口归因，分源保留扩充收益与 Teleop 保持能力。 |
| Risks | 低误差域可能覆盖不足。指定录包可能缺启动语义或与历史数据有重叠。旧工作区及命令来源必须隔离。EgoPro 几何、时间和动作分布需重新资格化。迭代受同一资源上限。 |
| Affected Requirements and Phases | 001–035 保持编号，新增 036–041。更新 A/B 的冻结和物理见证，C3 预算，D0–D3 与 E0–E3 技术门。 |
| Affected Contracts and Tests | RMB/M1、低误差域、记录角色、因果回放、诊断、WM 新域、预算和发布等级，实际执行将采用各阶段独立测试。本轮仅合成状态控制与设计检查。 |
| Compatibility | v2.0.0 冻结对象与历史检查脚本保留。原 P00–P02 凭据、HMB/RMB 契约身份和数据历史不变。M1 有界平滑候选只在硬阈值连续性未通过且训练前经审核才允许冻结。 |
| Remaining Hardware Obligations | C2-RT、P01-T、物理基座、真实 Servo、外部计量、真实场地、Shadow、HIL 各自审批。独立 Final Test 仍阻塞。 |
| Approval Status | DRAFT_READY_FOR_REVIEW。项目用户确认核心方向。本次补充的候选参数和认证范围待审核，Campaign 执行授权为 false。 |

[原冻结候选](https://github.com/FEGkkd/dexgym/blob/f78ee27f4762fedca9529bf697560bb267f8c704/docs/s0-5/design_freeze_candidate.json) · [新版路线](../ROADMAP.md) · [验收](../EVALUATION_PROTOCOL.md)
