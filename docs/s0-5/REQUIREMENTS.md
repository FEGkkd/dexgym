# S0-5 需求 v2.0.1

状态 DRAFT_READY_FOR_REVIEW。RETAINED 保持原技术义务，REVISED 修改责任或条件，DEFERRED 将原完整义务移到后续，SUPERSEDED 替换原主线义务，NEW 为新增。编号保持稳定。 [原 001–030](https://github.com/FEGkkd/dexgym/blob/b8d8b401bdda59ad00a37f0a3bf6878b462ed756/docs/s0-5/REQUIREMENTS.md)

| ID | 迁移 | 需求 | 责任 |
|---|---|---|---|
| S05-REQ-001 | REVISED | Teleop-only 主线与独立条件扩充 | V1-A |
| S05-REQ-002 | REVISED | 54 DoF 动作与零残差 B0 | V1-B |
| S05-REQ-003 | RETAINED | 共享因果 Actor | V1-B |
| S05-REQ-004 | REVISED | 独立映射、名义及可选可行层 | V1-A |
| S05-REQ-005 | SUPERSEDED | 现有名义链与公平 B0 | V1-D0 |
| S05-REQ-006 | REVISED | 静态精度和收敛 | V1-C1 |
| S05-REQ-007 | REVISED | 原速动态和困难段 | V1-C2 |
| S05-REQ-008 | RETAINED | 双手协调 | V1-D3 |
| S05-REQ-009 | REVISED | 实时推理与时间资格 | V1-D1 |
| S05-REQ-010 | RETAINED | 路径安全、回退与恢复 | V1-B |
| S05-REQ-011 | RETAINED | 来源、分区及可审计性 | V1-A |
| S05-REQ-012 | RETAINED | 标定、单位、时序和时钟资格 | V1-A |
| S05-REQ-013 | REVISED | 离线必要界与合法改善见证 | V1-A |
| S05-REQ-014 | RETAINED | 独立 FK 与 Jacobian 核对 | V1-B |
| S05-REQ-015 | REVISED | 仿真 Servo 与硬件对齐分门 | V1-B |
| S05-REQ-016 | RETAINED | Reward 单调、绑定拍和早退 | V1-B |
| S05-REQ-017 | RETAINED | 真实仿真 transition | V1-B |
| S05-REQ-018 | SUPERSEDED | SuperMimic 梯度及 bootstrap | V1-C0 |
| S05-REQ-019 | REVISED | 完整 checkpoint 与恢复身份 | V1-C0 |
| S05-REQ-020 | RETAINED | 单指、手型和对捏 | V1-C1 |
| S05-REQ-021 | REVISED | 变化、掉线与恢复 | V1-C3 |
| S05-REQ-022 | REVISED | 连续运行与泛化 | V1-D3 |
| S05-REQ-023 | DEFERRED | 全硬件分组随机化 | V1-C3 |
| S05-REQ-024 | RETAINED | 独立评分与失败分母 | V1-D3 |
| S05-REQ-025 | SUPERSEDED | SuperMimic 主线，PPO 可选对照 | V1-C0 |
| S05-REQ-026 | REVISED | V1 单种子与最终科研分离 | V1-D3 |
| S05-REQ-027 | REVISED | 独立 Actor 导出、SIM 运行 | V1-D1 |
| S05-REQ-028 | DEFERRED | 物理测量、遥测及真实部署 | 后续硬件 |
| S05-REQ-029 | REVISED | 一次授权、可审计推进与恢复 | 全阶段 |
| S05-REQ-030 | REVISED | 工程性能及最终挑战分门 | V1-D0 |
| S05-REQ-031 | NEW | RMB schema、覆盖和因果等价 | V1-A |
| S05-REQ-032 | NEW | WM 质量及真实有限差分 | V1-C0 |
| S05-REQ-033 | NEW | 模型漏洞利用检测和停训 | V1-C3 |
| S05-REQ-034 | NEW | 离线到实时名义映射差异 | V1-D1 |
| S05-REQ-035 | NEW | 有界自主调整和重试证据 | 全阶段 |
| S05-REQ-036 | NEW | 离线绝对精度与预 Actor 低误差域 | V1-D0 |
| S05-REQ-037 | NEW | 实际现有 MuJoCo Teleop 软件链路 | V1-D1 |
| S05-REQ-038 | NEW | 两份指定 MCAP 因果闭环回归 | V1-D1 |
| S05-REQ-039 | NEW | 部署反馈驱动的有界修订 | V1-D2 |
| S05-REQ-040 | NEW | 条件 EgoPro 独立准入与 WM 再资格 | V1-E2 |
| S05-REQ-041 | NEW | EgoPro 增益与原 Teleop 保留 | V1-E3 |

原 P02 静态结论和物理门均由原记录定义。完整验证方法、量纲、模块和证据见 [追踪矩阵](TRACEABILITY_MATRIX.md) 及 [评估协议](EVALUATION_PROTOCOL.md)。
