# S0-5 技术契约 v2.0.1

状态 DRAFT_READY_FOR_REVIEW。各子契约规定可序列化接口、单位、顺序、失败和版本锁；变更由 DCR-006 和 Campaign 协议治理。

| 契约 | 唯一责任边界 | 验证 |
|---|---|---|
| C1 数据与目标 | HMB 读入只读。RMB 区分 nominal、mapped、optional feasible，保留原始 lineage、M0 与计划分母。可行见证与评估目标独立于候选策略。| [RMB schema](specs/V2_rmb_contract.md) · [V1-A](specs/V1-A_offline_rmb.md) |
| C2 时间、因果与历史 | 20 ms 仿真控制拍、2 ms 物理子步为候选。保存四路捕获、到达、截止与资格。Actor 只读当前和历史，episode/索引/未来/Oracle 不进入张量。物理延迟与历史 SIMULATED 时钟独立取证。 | [任务契约](specs/V2_residual_task_contract.md) · [RMB schema](specs/V2_rmb_contract.md) |
| C3 命令、安全和 Servo | 54D 有名动作顺序左臂/右臂/左手/右手为 7/7/20/20。记录请求、下发、区间施加、实测和 Servo 记忆。相同 B0/B14/B54 安全与模型。确定性仓库安全和 Servo 拥有实际执行权。 | [V1-B](specs/V1-B_residual_environment.md) · [运行时](specs/V2_runtime_contract.md) |
| C4 共享观测与无 Bank 运行 | 同一 builder、关节 manifest、历史、mask、归一化与导出输入。缓存名义参考为开发资产，运行时由当前流式输入产生在线名义命令。 | [任务契约](specs/V2_residual_task_contract.md) · [运行时](specs/V2_runtime_contract.md) |
| C5 学习和归因 | 实际 MuJoCo transition 驱动 Replay，区分终止与截断。WM 输入为实际有效控制及状态，分离任务参考并按 causal imagination 推进。Actor 梯度和 Critic lambda-return 按契约验证。 | [学习契约](specs/V2_supermimic_contract.md) · [V1-C0](specs/V1-C0_world_model.md) |
| C6 版本、恢复、计量和发布 | identity 绑定设计、资产、配置、数据、提交、真实成本、checkpoint 和独立审查证据。变更传播到全部受影响产物，实际幸存分位数和失败复合分位数均保留。 | [自主研发协议](MULTI_AGENT_PROTOCOL.md) · [评估协议](EVALUATION_PROTOCOL.md) · [V1-D](specs/V1-D_evaluation_deployment.md) |
| C7 离线低误差域与实际 R/T 配对 | 训练前锁见证、支持、完整分母。D0 比较 B0/B54。D1/D3 由指定 MCAP 人端输入及允许事件驱动现有实际 MuJoCo 软件图，T1/T2 共用上下文，历史命令和反馈只读诊断。 | [回放规格](specs/V1-D_teleop_replay_and_refinement.md) |
| C8 分布扩充、血缘和留出 | Teleop-only 通过后以独立 EgoPro Bank 处理资格、几何、时钟、mask、采样和来源。支持差异、域内模型及原 Teleop 保留分别计量。部署回归源/派生不得新增至训练。 | [数据准入](specs/V1-E_conditional_egopro_augmentation.md) · [E2/E3](specs/V2_1_egopro_training_and_regression.md) |
| C9 诊断与等级 | 独立诊断导出须通过数据、功能、安全和因果检查。正式现有链路功能授 TELEOP_SIM_VALIDATED，增加 D0 绝对精度和 D3 双回归才可授 TELEOP_SIM_RECOMMENDED。 | [等级和证据](specs/V2_runtime_contract.md) |

术语 `physical rollout` 指执行 MuJoCo 方程并读回状态。P01-T、C2-RT、物理 robot-base、硬件 Servo 和外部测量继续各自独立审核。旧契约及量纲可按固定历史版本查证。

## C1 几何和数据

历史 P01 使用原 C1 口径。 [新版 RMB schema](specs/V2_rmb_contract.md)

## C6 版本及运行证据

历史工位规格使用原 C6 口径。 [新版自主研发协议](MULTI_AGENT_PROTOCOL.md) · [v0.1.2 契约](https://github.com/FEGkkd/dexgym/blob/b8d8b401bdda59ad00a37f0a3bf6878b462ed756/docs/s0-5/CONTRACTS.md)
