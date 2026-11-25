# S0-5 研发路线 v0.1.0

| 阶段 | 交付/阶段门 | 前序 |
|---|---|---|
| [P00](specs/P00_architecture_and_audit.md) | 仓库、运行环境、数据、历史行为和全套设计 | 用户授权 P00 |
| [P01](specs/P01_teleop_data.md) | Teleop-only 的独立数据和谱系 | P00 ACCEPTED |
| [P02](specs/P02_intent_and_feasibility.md) | 三层意图、可达性 Oracle、难段归因 | P01 ACCEPTED |
| [P03](specs/P03_nominal_controller.md) | 强传统基线、Servo、安全与冻结协议 | P02 ACCEPTED |
| [P04](specs/P04_rl_environment.md) | 54 DoF 因果任务、PPO 接口和真实 smoke | P03 ACCEPTED |
| [P05](specs/P05_static_precision.md) | 静态收敛与手指/对捏精度 | P04 ACCEPTED |
| [P06](specs/P06_dynamic_tracking.md) | 原速追踪、难段和双手协调 | P05 ACCEPTED |
| [P07](specs/P07_generalist_tracking.md) | 实时事件、未见动作和长时运行 | P06 ACCEPTED |
| [P08](specs/P08_robustness.md) | 臂手物理随机化和时序鲁棒 | P07 ACCEPTED |
| [P09](specs/P09_final_evaluation.md) | 强基线、多种子、消融、一次独立 Test | P08 ACCEPTED |
| [P10](specs/P10_deployment.md) | 独立部署、Shadow/HIL 与授权真机 | P09 ACCEPTED |

每阶段完成前置证据、审查和冻结当前阶段 Spec，获授权后再开工。每阶段有独立验收报告。正式 RL 之前 P03 冻结评估与相对收益门，动态训练之前 P05 静态验收，最终 Test 之前 P09 冻结全部候选和预算。P10 硬件审批独立。当前授权与状态见 [状态](PROJECT_STATE.md)。
