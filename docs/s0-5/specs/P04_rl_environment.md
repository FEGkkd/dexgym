# P04 54 DoF 因果任务、PPO 接口和真实 smoke

## 1. 阶段与版本

P04 v0.1.0，初稿。

## 2. 目标

建立独立 wholebody_s05 任务与 PPO adapter，真实闭环 smoke 验证语义。

## 3. 与主设计的关系

M4–M6、C2–C6。

## 4. 覆盖需求

002、003、016、017、018、019。

## 5. 前置条件

P03 ACCEPTED，B1+ 安全和 Servo、控制频率以及 E 协议冻结。

## 6. 现有实现审计

检查 Wholebody S0 的 TanhGaussian、RSL-RL 版本分支、Marvin54 物理及 S0-4 时序。对 adapter 单独做行级迁移审核。

## 7. 范围

54 动作四头、现/历史因果特征、手中心 reward、reset/终止/安全层，PPO 更新与恢复的最小实跑。

## 8. 范围外

正式性能训练、大规模 GPU campaign、SuperMimic 对照和真机。

## 9. 输入契约

同 P03 的独立目标与控制器、物理状态、固定训练 split、配置版本及 RNG。

## 10. 输出契约

task/action/observation/reward/env、PPO 对接、trace 和 checkpoint metadata，固定训练/部署可共享接口。

## 11. 数学定义

C5 多尺度、有界高斯 logπ、GAE 末值及实际 command/physics tick 定义。

## 12. 架构与模块

一个统一 task，按模块单向依赖数据、名义和 Servo；Actor 与 Critic 输入独立。checkpoint 兼容检查在载入前。

## 13. 实施任务

由数学和接口测试开始，再跑 GPU 少环境真实闭环、极短 rollout 和恢复，审查几何可控性和终止奖励。

## 14. 预期文件与接口

计划 task/{action,observation,reward,env}.py、learning/{ppo,checkpoint}.py 与契约标识。

## 15. 单元测试

54 轴审计、零动作、前缀不变、项目白名单、reward 单调与完美、超时 bootstrap、projection 与 reset 顺序。

## 16. 集成测试

检查 10 物理子步、实际控制动作、目标与状态绑定、PPO 比率与真实 transitions、恢复及部署接口同构。

## 17. 实际运行

少量环境真正执行 54 指令与最小的 PPO update，保存命令和退出日志。

## 18. 定量协议

指定偏差的 term 贡献与梯度方向、随机输入后每轴是否到达执行层、固定 seed 与同名义对照。

## 19. 强制验收

全部必需数学/接口语义且真实仿真 smoke 通过。缺实际执行或记录不能进入 P05。

## 20. 失败处理

对重采/批量终止保存失败轨迹；分布、时间或投影错先修契约再启动实验。

## 21. 风险与未决问题

TBD 具体 obs 字段、物理归一化和 loss 系数由 P04 根据可控方向和消费时效定，P05–P10 共用。

## 22. 可复现要求

固定 dtype、设备、环境版本、随机种子、日志、检查点哈希、实际 transitions。

## 23. 交付物

task、适配器、自动化测试、实跑证据及验收报告。

## 24. 退出门

功能证据审核完成，P05 正式训练须新授权。

## 25. 下游依赖

P05 静态训练以及 P09 等契约算法对照。
