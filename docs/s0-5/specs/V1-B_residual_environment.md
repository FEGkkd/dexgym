# S0-5 v2.0.1 V1-B 残差环境与 SuperMimic 适配

状态 DRAFT_READY_FOR_REVIEW，技术门 G-B。负责 Physics & Task Agent，独立 Reviewer 查验实际执行。依赖整体 Campaign 授权、锁定 V2.0 配置与技术证据。获授权后可与 V1-A 并行使用有身份的合成输入。正式 G-B 需 G-A 已通过，并完成真实 RMB 接线及实样本检查。

## 目标与范围

实现 54 DoF 同一环境、同一硬安全和 Servo 下的 B0/B14/B54。覆盖仿真模型、任务与单步学习适配。

## 前置和输入

仿真 profile 单独资格化，硬件计量另行处理。输入为模型和控制映射、仿真 Servo、合成或真实因果名义与独立几何、动作、状态和历史。真实 RMB 与 G-A 批次、schema 及资产 SHA 精确绑定。

## 数学和时间

资产采用当前经核验的模型及依赖资产 SHA，位置模式、正常重力、50 Hz/500 Hz 先按 [残差任务契约](V2_residual_task_contract.md)。手部独立驱动或被动耦合由有名 manifest 验证。单位、q_requested、q_issued、q_applied、q_measured 独立记录。几何目标随一个控制拍的实际动作绑定，反馈截断保留 pre-reset。

## 架构和实施

复用 WarpPhysics、CPU MuJoCo、Torch FK、S0-4 Servo 时钟和 SuperMimic 任务钩子。输出统一 Gym 风格 env、共享 Builder、数据接口、硬投影和回退、奖励、SuperMimic task adapter、trace 和恢复 checkpoint。建议放在 source/dexgym/tasks/direct/wholebody_s05/v1 下的 action、observation、reward、safety、env、supermimic_adapter 及 sim_asset_manifest 模块。实现依次锁定模型、关节表、硬安全、Builder、执行、奖励、任务记忆、适配和恢复。

## 单元测试

- 单元检查 54 轴顺序、角度与速度、关节/驱动闭包、耦合边界、同端点不同穿越路径、失效、硬超时、终止先后、reset、奖励梯度、正负效用排序及 replay 写末值。CPU FK 对已知标定姿态和 SciPy SO(3) 参照，接触与事件独立解析。
## 集成测试

- 检查 54 轴单独正负动作，给出每轴实际 ctrl/q/qd 响应和符号，覆盖保持与组合。对同 signed fixture 的 B54 零动作和 B0 检验 q_requested/q_issued/硬事件完全一致，实测 CPU/Warp 同阵列按共享容差。对同一足够长输入验证未来删除、替换、元数据扰动和在线/训练 Builder 逐拍一致。

## 实际运行及量化

- 在 1/8/16 环境真实运行 1、10、150 子步配对及不少于 512 个物理控制拍。验证 54 轴实际影响、10 个子步、反馈与事件、正常截断和早终止的预 reset 状态。真实采样和 WM 参数 smoke 合法。Actor/Critic 梯度与 optimizer smoke 用隔离、可丢弃的合成 transition 验证，空白新 Actor 于 C0 合格前保持未训练。物理 Actor/Critic 更新实证由 C1 首次真实学习获取。
## 预算

内部预检记录每种环境数的峰值显存、物理 step 时间、更新耗时、缓冲估算和依赖版本。实际 smoke 预算不超过 25,000 个真实物理 transition，纳入 C0 0.5M 子预算。CPU/GPU 受测容差及候选范围用 [参数登记](V2_parameter_registry_training.md)。

## G-B 技术验收

必须全 54 轴实际响应且通过统一硬安全；零残差语义和回退等价；接触、错误、timeout、反馈及独立层有 trace；CPU/Warp、Actor 因果负控、Builder 等价、恢复性 smoke 时序和可微测试通过。接入 G-A 锁定的 RMB，至少复验 16 源、每源 32 帧的 schema、单位、目标层、起止和 reset，检查运行时是否隔离特权标签。

## 失败和风险

缺物理模型、部分通道、限幅错误、坏梯度、片段末值错或任何关键审查未关均阻塞。最多两次复验，全部算原预算。风险包括 CPU/Warp 符号、滤波错一拍、掩码被标准化和软安全泄漏。

## 复现、交付和下游

固定 fixture、模型、seed、执行命令、退出状态、源与配置 SHA、日志、测试均由 v1/<campaign_id>/V1-B/<run_id>/gate_receipt.json 链接。交付运行 SHA、Builder 和真实 RMB 兼容记录。C0 拟合前再次确认身份和可控性。

计划任务模块命名采用追踪矩阵的 `wholebody_s05/v1/task/`，学习 adapter 由 `wholebody_s05/v1/learning/` 管理。

G-B 预留同一 CPU/Warp 模型、Servo、安全和初始状态供预 Actor 的连续低误差见证。见证采样属于真实仿真评估，计入 C0 评估分池，与 Actor 成绩隔离。 [冻结规则](V1-A_offline_rmb.md)
