# S0-5 v2.0.0 | P00 设计审核报告

**DESIGN_READY_FOR_REVIEW · DRAFT_READY_FOR_REVIEW · PENDING_HUMAN_REVIEW**

## A. Executive Summary

本设计把 S0-5 的主线设为现有 Teleop 上的共享 54 DoF SuperMimic 残差策略。第一版以独立的流式闭环仿真包为目标，使用现有历史 HMB 和 MuJoCo Warp，保留独立人体 mapped intent 作为主目标。本轮完成设计、源码和历史凭据审计、独立评阅和只读检查器。V1-A 至 V1-D 均为 NOT_STARTED，执行权限 false。项目用户与外部 Reviewer 接下来审核设计，Campaign 需单独批准。 [总体设计](../MASTER_DESIGN.md) · [DCR-005](../dcr/DCR-005_policy_first_supermimic.md)

## B. Original vs Revised Architecture

| 原 v0.1.x | V2.0.0 候选 |
|---|---|
| 新在线 54 DoF 联合 IK/QP 是学习前置 | 现有 Marvin IK + Wuji SDK 产生名义参考，B0 为同接口零残差。联合 IK/QP 是可选后续对照。 |
| PPO 主训练，后续增加 SuperMimic | SuperMimic 与独立 WM 资格为主线，PPO 是有预算才做的同条件对照。 |
| 真实硬件测量和注册阻塞正式控制开发 | V1 仿真实际计量和硬件资格分开。现有安全与各项真实资格保留。 |
| 小阶段逐一人工接受 | 设计批准加单独整体 Campaign 授权，阶段由独立技术证据推进，最终提交人工验收。 |
| P09/P10 承担科研评估和部署 | V1-D 优先交付真实合格的仿真等级，终评与物理部署另审。 |

## C. Preserved Research

远端 `s0-5` 初始 HEAD 和 Ubuntu 实际干净的 `/home/ubuntu/dexgym_worktrees/s0-5` 均为 `b8d8b401bdda59ad00a37f0a3bf6878b462ed756`。P00 原验收、Review Fix、P01-A–F、P02 原研究、复查、最终交接、DCR-001–003 与历史协议都在原 Git 对象，已有报告、证据、源码与签核的原始字节保留。旧活动文件在固定 Git 基准可查。独立 SIM0 工作树为 `c5af9af94d7b9d3f437c14df128441d1183bec56`，原签核对象随设计 Git 历史保留。该树的未跟踪源码及运行结果、用户当前脏的 `~/teleop_ws`（4sensor）和 Windows 既有目录均保留原样。在独立克隆完成本次修订。 [历史身份](../evidence/P00_V2/historical_inputs.json) · [P02 有限签核](https://github.com/FEGkkd/dexgym/blob/c5af9af94d7b9d3f437c14df128441d1183bec56/docs/s0-5/reports/approvals/P02_LIMITED_ACCEPTANCE_SIM0.md)

P02 Final Run `P02-FINAL-20261126-ecd12ec` 的 receipt、limited candidate 和 agent manifest，及 E1 和 F1→P02 交接五项均按既有 SHA 核对。Ubuntu E1 真实归档的 `review-1/hmb_manifest.json` 与仓库副本哈希相同。P00/P01 历史已接受，P02 历史静态有限接受沿原范围记录。M1 连续仿真应用为本次新审批项，正式物理评分独立待审。 [旧 P02 交接](P02_final_handoff.md) · [新版状态](../PROJECT_STATE.md)

## D. New Stage Roadmap

A 构建三层 RMB 并冻结域。B 可先用合成数据准备，待 A 通过后验证实际 RMB 和物理、动作、安全、观测、Reward。C0 资格化 WM，C1 静态、C2 原速、C3 混合迭代保持 WM 有效，D 经 C3 技术通过后独立配对回放、导出、流式部署。D 的工程性能未过仍可按真实资格报告较低等级。 [阶段与依赖](../ROADMAP.md)

## E. Algorithm Design

审计 `engine.py`、`config.py`、`buffer.py`、`networks.py`、`wholebody_s0_1/supermimic.py`、`wholebody_s04/train_aligned.py` 等实际 Git 源和 Ubuntu 文件。现引擎可供复用的部分包括 collect/Replay、预测更新、differentiable imagination、Actor 梯度及 Critic。V2 须实现独立源留出、54 轴有界激励、资格前禁止 Actor 更新、有效执行控制/SERVO 记忆一致、终止和截断 bootstrap，以及完整 checkpoint 恢复。历史 S0-4 的实运行配置 `action_noise=0` 缺乏手轴覆盖证据。V2 的奖励先归一到 [0,1]，hard failure 为 0，价值边界与 gamma 一致。 [学习契约](../specs/V2_supermimic_contract.md) · [任务契约](../specs/V2_residual_task_contract.md)

## F. Acceptance Criteria

功能门覆盖 RMB 身份、54 轴真实效果、B0 等价、因果、实际样本更新、独立 WM、安全与闭环部署。V1 性能候选要求静态和原速共同可改善域 P95 各相对改善至少 10%，完整有效动态域 P95 不超过 5% 退化，控制失败增量至多 1 个百分点，并通过功能非退化门。统计保留全计划分母、失败复合 P95、源等权及帧口径，接近零基线与非有限量走显式规则。最终静态 3 mm，原速动态 2/5/10 mm，困难段 10 mm，捏合 2 mm，双手相对 3 mm，稳定速度 1 mm/s，扰动恢复 300 ms，Actor P99 5 ms，真实端到端 50 ms，以及动态和难段 ≥20% 相对改善，均保留原挑战。Final Test 当前 BLOCKED。 [完整验收](../EVALUATION_PROTOCOL.md)

## G. Autonomous Campaign

主代理维护 DAG、资产和预算，分派 Data、Physics & Task、WM & Training、Deployment 和 Independent Reviewer。技术评审从同身份原始证据重算，独立 Reviewer 不参与被审核心模块的原始实现。每次失败有独立 Run ID、命令/PID/退出、已消费资源和重试。监督、恢复、账本与停止义务按协议执行，变更传播到所有受影响技术门。 [自主研发协议](../MULTI_AGENT_PROTOCOL.md) · [本轮 Agent 登记](../agent_work_manifest.json)

## H. Consistency Audit

V2 检查器读取活动版本、需求 35 项、7 个阶段、全部 DAG 边、链接、历史标签与文件、哈希、预算和设计-only状态。合成用例验证正常授权和技术门转换，以及无权、缺依赖、预算耗尽、指标篡改、失效 WM、历史凭据篡改、真实部署与正式/独立评分门禁。最初检查发现旧入口直接读取新版需求表时退出 1。独立快照重放在 b8d8b40 验证旧活动文件和 486 个本地链接。旧合成测试以引入版本 09c1c9e 运行 5/5，通过。旧五项用例直接套用 b8d8b40 时有一项在新版 P02 收据层先报缺模拟文件，保留该发现。初次 V2 集成还发现阶段章节和链接未齐，补齐后用最终 manifest 复验。 [实际执行结果](../evidence/P00_V2/verification_results.json) · [活动验证入口](../evidence/P00_V2/check_design.py) · [历史重放入口](../evidence/P00_V2/verify_legacy.py)

## I. Independent Reviewer Findings

独立 Reviewer 对三组设计、源码、历史证据和初版检查器进行了实查。高重要问题涉及 DAG/诊断分支、奖励和历史帧数、过期依赖证明及重试计数。中等与低重要问题涉及各层功能口径、证据路径、历史入口与链接、掌腕坐标以及计划模块。原发现与复核见证保存原文，处置结果纳入候选核验。 [独立设计复核](../evidence/P00_V2/independent_findings.md)

## J. Decisions Requiring Human Approval

- 批准 Policy-first / SuperMimic-first / simulation-first 和历史 P02 的范围登记。决定 M1 尺度 1 用于连续仿真的额外限定用途。保留 T0/历史 START 等工程对照及后续接线义务。 [V1-A](../specs/V1-A_offline_rmb.md)
- 审核参考覆盖、RMB 配准连续性、仿真 Servo 与 CPU/GPU 容差、完整或紧凑观测、残差尺度和 54 轴激励、WM 一步/多步/梯度及失败边界、分层评估与功能非退化容差。 [训练参数](../specs/V2_parameter_registry_training.md) · [WM 资格](../specs/V1-C0_world_model.md) · [评估参数](../specs/V2_parameter_registry_evaluation.md)
- 审核单主种子 49、20M 训练、2M 评估、1 GPU/48 GPU 小时、72 总小时含 24 小时等待、250 GiB 及重试上限。Ubuntu 只读快照见 RTX 5090 32,607 MiB 和约 640 GiB 可用磁盘，后续以授权当时实际预检为准。 [资源治理](../MULTI_AGENT_PROTOCOL.md)
- 真实时钟、白名单遥测、物理基座、实机 Servo、外部计量、独立 Final Test 和正式物理评分仍各自待审。生产 Teleop 的 START、时钟、命令所有权和真实动作另有门禁。 [硬件计划](../specs/HARDWARE_SHARED_MEASUREMENT_PLAN.md) · [运行时契约](../specs/V2_runtime_contract.md)

## K. Freeze Candidate Identity

版本 v2.0.0，复核基准 `b8d8b401bdda59ad00a37f0a3bf6878b462ed756`。冻结候选清单列出每份活动文件、审查材料、历史 Git 对象和源文件的 SHA。发布提交及清单自己的 SHA 由 Git ref 和发布后独立重算登记，避免把自引用写入待哈希文件。 [冻结候选](../design_freeze_candidate.json)
