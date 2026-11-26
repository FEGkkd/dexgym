# S0-5 V1 Autonomous Campaign 实施记录

**当前状态为 G-A、G-B 前置资格未通过，首版策略尚未交付。** 已保存实际启动和后续恢复运行。

Campaign ID 为 `S05-V1-20261127-490770e`。冻结设计为 `490770e1df45832d0693380563152bd19eefa3f4`，SHA-256 `cd5bff29a5b5f683fae328898aa9060b40fc4be00e193a22d9bf18cb2d10b60b`。执行配置 SHA-256 `955d80d09ffe674533cf5000e7bd227c3cdd7998cbbfb2737b42c273bc60e822`。DCR-007 范围独立核对通过，此审核只涵盖资源变更。 [执行登记](../campaign/README.md) · [资源审核凭据](../campaign/independent_resource_review.json) · [独立启动复核](../campaign/independent_launch_review.md)

## 恢复阶段证据

使用单独的内存目录替换触发 SIGBUS 的应用依赖，通过了 CUDA 运算和一环境 Warp 模型初始化。原生 CPU MuJoCo 随后正常退出，实际完成 136 个控制转移，54 轴正负响应、B0 零残差、非法动作回退和 reset 前反馈均通过。该次使用合成目标。G-B 仍需通过签名 RMB 接线、CPU/Warp 物理等价等独立检查。 [恢复证据索引](../campaign/resume_run_evidence.json)

相同模型的 CPU/Warp 自由响应通过 15 组对照，共 170 个实际物理区间，最大指尖差约 0.00112 mm。8 组合成安全边界对照也全部通过，涵盖命令拒绝、硬停和实际仿真自碰撞。CPU 与 Warp 分别通过 8 环境合成 World Model 接口烟测，各执行 176 个物理 transition、2 次 WM 更新。两次烟测里 Actor、Critic 和想象轨迹计数均为 0，C0 正式训练及独立模型资格仍未启动。 [CPU/Warp 自由响应证据](../campaign/evidence/warp-parity-03.json)

同一签名模型通过 64、256、1,024、2,048、4,096 个实际 Warp 环境的有界物理预检。最大规模实际执行两个控制拍，共 8,192 个有效物理 transition，约 683.53 transition/s，采样进程显存峰值 5,900 MiB。这一组数据覆盖短程物理控制，未包含 4,096 环境的 Replay、WM 或长程 learner 成本。 [规模预检](../campaign/evidence/v1b-scale-ladder.json)

全量签名 Train 的 228 个分区、86,338 个计划拍完成腕部可达性审计。在 85,382 个 HMB 联合有效拍中，至少一侧超出原 M1 精确腕姿所需的原生 SDK 连杆上界者有 63,967 拍。将 139 个未探测拍全部计为可能合格，精确双臂 IK 覆盖的数学上限仍为 21,415 / 85,382，即 25.08%，低于冻结的 85% 合法名义覆盖要求。此界限适用于原 M1 腕目标的精确 IK。现有 Teleop 的有界限速首步每侧 192/192 获得原生 SDK 解，该诊断没有证明整段命令合法或跟踪精度。 [全量审计数据](../campaign/evidence/train-radial-07.json) · [恢复阶段独立复核](../campaign/independent_resume_review.md)

对 1,002 个 Train 切换拍的独立几何审计也发现连续性不足。硬 M1 的额外步量 P95 / 最大值为 22.35 / 29.63 mm，三个预登记平滑候选最低为 17.33 / 23.05 mm。冻结门要求 P95 ≤5 mm 且最大值 ≤15 mm。此次没有选择候选或读取 Validation。 [连续性审计数据](../campaign/evidence/m1-continuity-08.json)

后续获准的 DCR-008 全量候选对照使用另一个固定原生命令提供器，保留原 M1 评分。308 个分区全部处理，候选 Train / Validation 主动覆盖为 37.68% / 34.96%，未达到冻结的 85% 数值门。26 个实际静态 Servo 配对中 20 个满足改善和功能条件。DCR 进入人工审查，签名参考契约与 G-A 保持原状态。 [DCR-008 候选验证与证据](DCR-008_causal_nominal_candidate_validation.md)

两份指定 MCAP 通过只读身份、schema 和记录顺序检查，共 286,327 条消息。START、模式和初态的 payload 尚未核验，实际 T0/T1/T2 闭环未运行。 [录包预检](../campaign/evidence/mcap-inventory-01.json)

## 首次启动运行与当时资格

| 用户验收项 | 首次启动结果 |
|---|---|
| RMB | 签名 HMB 可读取，24/8 母源，308 分区，116,613 计划拍。V1-A 正式范围未产完。首次 native 启动因库符号缺失失败，经确认切换到实际 Teleop vendor 库后，小样本处理 250 计划拍、247 映射有效、247 左臂 IK 成功、右臂 0 成功，双侧合法名义参考 0。两个前缀各 99 字段一致。硬捏合切换最大额外步量 17.77 mm，超过 15 mm 门。该候选为 SMOKE_ONLY。 |
| 环境与实际物理采样 | 两次单环境 CPU MuJoCo 烟测各写出 136 个控制转移，记录有名 54 轴正反响应、B0 零残差、回退和 pre-reset 检查为通过。两个进程均退出 135，记录只作诊断。物理烟测 272 拍全部计入首段训练资源。实际并行 4096 或降级配置仍无资格。 |
| 训练和更新 | 累计 learner iterations 0，imagined transitions 0，WM/Actor/Critic updates 0。GPU 完成过有界 seeded Torch 与 Warp kernel，打印了有效结果，进程退出 135。G-C0 无证据通过。尚无实际训练的 checkpoint 或最佳 Actor。 |
| 真值精度 | B0 与 Policy 的比较、静态、原速动态、困难段、S5/D10、功能保持与 D0 全部 NOT_RUN。 |
| 自主修正 | 保留旧失败，修复 Marvin 生产库链接，增加实际逐侧诊断和可达性探针。独立 Reviewer 发现 C0 读取未传 G-A 审核收据及环形 Replay 无法重建已淘汰的前缀。已修正源码并增加负控。后续实际 Ubuntu 回放须重新运行。 |
| 部署 | 两包身份只读校验成功。T0/T1/T2、实际 Teleop 回归与运行时等级均为 NOT_RUN。新增受限只读 MCAP 预检工具。 |
| EgoPro | 未触发，Teleop-only 硬门待通过。 |
| 发布等级 | 无 Policy 资格，实际机器人权限、正式物理评分和独立 Final Test 保持阻塞。 |
| 监控 | Ubuntu 绑定 `http://127.0.0.1:8765`。GET 首页 HTTP 200，API 返回实际 Campaign 和资源状态。初次服务 PID 141491，重启到 PID 143800 后，Windows 浏览器通过 SSH 转发确认已连接，能读取实际 BLOCKED 状态与 272 已用物理拍。 |

[运行证据与 SHA](../campaign/initial_run_evidence.json)

本机源码检查通过 27 项数据、Replay、C0 接口和原生 MuJoCo 测试，4 项录包预检测试，以及 6 项面板测试。原生 MuJoCo 测试使用同一模型和网格的临时 ASCII 路径副本，因为 Windows 原生 XML 加载器无法打开原工作区的中文路径。上述测试只支持已实现接口和工具的检查。实际 Ubuntu 物理及 GPU 进程仍需在可用存储上重新获得资格。

## 主机阻塞及恢复前置

Ubuntu 内核日志在 02:27:54 至 02:36:42 多次报告 `nvme0n1` 的 `critical medium error`，涉及不同扇区。根文件系统与现有依赖、HMB、Campaign、checkpoint 路径共用 Kingston NVMe。仅有的其他大容量挂载为易失 `/dev/shm`。已备份关键烟测证据，停止高 I/O、正式训练和持久化大量新产物。原工作树及已有训练模型未被改写。

恢复工作已定位并隔离受损依赖文件，下载的相同版本位于易失目录。持久盘介质错误仍见于内核日志，原始数据和关键证据保留原路径及核验摘要。后续 GPU 作业启动前须复核依赖来源和进程退出，持续保存关键日志。Campaign 台账保存实际资源扣账和阶段状态。

Ubuntu 复跑使用独立目录 `/dev/shm/s05-resume-490770e`。经实际退出验证的环境包含 NumPy 2.5.4、SciPy 1.18.1、PyTorch 2.11.0+cu130、MuJoCo 与 MuJoCo Warp 3.12.0、Warp 1.17.0 和 `nvidia-cusparselt-cu13` 0.8.0。隔离目录在设备重启后需重新建立。把修复后的 cuSPARSELt 同时置于 Python 的包搜索路径和动态链接器搜索路径，才能避免应用继续映射旧库。保持每次 Run 的实际命令和退出记录，后续在相同隔离环境复核。

## 本机面板

Ubuntu 访问地址为 `http://127.0.0.1:8765`。另一台电脑可通过已授权的 SSH 回环转发访问。

```bash
ssh -N -L 18765:127.0.0.1:8765 ubuntu@100.95.125.4
```

然后打开 `http://127.0.0.1:18765/`。Ubuntu 隔离目录中的服务使用 `scripts/s0_5_campaign_dashboard/server.py --campaign-root /home/ubuntu/dexgym_data/s0-5/campaign/S05-V1-20261127-490770e --port 8765 --tunnel-port 18765`，tmux session 为 `s05-v1-dashboard`。实时指标来自已记录运行。 [面板启动、停止与恢复说明](../../../scripts/s0_5_campaign_dashboard/README.md)
