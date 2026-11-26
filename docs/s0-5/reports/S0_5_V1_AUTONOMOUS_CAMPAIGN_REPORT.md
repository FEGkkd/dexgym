# S0-5 V1 Autonomous Campaign 实施记录

**当前状态：阶段已启动，后续执行因 Ubuntu 持久盘介质错误及未通过的前置资格阻塞。** 此报告保存本轮真实启动产物与失败，首版策略发布尚未达成。

Campaign ID 为 `S05-V1-20261127-490770e`。冻结设计为 `490770e1df45832d0693380563152bd19eefa3f4`，SHA-256 `cd5bff29a5b5f683fae328898aa9060b40fc4be00e193a22d9bf18cb2d10b60b`。执行配置 SHA-256 `955d80d09ffe674533cf5000e7bd227c3cdd7998cbbfb2737b42c273bc60e822`。DCR-007 范围独立核对通过，此审核只涵盖资源变更。 [执行登记](../campaign/README.md) · [资源审核凭据](../campaign/independent_resource_review.json) · [独立启动复核](../campaign/independent_launch_review.md)

## 实际运行与当前资格

| 用户验收项 | 目前实际结果 |
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

恢复时先在健康持久存储验证依赖实际入口和正常退出。按已冻结参考用右臂 SDK roundtrip、逐侧失败及现有软件的合法投影诊断定位 G-A 问题，先覆盖 Train 样本，随后独立核验近捏合切换、来源、窗口及连续见证。G-B 还需要实际 RMB 接口及 CPU/Warp 独立等价和环境梯度预检，G-C0 还需实际模型资格、S5/D10 和独立审核。当前进度以及失败历史继续由 Campaign 台账保存。

## 本机面板

Ubuntu 访问地址为 `http://127.0.0.1:8765`。另一台电脑可通过已授权的 SSH 回环转发访问。

```bash
ssh -N -L 18765:127.0.0.1:8765 ubuntu@100.95.125.4
```

然后打开 `http://127.0.0.1:18765/`。Ubuntu 隔离目录中的服务使用 `scripts/s0_5_campaign_dashboard/server.py --campaign-root /home/ubuntu/dexgym_data/s0-5/campaign/S05-V1-20261127-490770e --port 8765 --tunnel-port 18765`，tmux session 为 `s05-v1-dashboard`。实时指标来自已记录运行。 [面板启动、停止与恢复说明](../../../scripts/s0_5_campaign_dashboard/README.md)
