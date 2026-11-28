# DCR-009 六根整身扫掠诊断结果

运行 ID `HOME-ANCHORED-DIAGNOSTIC-DCR009-01`。六个 Train 母源均通过实测 Home 门并尝试 Home→START，均在 START 达标前停止，共 0 根进入原时间轴历史回放。三段隔离执行台账保持原状，前两段原始的异常退出记录未被改写。六次原生 STOP 后均未观察到迟到命令，启动者所属进程均退出，ROS Domain 213 均清空。[预登记名单与原分母](../campaign/evidence/dcr009-guarded-diagnostic/pilot_plan.json) · [合并回执](../campaign/evidence/dcr009-guarded-diagnostic/summary.json) · [独立复核](DCR-009_guarded_diagnostic_independent_review.md)

| 母源后缀 | 实测停止原因 | 许可并核对的双臂发布 | 未证明扫掠的请求 |
| --- | --- | ---: | ---: |
| `1bc59d07cb650ee219d5adbf` | 原生保护保持 | 159 | 79 |
| `2151546f1c0b35e04218a1bb` | 原生保护保持 | 156 | 79 |
| `2edc372773190c61446b634c` | 原生保护保持 | 168 | 81 |
| `3118754b1d9742a2718048e2` | 原生保护保持 | 136 | 75 |
| `b63bc72d2e8b2ee395e966f6` | 原生双臂命令被关节限位拦截 | 262 | 122 |
| `b70297470e5e9fc8d030706e` | 原生保护保持 | 209 | 87 |

1,090 条双臂 ROS 发布逐条匹配会话、命令序号、正向扫掠证明标识和 14 维位置 payload。523 条无法证明的请求中，左右手各 259 条，双臂 5 条。两手均无关节命令发布。本次新增守卫验证发布前实际状态、前一已下发目标和候选目标构成的双模型 54 轴包络，并在收到同一请求的许可后发布。驱动各物理子步的作用量及 Servo 动态超调未取得资格。未证明的包络保留原结果和原因，不能换算为不可达证明。[发令证据（一）](../campaign/evidence/dcr009-guarded-diagnostic/guard-audit-01.json) · [发令证据（二）](../campaign/evidence/dcr009-guarded-diagnostic/guard-audit-02.json) · [发令证据（三）](../campaign/evidence/dcr009-guarded-diagnostic/guard-audit-03.json)

三栏视频有 143 个原采样显示帧。左栏保留原 Tracker 注册腕与独立 MANUS 局部手型，中栏显示一次共同刚体映射后的注册腕目标，右栏按实测同时间戳 54 轴重建 MuJoCo 机器人。腕掌安装未知。视频期间人体输入为固定锚点。审阅版四倍慢放，各根结尾保持 0.6 s，保留原显示时刻标签。原始视频总长 5.72 s，审阅版 26.48 s，交付媒体包 SHA-256 为 `f1f9cee54bdcb96e09b0b71a0889913aa9fe1e7eab288c019837c54dbf975be6`。

六根原计划 19,854 拍，联合有效 19,649 拍，原签名五帧窗 19,019。签名数据、M0／M1 和各 split 分母原样保留。全部受控运行均无 85%／70% 分子贡献。没有运行全量 RMB、训练或实物。
