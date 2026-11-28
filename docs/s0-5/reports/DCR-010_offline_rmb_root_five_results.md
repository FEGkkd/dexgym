# DCR-010 第 5 根离线 RMB 诊断和 V1-B 准备

DCR-010 提议将离线数据、MuJoCo 和 SuperMimic 设为当前主线，异频实时证明、通信、驱动辨识、真实标定、HIL 和实物安全进入独立 Track B。本轮初始 GitHub 基线为 `548f04195d9f3815cc522d0f7f76c95b3bee8c46`。设计保持 `PENDING_HUMAN_REVIEW`，执行证据标记 `DIAGNOSTIC_ONLY`。 [DCR-010](../dcr/DCR-010_offline_rmb_first.md) · [v2.2 实施规格](../specs/V2_2_offline_rmb_track.md)

## 实际实现和证据边界

离线工具读取已签 50 Hz HMB，单次锁定两腕共同 START，直接调用真实 Marvin C++ Teleop 内核及 Wuji C SDK。每拍计算五次 4 ms 名义提案，每次合法下发后用签名模型做两个 2 ms 实际 MuJoCo 子步。该实施记录原始目标、独立形态目标、候选关节请求、合法名义参考、已下发及实际应用动作、Servo、反馈、净空、原始分母和故障。守卫使用 V2 连续证明，沿用签名仿真模型的 0.500 mm 净空、0.002 mm 数值余量、关节/耦合限位和实测物理检查。原 ROS 源事件和驱动物理链路适用范围另列。 [构建入口](../../../scripts/s0_5_campaign_teleop/build_offline_rmb_diagnostic.py) · [锁步执行](../../../scripts/s0_5_campaign_teleop/offline_rmb_lockstep.py) · [原生桥验证](../campaign/evidence/dcr010-offline-rmb/native-nominal-preflight.json)

签名与 Teleop 模型的命名腕 TCP 在既往 105 个 FK 探针相符。签名与 Teleop 的世界手部地标最大差异约左 6.703 mm、右 5.817 mm。本轮以签名模型统一仿真、地标和工程评分。第 5 根共有 4,801 个 START 后计划拍能得到完整工程形态目标，3 拍缺测。原双腕间距和相对 SE(3) 保持至浮点误差。固定虚拟轴、共同偏航、代理腕掌安装均属仿真工程假设，人体真实解剖腕掌外参仍为 `UNKNOWN`。 [几何审计](../campaign/evidence/dcr010-offline-rmb/root-five-geometry-audit.json)

## 两次有界尝试

初次执行 `056be67` 走完双腕预定位。右手无名指首个推进阶段于仿真 1.198 s 触发物理约束。第 46 轴实测速度 5.01918 rad/s，超过 5 rad/s。该指令的连续几何证明通过，失败子步实测净空 0.95687 mm、穿透 0。该次 4,804 个历史计划拍全部完成数据处理，实际历史物理拍为 0。初次总回执的末尾故障字段被分区纳秒级时轴判定覆盖，原始 START 事件保存了真实首因。 [初次回执](../campaign/evidence/dcr010-offline-rmb/run01-receipt.json) · [初次终止事件](../campaign/evidence/dcr010-offline-rmb/run01-terminal-start.jsonl)

第二次执行 `c2b5720` 把 START 的手指推进上限设置为 0.015 rad／4 ms，保留原物理速度、原始目标及历史指令配置。原 HMB 的分区边界相邻时戳最大偏差 +70 ns、最小 −14 ns，累积偏差 ≤96 ns。使用固定 +1 μs 的因果网格偏移，保留原时戳，按原 20 ms 控制周期处理；大于该容差的原始偏差才判定时间断裂。

11 个 START 阶段均通过，前置运动耗用 3.780 s 仿真时间，左右腕工程 TCP 位置误差 0.280／0.252 mm，最终手组实测关节到目标最大差 0.0233 rad。随后完整执行 4,804 个 20 ms 历史间隔，共 48,040 个实际子步、96.080 s 仿真时间。原录包 START 后区间为 96.068267 s，差异来自末个 HMB 控制间隔按完整 20 ms 执行。CPU 处理约 299 s。 [第二次回执](../campaign/evidence/dcr010-offline-rmb/run02-receipt.json) · [START 结果](../campaign/evidence/dcr010-offline-rmb/run02-start.json)

| 指标 | 第二次实际结果 |
|---|---:|
| 全母源／START 后计划拍 | 4,844／4,804 |
| 处理完成／物理 10 子步完成拍 | 4,804／4,804 |
| 确有合法主动名义参考 | 146／4,794，3.05% |
| HMB 五帧候选中有完整合法主动历史 | 139／4,708，2.95% |
| 名义或证明保持拍／输入保持拍 | 4,648／10 |
| 4 ms 提案有连续证明／未证明 | 1,115／22,905 |
| 每条下发均有证明的 4 ms 子周期／其中保持周期 | 24,020／22,955 |
| 仿真实际接触拍／已确认穿透记录 | 0／0 |
| 工程指尖目标未知拍 | 3 |
| 严格工作空间不可达 | 未独立证明，保留未知 |

22,905 次未证明全部达到离线深度 18 上限，均落在签名模型的右拇指与右食指末端传感器碰撞几何对。首次出现于 HMB 5.060 s（START 后 4.260 s），请求样点最小间隙 0.92377 mm。三个导出点的间隙分别为上拍实测 2.9074 mm、保持指令 2.7262 mm、新提案 2.4613 mm。这些端点不足以为混合关节状态包围盒内的全部姿态提供证明。拒绝后该拍使用另行获得许可的保持指令。全历史实际样本最小几何间隙 1.1706 mm。 [逐请求证明及覆盖核对](../campaign/evidence/dcr010-offline-rmb/run02-verified-audit.json) · [碰撞对和端点复算](../campaign/evidence/dcr010-offline-rmb/run02-pair-diagnosis.json)

共同工程手目标的每拍最差指尖误差在 4,801 个有目标和完整实测的拍中为 P50/P95/P99 148.70/234.07/318.83 mm。146 个主动拍为 54.74/57.56/76.60 mm。完整评分域的左右腕 TCP 位置 P95 为 76.86／83.67 mm，姿态 P95 为 59.25°／64.40°。评分用每拍末端实测对同拍开始的独立目标，完整结果包含已保留的失败区间。三栏视频中的人体输入与目标按原时间轴展示，实测栏显示保持时机器人确实停留的位置。三栏分别标注原 Tracker、注册腕代理和机器人 TCP，双腕距离不变性在注册腕代理映射前后核验。 [完整三栏视频](../campaign/evidence/dcr010-offline-rmb/root-five-offline.mp4) · [评分和活动分母](../campaign/evidence/dcr010-offline-rmb/run02-verified-audit.json) · [视频回执](../campaign/evidence/dcr010-offline-rmb/run02-video.json)

## 下游判定

第 5 根技术门为 `BLOCKED`。本实现一次联合提交 54 轴，单一近邻对未通过就保持整个指令。下一项最小对照是将现有 Teleop 的臂组、左手、右手所有权带入离线提交阶段，对每组仍用整体状态盒做证明，记录阻塞组和无关组是否可以独立合法推进。先用已封存的第一个未证明请求验证，保持目标、两类时间、签名模型和安全阈值。当前证据不足以启动六个 Train 母源的物理验证，正式全量 RMB 未授权；85%／70% 正式口径与全部历史分母保持。

V1-B 的合成测试在 Ubuntu 完成 40 条真实 MuJoCo transition，每拍 10 子步，五帧观测维数 8515，动作维数 54。SuperMimic 成功执行八步 World Model 推理，优化器更新和 checkpoint 均为 0。原物理 smoke 136 个 transition，原任务和 Replay 等回归 37 项及 5 个子检查通过。生产入口拒绝合成数据。GPU 对等尚未运行，探测时共享 GPU 使用率 96%。正式训练还需要合格 RMB、正式意图评分与数据准入、G-B CPU/Warp 对等和资源预算、独立 G-C0 与训练授权。 [V1-B 回执和运行命令](../campaign/evidence/dcr010-offline-rmb-first/v1-b-execution.json) · [独立审查](DCR-010_offline_rmb_independent_review.md)

## 复现

两次原始产物保存在 Ubuntu Campaign `S05-V1-20261127-490770e` 的 `runs/DCR010-OFFLINE-RMB-ROOT05-01` 和 `runs/DCR010-OFFLINE-RMB-ROOT05-02`。运行命令、固定资产和完整原始 trace SHA 见封存证据。实际执行代码早于后续保留 Segment 边界掩码的修订；只读复核检查已运行结果的活动速度 mask 在分区内跨 Segment 次数为 0。 [运行命令和身份](../campaign/evidence/dcr010-offline-rmb/root-five-command.md) · [首次 SHA](../campaign/evidence/dcr010-offline-rmb/run01-raw.sha256) · [第二次 SHA](../campaign/evidence/dcr010-offline-rmb/run02-raw.sha256)
