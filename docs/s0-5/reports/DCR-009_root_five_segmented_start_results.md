# DCR-009 第 5 根原中心 Home→START 分段诊断

状态为 **DIAGNOSTIC_ONLY / START_NOT_REACHED**。本轮只在隔离 MuJoCo 中冷启动一次第 5 根 `capture:b63bc72d2e8b2ee395e966f6` 的分段预定位。原方案来自同一母源已归档的零偏移试次。共同中心、冻结人体输入、注册腕目标、工程 TCP 目标及双模型扫掠配置一致，两个目标矩阵逐元素相同。采用已核验的 MuJoCo 虚拟坐标，腕掌外参仍为 UNKNOWN。两次的历史回放输入均为 0，原始 HMB 分母和正式 85%／70% 门保持有效，本报告不计入资格。 [执行约束](../campaign/approvals/DCR-009_root_five_segmented_start.md) · [配对记录](../campaign/evidence/dcr009-root-five-segmented/comparison.json)

## 原方案的关节与手部阻塞

最先出现并未恢复的限位是左臂 J6（`Joint6_L`）正向，在 Teleop 序号 211。候选 52.02078°，上一条命令 51.99922°，实测约 49.55°。原生物理上限为 60°，控制器保留 8° 后的上限为 52°。候选超出控制裕量 0.02078°，右臂同拍仍为 `OK`。上一拍同类内部探测经回退恢复。看门狗在预定位 1.303 s 记录 `JOINT_LIMIT`。 [原生左右臂诊断摘录](../campaign/evidence/dcr009-root-five-segmented/original-terminal-arm-diagnostics.jsonl)

从原日志还原同一缓存和证明输入后，左手首次拒绝序号 3、右手首次拒绝序号 4，均首先卡在实际加载的 MuJoCo 模型。原样证明复核没有前缀分歧。下表是首拒所在分支第 7 层、第 8 次探测中，中心间隙较保守阈值缺口最大的同手几何对。

| 手 | 首拒 | 代表几何对 | 探测中心间隙 | 该子区间的证明所需间隙 | 该点未获证明的几何对 |
| --- | ---: | --- | ---: | ---: | ---: |
| 左手 | 3 | 食指远节 ↔ 中指远节 | 19.41 mm | 52.15 mm | 70 |
| 右手 | 4 | 无名指远节 ↔ 小指指尖传感器框 | 8.05 mm | 61.84 mm | 105 |

中心采样间隙为正。第 7 层的保守运动包络仍不足以对整个区间出具许可，两次结果均为 `SWEEP_UNPROVEN`。逐对探测的最小中心间隙在实际加载模型上为左 2.337 mm、右 2.335 mm，在签名模型上为左 1.004 mm、右 1.187 mm。这些见证不支持已发生接触的结论。守卫仍使用原有 25 ms、最大深度 7、至多 63 次细分查询及双模型校验。 [逐对扫掠证明](../campaign/evidence/dcr009-root-five-segmented/original-hand-sweep-witness.json)

## 分段实测

双腕在原生 IK 请求前以不超过 60 mm/s、0.30 rad/s 的速度推进，手从上一条实际发布命令以不超过 0.60 rad/s 向原 Wuji 目标推进。名义目标保持，所有提议仍经原生限位和同一 54 轴扫掠守卫。两次均未观察到 START 交接。请求分母是本试次真实提交给守卫的组请求，与 HMB 资格无关。

| 指标 | 原方案 | 分段 |
| --- | ---: | ---: |
| 左手扫掠许可 | 0/59 | 6/133（4.51%） |
| 右手扫掠许可 | 0/59 | 4/133（3.01%） |
| 双臂组扫掠许可 | 279/279 | 731/732 |
| 同时段左腕 TCP 位置 P95 | 368.4 mm | 385.8 mm |
| 同时段右腕 TCP 位置 P95 | 305.7 mm | 339.2 mm |
| 同时段最差腕 TCP 姿态 P95 | 72.1° | 78.9° |
| 最后可用实测左／右腕 TCP 位置误差 | 237.9／168.6 mm（1.228 s） | 243.5／219.6 mm（2.928 s） |
| 看门狗停止 | 1.303 s | 2.950 s |
| START 交接 | 未观察到 | 未观察到 |

同时段统计是 0–1.30 s 内 44 个双方都有不同实测反馈的 20 ms 时间桶，无跨桶补帧。最后一行误差的两个采样时刻不同。分段后 731 条获许可的双臂原生发布有 727 条被 Python 订阅器观测，缺失 4 条保留为观测缺口。没有满足时序和同一发布证明的双手原 Wuji 终点误差记录，两侧仍是 UNKNOWN。 [配对记录及输入身份](../campaign/evidence/dcr009-root-five-segmented/comparison.json)

分段期间左 J6 原始 IK 仍请求 52.02729°，左臂保持而右臂前进。随后组命令 732 对应证明 998，守卫返回 `SWEEP_UNPROVEN`，发布标志为 0，证明耗时 2.400 ms。该提议只改变右臂 7 轴，左 J6 为 0.907566687 rad，仍在软件裕量内。控制器锁存 `COMMAND_STEP`，5.757 ms 后看门狗记录 `DRIVER_NOT_COMMANDABLE`。因此左 J6 裕量和最终扫掠拒绝是两次可区分的事件。原生 STOP 已回执，无 STOP 后观测到的命令，隔离 ROS 域已清空。 [证明 998](../campaign/evidence/dcr009-root-five-segmented/segmented-arm-terminal-sweep.json) · [事件序列](../campaign/evidence/dcr009-root-five-segmented/segmented-terminal-events.jsonl) · [原生摘要](../campaign/evidence/dcr009-root-five-segmented/segmented-terminal-native.txt)

## 录像和证据身份

并排视频左边为原方案、右边为分段，使用 0.22–1.22 s 的逐帧同时间采样，4 倍慢放。开始前的人体图是冻结的原 HMB START，机器人图由同一头时间戳的 54 轴实测反馈计算 FK。1.18 s 图中目标双腕间距为 565 mm，原方案 328 mm，分段 441 mm。间距单项更接近目标，但两侧腕位置 P95 在配对样本上均变大。另有分段末端 2.55–2.91 s 的实测短片。 [原方案渲染回执](../campaign/evidence/dcr009-root-five-segmented/original-render-receipt.json) · [分段渲染回执](../campaign/evidence/dcr009-root-five-segmented/segmented-render-receipt.json) · [末端渲染回执](../campaign/evidence/dcr009-root-five-segmented/segmented-end-render-receipt.json)

- 运行提交 `0b2bbef`，统计和渲染提交 `189d287`。唯一新试次退出后归档了实际隔离启动、守卫及回放命令。 [运行回执](../campaign/evidence/dcr009-root-five-segmented/run_receipt.json)
- 冻结计划 SHA-256 `3c0268291bdb8916a9a520d3fa73a1340fffd7f9e8a7b7364a1ef9950f9a0004`；原始实测轨迹 `6eaae4ff11202a636e5a06999eaa4ff2f763f57708f6aedc55cc731f4602711a`；分段实测轨迹 `c1de42da7d0a48a48261bf509decb86358bac7bcc8ed0496e17e9b8d7c3f91ba`。
- 本地交付压缩包 SHA-256 `05e2deeaff6901560e02f2c8605b815d8c880f8bc87adbd59e055534c3c7b260`，并排视频 `5be4af336d6e20d0843321186f82dc2eb0a9cdd388cdbe3bfae906faf1470f1c`，末端短片 `bc84a421c49c6f5134013db40f206c0f05ef176ae33999901b11514922d114cf`。 [编码核验](../campaign/evidence/dcr009-root-five-segmented/video_probe.json)

## 独立复核

独立 Reviewer 对下载后的原始关节诊断、两模型扫掠见证、守卫请求 998、终止事件、配对统计及三个渲染回执逐一交叉核验。序号 211 是首条记录到的左臂余量拒绝，与稍后的双臂保护停机分别记账。分段的左臂 J6 原始候选超出余量，最终停止的实际提议仅改变右臂。44 个共同时间桶的最差腕位置和姿态 P95 分别增加 17.40 mm 和 6.83°。三个渲染回执的九个媒体及逐帧文件 SHA-256 均核验通过，配对视频两侧各 26 帧，分段末端 10 帧。手部终点原 Wuji 误差仍为 UNKNOWN。 [配对结果](../campaign/evidence/dcr009-root-five-segmented/comparison.json) · [终止事件](../campaign/evidence/dcr009-root-five-segmented/segmented-terminal-events.jsonl)

本轮在第 5 根分段尝试后停止。没有开展其他母源、全量重建、历史动作回放、训练或真机运行。
