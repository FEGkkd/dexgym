# DCR-009 第 5 根同源时间轴回放与 STOP 核验

范围为独立 MuJoCo 的一次第 5 根诊断，母源 `capture:b63bc72d2e8b2ee395e966f6`。按上一次共同偏航和先稳双腕再逐指的方式进入 START，工程腕部目标、原 Wuji 手型和双模型整身扫掠守卫均保留。腕手完成 11 段驻留后，在 17.361 s 记录实际 `START_REACHED`。进入历史段后，沿用原生实时命令通道和历史段的原限位策略。没有其他母源、训练、真机、历史命令回灌或失败后的重启。 [实际运行回执](../campaign/evidence/dcr009-root-five-history/run_receipt.json) · [二进制及资产身份](../campaign/evidence/dcr009-root-five-history/runtime-assets.sha256)

## STOP 证据

新试次结果为 `STOP_BOUNDARY_PASS`。原生控制器在发布共用的锁内撤销旧会话，并在服务回应前写入终止序号和原生单调时间。审计逐项区分边界前发布、边界后才被观察器接收的旧命令，以及边界后实际新发布的命令。仅在原生发布、完整守卫载荷、同时间戳反馈、序号及会话身份都能匹配时，才把迟到消息归为旧命令。新的原生发布或无法证明身份的迟到观测仍报失败，即使观察器漏收也会检查原生日志。本次边界后新发布和迟到观测均为 0，隔离 ROS 域已清空。 [STOP 审计](../campaign/evidence/dcr009-root-five-history/stop-audit.json)

旧试次中，回执后约 8.441 ms 才收到的臂组命令 4290 与先前许可 5834 的载荷匹配。旧归档缺少新原生边界和完整同拍身份凭据，原 `ABORTED` 保留。本次运行前 7 项合成检查通过，覆盖已确认的迟到旧包、边界后新发布、证据缺失和带守卫的阶段关闭。最初的预热调用桩兼容性检查在仿真启动前失败并修复，因此没有增加实体回放次数。 [旧归档审计](../campaign/evidence/dcr009-root-five-history/archived-stop-audit.json) · [合成预检](../campaign/evidence/dcr009-root-five-history/preflight-tests.txt)

## 回放和跟踪

按冻结 START 从录包 HMB **0.80 s** 进入历史段，计划时长约 **96.068 s**。实际回放到 **2.6466 s** 时保护停机。已发包 **1,271/46,112**，全部匹配录包的顺序、原时间戳相对关系及原始 CDR 哈希，未发送余下 44,841 包。原母源 4,844 拍，START 后计划域 4,804 拍，其中 130 拍获得同时间戳、因果对齐的 54 轴反馈并计分，3 拍无合格配对反馈，4,671 拍位于未播放的原始后缀。误差只评价同一工程 TCP 口径下已实际播放的计分前缀。 [完整分母与前缀跟踪摘要](../campaign/evidence/dcr009-root-five-history/history-summary-compact.json) · [终止和时基](../campaign/evidence/dcr009-root-five-history/terminal-and-timing.jsonl)

| 前缀指标 | 左腕 | 右腕 |
| --- | ---: | ---: |
| 工程 TCP 位置 P95 | 0.760 mm | 0.925 mm |
| 工程 TCP 朝向 P95 | 0.473° | 0.600° |

在停止处，臂组第 4870 条提议对应证明 6985，两臂的关节目标均有变化。未改变的整身守卫首先在实际加载模型的**右无名指远节与右小指远节**间无法完成扫掠证明。该采样点间隙 **5.949 mm**，该子区间证明需要 **6.299 mm**，细分查询用尽后拒绝，未发布该提议。这个正间隙是保守证明不足的见证。系统锁存 `COMMAND_STEP` 并进入保护停机。已备案的原请求及完整前缀复放无差异。 [整身守卫见证](../campaign/evidence/dcr009-root-five-history/terminal-sweep-witness.json) · [未发布的提议](../campaign/evidence/dcr009-root-five-history/terminal-step.json)

并排短片左为录包对齐后的腕目标，右为同一时间轴的 MuJoCo 实测。时间支持为回放后 0–2.64 s，67 帧里 65 帧具备反馈。视频 SHA-256 `ecdd6bb3a0ce5fe2a9085460ea09a93461b562c7155a29e63841c2fc69c90c80`。本次技术状态为 `HISTORICAL_REPLAY_GUARDED_STOP`，全段完成和正式资格仍未通过。 [渲染回执](../campaign/evidence/dcr009-root-five-history/render-receipt.json)
