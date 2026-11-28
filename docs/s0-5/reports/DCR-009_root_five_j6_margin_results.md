# DCR-009 第 5 根分段 START 的左 J6 余量对照

结果为 **DIAGNOSTIC_ONLY / START_NOT_REACHED**。本次只冷启动一次第 5 根 `capture:b63bc72d2e8b2ee395e966f6`，使用原中心、原目标和原分段速率。基线为上次已归档的 8° 分段实测。仅在独立 MuJoCo 构建的 PREPOSITION 中，将左臂 `Joint6_L` 的软件余量从 8° 改为 5°，其余关节规则、硬限位、驱动程序、双模型扫掠守卫及反馈门未变。比较确认原始输入一致、两套目标矩阵逐元素相同、扫掠配置一致。 [配对证据](../campaign/evidence/dcr009-root-five-j6-margin/comparison.json) · [隔离原生补丁](../campaign/evidence/dcr009-root-five-j6-margin/native-change.json) · [实际输入和二进制 SHA](../campaign/evidence/dcr009-root-five-j6-margin/runtime-assets.sha256)

- **原 J6 拦截已解除。** 在实际观察到的命令中，最高值由 51.99974° 提高至 54.27175°，5° 试次有 59 条观察到的命令超过原来的 52°。 [观察统计](../campaign/evidence/dcr009-root-five-j6-margin/left-j6-observed.jsonl)
- **随后 IK 首次记录到的原始候选限位改为左臂 J2 下限。** 候选 −112.03369°，下限 −112°，前一命令 −111.65843°。该候选拒绝与随后实际整身停机分别记账。 [原生诊断](../campaign/evidence/dcr009-root-five-j6-margin/native-terminal.txt)
- **最终停机由整身扫掠拒绝造成。** 双臂组命令 775、证明 1059 只提议改变右臂七轴，左 J6 保持 54.27175°。实际加载模型许可，签名模型在第 7 层第 8 次检查时无法证明左腕 `l_wrist` 与左无名指近节 `l_ring_finger_proximal_abd` 的扫掠区间安全。采样间隙 1.28652 mm，证明所需间隙 1.30120 mm。复核完整前缀与记录没有分歧，最终命令发布标志为 0。看门狗在 3.12593 s 确认保护停机，基线为 2.95040 s。 [扫掠见证](../campaign/evidence/dcr009-root-five-j6-margin/arm-terminal-sweep-witness.json) · [命令变化](../campaign/evidence/dcr009-root-five-j6-margin/terminal-joint-delta.json) · [终止事件](../campaign/evidence/dcr009-root-five-j6-margin/terminal-events.jsonl)

112 个共同 20 ms 观察桶中，最差腕目标位置 P95 的 5° 减 8° 差为 +0.59 mm。基线左右手通过次数为 6/133 和 4/133，5° 试次为 6/142 和 4/142。两次均未观察到 START 交接，也没有历史动作输入。进程和隔离 ROS 域已收尾，无 STOP 后观测命令。正式 85%／70% 门和原数据分母不计入此次诊断。没有运行其他母源、全量重建、训练或真机。 [运行回执](../campaign/evidence/dcr009-root-five-j6-margin/run_receipt.json)

并排短片左为 8°、右为 5°。两侧各 67 个同时间实测画面，覆盖共同预定位时间 0.22–2.86 s，慢放两倍。下方数字为各自看门狗停机时刻。视频 SHA-256 为 `be967c0f52c653dee268aaa161af10c28693968742018d14f0d5631cdfd2cd24`。 [8° 渲染回执](../campaign/evidence/dcr009-root-five-j6-margin/8-deg-render-receipt.json) · [5° 渲染回执](../campaign/evidence/dcr009-root-five-j6-margin/5-deg-render-receipt.json)
