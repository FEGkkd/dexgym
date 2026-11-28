# DCR-009 第 5 根先稳双腕、再逐指的 START 对照

唯一新增运行是第 5 根 `capture:b63bc72d2e8b2ee395e966f6` 使用原中心和相同共同偏航的 MuJoCo 预定位。基线为已归档的共同偏航、腕手同步推进试次。当前试次保持原工程腕部目标、原 MANUS→Wuji 手型、左 J6 余量 5°、其他关节限位、原有限速、MuJoCo 驱动和双模型整身扫掠守卫。仅在相同守卫前加入逐指目标选择。配对证据核对两次目标、来源和守卫配置，活动期所有手组提议均经过原守卫。 [运行回执](../campaign/evidence/dcr009-root-five-wrist-first/run_receipt.json) · [原生准备回执](../campaign/evidence/dcr009-root-five-wrist-first/native-preparation.json) · [配对统计](../campaign/evidence/dcr009-root-five-wrist-first/comparison.json)

## 姿态结果

- **双腕先稳。** 头 1.949 s 内不改变手指下发目标。双腕各在 10 mm／5° 内稳住，完成 200 ms 新鲜实测驻留后才打开第一根手指。
- **右无名指未被拦截。** 1.949–3.258 s 的右无名指阶段有 59 条改变关节目标的请求，全部通过原守卫并下发。第一条是右手命令 115、证明 669，仅改变 `r_RFJ0–3`，最大单步 0.012 rad。右无名指对原 Wuji 手型完成实测驻留后，依次推进其余九指。保持命令没有计入这 59 条。 [右无名指原生记录](../campaign/evidence/dcr009-root-five-wrist-first/ring-and-last-arm-native.txt) · [逐阶段记录](../campaign/evidence/dcr009-root-five-wrist-first/lifecycle.jsonl)
- **完整 START 姿态实测达标。** 17.142 s 记录 `START_REACHED`，两腕的工程 TCP 位置误差约 0.0023／0.0016 mm，朝向误差约 0.00056°／0.00034°。双手对原 Wuji 目标的最大误差分别约 2.62×10⁻⁵ rad 和 1.19×10⁻⁷ rad，均有对应的守卫许可和实际已发布命令证据。数值来自签名 MuJoCo 模型及工程 TCP 契约。 [实测结果和命令匹配](../campaign/evidence/dcr009-root-five-wrist-first/comparison.json)

## STOP 收尾

自动化结果保留 **ABORTED**。观察器在 STOP 回执后 8.441 ms 才处理到旧会话的最后一条臂命令 4290。它匹配守卫证明 5834 及已记录的原生发布。该证明在回执前 6.180 ms 完成。原生控制器在发布共用的锁内先撤销会话再回复 STOP，同一旧会话的发布记录也早于驱动的待机状态 2.046 ms。独立复核据此确认观察器收到的是先前下发的命令。现有证据未定位消息延迟发生在哪一层，也未确认驱动消费的准确时点。原 STOP 判据未被更改，拥有的进程已停止，隔离 ROS 域已清空。 [STOP 及消息回执](../campaign/evidence/dcr009-root-five-wrist-first/lifecycle.jsonl) · [最后一条臂命令的守卫凭据](../campaign/evidence/dcr009-root-five-wrist-first/selected-sweep-proofs.jsonl) · [驱动待机证据](../campaign/evidence/dcr009-root-five-wrist-first/first-standby.json)

并排短片左为以前的共同偏航试次，停机后画面定格。右为本次先腕后指，画面来自 0.22–17.10 s 的实测反馈。实测 START 交接时刻以运行记录为准。视频 SHA-256 为 `bf97d89403f2d49ba9b8ade5aed87984615277ab6cd157cf5183e804dc74605b`。 [基线渲染](../campaign/evidence/dcr009-root-five-wrist-first/baseline-render-receipt.json) · [分段前段渲染](../campaign/evidence/dcr009-root-five-wrist-first/staged-first-render-receipt.json) · [分段后段渲染](../campaign/evidence/dcr009-root-five-wrist-first/staged-last-render-receipt.json)

仅执行一次单母源 START 尝试。历史回放输入为 0，未运行其他母源、训练、全量重建或真机。正式 85%／70% 资格不计入此轮诊断。运行前 4 项合成测试通过，覆盖阶段次序、拒绝条件和共同偏航几何守恒。 [合成检查记录](../campaign/evidence/dcr009-root-five-wrist-first/preflight-tests.txt)
