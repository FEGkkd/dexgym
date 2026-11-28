# DCR-009 第 5 根单条扫掠放行对照

此次只在隔离 MuJoCo 中复用第 5 根原中心、共同偏航、先腕后指和原时间轴。START 达标后，仿真诊断模式最多放行一条双臂组的 `SWEEP_UNPROVEN` 请求。两套原证明仍运行，明确检测到的碰撞、关节范围及其他失败仍拒绝。只执行一次，未运行其他母源、硬件、全量构建或训练。 [运行回执](../campaign/evidence/dcr009-root-five-one-unproven/run_receipt.json) · [诊断模式身份](../campaign/evidence/dcr009-root-five-one-unproven/guard-mode.json)

- **放行已发生。** 回放 2.6711 s 时，两套模型均返回 `SWEEP_UNPROVEN` 的证明 7020 获得本次唯一例外。原生双臂命令 4894 实际发布，并与观察到的 ROS 命令和精确时间戳匹配。原守卫只读复核确认，现场模型的右无名指与小指远节采样间隙 5.948 mm，所需证明间隙 6.309 mm。实际原守卫以相同前缀核验到此请求时，会以细分查询用尽拒绝。 [放行记录](../campaign/evidence/dcr009-root-five-one-unproven/guard-exception.jsonl) · [原守卫离线复核](../campaign/evidence/dcr009-root-five-one-unproven/strict-counterfactual-proof.json)
- **短时反馈中的两指仍有间隙。** 放行后 100 ms 内采到 13 组同时间戳的 54 轴反馈，包括停机收尾。用实际加载的 MuJoCo 模型计算，右无名指与小指远节的最小间隙约 5.720 mm，全部受检几何对的最小间隙约 2.335 mm。此结论覆盖已观测姿态。 [反馈及间隙](../campaign/evidence/dcr009-root-five-one-unproven/observed-effect.json)
- **下一条命令仍被拦下。** 约 9.145 ms 后，同类拒绝发生在双臂组命令 4895、证明 7023。这次回放到约 2.687 s 时停机，发送 1,288/46,112 包。先前严格试次发送 1,271 包，两次试次的调度及实测时点存在差异。STOP 边界核验通过。 [原生发布及停机日志](../campaign/evidence/dcr009-root-five-one-unproven/native-terminal.txt) · [下一条守卫请求](../campaign/evidence/dcr009-root-five-one-unproven/later_denials.jsonl) · [STOP 回执](../campaign/evidence/dcr009-root-five-one-unproven/stop-audit.json)

初始预检的五项合成检查通过。原守卫离线复核使用请求序列的只读分析副本，只把证明 7020 记录为严格条件原会产生的扫掠不确定，保留本次真实的放行及发布证据。无正式覆盖、整段完成或物理机器人资格。 [预检日志](../campaign/evidence/dcr009-root-five-one-unproven/preflight-tests.txt)
