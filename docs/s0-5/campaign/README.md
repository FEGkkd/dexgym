# S0-5 V1 Campaign 执行登记

设计基准为 v2.0.1。新授权和资源/节奏覆盖使用独立文件登记，冻结候选及其历史 SHA 仍按原提交复验。 [资源调整](../dcr/DCR-007_campaign_training_scale_monitoring.md) · [执行配置](execution_config.json)

持久证据根为 `/home/ubuntu/dexgym_data/s0-5/campaign/S05-V1-20261127-490770e`。主代理通过 Unix 文件锁更新 `ledger.json`，每次状态变更附 SHA 链 `events.jsonl`。阶段运行目录单独保存命令、PID、日志、过程结果与退出。原 HMB 归档只读，仿真工作树与用户工作树分开。训练阶段需按先前冻结的技术门和独立证据推进。

一次可恢复的服务器作业须有独占 Run ID 和实际退出码。进度源包括真实进程、训练日志和资格评估文件。未执行值留空或标明 NOT_RUN。

## DCR-008 候选验证

该轮授权覆盖候选试验。原精确名义方案和候选方案分别封存。配对统计使用全部原始计划帧及冻结 M1 评分。代码与参数在读取候选 Validation 结果前锁定。候选数据保存在 Campaign 的独立 Run 目录，无训练资格。 [DCR-008](../dcr/DCR-008_causal_teleop_nominal_reference_candidate.md) · [独立预登记](nominal_candidate_independent_protocol.md)

Ubuntu 原始全量对照位于 `runs/DCR-008-full-paired-01/data`。同目录存放配对汇总和因果前缀收据，每个分区的压缩数据带独立 SHA-256。全量原精确基线位于 `runs/DCR-008-exact-full-baseline-01/bank`。两次运行目录均保留启动命令、日志、进程身份和退出码。

原生阶段复用固定 Teleop Git 的纯 C++ 控制流程。原生桥、原精确基线和静态 Servo 诊断均有独立的 Run ID。只有独立复核和人工批准后，才能冻结正式参考契约并另行申请下游技术门。
