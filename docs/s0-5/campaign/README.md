# S0-5 V1 Campaign 执行登记

设计基准为 v2.0.1。新授权和资源/节奏覆盖使用独立文件登记，冻结候选及其历史 SHA 仍按原提交复验。 [资源调整](../dcr/DCR-007_campaign_training_scale_monitoring.md) · [执行配置](execution_config.json)

持久证据根为 `/home/ubuntu/dexgym_data/s0-5/campaign/S05-V1-20261127-490770e`。主代理通过 Unix 文件锁更新 `ledger.json`，每次状态变更附 SHA 链 `events.jsonl`。阶段运行目录单独保存命令、PID、日志、过程结果与退出。原 HMB 归档只读，仿真工作树与用户工作树分开。训练阶段需按先前冻结的技术门和独立证据推进。

一次可恢复的服务器作业须有独占 Run ID 和实际退出码。进度源包括真实进程、训练日志和资格评估文件。未执行值留空或标明 NOT_RUN。
