# S0-5 研发监控

面板使用 Python 标准库，只读取指定 Campaign 目录，并绑定 Ubuntu 的 `127.0.0.1`。页面每五秒重新读取已落盘产物。服务只提供读取路径。

## Ubuntu 启动与访问

先用 `ss -ltn` 检查端口。从 DexGym checkout 根目录启动。

```bash
CAMPAIGN_ROOT=/home/ubuntu/dexgym_data/s0-5/campaign/S05-V1-20261127-490770e
python3 -B scripts/s0_5_campaign_dashboard/server.py \
  --campaign-root "$CAMPAIGN_ROOT" --port 8765
```

本机地址为 `http://127.0.0.1:8765`。以下 SSH 转发命令在访问端运行。

```bash
ssh -N -L 127.0.0.1:8765:127.0.0.1:8765 ubuntu@100.95.125.4
```

如果访问端 8765 已占用，服务增加 `--tunnel-port 8766`，SSH 使用 `-L 127.0.0.1:8766:127.0.0.1:8765`。以前台 `Ctrl-C` 或对应 service unit 停止面板，通过同一根目录重启即可重新读取历史。面板进程应与训练独立运行，并单独保存启动命令、PID 和日志。

## 输入和状态

| 固定路径 | 读入 |
|---|---|
| `ledger.json`, `events.jsonl` | Campaign ID、Git、阶段、技术门、许可和预算 |
| `dashboard/status.json`, `dashboard/agent_status.json` | 当前运行、PID、heartbeat、Agent、资源告警 |
| `dashboard/data.json`, `dashboard/world_model.json`, `dashboard/deployment.json` | RMB 与物理资格、模型验证、部署与 EgoPro |
| `dashboard/training_metrics.jsonl` | `learner_iteration`、`recorded_at`、`metrics`，各更新、实际采样与想象计数分别记录 |
| `dashboard/evaluation_metrics.jsonl` | `evidence_kind`、`policy`、`learner_iteration`、`checkpoint_sha256`、`metrics` |
| `dashboard/iteration_history.jsonl` | iteration、checkpoint、审查意见、变更、复验、Run 和 Git |

评估记录经独立评估程序落盘后才能进入曲线。`evidence_kind` 标记为 `ACTUAL_MUJOCO` 的记录用于独立物理曲线，`ACTUAL_TELEOP_MUJOCO` 的用于录包页。`policy` 采用 `B0`、`CURRENT`、`BEST` 表示离线对照。实际录包记录使用 `bag_id` 和 `method`。精度字段包括 `s5_p95_mm`、`d10_p95_mm`、`static_p50_mm`、`static_p95_mm`、`static_p99_mm`、`dynamic_p50_mm`、`dynamic_p95_mm`、`dynamic_p99_mm` 和计分/计划拍数。训练回报独立绘制。缺值显示 `UNKNOWN`，没有运行记录时显示 `NOT_RUN`。时间戳使用带时区的 ISO 8601，浏览器统一按上海时区显示。

当前运行可在 `active_run` 对象或台账最新 `runs` 项中记录 `run_id`、`pid`、`pid_start_ticks`、`process_state` 和 `heartbeat_at`。面板检查进程身份及 60 秒心跳有效期，显示退出、PID 重用或过期。停止训练后仍可查看最后产物。JSON 快照优先采用同目录原子替换。JSONL 读取最近 1 MiB 内最多 450 条有效记录，跳过被截断的行。原始日志保留在 Campaign 目录。硬件查询仅读取 `/proc`、磁盘和 `nvidia-smi`。

## 验证

```bash
python3 -B scripts/s0_5_campaign_dashboard/test_dashboard.py -v
curl --fail --silent http://127.0.0.1:8765/api/snapshot
```

合成测试数据来自临时目录，用于验证文件接口、运行状态及访问边界。
