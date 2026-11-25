# P00 Review Fix 可复验证据

Run ID `S05-P00-REVIEW-FIX-20261126-CC05F09`。文件包含实际预检输出、内容哈希、负控及历史档案引用。

| 证据 | 角色 |
|---|---|
| [environment.json](environment.json) | 独立 Git checkout、解释器、ROS 依赖、真实模块解析、目录与系统资源 |
| [samples.json](samples.json) | 两种 MCAP 的有限 CDR 读取、包内 decoder、原始 log/publish、拓扑与遥测能力 |
| [historical.json](historical.json) · [Marvin54 现有归档](marvin54_archive_witness.json) | 各旧家族的用途发现；不为任何源签发独立终评资格 |
| [实际命令](commands.json) · [依赖锁](requirements.lock) | 执行与退出状态，实际版本、wheel hash |
| [初次导入失败](initial_environment.stderr.txt) · [首次下载超时](initial_dependency_download.txt) · [下载重试](dependency_download_retry.txt) · [依赖安装结果](dependency_install.txt) | 保留初次失败和恢复路径 |
| [历史 P00 快照](p00_snapshot_manifest.json) | 原始 Git commit 下 38 个文件的 Git blob 与 SHA-256 |
| [Review 证据哈希](review_evidence_manifest.json) · [Gate 引用](review_gates.json) | 当前不可变运行证据与矩阵索引 |
| [验证结果](review_validation.txt) · [修改清单](review_change_list.txt) | 一致性、历史回放与精确变更范围 |
| [预检](readiness_probe.py) · [启动 wrapper](run_readiness.sh) · [负控测试](test_review_checks.py) | 可重跑审计工具 |

采用 [执行基线](../../EXECUTION_BASELINE.md) 的 shell wrapper 和只读预检脚本，输出到新的 audit 临时目录。每个模式执行退出码和 stderr 独立保存。成功退出表示探查按预定范围结束，文件内部的 decoder 错误和缺失字段各自保留。

`python3 -B docs/s0-5/evidence/P00/check_documents.py --history-only --require-history` 从 Git 对象验证历史 P00，与当前状态无关。省略 `--history-only` 另外检查当前结构、哈希、引用和转换。`python3 -B docs/s0-5/evidence/P00_review_fix/test_review_checks.py` 运行负控。检查只验证证据一致性，P00 修订的技术和授权裁定由人工审核。[人工审查报告](../../reports/P00_review_fix.md)
